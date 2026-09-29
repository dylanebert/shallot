import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { build, type Plugin } from "../app";
import type { State } from "./state";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>, timeout = 5_000): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error(`${label} timed out after ${timeout} ms`)),
            timeout,
        );
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error: unknown) => {
                clearTimeout(timer);
                reject(new Error(`${label} rejected: ${String(error)}`, { cause: error }));
            },
        );
    });
}

async function waitAndValidate(state: State, label: string): Promise<void> {
    const device = state.gpu.device;
    await bounded(`${label} queue completion`, device.queue.onSubmittedWorkDone());
    const error = await bounded(`${label} validation scope`, device.popErrorScope());
    if (error) throw new Error(`${label}: ${error.message}`);
}

function median(values: number[]): number {
    const ordered = values.toSorted((a, b) => a - b);
    return ordered[Math.floor(ordered.length / 2)];
}

test("measure the opt-in eid-map cost against direct eid indexing at full population", async () => {
    let state!: State;
    const plugin: Plugin = {
        name: "TableMapCostProbe",
        features: ["timestamp-query"],
        initialize(current) {
            state = current;
        },
    };
    const app = await build({ defaults: false, plugins: [plugin] });

    try {
        const { class: adapterClass, identity } = state.gpu.adapter;
        console.info(`[gpu-table-perf] adapter class=${adapterClass} identity=${identity}`);
        expect(adapterClass).toBe("real");
        expect(identity.length).toBeGreaterThan(0);
        expect(identity.toLowerCase()).not.toContain("swiftshader");

        const device = state.gpu.device;
        const count = 100_000;
        const batchSize = 128;
        const table = state.table("mapped-eid-index", d.u32);
        const values = new Uint32Array(table.bytes.buffer);
        for (let eid = 0; eid < count; eid++) {
            const slot = table.acquire(eid);
            values[slot] = 1;
        }
        table.markRange(0, count);
        const eidToSlot = table.enableEidLookup();

        const direct = device.createBuffer({
            label: "direct-eid-index-records",
            size: table.buffer.size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        state.own(direct);
        const directValues = new Uint32Array(table.capacity);
        directValues.fill(1, 0, count);

        device.pushErrorScope("validation");
        device.queue.writeBuffer(direct, 0, directValues);
        table.upload();
        await waitAndValidate(state, "100%-population index inputs");
        const initialMapUploadBytes = table.lastMapUploadBytes;
        expect(initialMapUploadBytes).toBe(count * 4);
        const poseBytes = d.sizeOf(d.mat4x4f);
        const instanceBytes = d.sizeOf(d.struct({ transform: d.mat4x4f, color: d.vec4f }));
        const poseMapPercent = (eidToSlot.size / (table.capacity * poseBytes)) * 100;
        const instanceMapPercent = (eidToSlot.size / (table.capacity * instanceBytes)) * 100;
        device.pushErrorScope("validation");
        table.upload();
        await waitAndValidate(state, "steady-state map upload");
        expect(table.lastMapUploadBytes).toBe(0);
        console.info(
            `[gpu-table-perf] map bytes=${eidToSlot.size}; representative pose=${poseBytes} B/row (${poseMapPercent.toFixed(2)}% of allocated record bytes); instance=${instanceBytes} B/row (${instanceMapPercent.toFixed(2)}%); initial map upload=${initialMapUploadBytes} B; steady map upload=${table.lastMapUploadBytes} B`,
        );

        const output = device.createBuffer({
            label: "eid-map-cost-output",
            size: count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        state.own(output);
        const shader = device.createShaderModule({
            label: "eid-map-cost-shader",
            code: `
@group(0) @binding(0) var<storage, read> records: array<u32>;
@group(0) @binding(1) var<storage, read> eidToSlot: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
@compute @workgroup_size(64)
fn directEid(@builtin(global_invocation_id) id: vec3<u32>) {
    let eid = id.x;
    if (eid < ${count}u) { result[eid] = records[eid]; }
}
@compute @workgroup_size(64)
fn mappedEid(@builtin(global_invocation_id) id: vec3<u32>) {
    let eid = id.x;
    if (eid < ${count}u) {
        let slot = eidToSlot[eid] - 1u;
        result[eid] = records[slot];
    }
}`,
        });
        const layout = device.createBindGroupLayout({
            label: "eid-map-cost-layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const pipelineLayout = device.createPipelineLayout({
            label: "eid-map-cost-pipeline-layout",
            bindGroupLayouts: [layout],
        });
        const directPipeline = device.createComputePipeline({
            label: "direct-eid-index-pipeline",
            layout: pipelineLayout,
            compute: { module: shader, entryPoint: "directEid" },
        });
        const mappedPipeline = device.createComputePipeline({
            label: "mapped-eid-index-pipeline",
            layout: pipelineLayout,
            compute: { module: shader, entryPoint: "mappedEid" },
        });
        const directGroup = device.createBindGroup({
            label: "direct-eid-index-bindings",
            layout,
            entries: [
                { binding: 0, resource: { buffer: direct } },
                { binding: 1, resource: { buffer: eidToSlot } },
                { binding: 2, resource: { buffer: output } },
            ],
        });
        const mappedGroup = device.createBindGroup({
            label: "mapped-eid-index-bindings",
            layout,
            entries: [
                { binding: 0, resource: { buffer: table.buffer } },
                { binding: 1, resource: { buffer: eidToSlot } },
                { binding: 2, resource: { buffer: output } },
            ],
        });

        const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
        state.own(querySet);
        const resolveBuffer = device.createBuffer({
            label: "eid-map-cost-query-resolve",
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        state.own(resolveBuffer);
        const timestampBuffer = device.createBuffer({
            label: "eid-map-cost-timestamp-readback",
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        state.own(timestampBuffer);
        const sample = async (
            label: string,
            pipeline: GPUComputePipeline,
            group: GPUBindGroup,
            index: number,
        ): Promise<number> => {
            device.pushErrorScope("validation");
            const encoder = device.createCommandEncoder({ label: `${label}-batch` });
            const pass = encoder.beginComputePass({
                label: `${label}-pass`,
                timestampWrites: {
                    querySet,
                    beginningOfPassWriteIndex: 0,
                    endOfPassWriteIndex: 1,
                },
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group);
            for (let i = 0; i < batchSize; i++) {
                pass.dispatchWorkgroups(Math.ceil(count / 64));
            }
            pass.end();
            encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
            encoder.copyBufferToBuffer(resolveBuffer, 0, timestampBuffer, 0, 16);
            device.queue.submit([encoder.finish()]);
            await bounded(
                `${label} batch ${index} queue completion`,
                device.queue.onSubmittedWorkDone(),
            );
            await bounded(
                `${label} batch ${index} timestamp map`,
                timestampBuffer.mapAsync(GPUMapMode.READ),
            );
            const timestamps = new BigUint64Array(timestampBuffer.getMappedRange());
            const elapsedMs = Number(timestamps[1] - timestamps[0]) / 1_000_000 / batchSize;
            timestampBuffer.unmap();
            const error = await bounded(
                `${label} batch ${index} validation scope`,
                device.popErrorScope(),
            );
            if (error) throw new Error(`${label} batch ${index}: ${error.message}`);
            return elapsedMs;
        };

        for (let warmup = 0; warmup < 3; warmup++) {
            await sample("direct eid warmup", directPipeline, directGroup, warmup + 1);
            await sample("mapped eid warmup", mappedPipeline, mappedGroup, warmup + 1);
        }
        const directTimes: number[] = [];
        const mappedTimes: number[] = [];
        for (let index = 0; index < 5; index++) {
            if ((index & 1) === 0) {
                directTimes.push(
                    await sample("direct eid indexing", directPipeline, directGroup, index + 1),
                );
                mappedTimes.push(
                    await sample("eid-to-slot map", mappedPipeline, mappedGroup, index + 1),
                );
            } else {
                mappedTimes.push(
                    await sample("eid-to-slot map", mappedPipeline, mappedGroup, index + 1),
                );
                directTimes.push(
                    await sample("direct eid indexing", directPipeline, directGroup, index + 1),
                );
            }
        }
        const directMedian = median(directTimes);
        const mappedMedian = median(mappedTimes);
        const difference = (mappedMedian / directMedian - 1) * 100;
        console.info(
            `[gpu-table-perf] 100%-population dispatch median ms per ${count} rows; direct-eid=${directMedian.toFixed(4)} mapped-eid=${mappedMedian.toFixed(4)} difference=${difference.toFixed(1)}% recordBytes=${table.buffer.size} mapBytes=${eidToSlot.size} mapUploadBytes=${count * 4}`,
        );
    } finally {
        app.dispose();
    }
}, 30_000);
