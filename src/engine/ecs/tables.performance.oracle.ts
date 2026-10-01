import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { createApp, type Plugin } from "../app";
import { u32 } from "../index";
import type { World } from "./state";
import { registration } from "./traits";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

const UPLOAD_RECORD = d.struct({ value: d.u32, tag: d.u32 });
const MEMORY_RECORD = d.struct({ world: d.mat4x4f });

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

async function waitAndValidate(world: World, label: string): Promise<void> {
    const device = world.gpu.device;
    await bounded(`${label} queue completion`, device.queue.onSubmittedWorkDone());
    const error = await bounded(`${label} validation scope`, device.popErrorScope());
    if (error) throw new Error(`${label}: ${error.message}`);
}

function median(values: number[]): number {
    const ordered = values.toSorted((a, b) => a - b);
    return ordered[Math.floor(ordered.length / 2)];
}

async function timestampedPass(
    world: World,
    label: string,
    pipeline: GPUComputePipeline,
    bindings: GPUBindGroup,
    querySet: GPUQuerySet,
    resolveBuffer: GPUBuffer,
    readbackBuffer: GPUBuffer,
    workgroups: number,
    repetitions: number,
): Promise<number> {
    const device = world.gpu.device;
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder({ label: `${label}-timed-batch` });
    const pass = encoder.beginComputePass({
        label: `${label}-timed-pass`,
        timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    for (let i = 0; i < repetitions; i++) pass.dispatchWorkgroups(workgroups);
    pass.end();
    encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 16);
    device.queue.submit([encoder.finish()]);
    await bounded(`${label} queue completion`, device.queue.onSubmittedWorkDone());
    await bounded(`${label} timestamp map`, readbackBuffer.mapAsync(GPUMapMode.READ));
    const timestamps = new BigUint64Array(readbackBuffer.getMappedRange());
    const nanoseconds = Number(timestamps[1] - timestamps[0]) / repetitions;
    readbackBuffer.unmap();
    const error = await bounded(`${label} validation scope`, device.popErrorScope());
    if (error) throw new Error(`${label}: ${error.message}`);
    return nanoseconds;
}

test("measure the opt-in eid-map cost against direct eid indexing at full population", async () => {
    let world!: World;
    const plugin: Plugin = {
        name: "TableMapCostProbe",
        features: ["timestamp-query"],
        initialize(current) {
            world = current;
        },
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });

    try {
        const { class: adapterClass, identity } = world.gpu.adapter;
        console.info(`[gpu-table-perf] adapter class=${adapterClass} identity=${identity}`);
        expect(adapterClass).toBe("real");
        expect(identity.length).toBeGreaterThan(0);
        expect(identity.toLowerCase()).not.toContain("swiftshader");

        const device = world.gpu.device;
        const count = 100_000;
        const batchSize = 512;
        const transformRecord = d.struct({ transform: d.mat4x4f });
        const table = world.table("mapped-eid-index", transformRecord);
        const values = table.bytes;
        for (let eid = 0; eid < count; eid++) table.acquire(eid);
        values.fill(1, 0, count * d.sizeOf(transformRecord));
        table.markRange(0, count);
        const eidToSlot = table.enableEidLookup();

        const direct = device.createBuffer({
            label: "direct-eid-index-records",
            size: table.buffer.size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        world.own(direct);
        const wordsPerRow = table.rowBytes / 4;
        const directValues = new Uint32Array(table.capacity * wordsPerRow);
        directValues.fill(1, 0, count * wordsPerRow);

        device.pushErrorScope("validation");
        device.queue.writeBuffer(direct, 0, directValues);
        table.upload();
        await waitAndValidate(world, "100%-population index inputs");
        const initialMapUploadBytes = table.lastMapUploadBytes;
        expect(initialMapUploadBytes).toBe(count * 4);
        const globalTransformBytes = table.rowBytes;
        const instanceBytes = d.sizeOf(d.struct({ transform: d.mat4x4f, color: d.vec4f }));
        const globalTransformMapPercent =
            (eidToSlot.size / (table.capacity * globalTransformBytes)) * 100;
        const instanceMapPercent = (eidToSlot.size / (table.capacity * instanceBytes)) * 100;
        device.pushErrorScope("validation");
        table.upload();
        await waitAndValidate(world, "steady-state map upload");
        expect(table.lastMapUploadBytes).toBe(0);
        console.info(
            `[gpu-table-perf] map bytes=${eidToSlot.size}; representative GlobalTransform=${globalTransformBytes} B/row (${globalTransformMapPercent.toFixed(2)}% of allocated record bytes); instance=${instanceBytes} B/row (${instanceMapPercent.toFixed(2)}%); initial map upload=${initialMapUploadBytes} B; steady map upload=${table.lastMapUploadBytes} B`,
        );

        const output = device.createBuffer({
            label: "eid-map-cost-output",
            size: count * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        world.own(output);
        const shader = device.createShaderModule({
            label: "eid-map-cost-shader",
            code: `
struct GlobalTransform { transform: mat4x4f, };
@group(0) @binding(0) var<storage, read> records: array<GlobalTransform>;
@group(0) @binding(1) var<storage, read> eidToSlot: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4f>;
@compute @workgroup_size(64)
fn directEid(@builtin(global_invocation_id) id: vec3<u32>) {
    let eid = id.x;
    if (eid < ${count}u) {
        let transform = records[eid].transform;
        result[eid] = transform[0] + transform[1] + transform[2] + transform[3];
    }
}
@compute @workgroup_size(64)
fn mappedEid(@builtin(global_invocation_id) id: vec3<u32>) {
    let eid = id.x;
    if (eid < ${count}u) {
        let slot = eidToSlot[eid] - 1u;
        let transform = records[slot].transform;
        result[eid] = transform[0] + transform[1] + transform[2] + transform[3];
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
        world.own(querySet);
        const resolveBuffer = device.createBuffer({
            label: "eid-map-cost-query-resolve",
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        world.own(resolveBuffer);
        const timestampBuffer = device.createBuffer({
            label: "eid-map-cost-timestamp-readback",
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        world.own(timestampBuffer);
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
        for (let index = 0; index < 15; index++) {
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
            `[gpu-table-perf] 100%-population GlobalTransform dispatch median ms per ${count} rows; direct-eid=${directMedian.toFixed(4)} mapped-eid=${mappedMedian.toFixed(4)} difference=${difference.toFixed(1)}% directSamples=${directTimes.map((time) => time.toFixed(4)).join("/")} mappedSamples=${mappedTimes.map((time) => time.toFixed(4)).join("/")} recordBytes=${table.buffer.size} mapBytes=${eidToSlot.size} mapUploadBytes=${count * 4}`,
        );
    } finally {
        app.dispose();
    }
}, 0);

test("measure struct records against per-field arrays for GlobalTransform and light", async () => {
    let world!: World;
    const plugin: Plugin = {
        name: "TableRecordLayoutProbe",
        features: ["timestamp-query"],
        initialize(current) {
            world = current;
        },
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });

    try {
        const { class: adapterClass, identity } = world.gpu.adapter;
        console.info(`[gpu-table-layout] adapter class=${adapterClass} identity=${identity}`);
        expect(adapterClass).toBe("real");
        expect(identity.length).toBeGreaterThan(0);
        const device = world.gpu.device;
        const count = 100_000;
        const outputBytes = count * 16;
        const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
        world.own(querySet);
        const resolveBuffer = device.createBuffer({
            label: "table-layout-query-resolve",
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        world.own(resolveBuffer);
        const readbackBuffer = device.createBuffer({
            label: "table-layout-query-readback",
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        world.own(readbackBuffer);

        const scenarios = [
            {
                name: "global-transform",
                recordBytes: d.sizeOf(
                    d.struct({
                        position: d.vec4f,
                        rotation: d.vec4f,
                        scale: d.vec4f,
                    }),
                ),
                fieldBytes: [16, 16, 16],
                structShader: `struct Row { position: vec4f, rotation: vec4f, scale: vec4f, };
@group(0) @binding(0) var<storage, read> rows: array<Row>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${count}u) { output[id.x] = rows[id.x].position + rows[id.x].rotation + rows[id.x].scale; }
}`,
                fieldShader: `@group(0) @binding(0) var<storage, read> position: array<vec4f>;
@group(0) @binding(1) var<storage, read> rotation: array<vec4f>;
@group(0) @binding(2) var<storage, read> scale: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${count}u) { output[id.x] = position[id.x] + rotation[id.x] + scale[id.x]; }
}`,
            },
            {
                name: "light",
                recordBytes: d.sizeOf(d.struct({ color: d.vec4f, params: d.vec4f })),
                fieldBytes: [16, 16],
                structShader: `struct Row { color: vec4f, params: vec4f, };
@group(0) @binding(0) var<storage, read> rows: array<Row>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${count}u) { output[id.x] = rows[id.x].color + rows[id.x].params; }
}`,
                fieldShader: `@group(0) @binding(0) var<storage, read> color: array<vec4f>;
@group(0) @binding(1) var<storage, read> params: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < ${count}u) { output[id.x] = color[id.x] + params[id.x]; }
}`,
            },
        ];
        const results: string[] = [];

        for (const scenario of scenarios) {
            const totalBytes = count * scenario.recordBytes;
            expect(totalBytes).toBe(
                count * scenario.fieldBytes.reduce((sum, bytes) => sum + bytes, 0),
            );
            const recordBuffer = device.createBuffer({
                label: `${scenario.name}-struct-records`,
                size: totalBytes,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            world.own(recordBuffer);
            const recordSource = new Uint8Array(totalBytes);
            const fieldBuffers = scenario.fieldBytes.map((bytes, index) => {
                const buffer = device.createBuffer({
                    label: `${scenario.name}-field-${index}`,
                    size: count * bytes,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                world.own(buffer);
                return buffer;
            });
            const fieldSources = scenario.fieldBytes.map((bytes) => new Uint8Array(count * bytes));
            const structOutput = device.createBuffer({
                label: `${scenario.name}-struct-output`,
                size: outputBytes,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            world.own(structOutput);
            const fieldsOutput = device.createBuffer({
                label: `${scenario.name}-fields-output`,
                size: outputBytes,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            world.own(fieldsOutput);

            const measureUpload = async (shape: "struct" | "fields") => {
                const submitTimes: number[] = [];
                const cycleTimes: number[] = [];
                for (let sample = 0; sample < 7; sample++) {
                    device.pushErrorScope("validation");
                    const start = performance.now();
                    if (shape === "struct") device.queue.writeBuffer(recordBuffer, 0, recordSource);
                    else {
                        for (let i = 0; i < fieldBuffers.length; i++) {
                            device.queue.writeBuffer(fieldBuffers[i], 0, fieldSources[i]);
                        }
                    }
                    submitTimes.push(performance.now() - start);
                    await bounded(
                        `${scenario.name} ${shape} upload ${sample + 1} queue completion`,
                        device.queue.onSubmittedWorkDone(),
                    );
                    const error = await bounded(
                        `${scenario.name} ${shape} upload ${sample + 1} validation scope`,
                        device.popErrorScope(),
                    );
                    if (error)
                        throw new Error(`${scenario.name} ${shape} upload: ${error.message}`);
                    cycleTimes.push(performance.now() - start);
                }
                return { submitMs: median(submitTimes), cycleMs: median(cycleTimes) };
            };

            const structLayout = device.createBindGroupLayout({
                label: `${scenario.name}-struct-read-layout`,
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" },
                    },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                ],
            });
            const fieldLayout = device.createBindGroupLayout({
                label: `${scenario.name}-fields-read-layout`,
                entries: [
                    ...fieldBuffers.map((_, binding) => ({
                        binding,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "read-only-storage" as const },
                    })),
                    {
                        binding: fieldBuffers.length,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "storage" },
                    },
                ],
            });
            const pipelineLayout = (layout: GPUBindGroupLayout, label: string) =>
                device.createPipelineLayout({ label, bindGroupLayouts: [layout] });
            const structModule = device.createShaderModule({
                label: `${scenario.name}-struct-read-shader`,
                code: scenario.structShader,
            });
            const fieldsModule = device.createShaderModule({
                label: `${scenario.name}-fields-read-shader`,
                code: scenario.fieldShader,
            });
            const structPipeline = device.createComputePipeline({
                label: `${scenario.name}-struct-read-pipeline`,
                layout: pipelineLayout(structLayout, `${scenario.name}-struct-pipeline-layout`),
                compute: { module: structModule, entryPoint: "main" },
            });
            const fieldsPipeline = device.createComputePipeline({
                label: `${scenario.name}-fields-read-pipeline`,
                layout: pipelineLayout(fieldLayout, `${scenario.name}-fields-pipeline-layout`),
                compute: { module: fieldsModule, entryPoint: "main" },
            });
            const structGroup = device.createBindGroup({
                label: `${scenario.name}-struct-read-bindings`,
                layout: structLayout,
                entries: [
                    { binding: 0, resource: { buffer: recordBuffer } },
                    { binding: 1, resource: { buffer: structOutput } },
                ],
            });
            const fieldsGroup = device.createBindGroup({
                label: `${scenario.name}-fields-read-bindings`,
                layout: fieldLayout,
                entries: [
                    ...fieldBuffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
                    { binding: fieldBuffers.length, resource: { buffer: fieldsOutput } },
                ],
            });
            device.pushErrorScope("validation");
            device.queue.writeBuffer(recordBuffer, 0, recordSource);
            for (let i = 0; i < fieldBuffers.length; i++) {
                device.queue.writeBuffer(fieldBuffers[i], 0, fieldSources[i]);
            }
            await bounded(
                `${scenario.name} initial input upload`,
                device.queue.onSubmittedWorkDone(),
            );
            const pipelineError = await bounded(
                `${scenario.name} pipeline validation scope`,
                device.popErrorScope(),
            );
            if (pipelineError)
                throw new Error(`${scenario.name} layout pipeline: ${pipelineError.message}`);

            const structUpload = await measureUpload("struct");
            const fieldsUpload = await measureUpload("fields");
            const structGpu: number[] = [];
            const fieldsGpu: number[] = [];
            for (let warmup = 0; warmup < 2; warmup++) {
                await timestampedPass(
                    world,
                    `${scenario.name} struct warmup ${warmup + 1}`,
                    structPipeline,
                    structGroup,
                    querySet,
                    resolveBuffer,
                    readbackBuffer,
                    Math.ceil(count / 64),
                    32,
                );
                await timestampedPass(
                    world,
                    `${scenario.name} fields warmup ${warmup + 1}`,
                    fieldsPipeline,
                    fieldsGroup,
                    querySet,
                    resolveBuffer,
                    readbackBuffer,
                    Math.ceil(count / 64),
                    32,
                );
            }
            for (let sample = 0; sample < 7; sample++) {
                if ((sample & 1) === 0) {
                    structGpu.push(
                        await timestampedPass(
                            world,
                            `${scenario.name} struct read ${sample + 1}`,
                            structPipeline,
                            structGroup,
                            querySet,
                            resolveBuffer,
                            readbackBuffer,
                            Math.ceil(count / 64),
                            32,
                        ),
                    );
                    fieldsGpu.push(
                        await timestampedPass(
                            world,
                            `${scenario.name} fields read ${sample + 1}`,
                            fieldsPipeline,
                            fieldsGroup,
                            querySet,
                            resolveBuffer,
                            readbackBuffer,
                            Math.ceil(count / 64),
                            32,
                        ),
                    );
                } else {
                    fieldsGpu.push(
                        await timestampedPass(
                            world,
                            `${scenario.name} fields read ${sample + 1}`,
                            fieldsPipeline,
                            fieldsGroup,
                            querySet,
                            resolveBuffer,
                            readbackBuffer,
                            Math.ceil(count / 64),
                            32,
                        ),
                    );
                    structGpu.push(
                        await timestampedPass(
                            world,
                            `${scenario.name} struct read ${sample + 1}`,
                            structPipeline,
                            structGroup,
                            querySet,
                            resolveBuffer,
                            readbackBuffer,
                            Math.ceil(count / 64),
                            32,
                        ),
                    );
                }
            }
            results.push(
                `${scenario.name},${count},${scenario.recordBytes},${fieldBuffers.length},${structUpload.submitMs.toFixed(4)},${fieldsUpload.submitMs.toFixed(4)},${structUpload.cycleMs.toFixed(4)},${fieldsUpload.cycleMs.toFixed(4)},${median(structGpu).toFixed(1)},${median(fieldsGpu).toFixed(1)}`,
            );
        }

        console.info(
            `[gpu-table-layout] median values; columns=table,rows,recordBytes,fieldBuffers,structWriteBufferSubmitMs,fieldsWriteBufferSubmitMs,structUploadCycleMs,fieldsUploadCycleMs,structReadNsPerDispatch,fieldsReadNsPerDispatch\n${results.join("\n")}`,
        );
    } finally {
        app.dispose();
    }
}, 0);

test("measure dense table range uploads at 0.1%, 10%, and 100% changed", async () => {
    let world!: World;
    const plugin: Plugin = {
        name: "TableUploadCostProbe",
        initialize(current) {
            world = current;
        },
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });

    try {
        const { class: adapterClass, identity } = world.gpu.adapter;
        console.info(`[gpu-table-upload-perf] adapter class=${adapterClass} identity=${identity}`);
        expect(adapterClass).toBe("real");
        expect(identity.length).toBeGreaterThan(0);
        const results: string[] = [];
        const device = world.gpu.device;

        for (const count of [1_000, 10_000, 100_000]) {
            const table = world.table(`upload-cost-${count}`, UPLOAD_RECORD);
            for (let eid = 0; eid < count; eid++) table.acquire(eid);
            table.bytes.fill(0x5a);
            device.pushErrorScope("validation");
            table.markRange(0, count);
            table.upload();
            await bounded(
                `writeBuffer ${count} warmup queue completion`,
                device.queue.onSubmittedWorkDone(),
            );
            const warmupError = await bounded(
                `writeBuffer ${count} warmup validation scope`,
                device.popErrorScope(),
            );
            if (warmupError) throw new Error(`writeBuffer warmup: ${warmupError.message}`);
            expect(table.lastUploadPath).toBe("writeBuffer");

            for (const fraction of [0.001, 0.1, 1]) {
                const changed = Math.max(1, Math.floor(count * fraction));
                const submitTimes: number[] = [];
                const cycleTimes: number[] = [];
                for (let sample = 0; sample < 7; sample++) {
                    device.pushErrorScope("validation");
                    const start = performance.now();
                    table.markRange(0, changed);
                    table.upload();
                    submitTimes.push(performance.now() - start);
                    await bounded(
                        `writeBuffer ${count} rows ${changed} changed sample ${sample + 1} queue completion`,
                        device.queue.onSubmittedWorkDone(),
                    );
                    const error = await bounded(
                        `writeBuffer ${count} rows ${changed} changed sample ${sample + 1} validation scope`,
                        device.popErrorScope(),
                    );
                    if (error) throw new Error(`writeBuffer upload: ${error.message}`);
                    if (table.lastUploadPath !== "writeBuffer") {
                        throw new Error(`writeBuffer requested, got ${table.lastUploadPath}`);
                    }
                    cycleTimes.push(performance.now() - start);
                }
                results.push(
                    `${count},${(fraction * 100).toFixed(1)}%,${changed},${median(submitTimes).toFixed(4)},${median(cycleTimes).toFixed(4)}`,
                );
            }
        }

        console.info(
            `[gpu-table-upload-perf] median ms; columns=rows,changed,changedRows,submit,queueCycle\n${results.join("\n")}`,
        );
    } finally {
        app.dispose();
    }
}, 0);

test("measure dense table GPU memory at 1% and 100% population", async () => {
    let world!: World;
    const plugin: Plugin = {
        name: "TableMemoryProbe",
        initialize(current) {
            world = current;
        },
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });

    try {
        const { class: adapterClass, identity } = world.gpu.adapter;
        console.info(`[gpu-table-memory] adapter class=${adapterClass} identity=${identity}`);
        expect(adapterClass).toBe("real");
        const device = world.gpu.device;
        const results: string[] = [];
        for (const entityCount of [1_000, 10_000, 100_000]) {
            for (const fraction of [0.01, 1]) {
                const population = Math.max(1, Math.floor(entityCount * fraction));
                for (const usesMap of [false, true]) {
                    const table = world.table(
                        `table-memory-${entityCount}-${fraction}-${usesMap}`,
                        MEMORY_RECORD,
                    );
                    for (let item = 0; item < population; item++) {
                        table.acquire(Math.floor((item * entityCount) / population));
                    }
                    if (usesMap) table.enableEidLookup();
                    device.pushErrorScope("validation");
                    table.upload();
                    await bounded(
                        `table memory ${entityCount} ${fraction} ${usesMap} queue completion`,
                        device.queue.onSubmittedWorkDone(),
                    );
                    const error = await bounded(
                        `table memory ${entityCount} ${fraction} ${usesMap} validation scope`,
                        device.popErrorScope(),
                    );
                    if (error) throw new Error(`table memory setup: ${error.message}`);
                    const initialMapBytes = table.lastMapUploadBytes;
                    device.pushErrorScope("validation");
                    table.upload();
                    await bounded(
                        `table memory steady upload ${entityCount} ${fraction} ${usesMap}`,
                        device.queue.onSubmittedWorkDone(),
                    );
                    const steadyError = await bounded(
                        `table memory steady validation ${entityCount} ${fraction} ${usesMap}`,
                        device.popErrorScope(),
                    );
                    if (steadyError)
                        throw new Error(`table memory steady upload: ${steadyError.message}`);
                    expect(table.lastMapUploadBytes).toBe(0);
                    const records = table.buffer.size;
                    const active = table.activeRowsBuffer!.size;
                    const map = table.eidToRowBuffer?.size ?? 0;
                    results.push(
                        `${entityCount},${(fraction * 100).toFixed(0)}%,${population},${table.rowBytes},${table.capacity},${records},${active},${map},${records + active + map},${initialMapBytes},${table.lastMapUploadBytes}`,
                    );
                }
            }
        }
        console.info(
            `[gpu-table-memory] allocated GPU bytes; columns=entityRange,population,populated,recordBytes,slotCapacity,records,activeList,eidMap,total,initialMapUpload,steadyMapUpload\n${results.join("\n")}`,
        );
    } finally {
        app.dispose();
    }
}, 0);

test("measure component setter overhead against direct column writes", async () => {
    let world!: World;
    const Setter = { value: u32 };
    const count = 100_000;
    const plugin: Plugin = {
        name: "TableSetterCostProbe",
        components: [registration("Setter", Setter)],
        initialize(current) {
            world = current;
        },
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });

    try {
        const { class: adapterClass, identity } = world.gpu.adapter;
        console.info(`[gpu-table-setter] adapter class=${adapterClass} identity=${identity}`);
        expect(adapterClass).toBe("real");
        const eids = Array.from({ length: count }, () => world.create());
        const value = world.storage(Setter).value;
        const setterTimes: number[] = [];
        const directTimes: number[] = [];
        for (let repeat = 0; repeat < 7; repeat++) {
            let start = performance.now();
            for (let i = 0; i < count; i++) value.set(eids[i], i + repeat);
            setterTimes.push(performance.now() - start);
            const column = value.column;
            start = performance.now();
            for (let i = 0; i < count; i++) column[eids[i]] = i + repeat;
            directTimes.push(performance.now() - start);
        }
        console.info(
            `[gpu-table-setter] median ms per ${count} writes; setter=${median(setterTimes).toFixed(4)} direct-column=${median(directTimes).toFixed(4)} samplesSetter=${setterTimes.map((time) => time.toFixed(3)).join("/")} samplesDirect=${directTimes.map((time) => time.toFixed(3)).join("/")}`,
        );
    } finally {
        app.dispose();
    }
}, 0);
