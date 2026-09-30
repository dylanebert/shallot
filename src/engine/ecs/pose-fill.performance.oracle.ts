import { expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { build } from "../app";
import { probeBuffer } from "../runtime";
import { Xform } from "../utils";

const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

const layout = tgpu.bindGroupLayout({
    current: { storage: d.arrayOf(Xform), access: "readonly" },
    previous: { storage: d.arrayOf(Xform), access: "readonly" },
    output: { storage: d.arrayOf(Xform), access: "mutable" },
    params: { uniform: d.vec4f },
});
const kernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
})((args) => {
    "use gpu";
    const i = args.gid.x;
    if (i >= d.u32(layout.$.params.y)) return;
    const a = layout.$.params.x;
    const previous = layout.$.previous[i];
    const current = layout.$.current[i];
    const flip = std.select(d.f32(1), d.f32(-1), std.dot(previous.quat, current.quat) < 0);
    const blend = std.add(std.mul(previous.quat, flip * (1 - a)), std.mul(current.quat, a));
    const len = std.length(blend);
    let quat = d.vec4f(0, 0, 0, 1);
    if (len > 1e-12) quat = std.div(blend, len);
    const pos = std.mix(previous.pos, current.pos, a);
    const scale = std.mix(previous.scale, current.scale, a);
    layout.$.output[i] = Xform({
        pos: d.vec3f(pos.x, pos.y, pos.z),
        quat,
        scale: d.vec3f(scale.x, scale.y, scale.z),
    });
});

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label}: 5000 ms deadline`)), 5000);
        promise.then(
            (result) => {
                clearTimeout(timer);
                resolve(result);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}
function nlerpShortestInto(
    prev: Float32Array,
    p: number,
    curr: Float32Array,
    q: number,
    alpha: number,
    out: Float32Array,
    offset: number,
): void {
    const dot =
        prev[p] * curr[q] +
        prev[p + 1] * curr[q + 1] +
        prev[p + 2] * curr[q + 2] +
        prev[p + 3] * curr[q + 3];
    const flip = dot < 0 ? -1 : 1;
    const x = prev[p] * flip * (1 - alpha) + curr[q] * alpha;
    const y = prev[p + 1] * flip * (1 - alpha) + curr[q + 1] * alpha;
    const z = prev[p + 2] * flip * (1 - alpha) + curr[q + 2] * alpha;
    const w = prev[p + 3] * flip * (1 - alpha) + curr[q + 3] * alpha;
    const len = Math.sqrt(x * x + y * y + z * z + w * w);
    out[offset] = len > 1e-12 ? x / len : 0;
    out[offset + 1] = len > 1e-12 ? y / len : 0;
    out[offset + 2] = len > 1e-12 ? z / len : 0;
    out[offset + 3] = len > 1e-12 ? w / len : 1;
}
function median(values: number[]): number {
    const sorted = values.toSorted((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

// Deliberately manual: timing is reported, never asserted. Both alternatives fill the same Xform
// table, use dense rows, interpolate pos/scale and shortest-arc normalized quaternions, and start
// from the same six eid-indexed columns. GPU history lives only on the device, copied each tick.
// This measures a prototype, not production frames or a replacement for engine verification.
test("measure CPU interpolation against cached GPU pose-table fill", async () => {
    const app = await build({
        defaults: false,
        plugins: [{ name: "PoseFillMeasure", features: ["timestamp-query"] }],
    });
    const state = app.state;
    try {
        console.info(`[pose-fill] adapter=${JSON.stringify(state.gpu.adapter)}`);
        expect(state.gpu.adapter.class).toBe("real");
        const device = state.gpu.device;
        const root = state.gpu.root;
        const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
        state.own(querySet);
        const resolve = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        state.own(resolve);
        const read = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        state.own(read);
        const pipeline = root.createComputePipeline({ compute: kernel });
        const rawPipeline = root.unwrap(pipeline);
        for (const count of [1_000, 10_000, 100_000]) {
            const cpu = state.table(`pose-cpu-${count}`, Xform);
            const input = state.table(`pose-input-${count}`, Xform);
            const previous = state.table(`pose-history-${count}`, Xform, { gpuOnly: true });
            const gpu = state.table(`pose-gpu-${count}`, Xform, { gpuOnly: true });
            const eids = new Uint32Array(count);
            for (let row = 0; row < count; row++) {
                // A permuted eid domain ensures neither option relies on slot === eid.
                const eid = count - row - 1;
                eids[row] = eid;
                cpu.acquire(eid);
                input.acquire(eid);
                previous.acquire(eid);
                gpu.acquire(eid);
            }
            const columns = Array.from({ length: 6 }, () => new Float32Array(count * 4));
            for (let eid = 0; eid < count; eid++) {
                const j = eid * 4;
                columns[0][j] = eid;
                columns[0][j + 1] = 10;
                columns[0][j + 2] = -2;
                columns[3][j] = eid + 1;
                columns[3][j + 1] = 11;
                columns[3][j + 2] = -1;
                columns[1][j + 3] = eid % 2 ? -1 : 1;
                columns[4][j + 1] = 0.6;
                columns[4][j + 3] = 0.8;
                for (let k = 0; k < 3; k++) {
                    columns[2][j + k] = 1;
                    columns[5][j + k] = 2;
                }
            }
            const cpuWords = new Float32Array(cpu.bytes.buffer);
            const inputWords = new Float32Array(input.bytes.buffer);
            function fillCpu(alpha: number): void {
                const [pp, pq, ps, cp, cq, cs] = columns;
                for (let row = 0; row < count; row++) {
                    const eid = eids[row];
                    const j = eid * 4;
                    const b = row * 12;
                    for (let k = 0; k < 3; k++) {
                        cpuWords[b + k] = pp[j + k] * (1 - alpha) + cp[j + k] * alpha;
                        cpuWords[b + 8 + k] = ps[j + k] * (1 - alpha) + cs[j + k] * alpha;
                    }
                    nlerpShortestInto(pq, j, cq, j, alpha, cpuWords, b + 4);
                }
            }
            function gatherInputs(start = 3): void {
                for (let row = 0; row < count; row++) {
                    const j = eids[row] * 4;
                    const b = row * 12;
                    for (let field = 0; field < 3; field++) {
                        const source = columns[start + field];
                        const target = b + field * 4;
                        inputWords[target] = source[j];
                        inputWords[target + 1] = source[j + 1];
                        inputWords[target + 2] = source[j + 2];
                        inputWords[target + 3] = source[j + 3];
                    }
                }
            }
            const params = root
                .createBuffer(d.vec4f, d.vec4f(0.375, count, 0, 0))
                .$usage("uniform");
            const rawParams = root.unwrap(params);
            state.own(rawParams);
            const paramWords = new Float32Array([0.375, count, 0, 0]);
            const group = root.createBindGroup(layout, {
                current: input.typed,
                previous: previous.typed,
                output: gpu.typed,
                params,
            });
            const rawGroup = root.unwrap(group);
            function dispatch(repetitions = 1, timed = false): void {
                device.queue.writeBuffer(rawParams, 0, paramWords);
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginComputePass(
                    timed
                        ? {
                              timestampWrites: {
                                  querySet,
                                  beginningOfPassWriteIndex: 0,
                                  endOfPassWriteIndex: 1,
                              },
                          }
                        : {},
                );
                pass.setPipeline(rawPipeline);
                pass.setBindGroup(0, rawGroup);
                for (let i = 0; i < repetitions; i++)
                    pass.dispatchWorkgroups(Math.ceil(count / 64));
                pass.end();
                if (timed) {
                    encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
                    encoder.copyBufferToBuffer(resolve, 0, read, 0, 16);
                }
                device.queue.submit([encoder.finish()]);
            }
            device.pushErrorScope("validation");
            fillCpu(0.375);
            cpu.markRange(0, count);
            cpu.upload();
            gatherInputs(0);
            input.markRange(0, count);
            input.upload();
            function copyPrevious(): void {
                const encoder = device.createCommandEncoder();
                encoder.copyBufferToBuffer(input.buffer, 0, previous.buffer, 0, count * 48);
                device.queue.submit([encoder.finish()]);
            }
            copyPrevious();
            gatherInputs();
            input.markRange(0, count);
            input.upload();
            dispatch();
            await bounded("initial queue", device.queue.onSubmittedWorkDone());
            const actual = new Float32Array(
                (
                    await bounded(
                        "pose correctness",
                        probeBuffer(state, gpu.buffer, { size: count * 48 }),
                    )
                ).bytes,
            );
            for (let row = 0; row < count; row++) {
                // Pad lanes need not be defined by shader stores.
                for (const lane of [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]) {
                    if (Math.abs(actual[row * 12 + lane] - cpuWords[row * 12 + lane]) > 1e-4)
                        throw new Error(`pose mismatch count=${count} row=${row} lane=${lane}`);
                }
            }
            const error = await bounded("fill validation", device.popErrorScope());
            if (error) throw new Error(error.message);
            const cpuFill: number[] = [];
            const gpuGather: number[] = [];
            // Repeated pure CPU work, outside queue waits, reports arithmetic and gather separately.
            for (let i = 0; i < 400; i++) {
                let start = performance.now();
                fillCpu(0.375);
                const c = performance.now() - start;
                start = performance.now();
                gatherInputs();
                const g = performance.now() - start;
                if (i >= 300) {
                    cpuFill.push(c);
                    gpuGather.push(g);
                }
            }
            const gpuTimes: number[] = [];
            for (let sample = 0; sample < 9; sample++) {
                dispatch(512, true);
                await bounded("timestamp queue", device.queue.onSubmittedWorkDone());
                await bounded("timestamp map", read.mapAsync(GPUMapMode.READ));
                const t = new BigUint64Array(read.getMappedRange());
                gpuTimes.push(Number(t[1] - t[0]) / 512 / 1e6);
                read.unmap();
            }
            console.info(
                `[pose-fill] count=${count} CPU-fill-ms=${median(cpuFill).toFixed(6)} GPU-input-gather-ms=${median(gpuGather).toFixed(6)} GPU-dispatch-ms=${median(gpuTimes).toFixed(6)} CPU-upload-bytes=${count * 48} GPU-input-upload-bytes=${count * 48} GPU-extra-allocated-bytes=${input.buffer.size + previous.buffer.size}`,
            );
            for (const changed of [false, true]) {
                const cpuWall: number[] = [];
                const gpuWall: number[] = [];
                const cpuSubmit: number[] = [];
                const gpuSubmit: number[] = [];
                async function frame(kind: "cpu" | "gpu", sample: number): Promise<void> {
                    const alpha = 0.125 + (sample % 7) * 0.125;
                    const start = performance.now();
                    if (kind === "cpu") {
                        fillCpu(alpha);
                        cpu.markRange(0, count);
                        cpu.upload();
                    } else {
                        paramWords[0] = alpha;
                        if (changed) {
                            copyPrevious();
                            gatherInputs();
                            input.markRange(0, count);
                            input.upload();
                        }
                        dispatch();
                    }
                    const enqueue = performance.now() - start;
                    await bounded("frame queue", device.queue.onSubmittedWorkDone());
                    const wall = performance.now() - start;
                    if (sample >= 10) {
                        (kind === "cpu" ? cpuWall : gpuWall).push(wall);
                        (kind === "cpu" ? cpuSubmit : gpuSubmit).push(enqueue);
                    }
                }
                for (let i = 0; i < 41; i++) {
                    // Alternate order to avoid giving one path all the hot/cold samples.
                    if (i % 2) {
                        await frame("gpu", i);
                        await frame("cpu", i);
                    } else {
                        await frame("cpu", i);
                        await frame("gpu", i);
                    }
                }
                console.info(
                    `[pose-fill] count=${count} new-tick=${changed} CPU-submit-ms=${median(cpuSubmit).toFixed(6)} GPU-submit-ms=${median(gpuSubmit).toFixed(6)} CPU-drained-frame-ms=${median(cpuWall).toFixed(6)} GPU-drained-frame-ms=${median(gpuWall).toFixed(6)}`,
                );
            }
        }
    } finally {
        app.dispose();
    }
}, 0);
