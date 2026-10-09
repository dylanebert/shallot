import { expect, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { createApp } from "../../engine/app";
import { probeBuffer } from "../../engine/runtime";
import { Xform } from "../../engine/utils";

await setupGlobals();

const layout = tgpu.bindGroupLayout({
    current: { storage: d.arrayOf(Xform), access: "readonly" },
    previous: { storage: d.arrayOf(Xform), access: "readonly" },
    output: { storage: d.arrayOf(Xform), access: "mutable" },
    params: { uniform: d.vec4f },
});
const interpolate = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((args) => {
        "use gpu";
        const row = args.gid.x;
        if (row >= d.u32(layout.$.params.y)) return;
        const current = layout.$.current[row];
        const previous = layout.$.previous[row];
        const alpha = layout.$.params.x;
        const flip = std.select(d.f32(1), d.f32(-1), std.dot(previous.quat, current.quat) < 0);
        const blend = std.add(
            std.mul(previous.quat, flip * (1 - alpha)),
            std.mul(current.quat, alpha),
        );
        const length = std.length(blend);
        let quat = d.vec4f(0, 0, 0, 1);
        if (length > 1e-12) quat = std.div(blend, length);
        layout.$.output[row] = Xform({
            pos: std.mix(previous.pos, current.pos, alpha),
            quat,
            scale: std.mix(previous.scale, current.scale, alpha),
        });
    })
    .$name("globalTransformFillMeasure");

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} exceeded 5000 ms`)), 5000);
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}
function median(values: number[]): number {
    const sorted = values.toSorted((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}
function writeTransform(words: Float32Array, row: number, tick: number): void {
    const base = row * 12;
    words[base] = row + tick * 0.25;
    words[base + 1] = 10 + tick;
    words[base + 2] = -2;
    words[base + 3] = 0;
    // Alternate antipodal identity quaternions to prove shortest-arc normalization.
    words[base + 4] = 0;
    words[base + 5] = 0;
    words[base + 6] = 0;
    words[base + 7] = tick % 2 ? -1 : 1;
    words[base + 8] = 1 + tick * 0.1;
    words[base + 9] = 2;
    words[base + 10] = 3;
    words[base + 11] = 0;
}
function fillCpu(
    previous: Float32Array,
    current: Float32Array,
    out: Float32Array,
    count: number,
    alpha: number,
): void {
    for (let row = 0; row < count; row++) {
        const b = row * 12;
        for (let lane = 0; lane < 3; lane++) {
            out[b + lane] = previous[b + lane] * (1 - alpha) + current[b + lane] * alpha;
            out[b + 8 + lane] =
                previous[b + 8 + lane] * (1 - alpha) + current[b + 8 + lane] * alpha;
        }
        const dot =
            previous[b + 4] * current[b + 4] +
            previous[b + 5] * current[b + 5] +
            previous[b + 6] * current[b + 6] +
            previous[b + 7] * current[b + 7];
        const flip = dot < 0 ? -1 : 1;
        const x = previous[b + 4] * flip * (1 - alpha) + current[b + 4] * alpha;
        const y = previous[b + 5] * flip * (1 - alpha) + current[b + 5] * alpha;
        const z = previous[b + 6] * flip * (1 - alpha) + current[b + 6] * alpha;
        const w = previous[b + 7] * flip * (1 - alpha) + current[b + 7] * alpha;
        const len = Math.hypot(x, y, z, w);
        out[b + 4] = len > 1e-12 ? x / len : 0;
        out[b + 5] = len > 1e-12 ? y / len : 0;
        out[b + 6] = len > 1e-12 ? z / len : 0;
        out[b + 7] = len > 1e-12 ? w / len : 1;
        out[b + 11] = 0;
    }
}

// Manual real-adapter performance oracle. Both variants interpolate the same 48-byte decomposed
// GlobalTransform records and submit once per frame. The CPU variant uploads changed current rows and
// its filled renderer rows. The GPU variant uploads changed current rows to per-tick GPU staging,
// records current→previous then staging→current copies into the frame encoder, and computes renderer
// rows before that encoder's one submit. It has no CPU history shadow and does no history copy on no-tick.
test("measure CPU GlobalTransform fill against GPU-only tick history and frame interpolation", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [{ name: "GlobalTransformFillMeasure", gpu: { features: ["timestamp-query"] } }],
    });
    const world = app.world;
    try {
        console.info(`[global-transform-fill] adapter=${JSON.stringify(world.gpu.adapter)}`);
        expect(world.gpu.adapter.class).toBe("real");
        const device = world.gpu.device;
        const root = world.gpu.root;
        const pipeline = root.unwrap(root.createComputePipeline({ compute: interpolate }));
        for (const count of [1_000, 10_000, 100_000]) {
            const bytes = count * 48;
            const cpuPrev = new Float32Array(count * 12);
            const cpuCurrent = new Float32Array(count * 12);
            const cpuOutput = new Float32Array(count * 12);
            const gpuInitial = new Float32Array(count * 12);
            for (let row = 0; row < count; row++) {
                writeTransform(cpuPrev, row, 0);
                writeTransform(cpuCurrent, row, 1);
                writeTransform(gpuInitial, row, 1);
            }
            const make = (label: string, usage: GPUBufferUsageFlags) => {
                const buffer = device.createBuffer({ label, size: bytes, usage });
                world.own(buffer);
                return buffer;
            };
            const cpuCurrentGpu = make(
                `cpu-current-${count}`,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            );
            const cpuOutputGpu = make(
                `cpu-output-${count}`,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            );
            const current = make(
                `global-transform-current-${count}`,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            );
            const previous = make(
                `global-transform-previous-tick-${count}`,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            );
            const output = make(
                `global-transform-interpolated-${count}`,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            );
            const tickStage = make(
                `global-transform-tick-upload-${count}`,
                GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            );
            const params = root.unwrap(
                root.createBuffer(d.vec4f, d.vec4f(0.375, count, 0, 0)).$usage("uniform"),
            );
            world.own(params);
            device.queue.writeBuffer(cpuCurrentGpu, 0, cpuCurrent);
            device.queue.writeBuffer(cpuOutputGpu, 0, cpuOutput);
            device.queue.writeBuffer(current, 0, gpuInitial);
            device.queue.writeBuffer(previous, 0, cpuPrev);
            device.queue.writeBuffer(output, 0, gpuInitial);
            const group = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: current } },
                    { binding: 1, resource: { buffer: previous } },
                    { binding: 2, resource: { buffer: output } },
                    { binding: 3, resource: { buffer: params } },
                ],
            });
            const alpha = new Float32Array([0.375, count, 0, 0]);
            const dispatch = (changed: boolean, tick: number): GPUCommandBuffer => {
                if (changed) {
                    for (let row = 0; row < count; row++) writeTransform(gpuInitial, row, tick);
                    device.queue.writeBuffer(tickStage, 0, gpuInitial);
                }
                alpha[0] = 0.375 + (tick % 3) * 0.125;
                device.queue.writeBuffer(params, 0, alpha);
                const encoder = device.createCommandEncoder();
                if (changed) {
                    encoder.copyBufferToBuffer(current, 0, previous, 0, bytes);
                    encoder.copyBufferToBuffer(tickStage, 0, current, 0, bytes);
                }
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.dispatchWorkgroups(Math.ceil(count / 64));
                pass.end();
                return encoder.finish();
            };
            const cpuFrame = (changed: boolean, tick: number): GPUCommandBuffer => {
                if (changed) {
                    cpuPrev.set(cpuCurrent);
                    for (let row = 0; row < count; row++) writeTransform(cpuCurrent, row, tick);
                    device.queue.writeBuffer(cpuCurrentGpu, 0, cpuCurrent);
                }
                fillCpu(cpuPrev, cpuCurrent, cpuOutput, count, 0.375 + (tick % 3) * 0.125);
                device.queue.writeBuffer(cpuOutputGpu, 0, cpuOutput);
                const encoder = device.createCommandEncoder();
                return encoder.finish();
            };
            const gpuInitialCommand = dispatch(false, 0);
            device.queue.submit([gpuInitialCommand]);
            await bounded("initial fill completion", device.queue.onSubmittedWorkDone());
            const observed = new Float32Array(
                (
                    await bounded(
                        "fill comparison readback",
                        probeBuffer(world, output, { size: bytes }),
                    )
                ).bytes,
            );
            fillCpu(cpuPrev, cpuCurrent, cpuOutput, count, 0.375);
            for (let row = 0; row < count; row++) {
                for (const lane of [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]) {
                    if (Math.abs(observed[row * 12 + lane] - cpuOutput[row * 12 + lane]) > 1e-4)
                        throw new Error(
                            `GlobalTransform fill mismatch: ${count} rows, row ${row}, lane ${lane}`,
                        );
                }
            }
            const gpuWall: Record<string, number[]> = { stable: [], changed: [] };
            const cpuWall: Record<string, number[]> = { stable: [], changed: [] };
            for (let i = 0; i < 41; i++) {
                for (const changed of [false, true]) {
                    const tick = 2 + i;
                    const key = changed ? "changed" : "stable";
                    const runGpu = async () => {
                        const start = performance.now();
                        device.queue.submit([dispatch(changed, tick)]);
                        await bounded(
                            "GPU GlobalTransform frame",
                            device.queue.onSubmittedWorkDone(),
                        );
                        if (i >= 10) gpuWall[key].push(performance.now() - start);
                    };
                    const runCpu = async () => {
                        const start = performance.now();
                        device.queue.submit([cpuFrame(changed, tick)]);
                        await bounded(
                            "CPU GlobalTransform frame",
                            device.queue.onSubmittedWorkDone(),
                        );
                        if (i >= 10) cpuWall[key].push(performance.now() - start);
                    };
                    if ((i + Number(changed)) % 2) {
                        await runGpu();
                        await runCpu();
                    } else {
                        await runCpu();
                        await runGpu();
                    }
                }
            }
            console.info(
                `[global-transform-fill] count=${count} no-tick-drained-ms CPU=${median(cpuWall.stable).toFixed(6)} GPU=${median(gpuWall.stable).toFixed(6)} changed-tick-drained-ms CPU=${median(cpuWall.changed).toFixed(6)} GPU=${median(gpuWall.changed).toFixed(6)} upload-per-changed-row CPU=96B GPU=48B; extra-GPU-staging=${bytes}B`,
            );
        }
    } finally {
        app.dispose();
    }
}, 0);
