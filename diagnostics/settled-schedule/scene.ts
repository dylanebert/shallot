import { type Plugin, type System, type World } from "@dylanebert/shallot";

type ScheduleWindow = Window & {
    __shallotScheduleGpuWork(workgroups: number, iterations: number): void;
    __shallotScheduleGpuSample(durationMs: number): void;
};

const ITERATIONS = Number(new URLSearchParams(location.search).get("iterations") ?? 1_500);
if (!Number.isSafeInteger(ITERATIONS) || ITERATIONS < 1) {
    throw new Error("GPU-bound scene iterations must be a positive integer");
}
const GPU_SAMPLES = 32;
const SAMPLE_STRIDE = 256;
const WORKGROUP_SIZE = 256;

const shader = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    var value = f32(global_id.x + 1u) * 0.000001;
    for (var i = 0u; i < ${ITERATIONS}u; i += 1u) {
        value = sin(value + f32(i) * 0.000001);
    }
    if (local_id.x == 0u) {
        output[workgroup_id.x] = value;
    }
}
`;

const dispatchFrame: System = {
    group: "draw",
    name: "settled-schedule-gpu-work",
    update(world) {
        const state = world.resource(GpuWorkState);
        const encoder = world.frameEncoder();
        if (!encoder) throw new Error("GPU-bound scene has no frame encoder");

        const sample = state.submitted++;
        const pass = encoder.beginComputePass({
            label: "settled-schedule-gpu-work",
            timestampWrites:
                sample < GPU_SAMPLES
                    ? {
                          querySet: state.queries,
                          beginningOfPassWriteIndex: sample * 2,
                          endOfPassWriteIndex: sample * 2 + 1,
                      }
                    : undefined,
        });
        pass.setPipeline(state.pipeline);
        pass.setBindGroup(0, state.bindGroup);
        pass.dispatchWorkgroups(state.workgroups);
        pass.end();

        if (sample < GPU_SAMPLES) {
            const offset = sample * SAMPLE_STRIDE;
            encoder.resolveQuerySet(state.queries, sample * 2, 2, state.resolve, offset);
            encoder.copyBufferToBuffer(state.resolve, offset, state.readback, offset, 16);
            if (sample === GPU_SAMPLES - 1) world.onFrameSubmit(state.readSamples);
        }
    },
};

const GpuWorkState = {
    create(world: World) {
        const { device } = world.gpu;
        if (!device.features.has("timestamp-query")) {
            throw new Error("GPU-bound scene requires WebGPU timestamp-query");
        }
        const workgroups = device.limits.maxComputeWorkgroupsPerDimension;
        const pipeline = device.createComputePipeline({
            label: "settled-schedule-gpu-work",
            layout: "auto",
            compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
        });
        const output = device.createBuffer({
            label: "settled-schedule-gpu-output",
            size: workgroups * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: output } }],
        });
        const queries = device.createQuerySet({ type: "timestamp", count: GPU_SAMPLES * 2 });
        const resolve = device.createBuffer({
            size: SAMPLE_STRIDE * GPU_SAMPLES,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: SAMPLE_STRIDE * GPU_SAMPLES,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        for (const resource of [output, queries, resolve, readback]) world.own(resource);

        const readSamples = (): void => {
            void readback
                .mapAsync(GPUMapMode.READ, 0, SAMPLE_STRIDE * GPU_SAMPLES)
                .then(() => {
                    const values = new BigUint64Array(readback.getMappedRange());
                    for (let sample = 0; sample < GPU_SAMPLES; sample++) {
                        const index = (sample * SAMPLE_STRIDE) / 8;
                        const durationMs = Number(values[index + 1]! - values[index]!) / 1e6;
                        (window as ScheduleWindow).__shallotScheduleGpuSample(durationMs);
                    }
                    readback.unmap();
                })
                .catch((error: unknown) => {
                    console.error("[settled-schedule] timestamp readback failed", error);
                });
        };

        (window as ScheduleWindow).__shallotScheduleGpuWork(workgroups, ITERATIONS);
        return { workgroups, pipeline, bindGroup, queries, resolve, readback, readSamples, submitted: 0 };
    },
};

export const GpuWorkPlugin: Plugin = {
    name: "SettledScheduleGpuWork",
    gpu: { features: ["timestamp-query"] },
    initialize(world) {
        world.resource(GpuWorkState);
    },
    systems: [dispatchFrame],
};
