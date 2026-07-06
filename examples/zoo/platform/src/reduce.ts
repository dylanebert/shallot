import {
    Compute,
    capacity,
    f32,
    type Mirror,
    MirrorPlugin,
    mirror,
    type Plugin,
    SlabPlugin,
    type System,
    slab,
} from "@dylanebert/shallot";

// #doc:intro
// The platform layer is the substrate every built-in system is built on: the GPU device, per-entity
// buffers that upload to it, and readback that brings results home. Reach for it when a plugin needs its
// own compute pass — a simulation step, an aggregation, anything the standard systems don't cover.
//
// This specimen sums a per-entity value on the GPU and reads the total back to the CPU. Three pieces do
// the work: a `slab` field mirrors per-entity data up to the GPU, a compute pass over the device from
// `Compute` reduces it, and a `mirror` copies the one-number result back down.

let pipeline: GPUComputePipeline | null = null;
let bindGroup: GPUBindGroup | null = null;
let output: GPUBuffer | null = null;
let readback: Mirror | null = null;

// #doc:code source:platform/public/scenes/platform.scene
// ### The scene
//
// Three boxes over a ground, each carrying a `charge` — the per-entity value the pass sums. The scene
// authors the amounts; the code never touches the pose.

// #doc:code
// ### Per-entity data on the GPU
//
// `slab(f32)` stores one float per entity and mirrors it to a GPU buffer each frame, so a compute pass
// can read it — the write-only half of the platform's data path (the `Transform` firehose the renderer
// reads is the same primitive). Naming the slab `"charge"` publishes its buffer under that name in
// `Compute.buffers`, the registry the pass resolves it from.
// #region component
export const Charge = { amount: slab(f32, "charge") };
// #endregion

// one thread totals every slot. Unwritten slots are zero, so this sums the live charges with no
// membership gate; a scene that despawns entities would gate each slot on the "membership" buffer (the
// ECS page covers it). A real reduction parallelizes — standard/bvh's bounds reduce is the reference.
function sumWgsl(): string {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read> charge: array<f32>;
@group(0) @binding(1) var<storage, read_write> total: array<f32>;

@compute @workgroup_size(1)
fn main() {
    var sum = 0.0;
    for (var i = 0u; i < ${capacity}u; i++) { sum += charge[i]; }
    total[0] = sum;
}
`;
}

// #doc:code
// ### A custom compute pass
//
// `Compute.device` is the raw `GPUDevice`; build pipelines and buffers against it directly. `warm`
// compiles the sum kernel, allocates a one-float output buffer, and points a `mirror` at it. It runs
// once per build, so nothing here is per-frame.
// #region pass
async function build(): Promise<void> {
    const device = Compute.device;
    output = device.createBuffer({
        label: "reduce-total",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    pipeline = await device.createComputePipelineAsync({
        label: "reduce-sum",
        layout: "auto",
        compute: { entryPoint: "main", module: device.createShaderModule({ code: sumWgsl() }) },
    });
    readback = mirror(output);
}
// #endregion

// #doc:code
// ### Dispatch and read back
//
// A `simulation` system records the pass into an encoder and submits it each frame. Its `setup` runs on
// the first frame — after every plugin's `warm`, so the `"charge"` slab buffer is published in
// `Compute.buffers` by then — and binds it once. `mirror` copies the output into a CPU-mapped staging
// buffer, so its `snapshot` lands a frame or two later; read whatever's current.
// #region loop
const reduce = {
    name: "reduce",
    group: "simulation",
    setup() {
        bindGroup = Compute.device.createBindGroup({
            label: "reduce-sum",
            layout: pipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: Compute.buffers.get("charge")! } },
                { binding: 1, resource: { buffer: output! } },
            ],
        });
    },
    update() {
        if (!pipeline || !bindGroup || !readback) return;
        const device = Compute.device;
        const encoder = device.createCommandEncoder({ label: "reduce" });
        const pass = encoder.beginComputePass({ label: "reduce-sum" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([encoder.finish()]);

        if (readback.snapshot && Compute.frame % 60 === 0) {
            const total = new Float32Array(readback.snapshot.bytes)[0];
            console.log(`total charge: ${total.toFixed(2)}`);
        }
    },
} satisfies System;
// #endregion

export const Reduce = {
    name: "Reduce",
    components: { Charge },
    systems: [reduce],
    dependencies: [SlabPlugin, MirrorPlugin],
    traits: { Charge: { defaults: () => ({ amount: 0 }) } },
    warm: build,
    dispose() {
        output?.destroy();
        pipeline = null;
        bindGroup = null;
        output = null;
        readback = null;
    },
} satisfies Plugin;

export default Reduce;
