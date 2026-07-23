import {
    Compute,
    capacity,
    f32,
    type Mirror,
    MirrorPlugin,
    mirror,
    type Plugin,
    SlabPlugin,
    type State,
    type System,
    slab,
    Text,
    text,
} from "@dylanebert/shallot";

let pipeline: GPUComputePipeline | null = null;
let bindGroup: GPUBindGroup | null = null;
let output: GPUBuffer | null = null;
let readback: Mirror | null = null;

// `slab(f32)` mirrors one float per entity to a GPU buffer each frame — CPU→GPU per-entity data (the
// `Transform` firehose the renderer reads is the same primitive). Naming the
// slab "charge" publishes its buffer under that name in `Compute.buffers`, where the pass resolves it.
export const Charge = { amount: slab(f32, "charge") };

// a no-field marker selects the world-space label the readback total drives live
export const Readout = {};

// one thread totals every slot. Unwritten slots are zero, so this sums the live charges with no
// membership gate; a scene that despawns entities would gate each slot on the "membership" buffer. A
// real reduction parallelizes — standard/bvh's bounds reduce is the reference.
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

// `Compute.device` is the raw `GPUDevice`; build pipelines and buffers against it directly. Runs once
// per build, so nothing here is per-frame.
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

// `setup` runs on the first frame (after every plugin's `warm`, so the "charge" slab buffer is published
// in `Compute.buffers` by then) and binds once. `mirror`'s `snapshot` lands a frame or two behind the GPU
// write, so read whatever's current.
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
    update(state: State) {
        if (!pipeline || !bindGroup || !readback) return;
        const device = Compute.device;
        const encoder = device.createCommandEncoder({ label: "reduce" });
        const pass = encoder.beginComputePass({ label: "reduce-sum" });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([encoder.finish()]);

        if (readback.snapshot) {
            const total = new Float32Array(readback.snapshot.bytes)[0];
            const content = text(`total charge: ${total.toFixed(2)}`);
            for (const eid of state.query([Readout, Text])) Text.content.set(eid, content);
        }
    },
} satisfies System;

export const Reduce = {
    name: "Reduce",
    components: { Charge, Readout },
    systems: [reduce],
    dependencies: [SlabPlugin, MirrorPlugin],
    traits: {
        Charge: { defaults: () => ({ amount: 0 }) },
        Readout: { defaults: () => ({}) },
    },
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
