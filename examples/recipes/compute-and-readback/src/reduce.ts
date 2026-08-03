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
import tgpu, {
    type StorageFlag,
    type TgpuBindGroup,
    type TgpuBuffer,
    type TgpuComputePipeline,
} from "typegpu";
import * as d from "typegpu/data";

const ChargeData = d.arrayOf(d.f32, capacity);
const TotalData = d.arrayOf(d.f32, 1);

const reduceLayout = tgpu.bindGroupLayout({
    charge: { storage: ChargeData, access: "readonly" },
    total: { storage: TotalData, access: "mutable" },
});

const reduceKernel = tgpu
    .computeFn({ workgroupSize: [1] })(() => {
        "use gpu";
        let sum = d.f32(0);
        for (let i = d.u32(0); i < d.u32(capacity); i = i + d.u32(1)) {
            sum += reduceLayout.$.charge[i];
        }
        reduceLayout.$.total[0] = sum;
    })
    .$name("reduceSum");

let pipeline: TgpuComputePipeline | null = null;
let charge: (TgpuBuffer<typeof ChargeData> & StorageFlag) | null = null;
let output: (TgpuBuffer<typeof TotalData> & StorageFlag) | null = null;
let rawOutput: GPUBuffer | null = null;
let group: TgpuBindGroup | null = null;
let readback: Mirror | null = null;

// `slab(f32)` mirrors one float per entity to a GPU buffer each frame — CPU→GPU per-entity data (the
// `Transform` firehose the renderer reads is the same primitive). Naming the
// slab "charge" publishes its buffer under that name in `Compute.buffers`, where the pass resolves it.
export const Charge = { amount: slab(f32, "charge") };

// a no-field marker selects the world-space label the readback total drives live
export const Readout = {};

// `Compute.root` owns the typed wrappers and pipeline. `setup` runs after `SlabPlugin` warms, so the
// published charge buffer already exists when we wrap it and bind the compute layout once.
async function build(): Promise<void> {
    const root = Compute.root;
    rawOutput = Compute.device.createBuffer({
        label: "reduce-total",
        size: d.sizeOf(TotalData),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    output = root.createBuffer(TotalData, rawOutput).$usage("storage").$name("reduce-total");
    pipeline = root.createComputePipeline({ compute: reduceKernel }).$name("reduce-sum");
    readback = mirror(output);
}

// `setup` runs on the first frame (after every plugin's `warm`, so the "charge" slab buffer is published
// in `Compute.buffers` by then) and binds once. `mirror`'s `snapshot` lands a frame or two behind the GPU
// write, so read whatever's current.
const reduce = {
    name: "reduce",
    group: "simulation",
    setup() {
        const source = Compute.buffers.get("charge");
        if (!source || !output || !pipeline) return;
        charge = Compute.root.createBuffer(ChargeData, source).$usage("storage");
        group = Compute.root.createBindGroup(reduceLayout, {
            charge,
            total: output,
        });
    },
    update(state: State) {
        if (!pipeline || !charge || !output || !group || !readback) return;
        const device = Compute.device;
        const encoder = device.createCommandEncoder({ label: "reduce" });
        const pass = encoder.beginComputePass({ label: "reduce-sum" });
        pipeline.with(group).with(pass).dispatchWorkgroups(1);
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
        charge = null;
        group = null;
        output = null;
        rawOutput = null;
        readback = null;
    },
} satisfies Plugin;

export default Reduce;
