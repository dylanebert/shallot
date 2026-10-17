import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Meshes, type MeshHandle } from "../../core/mesh";
import type { World } from "../../engine";
import {
    beginComputePass,
    dispatchWorkgroups,
    endComputePass,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";

const descriptor = { label: "shallot-mesh-count" };
const sourceLayout = tgpu
    .bindGroupLayout({
        source: { storage: d.arrayOf(d.u32), access: "readonly" },
        rows: { storage: d.arrayOf(d.vec4u), access: "readonly" },
    })
    .$idx(0);
const outputLayout = tgpu
    .bindGroupLayout({
        args: { storage: d.arrayOf(d.u32), access: "mutable" },
    })
    .$idx(1);
const kernel = tgpu.computeFn({ workgroupSize: [64], in: { id: d.builtin.globalInvocationId } })(
    ({ id }) => {
        "use gpu";
        const dims = sourceLayout.$.rows[0];
        if (id.x >= dims.w) return;
        const row = sourceLayout.$.rows[id.x + 1];
        const count = std.min(sourceLayout.$.source[row.y], row.z);
        for (let slot = d.u32(0); slot < dims.z; slot++)
            for (let pair = row.x; pair < dims.y; pair += dims.x)
                outputLayout.$.args[(slot * dims.y + pair) * 5] = count;
    },
);
type Source = {
    buffer: GPUBuffer;
    table: GPUBuffer;
    group: GPUBindGroup;
    staging: Uint32Array;
    count: number;
    dirty: boolean;
    cursor: number;
};
const key = {
    create: () => ({
        pipeline: null as GPUComputePipeline | null,
        cache: new Map<GPUBuffer, Source>(),
        active: [] as Source[],
        activeCount: 0,
        sources: [] as Source[],
        target: null as GPUBuffer | null,
        output: null as GPUBindGroup | null,
    }),
};

/** Prepare all new bindings before publishing a new membership image. Cached source tables keep
 * their high-water capacity, so pooled producer buffers need no device allocation on reuse. */
export function rebuildMeshCounts(
    world: World,
    meshCount: number,
    pairs: number,
    slots: number,
): void {
    const state = world.resource(key);
    const meshes = world.resource(Meshes);
    for (let i = 0; i < state.sources.length; i++) state.sources[i]!.cursor = 0;
    state.activeCount = 0;
    for (let handle = 0; handle < meshCount; handle++) {
        const mesh = meshes.get(handle as MeshHandle);
        const source = mesh?.indirectIndexCount;
        if (!mesh || !source) continue;
        let entry = state.cache.get(source.buffer);
        if (!entry) {
            entry = {
                buffer: source.buffer,
                table: null!,
                group: null!,
                staging: new Uint32Array(0),
                count: 0,
                dirty: true,
                cursor: 0,
            };
            state.cache.set(source.buffer, entry);
            state.sources.push(entry);
        }
        if (entry.cursor === 0) {
            state.active[state.activeCount++] = entry;
            entry.cursor = 4;
            setCountLane(entry, 0, meshCount);
            setCountLane(entry, 1, pairs);
            setCountLane(entry, 2, slots);
        }
        const required = entry.cursor + 4;
        if (entry.staging.length < required) {
            let capacity = Math.max(8, entry.staging.length * 2);
            while (capacity < required) capacity *= 2;
            const staging = new Uint32Array(capacity);
            staging.set(entry.staging);
            const table = world.gpu.device.createBuffer({
                label: "shallot-mesh-count-rows",
                size: capacity * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            world.own(table);
            let group: GPUBindGroup;
            try {
                group = world.gpu.device.createBindGroup({
                    layout: world.gpu.root.unwrap(sourceLayout),
                    entries: [
                        { binding: 0, resource: { buffer: source.buffer } },
                        { binding: 1, resource: { buffer: table } },
                    ],
                });
            } catch (error) {
                table.destroy();
                throw error;
            }
            if (entry.table) world.retireGpuBuffer(entry.table);
            entry.table = table;
            entry.group = group;
            entry.staging = staging;
            entry.dirty = true;
            setCountLane(entry, 0, meshCount);
            setCountLane(entry, 1, pairs);
            setCountLane(entry, 2, slots);
        }
        setCountLane(entry, entry.cursor, handle);
        setCountLane(entry, entry.cursor + 1, (source.offset ?? 0) / 4);
        setCountLane(entry, entry.cursor + 2, mesh.indexCount);
        entry.cursor += 4;
    }
    if (state.activeCount && !state.pipeline)
        state.pipeline = world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: kernel }),
        );
    for (let i = 0; i < state.activeCount; i++) {
        const entry = state.active[i]!;
        entry.count = entry.cursor / 4 - 1;
        setCountLane(entry, 3, entry.count);
        if (entry.dirty) {
            world.gpu.device.queue.writeBuffer(entry.table, 0, entry.staging, 0, entry.cursor);
            entry.dirty = false;
        }
    }
}

function setCountLane(entry: Source, index: number, value: number): void {
    if (entry.staging[index] !== value) {
        entry.staging[index] = value;
        entry.dirty = true;
    }
}

export function publishMeshCounts(world: World, target: GPUBuffer): void {
    const state = world.resource(key);
    if (!state.activeCount) return;
    if (state.target !== target) {
        const group = world.gpu.device.createBindGroup({
            layout: world.gpu.root.unwrap(outputLayout),
            entries: [{ binding: 0, resource: { buffer: target } }],
        });
        state.output = group;
        state.target = target;
    }
    const pass = beginComputePass(world.frameEncoder()!, descriptor);
    setComputePipeline(pass, state.pipeline!);
    setComputeBindGroup(pass, 1, state.output!);
    for (let i = 0; i < state.activeCount; i++) {
        const entry = state.active[i]!;
        setComputeBindGroup(pass, 0, entry.group);
        dispatchWorkgroups(pass, Math.ceil(entry.count / 64));
    }
    endComputePass(pass);
}

// Sorted runs are known only after preprocessing. Read the canonical clamped count from the main
// records, rather than rebinding every producer for the transparent pass.
const transparentLayout = tgpu
    .bindGroupLayout({
        source: { storage: d.arrayOf(d.u32), access: "readonly" },
        runs: { storage: d.arrayOf(d.vec2u), access: "readonly" },
        target: { storage: d.arrayOf(d.u32), access: "mutable" },
    })
    .$idx(0);
const transparentKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { id: d.builtin.globalInvocationId },
})(({ id }) => {
    "use gpu";
    if (id.x >= transparentLayout.$.runs[0].x) return;
    const run = transparentLayout.$.runs[id.x + 1];
    transparentLayout.$.target[run.y] = transparentLayout.$.source[run.x];
});
const transparentKey = {
    create: () => ({
        pipeline: null as GPUComputePipeline | null,
        table: null as GPUBuffer | null,
        staging: new Uint32Array(0),
        group: null as GPUBindGroup | null,
        source: null as GPUBuffer | null,
        target: null as GPUBuffer | null,
    }),
};

export function publishTransparentCounts(
    world: World,
    source: GPUBuffer,
    target: GPUBuffer,
    records: readonly number[],
    offsets: readonly number[],
    count: number,
): void {
    const state = world.resource(transparentKey);
    const required = (count + 1) * 2;
    if (state.staging.length < required) {
        let capacity = 64;
        while (capacity < required) capacity *= 2;
        const table = world.gpu.device.createBuffer({
            label: "shallot-transparent-count-runs",
            size: capacity * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        world.own(table);
        if (state.table) world.retireGpuBuffer(state.table);
        state.table = table;
        state.staging = new Uint32Array(capacity);
        state.group = null;
    }
    state.pipeline ??= world.gpu.root.unwrap(
        world.gpu.root.createComputePipeline({ compute: transparentKernel }),
    );
    if (!state.group || state.source !== source || state.target !== target) {
        state.group = world.gpu.device.createBindGroup({
            layout: world.gpu.root.unwrap(transparentLayout),
            entries: [
                { binding: 0, resource: { buffer: source } },
                { binding: 1, resource: { buffer: state.table! } },
                { binding: 2, resource: { buffer: target } },
            ],
        });
        state.source = source;
        state.target = target;
    }
    state.staging[0] = count;
    for (let i = 0; i < count; i++) {
        state.staging[(i + 1) * 2] = records[i]! * 5;
        state.staging[(i + 1) * 2 + 1] = offsets[i]! / 4;
    }
    world.uploadGpuTable(state.table!, 0, state.staging.buffer, required * 4);
    const pass = beginComputePass(world.frameEncoder()!, descriptor);
    setComputePipeline(pass, state.pipeline);
    setComputeBindGroup(pass, 0, state.group);
    dispatchWorkgroups(pass, Math.ceil(count / 64));
    endComputePass(pass);
}
