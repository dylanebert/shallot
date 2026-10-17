import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Meshes } from "../../core/mesh";
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
};
const key = {
    create: () => ({
        pipeline: null as GPUComputePipeline | null,
        cache: new Map<GPUBuffer, Source>(),
        active: [] as Source[],
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
    const rows = new Map<GPUBuffer, number[]>();
    for (const [handle, mesh] of world.resource(Meshes).entries()) {
        const source = mesh.indirectIndexCount;
        if (!source) continue;
        let data = rows.get(source.buffer);
        if (!data) {
            data = [meshCount, pairs, slots, 0];
            rows.set(source.buffer, data);
        }
        data.push(handle, (source.offset ?? 0) / 4, mesh.indexCount, 0);
        data[3]!++;
    }
    const prepared: { entry: Source; data: number[]; previous?: Source }[] = [];
    try {
        for (const [buffer, data] of rows) {
            const previous = state.cache.get(buffer);
            let entry = previous;
            if (!entry || entry.staging.length < data.length) {
                let capacity = 8;
                while (capacity < data.length) capacity *= 2;
                const table = world.gpu.device.createBuffer({
                    label: "shallot-mesh-count-rows",
                    size: capacity * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                world.own(table);
                try {
                    const group = world.gpu.device.createBindGroup({
                        layout: world.gpu.root.unwrap(sourceLayout),
                        entries: [
                            { binding: 0, resource: { buffer } },
                            { binding: 1, resource: { buffer: table } },
                        ],
                    });
                    entry = { buffer, table, group, staging: new Uint32Array(capacity), count: 0 };
                } catch (error) {
                    table.destroy();
                    throw error;
                }
            }
            prepared.push({ entry, data, previous });
        }
        if (prepared.length && !state.pipeline)
            state.pipeline = world.gpu.root.unwrap(
                world.gpu.root.createComputePipeline({ compute: kernel }),
            );
    } catch (error) {
        for (const item of prepared) if (item.entry !== item.previous) item.entry.table.destroy();
        throw error;
    }
    const active: Source[] = [];
    for (const { entry, data, previous } of prepared) {
        let changed = entry.count !== data[3];
        for (let i = 0; i < data.length; i++)
            if (entry.staging[i] !== data[i]) {
                entry.staging[i] = data[i]!;
                changed = true;
            }
        if (changed)
            world.gpu.device.queue.writeBuffer(entry.table, 0, entry.staging, 0, data.length);
        entry.count = data[3]!;
        if (entry !== previous) {
            previous?.table.destroy();
            state.cache.set(entry.buffer, entry);
        }
        active.push(entry);
    }
    state.active = active;
}

export function publishMeshCounts(world: World, target: GPUBuffer): void {
    const state = world.resource(key);
    if (!state.active.length) return;
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
    for (const entry of state.active) {
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
        state.table?.destroy();
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
