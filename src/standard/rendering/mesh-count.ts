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

const passDescriptor = { label: "shallot-mesh-count" };

const layout = tgpu
    .bindGroupLayout({
        source: { storage: d.arrayOf(d.u32), access: "readonly" },
        result: { storage: d.arrayOf(d.u32), access: "mutable" },
        params: { uniform: d.vec2u },
    })
    .$idx(0);
const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    "use gpu";
    layout.$.result[0] = std.min(layout.$.source[layout.$.params.x], layout.$.params.y);
});
type Entry = { handle: MeshHandle; group: GPUBindGroup; result: GPUBuffer; params: GPUBuffer };
const key = {
    create: () => {
        return {
            pipeline: null as GPUComputePipeline | null,
            entries: [] as Entry[],
            byHandle: new Map<MeshHandle, Entry>(),
        };
    },
};

export function rebuildMeshCounts(world: World): void {
    const meshes = world.resource(Meshes);
    const state = world.resource(key);
    for (const entry of state.entries) {
        entry.result.destroy();
        entry.params.destroy();
    }
    state.entries.length = 0;
    state.byHandle.clear();
    for (const [handle, mesh] of meshes.entries()) {
        const source = mesh.indirectIndexCount;
        if (!source) continue;
        state.pipeline ??= world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: kernel }),
        );
        const offset = source.offset ?? 0;
        if (
            offset % 4 ||
            offset < 0 ||
            offset + 4 > source.buffer.size ||
            !(source.buffer.usage & GPUBufferUsage.STORAGE)
        )
            throw new Error(
                `mesh ${mesh.name}: indirectIndexCount requires a STORAGE buffer and an aligned in-range u32 byte offset`,
            );
        const result = world.gpu.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const params = world.gpu.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        world.own(result);
        world.own(params);
        world.gpu.device.queue.writeBuffer(
            params,
            0,
            new Uint32Array([offset / 4, mesh.indexCount]),
        );
        const group = world.gpu.device.createBindGroup({
            layout: world.gpu.root.unwrap(layout),
            entries: [
                { binding: 0, resource: { buffer: source.buffer } },
                { binding: 1, resource: { buffer: result } },
                { binding: 2, resource: { buffer: params } },
            ],
        });
        const entry = { handle, group, result, params };
        state.entries.push(entry);
        state.byHandle.set(handle, entry);
    }
}

export function publishMeshCounts(
    world: World,
    target: GPUBuffer,
    meshCount: number,
    pairs: number,
    slots: number,
): void {
    const state = world.resource(key);
    if (!state.entries.length) return;
    const encoder = world.frameEncoder()!;
    const pass = beginComputePass(encoder, passDescriptor);
    setComputePipeline(pass, state.pipeline!);
    for (const entry of state.entries) {
        setComputeBindGroup(pass, 0, entry.group);
        dispatchWorkgroups(pass, 1);
    }
    endComputePass(pass);
    for (const entry of state.entries)
        for (let slot = 0; slot < slots; slot++)
            for (let pair: number = entry.handle; pair < pairs; pair += meshCount)
                encoder.copyBufferToBuffer(entry.result, 0, target, (slot * pairs + pair) * 20, 4);
}

export function copyMeshCount(
    world: World,
    mesh: MeshHandle,
    target: GPUBuffer,
    offset: number,
): void {
    const entry = world.resource(key).byHandle.get(mesh);
    if (entry) world.frameEncoder()!.copyBufferToBuffer(entry.result, 0, target, offset, 4);
}
