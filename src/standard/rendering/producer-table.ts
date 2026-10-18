import tgpu, { type StorageFlag, type TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { MeshHandle } from "../../core/mesh";
import { CullVolumes, Xform } from "../../core/rendering";
import type { World } from "../../engine";
import type { MaterialHandle } from "./material";
import { MeshInstanceInput } from "./material-data";
import { ALPHA_PIPELINE_KEY_COUNT, AlphaModeCode, AlphaPipelineKey } from "./material-type";
import {
    CULL_STRIDE,
    CullParams,
    countLayout,
    MeshBounds,
    scatterLayout,
    visibility,
} from "./pack";

/** World-owned GPU instance columns. The producer owns initialization, count reset and frame ordering. */
export interface ProducerTable {
    readonly name: string;
    readonly capacity: number;
    readonly placements: TgpuBuffer<d.WgslArray<typeof Xform>> & StorageFlag;
    readonly meshInstances: TgpuBuffer<d.WgslArray<typeof MeshInstanceInput>> & StorageFlag;
    readonly count: TgpuBuffer<d.Atomic<d.U32>> & StorageFlag;
}

export const producerTables = {
    create: () => ({
        tables: [] as ProducerEntry[],
        draws: new Map<string, ProducerTable | null>(),
    }),
};
export interface ProducerEntry {
    table: ProducerTable;
    meshes: readonly MeshHandle[];
    materials: readonly MaterialHandle[];
}

/**
 * Register fixed-capacity GPU-only instance columns owned by this World; there is no removal API.
 * Write rows [0, count) before MeshPreprocessSystem on world.frameEncoder(), or during fixed.
 * Standard clamps count to capacity. Placement rows use Xform; meshInstances use
 * MeshInstanceInput, whose JSDoc gives each field's source and the silent failure a wrong value causes.
 * Material shaders read eid as 0xFFFFFFFF (the invalid entity id) and row as this table's row index.
 * Entity rows instead read their entity id in eid and 0xFFFFFFFF in row.
 * CPU picking, interpolation and snapshots do not include these rows. Blended materials are refused
 * because transparent sorting is on the CPU. Only declared mesh and material-type/alpha pairs draw.
 */
export function registerProducerTable(
    world: World,
    options: {
        name: string;
        capacity: number;
        meshes: readonly MeshHandle[];
        materials: readonly MaterialHandle[];
    },
): ProducerTable {
    const { name, capacity } = options;
    const fail = (cause: string): never => {
        throw new Error(`Producer table "${name}": ${cause}`);
    };
    if (!Number.isSafeInteger(capacity) || capacity < 1)
        fail("capacity must be a positive integer");
    if (capacity > 65535 * 64)
        fail("capacity exceeds one indirect dispatch (65535 workgroups × 64)");
    const limits = world.gpu.device.limits;
    for (const [column, bytes] of [
        ["placements", d.sizeOf(Xform)],
        ["meshInstances", d.sizeOf(MeshInstanceInput)],
    ] as const) {
        if (capacity * bytes > limits.maxStorageBufferBindingSize)
            fail(`${column} exceeds maxStorageBufferBindingSize`);
        if (capacity * bytes > limits.maxBufferSize) fail(`${column} exceeds maxBufferSize`);
    }
    const state = world.resource(producerTables);
    if (state.tables.some((entry) => entry.table.name === name)) fail("name is already registered");
    for (const material of options.materials) {
        if (
            material.alphaMode >= AlphaModeCode.Blend &&
            material.alphaMode <= AlphaModeCode.Multiply
        )
            fail(
                `material ${material.type}:${material.material} is blended; transparent sorting is on the CPU`,
            );
    }
    const root = world.gpu.root;
    const table: ProducerTable = {
        name,
        capacity,
        placements: root
            .createBuffer(d.arrayOf(Xform, capacity))
            .$usage("storage")
            .$name(`${name}:placements`),
        meshInstances: root
            .createBuffer(d.arrayOf(MeshInstanceInput, capacity))
            .$usage("storage")
            .$name(`${name}:meshInstances`),
        count: root.createBuffer(d.atomic(d.u32)).$usage("storage").$name(`${name}:count`),
    };
    state.tables.push({ table, meshes: [...options.meshes], materials: [...options.materials] });
    return table;
}

export const ProducerDispatch = d.struct({ x: d.u32, y: d.u32, z: d.u32, count: d.u32 });
export const ProducerParams = d.struct({ capacity: d.u32, pairOffset: d.u32, basePairs: d.u32 });
export const producerArgsLayout = tgpu
    .bindGroupLayout({
        count: { storage: d.atomic(d.u32), access: "mutable" },
        args: { storage: ProducerDispatch, access: "mutable" },
        producer: { uniform: ProducerParams },
        params: { uniform: CullParams },
    })
    .$idx(0);
export const producerCullLayout = tgpu
    .bindGroupLayout({
        instances: { storage: d.arrayOf(MeshInstanceInput), access: "readonly" },
        globalTransforms: { storage: d.arrayOf(Xform), access: "readonly" },
        meshBounds: { storage: d.arrayOf(MeshBounds), access: "readonly" },
        args: { storage: ProducerDispatch, access: "readonly" },
        producer: { uniform: ProducerParams },
        params: { uniform: CullParams },
        cullVolumes: { uniform: CullVolumes },
    })
    .$idx(0);
export const producerArgsKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    "use gpu";
    const n = std.min(
        std.atomicLoad(producerArgsLayout.$.count),
        producerArgsLayout.$.producer.capacity,
    );
    producerArgsLayout.$.args = ProducerDispatch({
        x: (n + 63) >>> 6,
        y: std.max(1, producerArgsLayout.$.params.viewCount),
        z: 1,
        count: n,
    });
});
const producerVisible = visibility(producerCullLayout);
const producerPair = tgpu.fn(
    [d.u32, d.u32],
    d.u32,
)((row, slot) => {
    "use gpu";
    const instance = producerCullLayout.$.instances[row];
    const mid = instance.mesh & 0xfffff;
    const params = producerCullLayout.$.params;
    const basePairs = producerCullLayout.$.producer.basePairs;
    if (
        mid >= params.meshCount ||
        producerCullLayout.$.meshBounds[mid].handle !== instance.mesh ||
        instance.materialType >= d.u32(basePairs / (ALPHA_PIPELINE_KEY_COUNT * params.meshCount)) ||
        (instance.alphaMode !== AlphaModeCode.Opaque &&
            instance.alphaMode !== AlphaModeCode.Mask &&
            instance.alphaMode !== AlphaModeCode.AlphaToCoverage)
    )
        return params.pairCount;
    if ((instance.flags & 1) !== 0 && producerCullLayout.$.cullVolumes[slot * CULL_STRIDE].y !== 0)
        return params.pairCount;
    if (!producerVisible(mid, row, row, slot)) return params.pairCount;
    let key: number = instance.alphaMode;
    if (key === AlphaModeCode.AlphaToCoverage) key = AlphaPipelineKey.AlphaToCoverage;
    return (
        producerCullLayout.$.producer.pairOffset +
        (instance.materialType * ALPHA_PIPELINE_KEY_COUNT + key) * params.meshCount +
        mid
    );
});
export const producerCountKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
    "use gpu";
    if (gid.x >= producerCullLayout.$.args.count) return;
    const pair = producerPair(gid.x, gid.y);
    if (pair >= producerCullLayout.$.params.pairCount) return;
    std.atomicAdd(countLayout.$.counts[gid.y * producerCullLayout.$.params.pairCount + pair], 1);
});
export const producerScatterKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
})(({ gid }) => {
    "use gpu";
    if (gid.x >= producerCullLayout.$.args.count) return;
    const pair = producerPair(gid.x, gid.y);
    if (pair >= producerCullLayout.$.params.pairCount) return;
    const idx = gid.y * producerCullLayout.$.params.pairCount + pair;
    const remaining = std.atomicSub(scatterLayout.$.drawArgs[idx].instanceCount, 1);
    const target = scatterLayout.$.drawArgs[idx].firstInstance + remaining - 1;
    scatterLayout.$.packedEids[target] = d.vec4u(0xffffffff, gid.x, gid.x + 1, 0);
    if (remaining === 1) {
        const count = d.u32(scatterLayout.$.drawArgs[idx].baseVertex);
        scatterLayout.$.drawArgs[idx].baseVertex = 0;
        std.atomicStore(scatterLayout.$.drawArgs[idx].instanceCount, count);
    }
});
