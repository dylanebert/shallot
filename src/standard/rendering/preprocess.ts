import type {
    StorageFlag,
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    UniformFlag,
} from "typegpu";
import { writeToArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Meshes, MeshInstance, meshResourcesKey } from "../../core/mesh";
import {
    BeginFrameSystem,
    globalTransformTable,
    MAX_SLOTS,
    NotShadowCaster,
    RenderContext,
} from "../../core/rendering";
import { ClearChangeMarksSystem, type Registry, type System, type World } from "../../engine";
import {
    beginComputePass,
    dispatchWorkgroups,
    dispatchWorkgroups2,
    endComputePass,
    precompile,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";
import { type MaterialType, MaterialTypes } from "./material";
import { MeshInstanceInput, MeshMaterial } from "./material-data";
import { ALPHA_PIPELINE_KEY_COUNT, copyMeshAlpha, TransparentMeshes } from "./material-type";
import { materialAttributeError } from "./mesh-attributes";
import { publishMeshCounts, rebuildMeshCounts } from "./mesh-count";
import {
    CullParams,
    countKernel,
    countLayout,
    cullLayout,
    scanKernel,
    scanLayout,
    scatterKernel,
    scatterLayout,
} from "./pack";
import { type Draw, DrawIndexedIndirect, Draws } from "./registry";

const DRAW_ARG_STRIDE = d.sizeOf(DrawIndexedIndirect);
const MESH_INSTANCES = [MeshInstance];
type InstanceBuffer = TgpuBuffer<d.WgslArray<d.Vec4u>> & StorageFlag;
type AtomicU32Buffer = TgpuBuffer<d.WgslArray<d.Atomic<d.U32>>> & StorageFlag;
type Vec4fBuffer = TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag;
type DrawBuffer = TgpuBuffer<d.WgslArray<typeof DrawIndexedIndirect>> &
    StorageFlag & { usableAsIndirect: true };

// Culling during per-view compaction keeps invisible instances out of indirect draws.

/**
 * GPU draw publication. drawArgs is slot-major at `slot*pairCount+pair`, with 20-byte indirect records.
 * registerDraws seeds indexCount/firstIndex/baseVertex; the GPU pack writes instanceCount/firstInstance
 * each frame. packedEids holds one row-capacity-sized region per slot, compacted into pair slices for
 * instance_index. pairCount is material-type count * ALPHA_PIPELINE_KEY_COUNT * mesh count.
 * Buffers grow with mesh registration, table capacity and active views; Draw.args.viewStride selects a slot.
 */
export interface MeshDrawBuffers {
    /** `DrawIndexedIndirect` records, slot-major (`slot * pairCount + pair`); null until the first frame's `syncBuffers` */
    drawArgs: DrawBuffer | null;
    /** packed entity identities, one dense list per view slot; null until `warmMeshPreprocess` */
    packedEids: InstanceBuffer | null;
    /** sorted transparent identities, bound in place of `eids` for the transparent draw */
    transparentEids: InstanceBuffer | null;
    /** per-view transparent indirect records, CPU-staged after sorting */
    transparentArgs: GPUBuffer | null;
}

interface MeshPreprocessState {
    meshDraws: MeshDrawBuffers;
    counts: AtomicU32Buffer | null;
    meshBounds: Vec4fBuffer | null;
    cullParams: (TgpuBuffer<typeof CullParams> & UniformFlag) | null;
    cullGroup: TgpuBindGroup<(typeof cullLayout)["entries"]> | null;
    countPipe: TgpuComputePipeline | null;
    scanPipe: TgpuComputePipeline | null;
    scatterPipe: TgpuComputePipeline | null;
    countBound: { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null;
    scanBound: { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null;
    scatterBound: { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null;
    materialTypeCount: number;
    /** each material type's `alphaKeys` when draws were last registered */
    alphaKeys: Int32Array;
    invalidPairs: Map<number, Map<number, { name: string; cause: string }>>;
    warnedPairs: Set<string>;
    meshCount: number;
    meshVersion: number;
    publishedDraws: Set<string>;
    pairCount: number;
    viewDim: number;
    packPass: GPUComputePassDescriptor;
    countsUnwrapped: AtomicU32Buffer | null;
    countsRaw: GPUBuffer | null;
    paramsTarget: (TgpuBuffer<typeof CullParams> & UniformFlag) | null;
    paramsViewCount: number;
    paramsPairCount: number;
    paramsInstanceCount: number;
    paramsInstanceCapacity: number;
    rowCapacity: number;
    inputGeneration: Int32Array;
}

const meshPreprocessKey = { create: createMeshPreprocessState };
const meshInstanceTableKey = { create: createMeshInstanceTable };

/** Dense MeshInstance records shared by preprocessing and surface stages.
 * External material sources may publish alphaMode, alphaCutoff and cullBounds after table upload and
 * before MeshPreprocessSystem. Preserve the other fields; address an entity through rowIndex(eid),
 * or iterate activeRowsTyped's [eid, row] pairs. GPU writes do not change the CPU component columns. */
export function meshInstanceTable(world: World) {
    return world.resource(meshInstanceTableKey);
}

function createMeshInstanceTable(world: World) {
    const table = world.table("meshInstances", MeshInstanceInput);
    table.setUploadPreparation(copyMeshAlpha);
    table.enableEidLookup();
    return table;
}

function createMeshPreprocessState(): MeshPreprocessState {
    return {
        meshDraws: {
            drawArgs: null,
            packedEids: null,
            transparentEids: null,
            transparentArgs: null,
        },
        counts: null,
        meshBounds: null,
        cullParams: null,
        cullGroup: null,
        countPipe: null,
        scanPipe: null,
        scatterPipe: null,
        countBound: null,
        scanBound: null,
        scatterBound: null,
        materialTypeCount: 0,
        alphaKeys: new Int32Array(0),
        invalidPairs: new Map(),
        warnedPairs: new Set(),
        meshCount: 0,
        meshVersion: -1,
        publishedDraws: new Set(),
        pairCount: 0,
        viewDim: 1,
        packPass: { label: "shallot-mesh-preprocess" },
        countsUnwrapped: null,
        countsRaw: null,
        paramsTarget: null,
        paramsViewCount: -1,
        paramsPairCount: -1,
        paramsInstanceCount: -1,
        paramsInstanceCapacity: -1,
        rowCapacity: 0,
        inputGeneration: new Int32Array(4).fill(-1),
    };
}

export function initializeMeshPreprocess(world: World): void {
    world.resource(meshPreprocessKey);
    const table = meshInstanceTable(world);
    table.bindComponent(MeshInstance, { mesh: "mesh", cullBounds: "cullBounds" });
    table.bindFields(MeshMaterial, {
        material: "material",
        materialType: "type",
        alphaMode: "alphaMode",
        alphaCutoff: "alphaCutoff",
    });
    table.bindPresence(NotShadowCaster, "flags", 1);
    const seedDefault = (eid: number) => {
        if (!world.has(eid, MeshMaterial)) {
            world.storage(MeshMaterial).material.set(eid, 0);
            world.storage(MeshMaterial).type.set(eid, 0);
            world.storage(MeshMaterial).alphaMode.set(eid, 0);
            world.storage(MeshMaterial).alphaCutoff.set(eid, 0.5);
        }
    };
    world.onDispose(
        world.observeMembership(MeshInstance, (eid, present) => {
            if (present) seedDefault(eid);
        }),
    );
    world.onDispose(
        world.observeMembership(MeshMaterial, (eid, present) => {
            if (!present && world.has(eid, MeshInstance)) seedDefault(eid);
        }),
    );
    for (const eid of world.query([MeshInstance])) seedDefault(eid);
}

export const MeshDrawBuffers: import("../../engine").Resource<MeshDrawBuffers> = {
    create: (world) => world.resource(meshPreprocessKey).meshDraws,
};

// Count and scatter use the same visibility test so their per-pair ranges agree.
export const MeshPreprocessSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world) {
        const _render = world.resource(RenderContext);
        const _meshPreprocess = world.resource(meshPreprocessKey);

        if (!_meshPreprocess.countPipe || !_meshPreprocess.scanPipe || !_meshPreprocess.scatterPipe)
            return;
        syncBuffers(world);
        if (_meshPreprocess.pairCount === 0) return;
        publishMeshCounts(world, world.gpu.root.unwrap(_meshPreprocess.meshDraws.drawArgs!));
        const count = bindCount(world);
        const scan = bindScan(world);
        const scatter = bindScatter(world);
        if (!count || !scan || !scatter) return;

        // viewCount + pairCount let the cull shader find a view's frustum and
        // index its slot's slice; slot ≥ viewCount means no frustum (headless),
        // packed unculled. The engine submits after this pass.
        const views = Math.max(1, _render.viewCount);
        const instanceCount = meshInstanceTable(world).count;
        if (
            _meshPreprocess.paramsTarget !== _meshPreprocess.cullParams ||
            _meshPreprocess.paramsViewCount !== _render.viewCount ||
            _meshPreprocess.paramsPairCount !== _meshPreprocess.pairCount ||
            _meshPreprocess.paramsInstanceCount !== instanceCount ||
            _meshPreprocess.paramsInstanceCapacity !== _meshPreprocess.rowCapacity
        ) {
            _meshPreprocess.cullParams!.write({
                viewCount: _render.viewCount,
                pairCount: _meshPreprocess.pairCount,
                meshCount: _meshPreprocess.meshCount,
                instanceCount,
                instanceCapacity: _meshPreprocess.rowCapacity,
            });
            _meshPreprocess.paramsTarget = _meshPreprocess.cullParams;
            _meshPreprocess.paramsViewCount = _render.viewCount;
            _meshPreprocess.paramsPairCount = _meshPreprocess.pairCount;
            _meshPreprocess.paramsInstanceCount = instanceCount;
            _meshPreprocess.paramsInstanceCapacity = _meshPreprocess.rowCapacity;
        }

        if (_meshPreprocess.countsUnwrapped !== _meshPreprocess.counts) {
            _meshPreprocess.countsUnwrapped = _meshPreprocess.counts;
            _meshPreprocess.countsRaw = world.gpu.root.unwrap(_meshPreprocess.counts!);
        }
        const encoder = world.frameEncoder()!;
        encoder.clearBuffer(_meshPreprocess.countsRaw!);
        _meshPreprocess.packPass.timestampWrites = world.gpu.span?.("mesh:preprocess");
        const pass = beginComputePass(encoder, _meshPreprocess.packPass);
        const rows = Math.ceil(instanceCount / 64);
        if (rows > 0) {
            setBound(pass, count);
            dispatchWorkgroups2(pass, rows, views);
        }
        // one workgroup per live view slot; slots past them are never drawn, and a slot that becomes
        // live again is scanned before its draw
        setBound(pass, scan);
        dispatchWorkgroups(pass, views);
        if (rows > 0) {
            setBound(pass, scatter);
            dispatchWorkgroups2(pass, rows, views);
        }
        endComputePass(pass);
    },
};

function setBound(
    pass: GPUComputePassEncoder,
    bound: { pipeline: GPUComputePipeline; groups: GPUBindGroup[] },
): void {
    setComputePipeline(pass, bound.pipeline);
    for (let i = 0; i < bound.groups.length; i++) setComputeBindGroup(pass, i, bound.groups[i]!);
}

// Bind dense MeshInstance and Transform tables, replacing groups only when one of their GPU buffers grows.
function cullGroup(world: World): TgpuBindGroup<(typeof cullLayout)["entries"]> | null {
    const _meshPreprocess = world.resource(meshPreprocessKey);

    if (!_meshPreprocess.cullParams || !_meshPreprocess.meshBounds) return null;
    const meshInstances = meshInstanceTable(world);
    const globalTransforms = globalTransformTable(world);
    const generation = _meshPreprocess.inputGeneration;
    if (
        generation[0] !== meshInstances.generation ||
        generation[1] !== meshInstances.activeGeneration ||
        generation[2] !== globalTransforms.generation ||
        generation[3] !== globalTransforms.mapGeneration
    ) {
        unbind(world);
        generation[0] = meshInstances.generation;
        generation[1] = meshInstances.activeGeneration;
        generation[2] = globalTransforms.generation;
        generation[3] = globalTransforms.mapGeneration;
    }
    if (_meshPreprocess.cullGroup) return _meshPreprocess.cullGroup;
    const cullVolumes = world.resource(RenderContext).cullVolumes;
    const instanceRows = meshInstances.activeRowsBuffer;
    const globalTransformRows = globalTransforms.eidToRowBuffer;
    if (!cullVolumes || !instanceRows || !globalTransformRows) {
        throw new Error(
            "[mesh-preprocess] dense table inputs missing: cull volumes, MeshInstance rows or GlobalTransform row lookup",
        );
    }
    _meshPreprocess.cullGroup = world.gpu.root.createBindGroup(cullLayout, {
        instanceRows,
        instances: meshInstances.buffer,
        globalTransforms: globalTransforms.buffer,
        globalTransformRows,
        meshBounds: _meshPreprocess.meshBounds,
        cullVolumes,
        params: _meshPreprocess.cullParams,
    });
    return _meshPreprocess.cullGroup;
}

function bindCount(world: World): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _meshPreprocess = world.resource(meshPreprocessKey);

    const cull = cullGroup(world);
    if (_meshPreprocess.countBound) return _meshPreprocess.countBound;
    if (!_meshPreprocess.countPipe || !cull || !_meshPreprocess.counts) return null;
    _meshPreprocess.countBound = {
        pipeline: world.gpu.root.unwrap(_meshPreprocess.countPipe),
        groups: [
            world.gpu.root.unwrap(cull),
            world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(countLayout, { counts: _meshPreprocess.counts }),
            ),
        ],
    };
    return _meshPreprocess.countBound;
}

function bindScan(world: World): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _meshPreprocess = world.resource(meshPreprocessKey);
    const _meshDraws = world.resource(MeshDrawBuffers);

    if (_meshPreprocess.scanBound) return _meshPreprocess.scanBound;
    if (
        !_meshPreprocess.scanPipe ||
        !_meshPreprocess.counts ||
        !_meshDraws.drawArgs ||
        !_meshPreprocess.cullParams
    )
        return null;
    _meshPreprocess.scanBound = {
        pipeline: world.gpu.root.unwrap(_meshPreprocess.scanPipe),
        groups: [
            world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(scanLayout, {
                    counts: _meshPreprocess.counts,
                    drawArgs: _meshDraws.drawArgs,
                    params: _meshPreprocess.cullParams,
                }),
            ),
        ],
    };
    return _meshPreprocess.scanBound;
}

function bindScatter(
    world: World,
): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _meshPreprocess = world.resource(meshPreprocessKey);
    const _meshDraws = world.resource(MeshDrawBuffers);

    const cull = cullGroup(world);
    if (_meshPreprocess.scatterBound) return _meshPreprocess.scatterBound;
    if (
        !_meshPreprocess.scatterPipe ||
        !cull ||
        !_meshPreprocess.counts ||
        !_meshDraws.drawArgs ||
        !_meshDraws.packedEids
    )
        return null;
    _meshPreprocess.scatterBound = {
        pipeline: world.gpu.root.unwrap(_meshPreprocess.scatterPipe),
        groups: [
            world.gpu.root.unwrap(cull),
            world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(scatterLayout, {
                    drawArgs: world.gpu.root.unwrap(_meshDraws.drawArgs),
                    packedEids: _meshDraws.packedEids,
                }),
            ),
        ],
    };
    return _meshPreprocess.scatterBound;
}

// every bound pipeline names at least one buffer `syncBuffers` can reallocate, so growth drops all of
// them together rather than tracking which buffer each one holds
function unbind(world: World): void {
    const _meshPreprocess = world.resource(meshPreprocessKey);

    _meshPreprocess.cullGroup = null;
    _meshPreprocess.countBound = null;
    _meshPreprocess.scanBound = null;
    _meshPreprocess.scatterBound = null;
}

// Mesh growth changes pair strides, requiring draw-record republication. drawArgs/counts scale with
// view capacity * pairCount; packed lists with view capacity * row capacity; mesh bounds with mesh count.
// Pipelines read cullParams/arrayLength, so growing any axis replaces buffers without recompilation.
// Submitted work retains old backing storage.
function syncBuffers(world: World): void {
    const _meshPreprocess = world.resource(meshPreprocessKey);
    const _meshDraws = world.resource(MeshDrawBuffers);

    if (_meshPreprocess.materialTypeCount === 0) return;
    const meshes = world.resource(meshResourcesKey).meshes;
    const meshCount = meshes.capacity;
    const changedMeshes = meshes.version !== _meshPreprocess.meshVersion;
    const viewCount = Math.max(1, world.resource(RenderContext).viewCount);
    const rowCapacity = meshInstanceTable(world).capacity;
    const growMesh = meshCount > _meshPreprocess.meshCount;
    const growView = viewCount > _meshPreprocess.viewDim;
    const growRows = rowCapacity > _meshPreprocess.rowCapacity;
    const growKeys = alphaKeysChanged(world);
    if (changedMeshes || growMesh || growView || growKeys) {
        const meshDim = Math.max(_meshPreprocess.meshCount, meshCount);
        const viewDim = growView
            ? Math.min(MAX_SLOTS, Math.max(viewCount, _meshPreprocess.viewDim * 2))
            : _meshPreprocess.viewDim;
        rebuildMeshCounts(
            world,
            meshDim,
            _meshPreprocess.materialTypeCount * ALPHA_PIPELINE_KEY_COUNT * meshDim,
            viewDim,
        );
    }
    if (!changedMeshes && !growMesh && !growView && !growRows && _meshDraws.drawArgs) {
        if (growKeys) registerDraws(world);
        return;
    }

    const device = world.gpu.device;
    _meshPreprocess.meshCount = Math.max(_meshPreprocess.meshCount, meshCount);
    // Geometric growth bounds reallocations when cameras arrive one at a time.
    if (growView)
        _meshPreprocess.viewDim = Math.min(
            MAX_SLOTS,
            Math.max(viewCount, _meshPreprocess.viewDim * 2),
        );
    _meshPreprocess.rowCapacity = Math.max(_meshPreprocess.rowCapacity, rowCapacity);
    _meshPreprocess.pairCount =
        _meshPreprocess.materialTypeCount * ALPHA_PIPELINE_KEY_COUNT * _meshPreprocess.meshCount;
    const records = _meshPreprocess.viewDim * _meshPreprocess.pairCount;

    const staleArgs: (DrawBuffer | AtomicU32Buffer | null)[] = [];
    const replaceArgs = growMesh || growView || !_meshDraws.drawArgs;
    if (replaceArgs) {
        staleArgs.push(_meshDraws.drawArgs, _meshPreprocess.counts);
        _meshDraws.drawArgs = world.gpu.root
            .createBuffer(d.arrayOf(DrawIndexedIndirect, records))
            .$usage("storage", "indirect")
            .$name("shallot-draw-args");
        _meshPreprocess.counts = world.gpu.root
            .createBuffer(d.arrayOf(d.atomic(d.u32), records))
            .$usage("storage")
            .$name("shallot-mesh-preprocess-counts");
    }

    let stalePacked: InstanceBuffer | null = null;
    let staleTransparentEids: InstanceBuffer | null = null;
    let staleTransparentArgs: GPUBuffer | null = null;
    if (growView || growRows || !_meshDraws.packedEids) {
        stalePacked = _meshDraws.packedEids;
        const listCapacity = _meshPreprocess.viewDim * _meshPreprocess.rowCapacity;
        _meshDraws.packedEids = world.gpu.root
            .createBuffer(d.arrayOf(d.vec4u, listCapacity))
            .$usage("storage")
            .$name("shallot-packed-eids");

        staleTransparentEids = _meshDraws.transparentEids;
        _meshDraws.transparentEids = world.gpu.root
            .createBuffer(d.arrayOf(d.vec4u, listCapacity))
            .$usage("storage")
            .$name("shallot-transparent-eids");
        staleTransparentArgs = _meshDraws.transparentArgs;
        _meshDraws.transparentArgs = device.createBuffer({
            label: "shallot-transparent-args",
            size: Math.max(20, listCapacity * d.sizeOf(DrawIndexedIndirect)),
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
        });
        world.own(_meshDraws.transparentArgs);
    }

    // Bounds follow membership even when a freed handle is reused without growth.
    let staleBounds: Vec4fBuffer | null = null;
    if (growMesh || !_meshPreprocess.meshBounds) {
        staleBounds = _meshPreprocess.meshBounds;
        _meshPreprocess.meshBounds = world.gpu.root
            .createBuffer(d.arrayOf(d.vec4f, _meshPreprocess.meshCount))
            .$usage("storage")
            .$name("shallot-mesh-bounds");
    }
    if (changedMeshes || staleBounds) writeMeshBounds(world, device);

    if (replaceArgs || growView || growRows || staleBounds) unbind(world);
    // draw records depend only on the meshes, the types' alpha keys and the args buffer, so rows
    // growth keeps them
    if (changedMeshes || replaceArgs || growKeys) registerDraws(world);
    _meshPreprocess.meshVersion = meshes.version;

    const stale = [...staleArgs, stalePacked, staleTransparentEids, staleBounds];
    // Replaced bindings cannot be used by this frame; submitted work retains its backing storage.
    for (const buffer of stale) buffer?.destroy();
    staleTransparentArgs?.destroy();
}

function alphaKeysChanged(world: World): boolean {
    world.resource(TransparentMeshes).growSources();
    const known = world.resource(meshPreprocessKey).alphaKeys;
    const sources = world.resource(MaterialTypes).sources;
    for (let type = 0; type < known.length; type++)
        if ((sources[type]?.alphaKeys ?? 0) !== known[type]) return true;
    return false;
}

/**
 * Fill the per-mesh local bounding sphere buffer (one `vec4` per
 * mesh handle: `xyz` center, `w` radius). A mesh without `bounds` (a procedural
 * producer that didn't supply one) gets a sentinel radius so the cull keeps it
 * always-visible rather than wrongly culling it
 */
function writeMeshBounds(world: World, device: GPUDevice): void {
    const _meshPreprocess = world.resource(meshPreprocessKey);
    const _meshes = world.resource(Meshes);

    const buffer = _meshPreprocess.meshBounds!;
    const data = new Float32Array(_meshPreprocess.meshCount * 4);
    for (const [handle, mesh] of _meshes.entries()) {
        if (mesh.bounds) data.set(mesh.bounds, handle * 4);
        else data[handle * 4 + 3] = 1e30; // never-cull sentinel
    }
    device.queue.writeBuffer(world.gpu.root.unwrap(buffer), 0, data as Float32Array<ArrayBuffer>);
}

type DrawRecord = {
    indexCount: number;
    instanceCount: number;
    firstIndex: number;
    baseVertex: number;
    firstInstance: number;
};

/**
 * Publishes (material type, alpha pipeline key, mesh) draw pairs for keys each type's rows have held.
 * Returns byte offsets and static indirect records for one view slot; the caller uploads them to
 * every slot. Device-free with explicit registries so ordering checks can exercise this same path. @internal
 */
export function publishMeshInstanceDraws(
    world: World,
    drawArgs: DrawBuffer,
    pairCount: number,
    registries: {
        materialTypes: readonly (MaterialType | undefined)[];
        sources: readonly ({ readonly alphaKeys: number } | undefined)[];
        meshes: ReturnType<typeof Meshes.create>;
        draws: Registry<Draw>;
    } = {
        materialTypes: world.resource(MaterialTypes).types,
        sources: world.resource(MaterialTypes).sources,
        meshes: world.resource(Meshes),
        draws: world.resource(Draws),
    },
): { offset: number; args: DrawRecord }[] {
    const { materialTypes, sources, meshes, draws } = registries;
    const invalidPairs = world.resource(meshPreprocessKey).invalidPairs;
    invalidPairs.clear();
    const published = world.resource(meshPreprocessKey).publishedDraws;
    for (const name of published) draws.delete(name);
    published.clear();
    const writes: { offset: number; args: DrawRecord }[] = [];
    const viewStride = pairCount * DRAW_ARG_STRIDE;
    for (const [materialType, definition] of materialTypes.entries()) {
        if (!definition) continue;
        const alphaKeys = sources[materialType]?.alphaKeys ?? 0;
        for (
            let alphaPipelineKey = 0;
            alphaPipelineKey < ALPHA_PIPELINE_KEY_COUNT;
            alphaPipelineKey++
        ) {
            // a key no row has held has no instance; its records stay unwritten and undrawn
            if (!(alphaKeys & (1 << alphaPipelineKey))) continue;
            for (const [handle, m] of meshes.entries()) {
                const cause = materialAttributeError(m, definition);
                if (cause) {
                    let rejected = invalidPairs.get(materialType);
                    if (!rejected) invalidPairs.set(materialType, (rejected = new Map()));
                    rejected.set(handle, {
                        name: `mesh:material:${definition.name}:${m.name}:${handle}`,
                        cause,
                    });
                    continue;
                }
                const pair =
                    (materialType * ALPHA_PIPELINE_KEY_COUNT + alphaPipelineKey) *
                        (pairCount / (materialTypes.length * ALPHA_PIPELINE_KEY_COUNT)) +
                    handle;
                const offset = pair * DRAW_ARG_STRIDE;
                // Indices are absolute vertex positions, so baseVertex remains zero.
                const args = {
                    indexCount: m.indexCount,
                    instanceCount: 0,
                    firstIndex: m.indexBase,
                    baseVertex: 0,
                    firstInstance: 0,
                };
                writes.push({ offset, args });
                const name = `mesh:material:${definition.name}:${alphaPipelineKey}:${m.name}:${handle}`;
                published.add(name);
                draws.register({
                    name,
                    materialType,
                    alphaPipelineKey,
                    mesh: handle,
                    args: { indirect: drawArgs, offset, viewStride },
                });
            }
        }
    }
    return writes;
}

function warnInvalidInstance(world: World, eid: number): void {
    if (!world.has(eid, MeshInstance)) return;
    const { invalidPairs, warnedPairs } = world.resource(meshPreprocessKey);
    const rejected = invalidPairs
        .get(world.storage(MeshMaterial).type.get(eid))
        ?.get(world.storage(MeshInstance).mesh.get(eid));
    if (!rejected || warnedPairs.has(rejected.name)) return;
    warnedPairs.add(rejected.name);
    console.warn(`standard: draw "${rejected.name}" skipped — ${rejected.cause}`);
}

export const MeshPairingSystem: System = {
    group: "draw",
    first: true,
    before: [ClearChangeMarksSystem],
    update(world) {
        if (world.resource(meshPreprocessKey).invalidPairs.size === 0) return;
        const meshes = world.fieldStorage(MeshInstance, "mesh").dirty;
        const types = world.fieldStorage(MeshMaterial, "type").dirty;
        const words = (world.entityHighWater + 31) >>> 5;
        for (let word = 0; word < words; word++) {
            let bits = (meshes[word] ?? 0) | (types[word] ?? 0);
            while (bits) {
                const bit = bits & -bits;
                warnInvalidInstance(world, (word << 5) + 31 - Math.clz32(bit));
                bits ^= bit;
            }
        }
    },
};

function registerDraws(world: World): void {
    const _meshDraws = world.resource(MeshDrawBuffers);
    const _meshPreprocess = world.resource(meshPreprocessKey);

    if (!world.gpu.device || !_meshDraws.drawArgs || _meshPreprocess.pairCount === 0) return;
    const viewStride = _meshPreprocess.pairCount * DRAW_ARG_STRIDE;
    // one slot's records packed once, then one write per view slot rather than one per record
    const slot = new Uint8Array(viewStride);
    const record = new ArrayBuffer(DRAW_ARG_STRIDE);
    const recordBytes = new Uint8Array(record);
    for (const { offset, args } of publishMeshInstanceDraws(
        world,
        _meshDraws.drawArgs,
        _meshPreprocess.pairCount,
    )) {
        writeToArrayBuffer(record, DrawIndexedIndirect, args);
        slot.set(recordBytes, offset);
    }
    // A new rejection can affect unchanged instances; ordinary frames check only marked pairings.
    if (_meshPreprocess.invalidPairs.size !== 0)
        for (const eid of world.query(MESH_INSTANCES)) warnInvalidInstance(world, eid);
    const target = world.gpu.root.unwrap(_meshDraws.drawArgs);
    for (let view = 0; view < _meshPreprocess.viewDim; view++)
        world.gpu.device.queue.writeBuffer(target, view * viewStride, slot);
    const sources = world.resource(MaterialTypes).sources;
    for (let type = 0; type < _meshPreprocess.alphaKeys.length; type++)
        _meshPreprocess.alphaKeys[type] = sources[type]?.alphaKeys ?? 0;
}

export function initMeshPreprocess(world: World): void {
    unbind(world);
}

// Warm runs after all initialize hooks, so the registered material-type count is final and can be
// baked into shaders. Mesh/view counts are not final; defer their buffer sizing to syncBuffers and
// read their dimensions from cullParams so later mesh registrations or cameras need no recompilation.
export function warmMeshPreprocess(world: World): void {
    const _meshPreprocess = world.resource(meshPreprocessKey);
    const _meshDraws = world.resource(MeshDrawBuffers);

    if (!world.gpu.device) return;
    const root = world.gpu.root;
    _meshPreprocess.materialTypeCount = world.resource(MaterialTypes).types.length;
    _meshPreprocess.alphaKeys = new Int32Array(_meshPreprocess.materialTypeCount);
    _meshPreprocess.meshCount = 0;
    _meshPreprocess.pairCount = 0;
    _meshPreprocess.viewDim = 1;
    _meshDraws.drawArgs = null;
    _meshDraws.packedEids = null;
    _meshDraws.transparentEids = null;
    _meshDraws.transparentArgs = null;
    _meshPreprocess.counts = null;
    _meshPreprocess.meshBounds = null;
    _meshPreprocess.rowCapacity = 0;
    _meshPreprocess.inputGeneration.fill(-1);
    _meshPreprocess.paramsViewCount = -1;
    _meshPreprocess.paramsPairCount = -1;
    _meshPreprocess.paramsInstanceCount = -1;
    _meshPreprocess.paramsInstanceCapacity = -1;
    unbind(world);

    _meshPreprocess.cullParams = root
        .createBuffer(CullParams)
        .$usage("uniform")
        .$name("shallot-mesh-preprocess-cull-params");
    if (_meshPreprocess.materialTypeCount === 0) return;

    _meshPreprocess.countPipe = root
        .createComputePipeline({ compute: countKernel(_meshPreprocess.materialTypeCount) })
        .$name("shallot-mesh-preprocess-count");
    _meshPreprocess.scanPipe = root
        .createComputePipeline({ compute: scanKernel() })
        .$name("shallot-mesh-preprocess-scan");
    _meshPreprocess.scatterPipe = root
        .createComputePipeline({ compute: scatterKernel(_meshPreprocess.materialTypeCount) })
        .$name("shallot-mesh-preprocess-scatter");

    // Warm hooks run concurrently under Promise.all. Defer allocation and binding to the precompile
    // drain after all hooks resolve, when meshes, dense tables, transform lookup and cull volumes are
    // published. Separate forcers give each pipeline its own compile-report row.
    precompile(world, "shallot-mesh-preprocess-count", () => {
        syncBuffers(world);
        const bound = bindCount(world);
        return bound && [bound.pipeline];
    });
    precompile(world, "shallot-mesh-preprocess-scan", () => {
        const bound = bindScan(world);
        return bound && [bound.pipeline];
    });
    precompile(world, "shallot-mesh-preprocess-scatter", () => {
        const bound = bindScatter(world);
        return bound && [bound.pipeline];
    });
}
