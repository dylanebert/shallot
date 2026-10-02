import type {
    StorageFlag,
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    UniformFlag,
} from "typegpu";
import { writeToArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import { type Mesh, Meshes, MeshInstance } from "../../core/mesh";
import { BeginFrameSystem, NotShadowCaster, Render } from "../../core/rendering";
import type { Registry, System, World } from "../../engine";
import { globalTransformTable } from "../../engine";
import { precompile } from "../../engine/runtime";
import type { Surface } from "./contract";
import { Surfaces } from "./contract";
import { MeshMaterial, materialTable } from "./material";
import {
    CullParams,
    countKernel,
    countLayout,
    cullLayout,
    PartRecord,
    scanKernel,
    scanLayout,
    scatterKernel,
    scatterLayout,
} from "./pack";
import { type Draw, DrawIndexedIndirect, Draws } from "./registry";

// stride derived from the schema (a second hand-authored stride is layout drift waiting to
// happen).
const DRAW_ARG_STRIDE = d.sizeOf(DrawIndexedIndirect);
type InstanceBuffer = TgpuBuffer<d.WgslArray<d.Vec4u>> & StorageFlag;
type AtomicU32Buffer = TgpuBuffer<d.WgslArray<d.Atomic<d.U32>>> & StorageFlag;
type Vec4fBuffer = TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag;
type DrawBuffer = TgpuBuffer<d.WgslArray<typeof DrawIndexedIndirect>> &
    StorageFlag & { usableAsIndirect: true };

// Pack is cull → count → scan → scatter, run per active view: count tallies the
// frustum-visible parts per (view, pair), the single-thread scan turns counts
// into each (view, pair)'s instanceCount + compacted firstInstance (written
// into drawArgs), scatter appends each surviving eid into its slice of
// packedEids. The cull test (instance bound vs the slot's `cullVolumes[slot]`) gates both
// count and scatter, so off-screen parts never reach the indirect args — this
// is niagara's cull → compact → drawIndirect spine. Output is slot-major: each
// camera owns its own drawArgs records + packedEids region, so the four-up
// example culls each view independently and the shadow pass reuses
// the same pack against the sun's frustum as one more slot. registerDraws
// writes the static indexCount + firstIndex; the per-view dimension grows
// lazily with the active camera count, the pair dimension with mesh count.
// Culling lives here, in the producer's per-view compaction, shadow slots
// included, because a consumer that culled would already have paid for every
// instance it never draws.
// Per-world GPU pack state is created by PartPlugin.initialize below.

/**
 * GPU-resident MeshInstance draw publication. `drawArgs` holds `DrawIndexedIndirect` entries
 * (20 bytes) laid out slot-major (`slot * pairCount + pair`), so each camera
 * has its own per-pair records: static indexCount/firstIndex/baseVertex from
 * `registerDraws`, per-frame instanceCount/firstInstance from the pack. StandardRenderer
 * reads `slot`'s records via `Draw.args.viewStride`. `packedEids` is one list
 * partitioned into a `capacity`-sized region per slot, each region compacted
 * into per-pair slices, read by the VS at `instance_index`. The slot dimension
 * grows with the active camera count, the pair dimension (`Surfaces.size ×
 * Meshes.size`) with mesh registration: no fixed upper bound on either
 *
 * @expand
 */
export interface MeshDrawBuffers {
    /** `DrawIndexedIndirect` records, slot-major (`slot * pairCount + pair`); null until the first frame's `syncBuffers` */
    drawArgs: DrawBuffer | null;
    /** packed entity identities, one dense list per view slot; null until `warmPart` */
    packedEids: InstanceBuffer | null;
}

interface PartGpuState {
    parts: MeshDrawBuffers;
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
    surfaceCount: number;
    meshCount: number;
    pairCount: number;
    viewDim: number;
    packPass: GPUComputePassDescriptor;
    countsUnwrapped: AtomicU32Buffer | null;
    countsRaw: GPUBuffer | null;
    paramsTarget: (TgpuBuffer<typeof CullParams> & UniformFlag) | null;
    paramsViewCount: number;
    paramsPairCount: number;
    paramsPartCount: number;
    paramsPartCapacity: number;
    rowCapacity: number;
    inputGeneration: Int32Array;
}

const partGpuKey = { create: createPartGpuState };
const partTableKey = { create: createPartTable };

/** Dense MeshInstance records shared by the GPU pack and typed surface stages. */
export function partTable(world: World) {
    return world.resource(partTableKey);
}

function createPartTable(world: World) {
    const table = world.table("partInputs", PartRecord);
    table.enableEidLookup();
    const publishMap = (buffer: GPUBuffer) => {
        world.gpu.buffers.set("partRowMap", buffer);
        world.gpu.typed.set("partRowMap", table.eidToRowTyped!);
    };
    table.subscribeMap(publishMap);
    publishMap(table.eidToRowBuffer!);
    return table;
}

function createPartGpuState(): PartGpuState {
    return {
        parts: { drawArgs: null, packedEids: null },
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
        surfaceCount: 0,
        meshCount: 0,
        pairCount: 0,
        viewDim: 1,
        packPass: { label: "shallot-part-pack" },
        countsUnwrapped: null,
        countsRaw: null,
        paramsTarget: null,
        paramsViewCount: -1,
        paramsPairCount: -1,
        paramsPartCount: -1,
        paramsPartCapacity: -1,
        rowCapacity: 0,
        inputGeneration: new Int32Array(5).fill(-1),
    };
}

function _partGpuState(world: World): PartGpuState {
    return world.resource(partGpuKey);
}

export function initializePartState(world: World): void {
    world.resource(partGpuKey);
    const table = partTable(world);
    table.bindComponent(MeshInstance, { mesh: "mesh" });
    materialTable(world);
    table.bindFields(MeshMaterial, { material: "material" });
    table.bindPresence(NotShadowCaster, "flags", 1);
    const seedDefault = (eid: number) => {
        if (!world.has(eid, MeshMaterial)) world.storage(MeshMaterial).material.set(eid, 0);
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
    create: (world) => world.resource(partGpuKey).parts,
};

/**
 * per-frame MeshInstance pack. Clears the counts, then cull → count → scan → scatter
 * over active MeshInstance table rows and view slots. No CPU iteration over
 * mesh instances: every thread culls an active row against the view's frustum. The
 * count + scatter dispatch a row of workgroups per active view (`gid.y` =
 * slot); the scan dispatches one workgroup per slot, each scanning its row in
 * parallel
 */
export const PartSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world) {
        const _render = world.resource(Render);
        const _partGpu = world.resource(partGpuKey);

        if (!_render.encoder || !_partGpu.countPipe || !_partGpu.scanPipe || !_partGpu.scatterPipe)
            return;
        syncBuffers(world);
        if (_partGpu.pairCount === 0) return;
        const count = bindCount(world);
        const scan = bindScan(world);
        const scatter = bindScatter(world);
        if (!count || !scan || !scatter) return;

        // viewCount + pairCount let the cull shader find a view's frustum and
        // index its slot's slice; slot ≥ viewCount means no frustum (headless),
        // packed unculled. Queued before EndFrameSystem submits the encoder, so
        // it lands before the pack executes
        const views = Math.max(1, _render.viewCount);
        // a two-word uniform written when either word changes: the typed write is the idiomatic path here.
        // The "CPU truth stays typed arrays" law governs the per-entity firehoses, where the
        // schema serializer is orders slower than a bulk `Float32Array.set`; two scalars are not that
        const partCount = partTable(world).count;
        if (
            _partGpu.paramsTarget !== _partGpu.cullParams ||
            _partGpu.paramsViewCount !== _render.viewCount ||
            _partGpu.paramsPairCount !== _partGpu.pairCount ||
            _partGpu.paramsPartCount !== partCount ||
            _partGpu.paramsPartCapacity !== _partGpu.rowCapacity
        ) {
            _partGpu.cullParams!.write({
                viewCount: _render.viewCount,
                pairCount: _partGpu.pairCount,
                partCount,
                partCapacity: _partGpu.rowCapacity,
            });
            _partGpu.paramsTarget = _partGpu.cullParams;
            _partGpu.paramsViewCount = _render.viewCount;
            _partGpu.paramsPairCount = _partGpu.pairCount;
            _partGpu.paramsPartCount = partCount;
            _partGpu.paramsPartCapacity = _partGpu.rowCapacity;
        }

        if (_partGpu.countsUnwrapped !== _partGpu.counts) {
            _partGpu.countsUnwrapped = _partGpu.counts;
            _partGpu.countsRaw = world.gpu.root.unwrap(_partGpu.counts!);
        }
        _render.encoder.clearBuffer(_partGpu.countsRaw!);
        _partGpu.packPass.timestampWrites = world.gpu.span?.("part:pack");
        const pass = _render.encoder.beginComputePass(_partGpu.packPass);
        const rows = Math.ceil(partCount / 64);
        if (rows > 0) {
            setBound(pass, count);
            pass.dispatchWorkgroups(rows, views);
        }
        // one workgroup per allocated view slot (the counts buffer spans _part.viewDim ×
        // pairCount); slots past the active views carry zero counts → zero instanceCount
        setBound(pass, scan);
        pass.dispatchWorkgroups(_partGpu.viewDim);
        if (rows > 0) {
            setBound(pass, scatter);
            pass.dispatchWorkgroups(rows, views);
        }
        pass.end();
    },
};

// bind one pass's pipeline and its groups, each at the index its layout declares
function setBound(
    pass: GPUComputePassEncoder,
    bound: { pipeline: GPUComputePipeline; groups: GPUBindGroup[] },
): void {
    pass.setPipeline(bound.pipeline);
    for (let i = 0; i < bound.groups.length; i++) pass.setBindGroup(i, bound.groups[i]);
}

// Bind dense MeshInstance and Transform tables, replacing groups only when one of their GPU buffers grows.
function cullGroup(world: World): TgpuBindGroup<(typeof cullLayout)["entries"]> | null {
    const _partGpu = world.resource(partGpuKey);

    if (!_partGpu.cullParams || !_partGpu.meshBounds) return null;
    const parts = partTable(world);
    const globalTransforms = globalTransformTable(world);
    const materials = materialTable(world);
    const generation = _partGpu.inputGeneration;
    if (
        generation[0] !== parts.generation ||
        generation[1] !== parts.activeGeneration ||
        generation[2] !== globalTransforms.generation ||
        generation[3] !== globalTransforms.mapGeneration ||
        generation[4] !== materials.generation
    ) {
        unbind(world);
        generation[0] = parts.generation;
        generation[1] = parts.activeGeneration;
        generation[2] = globalTransforms.generation;
        generation[3] = globalTransforms.mapGeneration;
        generation[4] = materials.generation;
    }
    if (_partGpu.cullGroup) return _partGpu.cullGroup;
    const cullVolumes = world.gpu.buffers.get("cullVolumes");
    const partRows = parts.activeRowsBuffer;
    const globalTransformRows = globalTransforms.eidToRowBuffer;
    if (!cullVolumes || !partRows || !globalTransformRows) {
        throw new Error(
            "[part] dense table inputs missing: cull volumes, MeshInstance rows or GlobalTransform row lookup",
        );
    }
    _partGpu.cullGroup = world.gpu.root.createBindGroup(cullLayout, {
        partRows,
        parts: parts.buffer,
        materials: materials.buffer,
        globalTransforms: globalTransforms.buffer,
        globalTransformRows,
        meshBounds: _partGpu.meshBounds,
        cullVolumes,
        params: _partGpu.cullParams,
    });
    return _partGpu.cullGroup;
}

function bindCount(world: World): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _partGpu = world.resource(partGpuKey);

    const cull = cullGroup(world);
    if (_partGpu.countBound) return _partGpu.countBound;
    if (!_partGpu.countPipe || !cull || !_partGpu.counts) return null;
    _partGpu.countBound = {
        pipeline: world.gpu.root.unwrap(_partGpu.countPipe),
        groups: [
            world.gpu.root.unwrap(cull),
            world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(countLayout, { counts: _partGpu.counts }),
            ),
        ],
    };
    return _partGpu.countBound;
}

function bindScan(world: World): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _partGpu = world.resource(partGpuKey);
    const _parts = world.resource(MeshDrawBuffers);

    if (_partGpu.scanBound) return _partGpu.scanBound;
    if (!_partGpu.scanPipe || !_partGpu.counts || !_parts.drawArgs || !_partGpu.cullParams)
        return null;
    _partGpu.scanBound = {
        pipeline: world.gpu.root.unwrap(_partGpu.scanPipe),
        groups: [
            world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(scanLayout, {
                    counts: _partGpu.counts,
                    drawArgs: _parts.drawArgs,
                    params: _partGpu.cullParams,
                }),
            ),
        ],
    };
    return _partGpu.scanBound;
}

function bindScatter(
    world: World,
): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _partGpu = world.resource(partGpuKey);
    const _parts = world.resource(MeshDrawBuffers);

    const cull = cullGroup(world);
    if (_partGpu.scatterBound) return _partGpu.scatterBound;
    if (
        !_partGpu.scatterPipe ||
        !cull ||
        !_partGpu.counts ||
        !_parts.drawArgs ||
        !_parts.packedEids
    )
        return null;
    _partGpu.scatterBound = {
        pipeline: world.gpu.root.unwrap(_partGpu.scatterPipe),
        groups: [
            world.gpu.root.unwrap(cull),
            world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(scatterLayout, {
                    drawArgs: world.gpu.root.unwrap(_parts.drawArgs),
                    packedEids: _parts.packedEids,
                }),
            ),
        ],
    };
    return _partGpu.scatterBound;
}

// every bound pipeline names at least one buffer `syncBuffers` can reallocate, so growth drops all of
// them together rather than tracking which buffer each one holds
function unbind(world: World): void {
    const _partGpu = world.resource(partGpuKey);

    _partGpu.cullGroup = null;
    _partGpu.countBound = null;
    _partGpu.scanBound = null;
    _partGpu.scatterBound = null;
}

/**
 * size the pack's buffers to the live mesh count (the pair dimension) and
 * active MeshInstance table row capacity and camera count, growing when any axis rises
 * after warm. `drawArgs` + `counts` scale with `viewDim × pairCount`; dense
 * output lists scale with `viewDim × rowCapacity`; mesh bounds scale with mesh count.
 * Pair growth only appends slots
 * (`mid * surfaceCount + sid`) so existing offsets hold, and the pipelines read
 * both dimensions from `cullParams` + `arrayLength`, never recompiling. Old
 * buffers free behind the submit fence: a prior frame may still reference them
 */
function syncBuffers(world: World): void {
    const _partGpu = world.resource(partGpuKey);
    const _parts = world.resource(MeshDrawBuffers);

    if (_partGpu.surfaceCount === 0) return;
    const meshCount = world.resource(Meshes).size;
    const viewDim = Math.max(1, world.resource(Render).viewCount);
    const rowCapacity = partTable(world).capacity;
    const growMesh = meshCount > _partGpu.meshCount;
    const growView = viewDim > _partGpu.viewDim;
    const growRows = rowCapacity > _partGpu.rowCapacity;
    if (!growMesh && !growView && !growRows && _parts.drawArgs) return;

    const device = world.gpu.device;
    _partGpu.meshCount = Math.max(_partGpu.meshCount, meshCount);
    _partGpu.viewDim = Math.max(_partGpu.viewDim, viewDim);
    _partGpu.rowCapacity = Math.max(_partGpu.rowCapacity, rowCapacity);
    _partGpu.pairCount = _partGpu.surfaceCount * _partGpu.meshCount;
    const records = _partGpu.viewDim * _partGpu.pairCount;

    const staleArgs: (DrawBuffer | AtomicU32Buffer | null)[] = [];
    if (growMesh || growView || !_parts.drawArgs) {
        staleArgs.push(_parts.drawArgs, _partGpu.counts);
        _parts.drawArgs = world.gpu.root
            .createBuffer(d.arrayOf(DrawIndexedIndirect, records))
            .$usage("storage", "indirect")
            .$name("shallot-draw-args");
        _partGpu.counts = world.gpu.root
            .createBuffer(d.arrayOf(d.atomic(d.u32), records))
            .$usage("storage")
            .$name("shallot-part-counts");
    }

    let stalePacked: InstanceBuffer | null = null;
    if (growView || growRows || !_parts.packedEids) {
        stalePacked = _parts.packedEids;
        const listCapacity = _partGpu.viewDim * _partGpu.rowCapacity;
        _parts.packedEids = world.gpu.root
            .createBuffer(d.arrayOf(d.vec4u, listCapacity))
            .$usage("storage")
            .$name("shallot-packed-eids");
        world.gpu.buffers.set("eids", world.gpu.root.unwrap(_parts.packedEids));
        world.gpu.typed.set("eids", _parts.packedEids);
    }

    // meshBounds is indexed by mesh id — rebuild only when a mesh registers
    let staleBounds: Vec4fBuffer | null = null;
    if (growMesh || !_partGpu.meshBounds) {
        staleBounds = _partGpu.meshBounds;
        _partGpu.meshBounds = writeMeshBounds(world, device);
    }

    unbind(world);
    registerDraws(world);

    const stale = [...staleArgs, stalePacked, staleBounds];
    // Replaced bindings cannot be used by this frame; submitted work retains its backing storage.
    for (const buffer of stale) buffer?.destroy();
}

/**
 * allocate + fill the per-mesh local bounding sphere buffer (one `vec4` per
 * mesh id: `xyz` center, `w` radius). A mesh without `bounds` (a procedural
 * producer that didn't supply one) gets a sentinel radius so the cull keeps it
 * always-visible rather than wrongly culling it
 */
function writeMeshBounds(world: World, device: GPUDevice): Vec4fBuffer {
    const _partGpu = world.resource(partGpuKey);
    const _meshes = world.resource(Meshes);

    const buffer = world.gpu.root
        .createBuffer(d.arrayOf(d.vec4f, _partGpu.meshCount))
        .$usage("storage")
        .$name("shallot-mesh-bounds");
    const data = new Float32Array(_partGpu.meshCount * 4);
    for (const m of _meshes) {
        const id = _meshes.id(m.name)!;
        if (m.bounds) data.set(m.bounds, id * 4);
        else data[id * 4 + 3] = 1e30; // never-cull sentinel
    }
    device.queue.writeBuffer(world.gpu.root.unwrap(buffer), 0, data as Float32Array<ArrayBuffer>);
    return buffer;
}

/** publish MeshInstance's `(surface, mesh)` draw pairs and return the indirect records the GPU buffer needs.
 * Device-free so ordering tests can exercise the production publication seam without an adapter.
 * @internal */
type DrawRecord = {
    indexCount: number;
    instanceCount: number;
    firstIndex: number;
    baseVertex: number;
    firstInstance: number;
};

export function publishPartDraws(
    world: World,
    drawArgs: DrawBuffer,
    surfaceCount: number,
    pairCount: number,
    registries: {
        surfaces: Registry<Surface>;
        meshes: Registry<Mesh>;
        draws: Registry<Draw>;
    } = {
        surfaces: world.resource(Surfaces),
        meshes: world.resource(Meshes),
        draws: world.resource(Draws),
    },
): { offset: number; args: DrawRecord }[] {
    const { surfaces, meshes, draws } = registries;
    const writes: { offset: number; args: DrawRecord }[] = [];
    const viewStride = pairCount * DRAW_ARG_STRIDE;
    for (const surface of surfaces) {
        const entries = surface.layout.entries;
        if (!("eids" in entries) || !("globalTransforms" in entries)) continue;
        const sid = surfaces.id(surface.name)!;
        for (const m of meshes) {
            const pair = meshes.id(m.name)! * surfaceCount + sid;
            const offset = pair * DRAW_ARG_STRIDE;
            // DrawIndexedIndirect: indexCount, instanceCount (pack), firstIndex, baseVertex (0 — indices
            // are absolute vertex positions), firstInstance (pack)
            const args = {
                indexCount: m.indexCount,
                instanceCount: 0,
                firstIndex: m.indexBase,
                baseVertex: 0,
                firstInstance: 0,
            };
            writes.push({ offset, args });
            draws.register({
                name: `part:${surface.name}:${m.name}`,
                surface: surface.name,
                mesh: m.name,
                args: { indirect: drawArgs, offset, viewStride },
            });
        }
    }
    return writes;
}

function registerDraws(world: World): void {
    const _parts = world.resource(MeshDrawBuffers);
    const _partGpu = world.resource(partGpuKey);

    if (!world.gpu.device || !_parts.drawArgs || _partGpu.pairCount === 0) return;
    const viewStride = _partGpu.pairCount * DRAW_ARG_STRIDE;
    for (const { offset, args } of publishPartDraws(
        world,
        _parts.drawArgs,
        _partGpu.surfaceCount,
        _partGpu.pairCount,
    )) {
        const bytes = new ArrayBuffer(DRAW_ARG_STRIDE);
        writeToArrayBuffer(bytes, DrawIndexedIndirect, args);
        for (let slot = 0; slot < _partGpu.viewDim; slot++) {
            world.gpu.device.queue.writeBuffer(
                world.gpu.root.unwrap(_parts.drawArgs),
                slot * viewStride + offset,
                bytes,
            );
        }
    }
}

/** Reset cached bind groups for a newly built world. */
export function initPart(world: World): void {
    unbind(world);
}

/**
 * compile the pack pipelines + allocate `packedEids`'s first slot. Runs at warm
 * (after every `initialize`), so `Surfaces.size` is final: surfaces are WGSL
 * shading programs declared in code, never data-driven, so the surface count is
 * the one axis safe to bake into the shaders. The pair count + view count come
 * from `cullParams` each frame, so the pipelines never recompile when meshes
 * register or cameras attach. `drawArgs` + `counts` + `meshBounds` size lazily
 * (`syncBuffers`), not here: neither `Meshes.size` nor the camera count is
 * final at warm
 */
export function warmPart(world: World): void {
    const _partGpu = world.resource(partGpuKey);
    const _parts = world.resource(MeshDrawBuffers);

    if (!world.gpu.device) return;
    const root = world.gpu.root;
    _partGpu.surfaceCount = world.resource(Surfaces).size;
    _partGpu.meshCount = 0;
    _partGpu.pairCount = 0;
    _partGpu.viewDim = 1;
    _parts.drawArgs = null;
    _parts.packedEids = null;
    _partGpu.counts = null;
    _partGpu.meshBounds = null;
    _partGpu.rowCapacity = 0;
    _partGpu.inputGeneration.fill(-1);
    _partGpu.paramsViewCount = -1;
    _partGpu.paramsPairCount = -1;
    _partGpu.paramsPartCount = -1;
    _partGpu.paramsPartCapacity = -1;
    unbind(world);

    _partGpu.cullParams = root
        .createBuffer(CullParams)
        .$usage("uniform")
        .$name("shallot-part-cull-params");
    if (_partGpu.surfaceCount === 0) return;

    _partGpu.countPipe = root
        .createComputePipeline({ compute: countKernel(_partGpu.surfaceCount) })
        .$name("shallot-part-count");
    _partGpu.scanPipe = root
        .createComputePipeline({ compute: scanKernel() })
        .$name("shallot-part-scan");
    _partGpu.scatterPipe = root
        .createComputePipeline({ compute: scatterKernel(_partGpu.surfaceCount) })
        .$name("shallot-part-scatter");

    // both the allocation and the bind are deferred into the forcers, not done here. The drain runs
    // after every plugin's warm has resolved (warm hooks run under `Promise.all`), which is the first
    // moment meshes, dense table buffers, the transform lookup, and cull volumes are published — so
    // `syncBuffers` can size the pack's buffers there, and the pipeline that forces the compile has
    // something to bind. One forcer per pipeline, so each gets its own row in the compile table
    precompile(world, "shallot-part-count", () => {
        syncBuffers(world);
        const bound = bindCount(world);
        return bound && [bound.pipeline];
    });
    precompile(world, "shallot-part-scan", () => {
        const bound = bindScan(world);
        return bound && [bound.pipeline];
    });
    precompile(world, "shallot-part-scatter", () => {
        const bound = bindScatter(world);
        return bound && [bound.pipeline];
    });
}
