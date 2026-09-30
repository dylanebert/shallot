import type {
    StorageFlag,
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    UniformFlag,
} from "typegpu";
import { writeToArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { Draw, Mesh, Surface } from "../../core/rendering";
import {
    BeginFrameSystem,
    DrawIndexedIndirect,
    Draws,
    Meshes,
    Render,
    Surfaces,
} from "../../core/rendering";
import type { Registry, World, System } from "../../engine";
import { GlobalTransform, globalTransformTable, u32, vec4 } from "../../engine";
import { precompile } from "../../engine/runtime";
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

// stride derived from the schema (a second hand-authored stride is layout drift waiting to
// happen).
const DRAW_ARG_STRIDE = d.sizeOf(DrawIndexedIndirect);
type InstanceBuffer = TgpuBuffer<d.WgslArray<d.Vec4u>> & StorageFlag;
type AtomicU32Buffer = TgpuBuffer<d.WgslArray<d.Atomic<d.U32>>> & StorageFlag;
type Vec4fBuffer = TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag;
type DrawBuffer = TgpuBuffer<d.WgslArray<typeof DrawIndexedIndirect>> &
    StorageFlag & { usableAsIndirect: true };

/**
 * ECS-shaped opt-in for Part rendering. `surface` holds the {@link Surfaces}
 * ID for the entity's shading; `mesh` holds the {@link Meshes} ID for its
 * geometry. A dense struct table feeds the GPU pack, which groups Parts by
 * `(surface, mesh)` and emits one indirect draw per used pair, so a surface is
 * shading only and renders any mesh. `surface` defaults to `"default"`, `mesh`
 * to `"cube"`; scenes pick others via `<a part="surface: checker; mesh: wall" />`
 *
 * @example
 * ```
 * <a part transform="pos: 0 0 0" color="rgba: 1 0.5 0.2 1" />
 * <a part="surface: checker; mesh: wall" transform="pos: 2 0 0" />
 * ```
 */
export const Part = {
    surface: u32,
    mesh: u32,
};

/**
 * per-entity base color, authored and stored as linear RGBA in its Part table record. Alpha is reserved for transparency.
 */
export const Color = {
    rgba: vec4,
};

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
 * GPU-resident Part draw publication. `drawArgs` holds `DrawIndexedIndirect` entries
 * (20 bytes) laid out slot-major (`slot * pairCount + pair`), so each camera
 * has its own per-pair records: static indexCount/firstIndex/baseVertex from
 * `registerDraws`, per-frame instanceCount/firstInstance from the pack. Sear
 * reads `slot`'s records via `Draw.args.viewStride`. `packedEids` is one list
 * partitioned into a `capacity`-sized region per slot, each region compacted
 * into per-pair slices, read by the VS at `instance_index`. The slot dimension
 * grows with the active camera count, the pair dimension (`Surfaces.size ×
 * Meshes.size`) with mesh registration: no fixed upper bound on either
 *
 * @expand
 */
export interface Parts {
    /** `DrawIndexedIndirect` records, slot-major (`slot * pairCount + pair`); null until the first frame's `syncBuffers` */
    drawArgs: DrawBuffer | null;
    /** packed entity identities, one dense list per view slot; null until `warmPart` */
    packedEids: InstanceBuffer | null;
}

interface PartGpuState {
    parts: Parts;
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

/** Dense Part records shared by the GPU pack and typed surface stages. */
export function partTable(state: World) {
    return state.resource(partTableKey);
}

function createPartTable(state: World) {
    const table = state.table("partInputs", PartRecord);
    table.enableEidLookup();
    const publishMap = (buffer: GPUBuffer) => {
        state.gpu.buffers.set("partRowMap", buffer);
        state.gpu.typed.set("partRowMap", table.eidToRowTyped!);
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
        inputGeneration: new Int32Array(4).fill(-1),
    };
}

function _partGpuState(state: World): PartGpuState {
    return state.resource(partGpuKey);
}

export function initializePartState(state: World): void {
    state.resource(partGpuKey);
    const table = partTable(state);
    table.bindComponent(Part, { surface: "surface", mesh: "mesh" });
    table.bindFields(Color, { color: "rgba" });
    const seedMissingColor = (eid: number) => {
        if (!state.has(eid, Color)) state.of(Color).rgba.set(eid, 1, 0, 1, 1);
    };
    const removeMissingColorDefault = state.observeMembership(Part, (eid, present) => {
        if (present) seedMissingColor(eid);
    });
    for (const eid of state.query([Part])) seedMissingColor(eid);
    state.onDispose(removeMissingColorDefault);
}

export const Parts: import("../../engine").Resource<Parts> = {
    create: (state) => state.resource(partGpuKey).parts,
};

/**
 * per-frame Part pack. Clears the counts, then cull → count → scan → scatter
 * over `(eid, view slot)`. No CPU iteration over Parts: every thread gates on
 * the mirrored component-membership bit, then on the view's frustum. The
 * count + scatter dispatch a row of workgroups per active view (`gid.y` =
 * slot); the scan dispatches one workgroup per slot, each scanning its row in
 * parallel
 */
export const PartSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(state) {
        const _render = state.resource(Render);
        const _partGpu = state.resource(partGpuKey);

        if (!_render.encoder || !_partGpu.countPipe || !_partGpu.scanPipe || !_partGpu.scatterPipe)
            return;
        syncBuffers(state);
        if (_partGpu.pairCount === 0) return;
        const count = bindCount(state);
        const scan = bindScan(state);
        const scatter = bindScatter(state);
        if (!count || !scan || !scatter) return;

        // viewCount + pairCount let the cull shader find a view's frustum and
        // index its slot's slice; slot ≥ viewCount means no frustum (headless),
        // packed unculled. Queued before EndFrameSystem submits the encoder, so
        // it lands before the pack executes
        const views = Math.max(1, _render.viewCount);
        // a two-word uniform written when either word changes: the typed write is the idiomatic path here.
        // The "CPU truth stays typed arrays" law governs the per-entity firehoses, where the
        // schema serializer is orders slower than a bulk `Float32Array.set`; two scalars are not that
        const partCount = partTable(state).count;
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
            _partGpu.countsRaw = state.gpu.root.unwrap(_partGpu.counts!);
        }
        _render.encoder.clearBuffer(_partGpu.countsRaw!);
        _partGpu.packPass.timestampWrites = state.gpu.span?.("part:pack");
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

// Bind dense Part and Transform tables, replacing groups only when one of their GPU buffers grows.
function cullGroup(state: World): TgpuBindGroup<(typeof cullLayout)["entries"]> | null {
    const _partGpu = state.resource(partGpuKey);

    if (!_partGpu.cullParams || !_partGpu.meshBounds) return null;
    const parts = partTable(state);
    const globalTransforms = globalTransformTable(state);
    const generation = _partGpu.inputGeneration;
    if (
        generation[0] !== parts.generation ||
        generation[1] !== parts.activeGeneration ||
        generation[2] !== globalTransforms.generation ||
        generation[3] !== globalTransforms.mapGeneration
    ) {
        unbind(state);
        generation[0] = parts.generation;
        generation[1] = parts.activeGeneration;
        generation[2] = globalTransforms.generation;
        generation[3] = globalTransforms.mapGeneration;
    }
    if (_partGpu.cullGroup) return _partGpu.cullGroup;
    const cullVolumes = state.gpu.buffers.get("cullVolumes");
    const partRows = parts.activeRowsBuffer;
    const globalTransformRows = globalTransforms.eidToRowBuffer;
    if (!cullVolumes || !partRows || !globalTransformRows) {
        throw new Error(
            "[part] dense table inputs missing: cull volumes, Part rows or GlobalTransform row lookup",
        );
    }
    _partGpu.cullGroup = state.gpu.root.createBindGroup(cullLayout, {
        partRows,
        parts: parts.buffer,
        globalTransforms: globalTransforms.buffer,
        globalTransformRows,
        meshBounds: _partGpu.meshBounds,
        cullVolumes,
        params: _partGpu.cullParams,
    });
    return _partGpu.cullGroup;
}

function bindCount(state: World): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _partGpu = state.resource(partGpuKey);

    const cull = cullGroup(state);
    if (_partGpu.countBound) return _partGpu.countBound;
    if (!_partGpu.countPipe || !cull || !_partGpu.counts) return null;
    _partGpu.countBound = {
        pipeline: state.gpu.root.unwrap(_partGpu.countPipe),
        groups: [
            state.gpu.root.unwrap(cull),
            state.gpu.root.unwrap(
                state.gpu.root.createBindGroup(countLayout, { counts: _partGpu.counts }),
            ),
        ],
    };
    return _partGpu.countBound;
}

function bindScan(state: World): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _partGpu = state.resource(partGpuKey);
    const _parts = state.resource(Parts);

    if (_partGpu.scanBound) return _partGpu.scanBound;
    if (!_partGpu.scanPipe || !_partGpu.counts || !_parts.drawArgs || !_partGpu.cullParams)
        return null;
    _partGpu.scanBound = {
        pipeline: state.gpu.root.unwrap(_partGpu.scanPipe),
        groups: [
            state.gpu.root.unwrap(
                state.gpu.root.createBindGroup(scanLayout, {
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
    state: World,
): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const _partGpu = state.resource(partGpuKey);
    const _parts = state.resource(Parts);

    const cull = cullGroup(state);
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
        pipeline: state.gpu.root.unwrap(_partGpu.scatterPipe),
        groups: [
            state.gpu.root.unwrap(cull),
            state.gpu.root.unwrap(
                state.gpu.root.createBindGroup(scatterLayout, {
                    drawArgs: _parts.drawArgs,
                    counts: _partGpu.counts,
                    packedEids: _parts.packedEids,
                }),
            ),
        ],
    };
    return _partGpu.scatterBound;
}

// every bound pipeline names at least one buffer `syncBuffers` can reallocate, so growth drops all of
// them together rather than tracking which buffer each one holds
function unbind(state: World): void {
    const _partGpu = state.resource(partGpuKey);

    _partGpu.cullGroup = null;
    _partGpu.countBound = null;
    _partGpu.scanBound = null;
    _partGpu.scatterBound = null;
}

/**
 * size the pack's buffers to the live mesh count (the pair dimension) and
 * active Part table row capacity and camera count, growing when any axis rises
 * after warm. `drawArgs` + `counts` scale with `viewDim × pairCount`; dense
 * output lists scale with `viewDim × rowCapacity`; mesh bounds scale with mesh count.
 * Pair growth only appends slots
 * (`mid * surfaceCount + sid`) so existing offsets hold, and the pipelines read
 * both dimensions from `cullParams` + `arrayLength`, never recompiling. Old
 * buffers free behind the submit fence: a prior frame may still reference them
 */
function syncBuffers(state: World): void {
    const _partGpu = state.resource(partGpuKey);
    const _parts = state.resource(Parts);

    if (_partGpu.surfaceCount === 0) return;
    const meshCount = state.resource(Meshes).size;
    const viewDim = Math.max(1, state.resource(Render).viewCount);
    const rowCapacity = partTable(state).capacity;
    const growMesh = meshCount > _partGpu.meshCount;
    const growView = viewDim > _partGpu.viewDim;
    const growRows = rowCapacity > _partGpu.rowCapacity;
    if (!growMesh && !growView && !growRows && _parts.drawArgs) return;

    const device = state.gpu.device;
    _partGpu.meshCount = Math.max(_partGpu.meshCount, meshCount);
    _partGpu.viewDim = Math.max(_partGpu.viewDim, viewDim);
    _partGpu.rowCapacity = Math.max(_partGpu.rowCapacity, rowCapacity);
    _partGpu.pairCount = _partGpu.surfaceCount * _partGpu.meshCount;
    const records = _partGpu.viewDim * _partGpu.pairCount;

    const staleArgs: (DrawBuffer | AtomicU32Buffer | null)[] = [];
    if (growMesh || growView || !_parts.drawArgs) {
        staleArgs.push(_parts.drawArgs, _partGpu.counts);
        _parts.drawArgs = state.gpu.root
            .createBuffer(d.arrayOf(DrawIndexedIndirect, records))
            .$usage("storage", "indirect")
            .$name("shallot-draw-args");
        _partGpu.counts = state.gpu.root
            .createBuffer(d.arrayOf(d.atomic(d.u32), records))
            .$usage("storage")
            .$name("shallot-part-counts");
    }

    let stalePacked: InstanceBuffer | null = null;
    if (growView || growRows || !_parts.packedEids) {
        stalePacked = _parts.packedEids;
        const listCapacity = _partGpu.viewDim * _partGpu.rowCapacity;
        _parts.packedEids = state.gpu.root
            .createBuffer(d.arrayOf(d.vec4u, listCapacity))
            .$usage("storage")
            .$name("shallot-packed-eids");
        state.gpu.buffers.set("eids", state.gpu.root.unwrap(_parts.packedEids));
        state.gpu.typed.set("eids", _parts.packedEids);
    }

    // meshBounds is indexed by mesh id — rebuild only when a mesh registers
    let staleBounds: Vec4fBuffer | null = null;
    if (growMesh || !_partGpu.meshBounds) {
        staleBounds = _partGpu.meshBounds;
        _partGpu.meshBounds = writeMeshBounds(state, device);
    }

    unbind(state);
    registerDraws(state);

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
function writeMeshBounds(state: World, device: GPUDevice): Vec4fBuffer {
    const _partGpu = state.resource(partGpuKey);
    const _meshes = state.resource(Meshes);

    const buffer = state.gpu.root
        .createBuffer(d.arrayOf(d.vec4f, _partGpu.meshCount))
        .$usage("storage")
        .$name("shallot-mesh-bounds");
    const data = new Float32Array(_partGpu.meshCount * 4);
    for (const m of _meshes) {
        const id = _meshes.id(m.name)!;
        if (m.bounds) data.set(m.bounds, id * 4);
        else data[id * 4 + 3] = 1e30; // never-cull sentinel
    }
    device.queue.writeBuffer(state.gpu.root.unwrap(buffer), 0, data as Float32Array<ArrayBuffer>);
    return buffer;
}

/** publish Part's `(surface, mesh)` draw pairs and return the indirect records the GPU buffer needs.
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
    state: World,
    drawArgs: DrawBuffer,
    surfaceCount: number,
    pairCount: number,
    registries: {
        surfaces: Registry<Surface>;
        meshes: Registry<Mesh>;
        draws: Registry<Draw>;
    } = {
        surfaces: state.resource(Surfaces),
        meshes: state.resource(Meshes),
        draws: state.resource(Draws),
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

function registerDraws(state: World): void {
    const _parts = state.resource(Parts);
    const _partGpu = state.resource(partGpuKey);

    if (!state.gpu.device || !_parts.drawArgs || _partGpu.pairCount === 0) return;
    const viewStride = _partGpu.pairCount * DRAW_ARG_STRIDE;
    for (const { offset, args } of publishPartDraws(
        state,
        _parts.drawArgs,
        _partGpu.surfaceCount,
        _partGpu.pairCount,
    )) {
        const bytes = new ArrayBuffer(DRAW_ARG_STRIDE);
        writeToArrayBuffer(bytes, DrawIndexedIndirect, args);
        for (let slot = 0; slot < _partGpu.viewDim; slot++) {
            state.gpu.device.queue.writeBuffer(
                state.gpu.root.unwrap(_parts.drawArgs),
                slot * viewStride + offset,
                bytes,
            );
        }
    }
}

/** Reset cached bind groups for a newly built world. */
export function initPart(state: World): void {
    unbind(state);
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
export function warmPart(state: World): void {
    const _partGpu = state.resource(partGpuKey);
    const _parts = state.resource(Parts);

    if (!state.gpu.device) return;
    const root = state.gpu.root;
    _partGpu.surfaceCount = state.resource(Surfaces).size;
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
    unbind(state);

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
    precompile(state, "shallot-part-count", () => {
        syncBuffers(state);
        const bound = bindCount(state);
        return bound && [bound.pipeline];
    });
    precompile(state, "shallot-part-scan", () => {
        const bound = bindScan(state);
        return bound && [bound.pipeline];
    });
    precompile(state, "shallot-part-scatter", () => {
        const bound = bindScatter(state);
        return bound && [bound.pipeline];
    });
}

export const PartTraits = {
    requires: [GlobalTransform],
    defaults: (state: World) => {
        const _surfaces = state.resource(Surfaces);
        const _meshes = state.resource(Meshes);

        // a missing "default" surface or "cube" mesh is a wiring bug — but only when the registry is
        // populated. With no SearPlugin the surface registry is empty (`Surfaces.size === 0`), so id 0 is
        // inert: there is no sear pass to marshal it to. That build is sanctioned (the conformance roster
        // exercises exactly that — `tests/conformance.test.ts:269`, plugins `Slab/Transforms/Render/Part`,
        // no Sear), and a warn there is a false alarm that trains the reader to ignore it. The genuine
        // wiring bug is the ordering case: surfaces *are* registered but "default" is absent (a SearPlugin
        // or surface owner forgot to register the default). So: warn only when the registry is populated
        // and the named default is missing (`Surfaces.size > 0 && Surfaces.id("default") === undefined`),
        // and likewise for `Meshes`/"cube". The fallback return is still 0 — the same value the pre-fix
        // `?? 0` produced, but now named at the call site as a wiring bug instead of silently binding
        // whatever surface/mesh holds registry id 0.
        const surface = _surfaces.id("default");
        const mesh = _meshes.id("cube");
        if (_surfaces.size > 0 && surface === undefined)
            console.warn(
                '[part] default surface "default" is not registered — a SearPlugin or surface owner must register it; Part entities will bind whatever surface holds registry id 0',
            );
        if (_meshes.size > 0 && mesh === undefined)
            console.warn(
                '[part] default mesh "cube" is not registered — PartPlugin.initialize() registers it via initMeshes(); Part entities will bind whatever mesh holds registry id 0',
            );
        return { surface: surface ?? 0, mesh: mesh ?? 0 };
    },
    parse: {
        surface: (value: string, state: World) => state.resource(Surfaces).id(value),
        mesh: (value: string, state: World) => state.resource(Meshes).id(value),
    },
    format: {
        surface: (value: number, state: World) => state.resource(Surfaces).name(value),
        mesh: (value: number, state: World) => state.resource(Meshes).name(value),
    },
};

export const ColorTraits = {
    defaults: () => ({ rgba: [1, 1, 1, 1] }),
};
