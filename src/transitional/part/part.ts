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
import type { Registry, State, System } from "../../engine";
import { Compute, field, GlobalTransform, globalTransformTable, u32, vec4 } from "../../engine";
import { precompile, worldResource } from "../../engine/runtime";
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
    surface: field(u32),
    mesh: field(u32),
};

/**
 * per-entity base color, authored and stored as linear RGBA in its Part table record. Alpha is reserved for transparency.
 */
export const Color = {
    rgba: field(vec4),
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

const partGpuKey = Symbol("shallot.part");
const partTableKey = Symbol("shallot.part-table");

/** Dense Part records shared by the GPU pack and typed surface stages. */
export function partTable(state: State) {
    return state.resource(partTableKey, createPartTable);
}

function createPartTable(state: State) {
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

function partGpuState(): PartGpuState {
    return worldResource(partGpuKey, createPartGpuState);
}

export function initializePartState(state: State): void {
    state.resource(partGpuKey, createPartGpuState);
    const table = partTable(state);
    table.bindComponent(Part, { surface: "surface", mesh: "mesh" });
    table.bindFields(Color, { color: "rgba" });
    const seedMissingColor = (eid: number) => {
        if (!state.has(eid, Color)) Color.rgba.set(eid, 1, 0, 1, 1);
    };
    const removeMissingColorDefault = state.observeMembership(Part, (eid, present) => {
        if (present) seedMissingColor(eid);
    });
    for (const eid of state.query([Part])) seedMissingColor(eid);
    state.onDispose(removeMissingColorDefault);
}

const _part = new Proxy({} as Omit<PartGpuState, "parts">, {
    get(_target, key) {
        return partGpuState()[key as keyof PartGpuState] as never;
    },
    set(_target, key, value) {
        (partGpuState() as unknown as Record<PropertyKey, unknown>)[key] = value;
        return true;
    },
});

export const Parts: Parts = new Proxy({} as Parts, {
    get(_target, key) {
        return partGpuState().parts[key as keyof Parts];
    },
    set(_target, key, value) {
        (partGpuState().parts as unknown as Record<PropertyKey, unknown>)[key] = value;
        return true;
    },
});

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
        if (!Render.encoder || !_part.countPipe || !_part.scanPipe || !_part.scatterPipe) return;
        syncBuffers(state);
        if (_part.pairCount === 0) return;
        const count = bindCount(state);
        const scan = bindScan();
        const scatter = bindScatter(state);
        if (!count || !scan || !scatter) return;

        // viewCount + pairCount let the cull shader find a view's frustum and
        // index its slot's slice; slot ≥ viewCount means no frustum (headless),
        // packed unculled. Queued before EndFrameSystem submits the encoder, so
        // it lands before the pack executes
        const views = Math.max(1, Render.viewCount);
        // a two-word uniform written when either word changes: the typed write is the idiomatic path here.
        // The "CPU truth stays typed arrays" law governs the per-entity firehoses, where the
        // schema serializer is orders slower than a bulk `Float32Array.set`; two scalars are not that
        const partCount = partTable(state).count;
        if (
            _part.paramsTarget !== _part.cullParams ||
            _part.paramsViewCount !== Render.viewCount ||
            _part.paramsPairCount !== _part.pairCount ||
            _part.paramsPartCount !== partCount ||
            _part.paramsPartCapacity !== _part.rowCapacity
        ) {
            _part.cullParams!.write({
                viewCount: Render.viewCount,
                pairCount: _part.pairCount,
                partCount,
                partCapacity: _part.rowCapacity,
            });
            _part.paramsTarget = _part.cullParams;
            _part.paramsViewCount = Render.viewCount;
            _part.paramsPairCount = _part.pairCount;
            _part.paramsPartCount = partCount;
            _part.paramsPartCapacity = _part.rowCapacity;
        }

        if (_part.countsUnwrapped !== _part.counts) {
            _part.countsUnwrapped = _part.counts;
            _part.countsRaw = Compute.root.unwrap(_part.counts!);
        }
        Render.encoder.clearBuffer(_part.countsRaw!);
        _part.packPass.timestampWrites = Compute.span?.("part:pack");
        const pass = Render.encoder.beginComputePass(_part.packPass);
        const rows = Math.ceil(partCount / 64);
        if (rows > 0) {
            setBound(pass, count);
            pass.dispatchWorkgroups(rows, views);
        }
        // one workgroup per allocated view slot (the counts buffer spans _part.viewDim ×
        // pairCount); slots past the active views carry zero counts → zero instanceCount
        setBound(pass, scan);
        pass.dispatchWorkgroups(_part.viewDim);
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
function cullGroup(state: State): TgpuBindGroup<(typeof cullLayout)["entries"]> | null {
    if (!_part.cullParams || !_part.meshBounds) return null;
    const parts = partTable(state);
    const globalTransforms = globalTransformTable(state);
    const generation = _part.inputGeneration;
    if (
        generation[0] !== parts.generation ||
        generation[1] !== parts.activeGeneration ||
        generation[2] !== globalTransforms.generation ||
        generation[3] !== globalTransforms.mapGeneration
    ) {
        unbind();
        generation[0] = parts.generation;
        generation[1] = parts.activeGeneration;
        generation[2] = globalTransforms.generation;
        generation[3] = globalTransforms.mapGeneration;
    }
    if (_part.cullGroup) return _part.cullGroup;
    const cullVolumes = Compute.buffers.get("cullVolumes");
    const partRows = parts.activeRowsBuffer;
    const globalTransformRows = globalTransforms.eidToRowBuffer;
    if (!cullVolumes || !partRows || !globalTransformRows) {
        throw new Error(
            "[part] dense table inputs missing: cull volumes, Part rows or GlobalTransform row lookup",
        );
    }
    _part.cullGroup = Compute.root.createBindGroup(cullLayout, {
        partRows,
        parts: parts.buffer,
        globalTransforms: globalTransforms.buffer,
        globalTransformRows,
        meshBounds: _part.meshBounds,
        cullVolumes,
        params: _part.cullParams,
    });
    return _part.cullGroup;
}

function bindCount(state: State): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const cull = cullGroup(state);
    if (_part.countBound) return _part.countBound;
    if (!_part.countPipe || !cull || !_part.counts) return null;
    _part.countBound = {
        pipeline: Compute.root.unwrap(_part.countPipe),
        groups: [
            Compute.root.unwrap(cull),
            Compute.root.unwrap(
                Compute.root.createBindGroup(countLayout, { counts: _part.counts }),
            ),
        ],
    };
    return _part.countBound;
}

function bindScan(): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    if (_part.scanBound) return _part.scanBound;
    if (!_part.scanPipe || !_part.counts || !Parts.drawArgs || !_part.cullParams) return null;
    _part.scanBound = {
        pipeline: Compute.root.unwrap(_part.scanPipe),
        groups: [
            Compute.root.unwrap(
                Compute.root.createBindGroup(scanLayout, {
                    counts: _part.counts,
                    drawArgs: Parts.drawArgs,
                    params: _part.cullParams,
                }),
            ),
        ],
    };
    return _part.scanBound;
}

function bindScatter(
    state: State,
): { pipeline: GPUComputePipeline; groups: GPUBindGroup[] } | null {
    const cull = cullGroup(state);
    if (_part.scatterBound) return _part.scatterBound;
    if (!_part.scatterPipe || !cull || !_part.counts || !Parts.drawArgs || !Parts.packedEids)
        return null;
    _part.scatterBound = {
        pipeline: Compute.root.unwrap(_part.scatterPipe),
        groups: [
            Compute.root.unwrap(cull),
            Compute.root.unwrap(
                Compute.root.createBindGroup(scatterLayout, {
                    drawArgs: Parts.drawArgs,
                    counts: _part.counts,
                    packedEids: Parts.packedEids,
                }),
            ),
        ],
    };
    return _part.scatterBound;
}

// every bound pipeline names at least one buffer `syncBuffers` can reallocate, so growth drops all of
// them together rather than tracking which buffer each one holds
function unbind(): void {
    _part.cullGroup = null;
    _part.countBound = null;
    _part.scanBound = null;
    _part.scatterBound = null;
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
function syncBuffers(state: State): void {
    if (_part.surfaceCount === 0) return;
    const meshCount = Meshes.size;
    const viewDim = Math.max(1, Render.viewCount);
    const rowCapacity = partTable(state).capacity;
    const growMesh = meshCount > _part.meshCount;
    const growView = viewDim > _part.viewDim;
    const growRows = rowCapacity > _part.rowCapacity;
    if (!growMesh && !growView && !growRows && Parts.drawArgs) return;

    const device = Compute.device;
    _part.meshCount = Math.max(_part.meshCount, meshCount);
    _part.viewDim = Math.max(_part.viewDim, viewDim);
    _part.rowCapacity = Math.max(_part.rowCapacity, rowCapacity);
    _part.pairCount = _part.surfaceCount * _part.meshCount;
    const records = _part.viewDim * _part.pairCount;

    const staleArgs: (DrawBuffer | AtomicU32Buffer | null)[] = [];
    if (growMesh || growView || !Parts.drawArgs) {
        staleArgs.push(Parts.drawArgs, _part.counts);
        Parts.drawArgs = Compute.root
            .createBuffer(d.arrayOf(DrawIndexedIndirect, records))
            .$usage("storage", "indirect")
            .$name("shallot-draw-args");
        _part.counts = Compute.root
            .createBuffer(d.arrayOf(d.atomic(d.u32), records))
            .$usage("storage")
            .$name("shallot-part-counts");
    }

    let stalePacked: InstanceBuffer | null = null;
    if (growView || growRows || !Parts.packedEids) {
        stalePacked = Parts.packedEids;
        const listCapacity = _part.viewDim * _part.rowCapacity;
        Parts.packedEids = Compute.root
            .createBuffer(d.arrayOf(d.vec4u, listCapacity))
            .$usage("storage")
            .$name("shallot-packed-eids");
        Compute.buffers.set("eids", Compute.root.unwrap(Parts.packedEids));
        Compute.typed.set("eids", Parts.packedEids);
    }

    // meshBounds is indexed by mesh id — rebuild only when a mesh registers
    let staleBounds: Vec4fBuffer | null = null;
    if (growMesh || !_part.meshBounds) {
        staleBounds = _part.meshBounds;
        _part.meshBounds = writeMeshBounds(device);
    }

    unbind();
    registerDraws();

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
function writeMeshBounds(device: GPUDevice): Vec4fBuffer {
    const buffer = Compute.root
        .createBuffer(d.arrayOf(d.vec4f, _part.meshCount))
        .$usage("storage")
        .$name("shallot-mesh-bounds");
    const data = new Float32Array(_part.meshCount * 4);
    for (const m of Meshes) {
        const id = Meshes.id(m.name)!;
        if (m.bounds) data.set(m.bounds, id * 4);
        else data[id * 4 + 3] = 1e30; // never-cull sentinel
    }
    device.queue.writeBuffer(Compute.root.unwrap(buffer), 0, data as Float32Array<ArrayBuffer>);
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
    drawArgs: DrawBuffer,
    surfaceCount: number,
    pairCount: number,
    registries: {
        surfaces: Registry<Surface>;
        meshes: Registry<Mesh>;
        draws: Registry<Draw>;
    } = { surfaces: Surfaces, meshes: Meshes, draws: Draws },
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

function registerDraws(): void {
    if (!Compute.device || !Parts.drawArgs || _part.pairCount === 0) return;
    const viewStride = _part.pairCount * DRAW_ARG_STRIDE;
    for (const { offset, args } of publishPartDraws(
        Parts.drawArgs,
        _part.surfaceCount,
        _part.pairCount,
    )) {
        const bytes = new ArrayBuffer(DRAW_ARG_STRIDE);
        writeToArrayBuffer(bytes, DrawIndexedIndirect, args);
        for (let slot = 0; slot < _part.viewDim; slot++) {
            Compute.device.queue.writeBuffer(
                Compute.root.unwrap(Parts.drawArgs),
                slot * viewStride + offset,
                bytes,
            );
        }
    }
}

/** Reset cached bind groups for a newly built world. */
export function initPart(): void {
    unbind();
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
export function warmPart(state: State): void {
    if (!Compute.device) return;
    const root = Compute.root;
    _part.surfaceCount = Surfaces.size;
    _part.meshCount = 0;
    _part.pairCount = 0;
    _part.viewDim = 1;
    Parts.drawArgs = null;
    Parts.packedEids = null;
    _part.counts = null;
    _part.meshBounds = null;
    _part.rowCapacity = 0;
    _part.inputGeneration.fill(-1);
    _part.paramsViewCount = -1;
    _part.paramsPairCount = -1;
    _part.paramsPartCount = -1;
    _part.paramsPartCapacity = -1;
    unbind();

    _part.cullParams = root
        .createBuffer(CullParams)
        .$usage("uniform")
        .$name("shallot-part-cull-params");
    if (_part.surfaceCount === 0) return;

    _part.countPipe = root
        .createComputePipeline({ compute: countKernel(_part.surfaceCount) })
        .$name("shallot-part-count");
    _part.scanPipe = root
        .createComputePipeline({ compute: scanKernel() })
        .$name("shallot-part-scan");
    _part.scatterPipe = root
        .createComputePipeline({ compute: scatterKernel(_part.surfaceCount) })
        .$name("shallot-part-scatter");

    // both the allocation and the bind are deferred into the forcers, not done here. The drain runs
    // after every plugin's warm has resolved (warm hooks run under `Promise.all`), which is the first
    // moment meshes, dense table buffers, the transform lookup, and cull volumes are published — so
    // `syncBuffers` can size the pack's buffers there, and the pipeline that forces the compile has
    // something to bind. One forcer per pipeline, so each gets its own row in the compile table
    precompile("shallot-part-count", () => {
        syncBuffers(state);
        const bound = bindCount(state);
        return bound && [bound.pipeline];
    });
    precompile("shallot-part-scan", () => {
        const bound = bindScan();
        return bound && [bound.pipeline];
    });
    precompile("shallot-part-scatter", () => {
        const bound = bindScatter(state);
        return bound && [bound.pipeline];
    });
}

export const PartTraits = {
    requires: [GlobalTransform],
    defaults: () => {
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
        const surface = Surfaces.id("default");
        const mesh = Meshes.id("cube");
        if (Surfaces.size > 0 && surface === undefined)
            console.warn(
                '[part] default surface "default" is not registered — a SearPlugin or surface owner must register it; Part entities will bind whatever surface holds registry id 0',
            );
        if (Meshes.size > 0 && mesh === undefined)
            console.warn(
                '[part] default mesh "cube" is not registered — PartPlugin.initialize() registers it via initMeshes(); Part entities will bind whatever mesh holds registry id 0',
            );
        return { surface: surface ?? 0, mesh: mesh ?? 0 };
    },
    parse: {
        surface: (value: string) => Surfaces.id(value),
        mesh: (value: string) => Meshes.id(value),
    },
    format: {
        surface: (value: number) => Surfaces.name(value),
        mesh: (value: number) => Meshes.name(value),
    },
};

export const ColorTraits = {
    defaults: () => ({ rgba: [1, 1, 1, 1] }),
};
