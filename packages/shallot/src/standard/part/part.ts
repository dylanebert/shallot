import type {
    StorageFlag,
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    UniformFlag,
} from "typegpu";
import { writeToArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { Registry, State, System } from "../../engine";
import { Compute, capacity, srgb8x4, u32 } from "../../engine";
import { precompile } from "../../engine/runtime";
import type { Draw, Mesh, Surface } from "../render/core";
import {
    BeginFrameSystem,
    DrawIndexedIndirect,
    Draws,
    Meshes,
    Render,
    Surfaces,
} from "../render/core";
import { slab } from "../slab";
import { Transform } from "../transforms";
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

// stride derived from the schema (gpu.md: a second hand-authored stride is layout drift waiting to
// happen).
const DRAW_ARG_STRIDE = d.sizeOf(DrawIndexedIndirect);
type U32Buffer = TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
type AtomicU32Buffer = TgpuBuffer<d.WgslArray<d.Atomic<d.U32>>> & StorageFlag;
type Vec4fBuffer = TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag;
type DrawBuffer = TgpuBuffer<d.WgslArray<typeof DrawIndexedIndirect>> &
    StorageFlag & { usableAsIndirect: true };

/**
 * ECS-shaped opt-in for Part rendering. `surface` holds the {@link Surfaces}
 * ID for the entity's shading; `mesh` holds the {@link Meshes} ID for its
 * geometry: both `slab(u32)` the pack reads on GPU. The pack groups Parts by
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
    surface: slab(u32),
    mesh: slab(u32),
};

/**
 * per-entity base color. Authored CPU-side as a linear {@link Quad}, but mirrored to the GPU as one
 * sRGB-packed u32 ({@link srgb8x4}, 16 B → 4 B), published to `Compute.buffers` under the name
 * `"color"`; a surface reads it as `unpackLdrColor(color[eid])`. Alpha is reserved for transparency
 */
export const Color = {
    rgba: slab(srgb8x4, "color"),
};

// Pack is cull → count → scan → scatter, run per active view: count tallies the
// frustum-visible parts per (view, pair), the single-thread scan turns counts
// into each (view, pair)'s instanceCount + compacted firstInstance (written
// into drawArgs), scatter appends each surviving eid into its slice of
// packedEids. The cull test (instance bound vs the slot's `cullVolumes[slot]`) gates both
// count and scatter, so off-screen parts never reach the indirect args — this
// is niagara's cull → compact → drawIndirect spine. Output is slot-major: each
// camera owns its own drawArgs records + packedEids region, so the four-up
// example culls each view independently and the shadow pass (Phase 2) reuses
// the same pack against the sun's frustum as one more slot. registerDraws
// writes the static indexCount + firstIndex; the per-view dimension grows
// lazily with the active camera count, the pair dimension with mesh count.
let _counts: AtomicU32Buffer | null = null;
let _meshBounds: Vec4fBuffer | null = null;
let _cullParams: (TgpuBuffer<typeof CullParams> & UniformFlag) | null = null;
let _cullGroup: TgpuBindGroup<(typeof cullLayout)["entries"]> | null = null;
let _countPipe: TgpuComputePipeline | null = null;
let _scanPipe: TgpuComputePipeline | null = null;
let _scatterPipe: TgpuComputePipeline | null = null;
let _countBound: TgpuComputePipeline | null = null;
let _scanBound: TgpuComputePipeline | null = null;
let _scatterBound: TgpuComputePipeline | null = null;
let _surfaceCount = 0;
let _meshCount = 0;
let _pairCount = 0;
// monotonic high-water of the active camera count — the slot dimension of
// drawArgs / counts / packedEids. Starts at 1 so a headless frame (no camera)
// still packs into slot 0, where the cull is a no-op (`visible` returns true
// once `slot >= viewCount`)
let _viewDim = 1;

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
    /** packed survivor eids, one `capacity`-sized region per view slot; null until `warmPart` */
    packedEids: U32Buffer | null;
}

export const Parts: Parts = {
    drawArgs: null,
    packedEids: null,
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
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    update() {
        if (!Render.encoder || !_countPipe || !_scanPipe || !_scatterPipe) return;
        syncBuffers();
        if (_pairCount === 0) return;
        const count = bindCount();
        const scan = bindScan();
        const scatter = bindScatter();
        if (!count || !scan || !scatter) return;

        // viewCount + pairCount let the cull shader find a view's frustum and
        // index its slot's slice; slot ≥ viewCount means no frustum (headless),
        // packed unculled. Queued before EndFrameSystem submits the encoder, so
        // it lands before the pack executes
        const views = Math.max(1, Render.viewCount);
        // a two-word uniform written once a frame: the typed write is the idiomatic path here. The
        // "CPU truth stays typed arrays" law (gpu.md) governs the per-entity firehoses, where the
        // schema serializer is orders slower than a bulk `Float32Array.set`; two scalars are not that
        _cullParams!.write({ viewCount: Render.viewCount, pairCount: _pairCount });

        Render.encoder.clearBuffer(Compute.root.unwrap(_counts!));
        const pass = Render.encoder.beginComputePass({
            label: "shallot-part-pack",
            timestampWrites: Compute.span?.("part:pack"),
        });
        const rows = Math.ceil(capacity / 64);
        count.with(pass).dispatchWorkgroups(rows, views);
        // one workgroup per allocated view slot (the counts buffer spans _viewDim ×
        // pairCount); slots past the active views carry zero counts → zero instanceCount
        scan.with(pass).dispatchWorkgroups(_viewDim);
        scatter.with(pass).dispatchWorkgroups(rows, views);
        pass.end();
    },
};

// the one shared cull bind group both count and scatter reference. Its inputs are the per-entity slabs +
// membership mirror, the world-transform firehose and the per-view cull volumes — all stable,
// fixed-capacity identities published by their owners before any draw-group consumer runs, so a missing
// one is a wiring bug and throws. `_cullParams` / `_meshBounds` are this module's own and genuinely late
// (`syncBuffers` sizes them off the final mesh count), which is the one null the callers tolerate.
// A typed bind group takes a raw GPUBuffer, which is what keeps those publishers' reach-in open
function cullGroup(): TgpuBindGroup<(typeof cullLayout)["entries"]> | null {
    if (_cullGroup) return _cullGroup;
    if (!_cullParams || !_meshBounds) return null;
    const surface = Part.surface.gpu;
    const mesh = Part.mesh.gpu;
    const membership = Compute.buffers.get("membership");
    const transforms = Compute.buffers.get("transforms");
    const cullVolumes = Compute.buffers.get("cullVolumes");
    if (!surface || !mesh || !membership || !transforms || !cullVolumes) {
        throw new Error(
            "[part] cull inputs missing — declare RenderPlugin + SlabPlugin as dependencies",
        );
    }
    _cullGroup = Compute.root.createBindGroup(cullLayout, {
        surfaceField: surface,
        meshField: mesh,
        membership,
        transforms,
        meshBounds: _meshBounds,
        cullVolumes,
        params: _cullParams,
    });
    return _cullGroup;
}

function bindCount(): TgpuComputePipeline | null {
    if (_countBound) return _countBound;
    const cull = cullGroup();
    if (!_countPipe || !cull || !_counts) return null;
    _countBound = _countPipe
        .with(cull)
        .with(Compute.root.createBindGroup(countLayout, { counts: _counts }));
    return _countBound;
}

function bindScan(): TgpuComputePipeline | null {
    if (_scanBound) return _scanBound;
    if (!_scanPipe || !_counts || !Parts.drawArgs || !_cullParams) return null;
    _scanBound = _scanPipe.with(
        Compute.root.createBindGroup(scanLayout, {
            counts: _counts,
            drawArgs: Parts.drawArgs,
            params: _cullParams,
        }),
    );
    return _scanBound;
}

function bindScatter(): TgpuComputePipeline | null {
    if (_scatterBound) return _scatterBound;
    const cull = cullGroup();
    if (!_scatterPipe || !cull || !_counts || !Parts.drawArgs || !Parts.packedEids) return null;
    _scatterBound = _scatterPipe.with(cull).with(
        Compute.root.createBindGroup(scatterLayout, {
            drawArgs: Parts.drawArgs,
            counts: _counts,
            packedEids: Parts.packedEids,
        }),
    );
    return _scatterBound;
}

// every bound pipeline names at least one buffer `syncBuffers` can reallocate, so growth drops all of
// them together rather than tracking which buffer each one holds
function unbind(): void {
    _cullGroup = null;
    _countBound = null;
    _scanBound = null;
    _scatterBound = null;
}

/**
 * size the pack's buffers to the live mesh count (the pair dimension) and
 * active camera count (the view/slot dimension), growing when either rises
 * after warm: procedural producers size geometry from scene data and cameras
 * attach at runtime, so neither is final at warm. Called each frame; the two
 * `<=` compares are ints, not per-entity dirty tracking. `drawArgs` + `counts`
 * scale with `viewDim × pairCount`; `packedEids` with `viewDim × capacity`;
 * `meshBounds` with mesh count. Pair growth only appends slots
 * (`mid * surfaceCount + sid`) so existing offsets hold, and the pipelines read
 * both dimensions from `cullParams` + `arrayLength`, never recompiling. Old
 * buffers free behind the submit fence: a prior frame may still reference them
 */
function syncBuffers(): void {
    if (_surfaceCount === 0) return;
    const meshCount = Meshes.size;
    const viewDim = Math.max(1, Render.viewCount);
    const growMesh = meshCount > _meshCount;
    const growView = viewDim > _viewDim;
    if (!growMesh && !growView && Parts.drawArgs) return;

    const device = Compute.device;
    _meshCount = Math.max(_meshCount, meshCount);
    _viewDim = Math.max(_viewDim, viewDim);
    _pairCount = _surfaceCount * _meshCount;
    const records = _viewDim * _pairCount;

    // drawArgs + counts span every (view, pair) — realloc when either dimension
    // grows. COPY_SRC for GPU-debug readback (gpu.md) + the pack tests
    const staleArgs = [Parts.drawArgs, _counts];
    Parts.drawArgs = Compute.root
        .createBuffer(d.arrayOf(DrawIndexedIndirect, records))
        .$usage("storage", "indirect")
        .$name("shallot-draw-args");
    _counts = Compute.root
        .createBuffer(d.arrayOf(d.atomic(d.u32), records))
        .$usage("storage")
        .$name("shallot-part-counts");

    // packedEids holds one capacity-sized region per view — realloc only when
    // the view dimension grows, so a mesh registering doesn't churn the buffer
    // sear binds (its identity invalidates the bind-group cache)
    let stalePacked: U32Buffer | null = null;
    if (growView || !Parts.packedEids) {
        stalePacked = Parts.packedEids;
        Parts.packedEids = Compute.root
            .createBuffer(d.arrayOf(d.u32, _viewDim * capacity))
            .$usage("storage")
            .$name("shallot-packed-eids");
        Compute.buffers.set("eids", Compute.root.unwrap(Parts.packedEids));
        Compute.typed.set("eids", Parts.packedEids);
    }

    // meshBounds is indexed by mesh id — rebuild only when a mesh registers
    let staleBounds: Vec4fBuffer | null = null;
    if (growMesh || !_meshBounds) {
        staleBounds = _meshBounds;
        _meshBounds = writeMeshBounds(device);
    }

    unbind();
    registerDraws();

    const stale = [...staleArgs, stalePacked, staleBounds];
    if (stale.some(Boolean)) {
        device.queue.onSubmittedWorkDone().then(() => {
            for (const b of stale) b?.destroy();
        });
    }
}

/**
 * allocate + fill the per-mesh local bounding sphere buffer (one `vec4` per
 * mesh id: `xyz` center, `w` radius). A mesh without `bounds` (a procedural
 * producer that didn't supply one) gets a sentinel radius so the cull keeps it
 * always-visible rather than wrongly culling it
 */
function writeMeshBounds(device: GPUDevice): Vec4fBuffer {
    const buffer = Compute.root
        .createBuffer(d.arrayOf(d.vec4f, _meshCount))
        .$usage("storage")
        .$name("shallot-mesh-bounds");
    const data = new Float32Array(_meshCount * 4);
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
        if (!("eids" in entries) || !("transforms" in entries)) continue;
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
    if (!Compute.device || !Parts.drawArgs || _pairCount === 0) return;
    const viewStride = _pairCount * DRAW_ARG_STRIDE;
    for (const { offset, args } of publishPartDraws(Parts.drawArgs, _surfaceCount, _pairCount)) {
        const bytes = new ArrayBuffer(DRAW_ARG_STRIDE);
        writeToArrayBuffer(bytes, DrawIndexedIndirect, args);
        for (let slot = 0; slot < _viewDim; slot++) {
            Compute.device.queue.writeBuffer(
                Compute.root.unwrap(Parts.drawArgs),
                slot * viewStride + offset,
                bytes,
            );
        }
    }
}

/** seed Part defaults. The slab arrays are allocated by SlabPlugin (a dependency) before this runs */
export function initPart(): void {
    // base every slot in magenta — the visible "Part without an explicit Color" indicator (an entity
    // with Color overwrites its slot via the white trait default on add). `Part.surface`/`mesh`/
    // `Color.rgba` are declared inline (`slab(...)`); collect() in SlabPlugin.initialize allocated
    // their arrays already. The pack gates each slot on the Part-membership bit, so a destroyed or
    // non-Part slot is skipped regardless of the stale ids it holds.
    for (let i = 0; i < capacity; i++) Color.rgba.set(i, 1, 0, 1, 1);

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
    _surfaceCount = Surfaces.size;
    _meshCount = 0;
    _pairCount = 0;
    _viewDim = 1;
    Parts.drawArgs = null;
    _counts = null;
    _meshBounds = null;
    unbind();

    // one capacity-sized region (slot 0); syncBuffers grows it as cameras attach.
    // COPY_SRC for GPU-debug readback (gpu.md) + the pack tests
    Parts.packedEids = root
        .createBuffer(d.arrayOf(d.u32, capacity))
        .$usage("storage")
        .$name("shallot-packed-eids");
    Compute.buffers.set("eids", root.unwrap(Parts.packedEids));
    Compute.typed.set("eids", Parts.packedEids);

    _cullParams = root.createBuffer(CullParams).$usage("uniform").$name("shallot-part-cull-params");
    if (_surfaceCount === 0) return;

    const part = state.membership.bit(Part);
    const base = part.gen * capacity;
    _countPipe = root
        .createComputePipeline({ compute: countKernel(base, part.mask, _surfaceCount) })
        .$name("shallot-part-count");
    _scanPipe = root.createComputePipeline({ compute: scanKernel }).$name("shallot-part-scan");
    _scatterPipe = root
        .createComputePipeline({ compute: scatterKernel(base, part.mask, _surfaceCount) })
        .$name("shallot-part-scatter");

    // both the allocation and the bind are deferred into the forcers, not done here. The drain runs
    // after every plugin's warm has resolved (warm hooks run under `Promise.all`), which is the first
    // moment `Meshes` is flushed and `membership` / `transforms` / `cullVolumes` are published — so
    // `syncBuffers` can size the pack's buffers there, and the dispatch that forces the compile has
    // something to bind. One forcer per pipeline, so each gets its own row in the compile table
    precompile("shallot-part-count", () => {
        syncBuffers();
        const bound = bindCount();
        bound?.dispatchWorkgroups(0);
        return bound;
    });
    precompile("shallot-part-scan", () => {
        const bound = bindScan();
        bound?.dispatchWorkgroups(0);
        return bound;
    });
    precompile("shallot-part-scatter", () => {
        const bound = bindScatter();
        bound?.dispatchWorkgroups(0);
        return bound;
    });
}

export const PartTraits = {
    requires: [Transform],
    defaults: () => ({ surface: Surfaces.id("default") ?? 0, mesh: Meshes.id("cube") ?? 0 }),
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
