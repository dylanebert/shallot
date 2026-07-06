import type { State, System } from "../../engine";
import { Compute, capacity, srgb8x4, u32 } from "../../engine";
import { XFORM_WGSL } from "../../engine/utils/core";
import {
    BeginFrameSystem,
    CULL_FRUSTUM,
    CULL_VOLUME_FLOATS,
    Draws,
    Meshes,
    Render,
    Surfaces,
} from "../render/core";
import { slab } from "../slab";
import { Transform } from "../transforms";

const DRAW_ARG_STRIDE = 20;

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
let _counts: GPUBuffer | null = null;
let _meshBounds: GPUBuffer | null = null;
let _cullParams: GPUBuffer | null = null;
let _cullStaging: Uint32Array | null = null;
let _countPipe: GPUComputePipeline | null = null;
let _scanPipe: GPUComputePipeline | null = null;
let _scatterPipe: GPUComputePipeline | null = null;
let _countLayout: GPUBindGroupLayout | null = null;
let _scanLayout: GPUBindGroupLayout | null = null;
let _scatterLayout: GPUBindGroupLayout | null = null;
let _countGroup: GPUBindGroup | null = null;
let _scanGroup: GPUBindGroup | null = null;
let _scatterGroup: GPUBindGroup | null = null;
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
    drawArgs: GPUBuffer | null;
    /** packed survivor eids, one `capacity`-sized region per view slot; null until `warmPart` */
    packedEids: GPUBuffer | null;
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
        _countGroup ??= countGroup();
        _scanGroup ??= scanGroup();
        _scatterGroup ??= scatterGroup();

        // viewCount + pairCount let the cull shader find a view's frustum and
        // index its slot's slice; slot ≥ viewCount means no frustum (headless),
        // packed unculled. Queued before EndFrameSystem submits the encoder, so
        // it lands before the pack executes
        const views = Math.max(1, Render.viewCount);
        _cullStaging![0] = Render.viewCount;
        _cullStaging![1] = _pairCount;
        Compute.device.queue.writeBuffer(
            _cullParams!,
            0,
            _cullStaging! as Uint32Array<ArrayBuffer>,
        );

        Render.encoder.clearBuffer(_counts!);
        const pass = Render.encoder.beginComputePass({
            label: "kitchen-part-pack",
            timestampWrites: Compute.span?.("part:pack"),
        });
        const rows = Math.ceil(capacity / 64);
        pass.setPipeline(_countPipe);
        pass.setBindGroup(0, _countGroup);
        pass.dispatchWorkgroups(rows, views);
        // one workgroup per allocated view slot (the counts buffer spans _viewDim ×
        // pairCount); slots past the active views carry zero counts → zero instanceCount
        pass.setPipeline(_scanPipe);
        pass.setBindGroup(0, _scanGroup!);
        pass.dispatchWorkgroups(_viewDim);
        pass.setPipeline(_scatterPipe);
        pass.setBindGroup(0, _scatterGroup);
        pass.dispatchWorkgroups(rows, views);
        pass.end();
    },
};

// the buffers the cull reads: per-entity slabs + membership mirror, plus the
// world-matrix firehose and the per-view cull volumes the visibility test
// needs. All stable, fixed-capacity identities — the slab `.gpu` from
// SlabPlugin.warm, membership/transforms/cullVolumes published by their owners
// before any draw-group consumer runs. Read when the pack first builds its bind
// groups (and on growth); a missing one is a wiring bug, not a frame to skip
interface CullInputs {
    surface: GPUBuffer;
    mesh: GPUBuffer;
    membership: GPUBuffer;
    transforms: GPUBuffer;
    cullVolumes: GPUBuffer;
}

function partInputs(): CullInputs {
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
    return { surface, mesh, membership, transforms, cullVolumes };
}

function bind(buffers: GPUBuffer[], label: string, layout: GPUBindGroupLayout): GPUBindGroup {
    return Compute.device.createBindGroup({
        label,
        layout,
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
}

// count + scatter share the same cull inputs in the same binding order; scatter
// appends drawArgs + packedEids after the read-write counts
function cullPrefix(inp: CullInputs): GPUBuffer[] {
    return [
        inp.surface,
        inp.mesh,
        inp.membership,
        inp.transforms,
        _meshBounds!,
        inp.cullVolumes,
        _cullParams!,
    ];
}

function countGroup(): GPUBindGroup {
    return bind([...cullPrefix(partInputs()), _counts!], "kitchen-part-count", _countLayout!);
}

function scanGroup(): GPUBindGroup {
    return bind([_counts!, Parts.drawArgs!, _cullParams!], "kitchen-part-scan", _scanLayout!);
}

function scatterGroup(): GPUBindGroup {
    return bind(
        [...cullPrefix(partInputs()), Parts.drawArgs!, _counts!, Parts.packedEids!],
        "kitchen-part-scatter",
        _scatterLayout!,
    );
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
    Parts.drawArgs = device.createBuffer({
        label: "kitchen-draw-args",
        size: records * DRAW_ARG_STRIDE,
        usage:
            GPUBufferUsage.INDIRECT |
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
    });
    _counts = device.createBuffer({
        label: "kitchen-part-counts",
        size: records * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // packedEids holds one capacity-sized region per view — realloc only when
    // the view dimension grows, so a mesh registering doesn't churn the buffer
    // sear binds (its identity invalidates the bind-group cache)
    let stalePacked: GPUBuffer | null = null;
    if (growView || !Parts.packedEids) {
        stalePacked = Parts.packedEids;
        Parts.packedEids = device.createBuffer({
            label: "kitchen-packed-eids",
            size: _viewDim * capacity * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        Compute.buffers.set("eids", Parts.packedEids);
    }

    // meshBounds is indexed by mesh id — rebuild only when a mesh registers
    let staleBounds: GPUBuffer | null = null;
    if (growMesh || !_meshBounds) {
        staleBounds = _meshBounds;
        _meshBounds = writeMeshBounds(device);
    }

    _countGroup = null;
    _scanGroup = null;
    _scatterGroup = null;
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
function writeMeshBounds(device: GPUDevice): GPUBuffer {
    const buffer = device.createBuffer({
        label: "kitchen-mesh-bounds",
        size: _meshCount * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const data = new Float32Array(_meshCount * 4);
    for (const m of Meshes) {
        const id = Meshes.id(m.name)!;
        if (m.bounds) data.set(m.bounds, id * 4);
        else data[id * 4 + 3] = 1e30; // never-cull sentinel
    }
    device.queue.writeBuffer(buffer, 0, data as Float32Array<ArrayBuffer>);
    return buffer;
}

/**
 * register one Draw per `(Part-compatible surface × mesh)` pair and seed every
 * slot's static indexed-indirect args (indexCount = `mesh.indexCount`, firstIndex =
 * `mesh.indexBase`, baseVertex = 0): the per-frame pack fills instanceCount + the
 * compacted firstInstance. The static args repeat across all `_viewDim` slots since a
 * camera attaching later reuses a slot whose geometry never changes; the Draw
 * points at the pair's slot-0 record and carries `viewStride = pairCount × 20`,
 * so sear reads `slot`'s record as `offset + slot * viewStride`. A surface is
 * instanced when it declares the `eids` + `transforms` bindings (sear applies
 * the standard transform). Every combination gets a Draw; the ones no entity uses pack to
 * `instanceCount: 0` and `drawIndirect` no-ops them. So a producer just
 * registers its mesh + surface and spawns Parts: the draw it needs already
 * exists, whenever its entities appear. (When `multi-draw-indirect` lands,
 * these per-pair draws fold into one call per surface on the GPU.) Re-run by
 * `syncBuffers` whenever a mesh registers or the view count grows: it re-seeds
 * every slot and repoints the Draws at the freshly grown `drawArgs`. The pair
 * offset is `mid * surfaceCount + sid`, so a new mesh only appends slots
 */
function registerDraws(): void {
    if (!Compute.device || _pairCount === 0) return;
    const viewStride = _pairCount * DRAW_ARG_STRIDE;
    for (const surface of Surfaces) {
        if (!surface.bindings?.eids || !surface.bindings?.transforms) continue;
        const sid = Surfaces.id(surface.name)!;
        for (const m of Meshes) {
            const pair = Meshes.id(m.name)! * _surfaceCount + sid;
            const offset = pair * DRAW_ARG_STRIDE;
            // DrawIndexedIndirect: indexCount, instanceCount (pack), firstIndex, baseVertex (0 — indices
            // are absolute vertex positions), firstInstance (pack)
            const args = new Uint32Array([m.indexCount, 0, m.indexBase, 0, 0]);
            for (let slot = 0; slot < _viewDim; slot++) {
                Compute.device.queue.writeBuffer(Parts.drawArgs!, slot * viewStride + offset, args);
            }
            Draws.register({
                name: `part:${surface.name}:${m.name}`,
                surface: surface.name,
                mesh: m.name,
                args: { indirect: Parts.drawArgs!, offset, viewStride },
            });
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

    _countGroup = null;
    _scanGroup = null;
    _scatterGroup = null;
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
export async function warmPart(state: State): Promise<void> {
    if (!Compute.device) return;
    const device = Compute.device;
    _surfaceCount = Surfaces.size;
    _meshCount = 0;
    _pairCount = 0;
    _viewDim = 1;
    Parts.drawArgs = null;
    _counts = null;
    _meshBounds = null;

    // one capacity-sized region (slot 0); syncBuffers grows it as cameras attach.
    // COPY_SRC for GPU-debug readback (gpu.md) + the pack tests
    Parts.packedEids = device.createBuffer({
        label: "kitchen-packed-eids",
        size: capacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    Compute.buffers.set("eids", Parts.packedEids);

    // { viewCount, pairCount } — written each frame, read by all three passes
    _cullParams = device.createBuffer({
        label: "kitchen-part-cull-params",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cullStaging = new Uint32Array(4);
    if (_surfaceCount === 0) return;

    const part = state.membership.bit(Part);
    _countLayout = layout("kitchen-part-count", "rrrrrruw");
    _scanLayout = layout("kitchen-part-scan", "wwu");
    _scatterLayout = layout("kitchen-part-scatter", "rrrrrrurww");

    const gate = `if ((membership[${part.gen}u * ${capacity}u + eid] & ${part.mask}u) == 0u) { return; }`;
    // resolve the entity's within-view (surface, mesh) pair. surfaceCount is
    // baked (stable); pairCount comes from cullParams, so a late mesh whose pair
    // sits past the current grid is skipped until syncBuffers grows it
    const readPair = `
    let sid = surfaceField[eid];
    let mid = meshField[eid];
    if (sid >= ${_surfaceCount}u) { return; }
    let pair = mid * ${_surfaceCount}u + sid;
    if (pair >= params.pairCount) { return; }`;

    // the cull inputs (bindings 0..6) + the frustum test, shared by count +
    // scatter. `visible` transforms the mesh's local bounding sphere by the
    // instance transform (center by `xformPoint`, radius by the largest |scale|
    // axis — conservative for non-uniform scale) and tests it against this
    // view's six planes. slot ≥ viewCount means no frustum exists (headless or a
    // synthetic slot): keep everything, so the pack degrades to plain compaction
    const cullDecls =
        XFORM_WGSL +
        /* wgsl */ `
struct CullParams { viewCount: u32, pairCount: u32 }

@group(0) @binding(0) var<storage, read> surfaceField: array<u32>;
@group(0) @binding(1) var<storage, read> meshField: array<u32>;
@group(0) @binding(2) var<storage, read> membership: array<u32>;
@group(0) @binding(3) var<storage, read> transforms: array<Xform>;
@group(0) @binding(4) var<storage, read> meshBounds: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> cullVolumes: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: CullParams;

const CULL_STRIDE: u32 = ${CULL_VOLUME_FLOATS / 4}u; // header vec4 + the 6 frustum planes

// test the instance's world bounding sphere against this slot's frustum cull volume, dispatched on the
// leading tag word. tag ${CULL_FRUSTUM} = a 6-plane AND (every camera, the sun, and each point/spot shadow
// combo's depth view). An unknown tag (an unwritten slot) keeps the instance.
fn visible(eid: u32, mid: u32, slot: u32) -> bool {
    if (slot >= params.viewCount) { return true; }
    let xf = transforms[eid];
    let b = meshBounds[mid];
    let center = xformPoint(xf, b.xyz);
    let radius = b.w * max(abs(xf.scale.x), max(abs(xf.scale.y), abs(xf.scale.z)));
    let base = slot * CULL_STRIDE;
    let header = cullVolumes[base];
    switch u32(header.x) {
        case ${CULL_FRUSTUM}u: {
            // planes follow the header vec4 at base + 1
            for (var i = 0u; i < 6u; i = i + 1u) {
                let pl = cullVolumes[base + 1u + i];
                if (dot(pl.xyz, center) + pl.w < -radius) { return false; }
            }
            return true;
        }
        default: { return true; }
    }
}`;

    const countWgsl = /* wgsl */ `
${cullDecls}
@group(0) @binding(7) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    let slot = gid.y;
    if (eid >= ${capacity}u) { return; }
    ${gate}
    ${readPair}
    if (!visible(eid, mid, slot)) { return; }
    atomicAdd(&counts[slot * params.pairCount + pair], 1u);
}
`;

    // exclusive prefix sum, one workgroup per view slot — `dispatchWorkgroups(viewDim)`.
    // Each slot's row is scanned in parallel: a `SCAN_WG`-wide LDS Hillis-Steele scan
    // walks the row in tiles, a `carry` threading the running offset across tiles, so
    // the slot's packedEids region starts at slot * capacity. Writes instanceCount +
    // compacted firstInstance, resets counts so scatter reuses it as a cursor, and
    // leaves the static indexCount / firstIndex (lanes 0, 2) alone. Pure LDS (no
    // subgroup ops) — the part pack stays inside the base feature floor, so a
    // physics-free app never needs `subgroups`. One workgroup per slot keeps the
    // pass independent of the view-slot count: every slot's row scans concurrently
    const scanWgsl = /* wgsl */ `
struct CullParams { viewCount: u32, pairCount: u32 }
@group(0) @binding(0) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> drawArgs: array<u32>;
@group(0) @binding(2) var<uniform> params: CullParams;

const SCAN_WG: u32 = 256u;
var<workgroup> temp: array<u32, SCAN_WG>;
var<workgroup> carry: u32;

@compute @workgroup_size(SCAN_WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let slot = wid.x;
    let pairCount = params.pairCount;
    let local = lid.x;
    let base = slot * pairCount;

    if (local == 0u) { carry = 0u; }
    workgroupBarrier();

    // tile the row: each pass scans SCAN_WG counts, carry holds the prefix of all
    // prior tiles. Outer condition is workgroup-uniform (params + constants), so the
    // in-loop barriers are legal
    var tileBase = 0u;
    loop {
        if (tileBase >= pairCount) { break; }
        let p = tileBase + local;
        let inRange = p < pairCount;
        let idx = base + p;
        var c = 0u;
        if (inRange) { c = atomicLoad(&counts[idx]); }

        // inclusive Hillis-Steele scan of c across the workgroup into temp
        temp[local] = c;
        workgroupBarrier();
        var offset = 1u;
        loop {
            if (offset >= SCAN_WG) { break; }
            var add = 0u;
            if (local >= offset) { add = temp[local - offset]; }
            workgroupBarrier();
            temp[local] = temp[local] + add;
            workgroupBarrier();
            offset = offset * 2u;
        }
        let excl = temp[local] - c;            // inclusive − own = exclusive prefix
        let tileTotal = temp[SCAN_WG - 1u];    // every out-of-range lane added 0

        if (inRange) {
            drawArgs[idx * 5u + 1u] = c;
            drawArgs[idx * 5u + 4u] = slot * ${capacity}u + carry + excl;
            atomicStore(&counts[idx], 0u);
        }
        workgroupBarrier();
        if (local == 0u) { carry = carry + tileTotal; }
        workgroupBarrier();

        tileBase = tileBase + SCAN_WG;
    }
}
`;

    const scatterWgsl = /* wgsl */ `
${cullDecls}
@group(0) @binding(7) var<storage, read> drawArgs: array<u32>;
@group(0) @binding(8) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> packedEids: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    let slot = gid.y;
    if (eid >= ${capacity}u) { return; }
    ${gate}
    ${readPair}
    if (!visible(eid, mid, slot)) { return; }
    let idx = slot * params.pairCount + pair;
    let local = atomicAdd(&counts[idx], 1u);
    packedEids[drawArgs[idx * 5u + 4u] + local] = eid;
}
`;

    const compile = (label: string, code: string, l: GPUBindGroupLayout) =>
        device.createComputePipelineAsync({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts: [l] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
        });
    [_countPipe, _scanPipe, _scatterPipe] = await Promise.all([
        compile("kitchen-part-count", countWgsl, _countLayout),
        compile("kitchen-part-scan", scanWgsl, _scanLayout),
        compile("kitchen-part-scatter", scatterWgsl, _scatterLayout),
    ]);
}

// build a compute bind-group layout from a kinds string: `r` = read-only
// storage, `w` = read-write storage, `u` = uniform, one char per binding
function layout(label: string, kinds: string): GPUBindGroupLayout {
    return Compute.device.createBindGroupLayout({
        label,
        entries: [...kinds].map((k, binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: k === "u" ? "uniform" : k === "w" ? "storage" : "read-only-storage",
            } as const,
        })),
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
