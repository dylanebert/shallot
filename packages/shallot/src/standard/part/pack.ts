import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { capacity } from "../../engine";
import { Xform, xformPoint } from "../../engine/utils/core";
import { CULL_FRUSTUM, CULL_VOLUME_FLOATS, DrawIndexedIndirect } from "../render/core";

// The pack kernels: cull → count → scan → scatter, the compute half of the Part producer. Count and
// scatter share the same cull inputs, so those are ONE bind group layout both kernels reference (and the
// shared `visible` test closes over) — a typed bind group is bound by layout identity, never by group
// index, so the group a kernel's own I/O lands in is invisible to the CPU side and the prefix needs no
// re-declaration. The scan is a third pipeline with no cull inputs at all.

/** `{ viewCount, pairCount }` — written each frame, read by all three passes @internal */
export const CullParams = d.struct({ viewCount: d.u32, pairCount: d.u32 });

/** the cull inputs shared by count + scatter: the per-entity slabs + membership mirror, the world-transform
 *  firehose, the per-mesh bounds, and the per-view cull volumes the visibility test needs.
 *  @internal */
export const cullLayout = tgpu.bindGroupLayout({
    surfaceField: { storage: d.arrayOf(d.u32), access: "readonly" },
    meshField: { storage: d.arrayOf(d.u32), access: "readonly" },
    membership: { storage: d.arrayOf(d.u32), access: "readonly" },
    transforms: { storage: d.arrayOf(Xform), access: "readonly" },
    meshBounds: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    cullVolumes: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    params: { uniform: CullParams },
});

/** the count pass's own output: one atomic tally per (view slot, pair) @internal */
export const countLayout = tgpu.bindGroupLayout({
    counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
});

/** the scan pass's I/O — no cull inputs, so it references neither shared layout @internal */
export const scanLayout = tgpu.bindGroupLayout({
    counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
    drawArgs: { storage: d.arrayOf(DrawIndexedIndirect), access: "mutable" },
    params: { uniform: CullParams },
});

/** the scatter pass's own I/O: the scanned args it reads bases from, the counts it reuses as a cursor,
 *  and the compacted survivor list it appends into @internal */
export const scatterLayout = tgpu.bindGroupLayout({
    drawArgs: { storage: d.arrayOf(DrawIndexedIndirect), access: "readonly" },
    counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
    packedEids: { storage: d.arrayOf(d.u32), access: "mutable" },
});

// header vec4 + the 6 frustum planes
const CULL_STRIDE = CULL_VOLUME_FLOATS / 4;

/**
 * test the instance's world bounding sphere against this slot's frustum cull volume, dispatched on the
 * leading tag word. Tag `CULL_FRUSTUM` = a 6-plane AND (every camera, the sun, and each point/spot shadow
 * combo's depth view). An unknown tag (an unwritten slot) keeps the instance, and so does a slot past the
 * active view count (headless, or the synthetic slot 0 when no camera exists) — the pack then degrades to
 * plain compaction. The radius scales by the largest |scale| axis, conservative for non-uniform scale.
 * @internal
 */
export const visible = tgpu.fn(
    [d.u32, d.u32, d.u32],
    d.bool,
)((eid, mid, slot) => {
    "use gpu";
    if (slot >= cullLayout.$.params.viewCount) return true;
    const xf = cullLayout.$.transforms[eid];
    const b = cullLayout.$.meshBounds[mid];
    const center = xformPoint(xf, d.vec3f(b.x, b.y, b.z));
    const radius =
        b.w * std.max(std.abs(xf.scale.x), std.max(std.abs(xf.scale.y), std.abs(xf.scale.z)));
    const base = slot * CULL_STRIDE;
    const header = cullLayout.$.cullVolumes[base];
    if (d.u32(header.x) !== CULL_FRUSTUM) return true;
    // planes follow the header vec4 at base + 1
    let i = d.u32(0);
    while (i < 6) {
        const pl = cullLayout.$.cullVolumes[base + 1 + i];
        if (std.dot(d.vec3f(pl.x, pl.y, pl.z), center) + pl.w < -radius) return false;
        i = i + 1;
    }
    return true;
});

// the entity's within-view (surface, mesh) pair plus the mesh id the frustum test needs, read once each.
// TGSL has no optional, so a miss returns `pair = params.pairCount` — an out-of-range index the callers
// already test against.
const Pair = d.struct({ pair: d.u32, mid: d.u32 });

// surfaceCount is baked (surfaces are WGSL programs declared in code, so the count is final at warm);
// pairCount comes from the uniform, so a late mesh whose pair sits past the current grid is skipped until
// `syncBuffers` grows it
function pairFactory(surfaceCount: number) {
    return tgpu
        .fn(
            [d.u32],
            Pair,
        )((eid) => {
            "use gpu";
            const sid = cullLayout.$.surfaceField[eid];
            const mid = cullLayout.$.meshField[eid];
            if (sid >= surfaceCount) return Pair({ pair: cullLayout.$.params.pairCount, mid });
            return Pair({ pair: mid * surfaceCount + sid, mid });
        })
        .$name("partPair");
}

/**
 * tally the frustum-visible parts per (view slot, pair). One thread per (eid, slot): every thread gates on
 * the mirrored component-membership bit, then on the view's frustum — no CPU iteration over Parts. `base`
 * (the Part component's membership generation row), `mask`, and `surfaceCount` are captured numbers, so
 * they fold to literals.
 * @internal
 */
export function countKernel(base: number, mask: number, surfaceCount: number) {
    const pair = pairFactory(surfaceCount);
    // a factory-returned kernel has no binding for `names: "strict"` to read, so it resolves to `fn item`
    // unless named — and that name is what a Tint compile error and every GPU `console.log` line reports
    return tgpu
        .computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const eid = input.gid.x;
            const slot = input.gid.y;
            if (eid >= capacity) return;
            if ((cullLayout.$.membership[base + eid] & mask) === 0) return;
            const g = pair(eid);
            if (g.pair >= cullLayout.$.params.pairCount) return;
            if (!visible(eid, g.mid, slot)) return;
            std.atomicAdd(countLayout.$.counts[slot * cullLayout.$.params.pairCount + g.pair], 1);
        })
        .$name("partCount");
}

const SCAN_WG = 256;

const temp = tgpu.workgroupVar(d.arrayOf(d.u32, SCAN_WG));
const carry = tgpu.workgroupVar(d.u32);

/**
 * exclusive prefix sum, one workgroup per view slot. Each slot's row is scanned in parallel: a `SCAN_WG`-wide
 * LDS Hillis-Steele scan walks the row in tiles, a `carry` threading the running offset across tiles, so the
 * slot's packedEids region starts at `slot * capacity`. Writes instanceCount + the compacted firstInstance,
 * resets counts so scatter reuses them as a cursor, and leaves the static indexCount / firstIndex (lanes 0,
 * 2) alone. Pure LDS (no subgroup ops) — the part pack stays inside the base feature floor, so a
 * physics-free app never needs `subgroups`. One workgroup per slot keeps the pass independent of the
 * view-slot count.
 * @internal
 */
export const scanKernel = tgpu.computeFn({
    workgroupSize: [SCAN_WG],
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
})((input) => {
    "use gpu";
    const slot = input.wid.x;
    const pairCount = scanLayout.$.params.pairCount;
    const local = input.lid.x;
    const base = slot * pairCount;

    if (local === 0) carry.$ = 0;
    std.workgroupBarrier();

    // tile the row: each pass scans SCAN_WG counts, carry holds the prefix of all prior tiles. The outer
    // condition is workgroup-uniform (a uniform field + constants), so the in-loop barriers are legal
    let tileBase = d.u32(0);
    while (tileBase < pairCount) {
        const p = tileBase + local;
        const inRange = p < pairCount;
        const idx = base + p;
        let c = d.u32(0);
        if (inRange) c = std.atomicLoad(scanLayout.$.counts[idx]);

        // inclusive Hillis-Steele scan of c across the workgroup into temp
        temp.$[local] = c;
        std.workgroupBarrier();
        let offset = d.u32(1);
        while (offset < SCAN_WG) {
            let add = d.u32(0);
            if (local >= offset) add = temp.$[local - offset];
            std.workgroupBarrier();
            temp.$[local] = temp.$[local] + add;
            std.workgroupBarrier();
            offset = offset * 2;
        }
        const excl = temp.$[local] - c; // inclusive − own = exclusive prefix
        const tileTotal = temp.$[SCAN_WG - 1]; // every out-of-range lane added 0

        if (inRange) {
            scanLayout.$.drawArgs[idx].instanceCount = c;
            scanLayout.$.drawArgs[idx].firstInstance = slot * capacity + carry.$ + excl;
            std.atomicStore(scanLayout.$.counts[idx], 0);
        }
        std.workgroupBarrier();
        if (local === 0) carry.$ = carry.$ + tileTotal;
        std.workgroupBarrier();

        tileBase = tileBase + SCAN_WG;
    }
});

/**
 * append each surviving eid into its slice of packedEids, re-running the identical membership + pair +
 * frustum gate the count pass ran so the two agree exactly. `counts`, zeroed by the scan, is the per-slice
 * cursor.
 * @internal
 */
export function scatterKernel(base: number, mask: number, surfaceCount: number) {
    const pair = pairFactory(surfaceCount);
    return tgpu
        .computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const eid = input.gid.x;
            const slot = input.gid.y;
            if (eid >= capacity) return;
            if ((cullLayout.$.membership[base + eid] & mask) === 0) return;
            const g = pair(eid);
            if (g.pair >= cullLayout.$.params.pairCount) return;
            if (!visible(eid, g.mid, slot)) return;
            const idx = slot * cullLayout.$.params.pairCount + g.pair;
            const local = std.atomicAdd(scatterLayout.$.counts[idx], 1);
            scatterLayout.$.packedEids[scatterLayout.$.drawArgs[idx].firstInstance + local] = eid;
        })
        .$name("partScatter");
}

/** the emitted pack WGSL — the device-free structural seam the pack tests resolve.
 *  @internal */
export function packWgsl(
    base: number,
    mask: number,
    surfaceCount: number,
): { count: string; scan: string; scatter: string } {
    return {
        count: tgpu.resolve([countKernel(base, mask, surfaceCount)], { names: "strict" }),
        scan: tgpu.resolve([scanKernel], { names: "strict" }),
        scatter: tgpu.resolve([scatterKernel(base, mask, surfaceCount)], { names: "strict" }),
    };
}
