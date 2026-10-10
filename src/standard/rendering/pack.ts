import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { CULL_FRUSTUM, CULL_VOLUME_FLOATS, CullVolumes } from "../../core/rendering";
import { Xform, xformPoint } from "../../engine/utils";
import { MeshInstanceInput } from "./material-data";
import { DrawIndexedIndirect } from "./registry";

// The pack kernels: cull → count → scan → scatter, the compute half of the MeshInstance producer. Count and
// scatter share the same cull inputs, so those are ONE bind group layout both kernels reference (and the
// shared `visible` test closes over), and the prefix needs no re-declaration. The scan is a third pipeline
// with no cull inputs at all. Each layout pins its group index with `$idx`: the dispatches are issued on a
// raw compute pass, which addresses a bind group by index, so the index is declared here rather than left
// to resolution order. This displaces the note that the group index is invisible to the CPU side.

/** dense MeshInstance row counts and view/pair dimensions, written once per changed frame @internal */
export const CullParams = d.struct({
    viewCount: d.u32,
    pairCount: d.u32,
    instanceCount: d.u32,
    instanceCapacity: d.u32,
});

/** shared dense inputs for count + scatter, plus mesh bounds and per-view cull volumes @internal */
export const cullLayout = tgpu
    .bindGroupLayout({
        instanceRows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
        instances: { storage: d.arrayOf(MeshInstanceInput), access: "readonly" },
        globalTransforms: { storage: d.arrayOf(Xform), access: "readonly" },
        globalTransformRows: { storage: d.arrayOf(d.u32), access: "readonly" },
        meshBounds: { storage: d.arrayOf(d.vec4f), access: "readonly" },
        cullVolumes: { uniform: CullVolumes },
        params: { uniform: CullParams },
    })
    .$idx(0);

/** the count pass's own output: one atomic tally per (view slot, pair) @internal */
export const countLayout = tgpu
    .bindGroupLayout({
        counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
    })
    .$idx(1);

/** the scan pass's I/O — no cull inputs, so it references neither shared layout @internal */
export const scanLayout = tgpu
    .bindGroupLayout({
        counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        drawArgs: { storage: d.arrayOf(DrawIndexedIndirect), access: "mutable" },
        params: { uniform: CullParams },
    })
    .$idx(0);

/** Scatter reuses the indirect instance count as its cursor, restoring the draw record before rendering.
 * The atomic view has the canonical indirect record's byte layout. @internal */
const ScatterArgs = d.struct({
    ...DrawIndexedIndirect.propTypes,
    instanceCount: d.atomic(d.u32),
});
export const scatterLayout = tgpu
    .bindGroupLayout({
        drawArgs: { storage: d.arrayOf(ScatterArgs), access: "mutable" },
        packedEids: { storage: d.arrayOf(d.vec4u), access: "mutable" },
    })
    .$idx(1);

// header vec4 + the 6 frustum planes
const CULL_STRIDE = CULL_VOLUME_FLOATS / 4;

/**
 * test the instance's world bounding sphere against this slot's frustum cull volume, dispatched on the
 * leading tag word. `CULL_FRUSTUM` = a 6-plane AND (every camera, the sun, and each point/spot shadow
 * combo's depth view). An unknown tag (an unwritten slot) keeps the instance, and so does a slot past the
 * active view count (headless, or the synthetic slot 0 when no camera exists) — the pack then degrades to
 * plain compaction. The radius scales by the largest |scale| axis, conservative for non-uniform scale.
 * @internal
 */
export const visible = tgpu.fn(
    [d.u32, d.u32, d.u32],
    d.bool,
)((mid, globalTransformRow, slot) => {
    "use gpu";
    if (slot >= cullLayout.$.params.viewCount) return true;
    const xf = cullLayout.$.globalTransforms[globalTransformRow];
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

// Resolve one active table row to its pair, identity and GlobalTransform row. A miss is out of range.
const Pair = d.struct({
    pair: d.u32,
    mid: d.u32,
    eid: d.u32,
    row: d.u32,
    globalTransformRow: d.u32,
});

// Material-type count is baked; the active row list and pair grid remain dynamic.
function pairFactory(materialTypeCount: number) {
    return tgpu
        .fn(
            [d.u32],
            Pair,
        )((index) => {
            "use gpu";
            const entry = cullLayout.$.instanceRows[index];
            const eid = entry.x;
            const row = entry.y;
            const instance = cullLayout.$.instances[row];
            const encodedGlobalTransform = cullLayout.$.globalTransformRows[eid];
            const invalidPair = cullLayout.$.params.pairCount;
            const materialType = instance.materialType;
            if (materialType >= materialTypeCount || encodedGlobalTransform === 0) {
                return Pair({
                    pair: invalidPair,
                    mid: instance.mesh,
                    eid,
                    row,
                    globalTransformRow: 0,
                });
            }
            return Pair({
                pair: instance.mesh * materialTypeCount + materialType,
                mid: instance.mesh,
                eid,
                row,
                globalTransformRow: encodedGlobalTransform - 1,
            });
        })
        .$name("instancePair");
}

/** Tally frustum-visible active MeshInstance rows per (view slot, pair); no entity-capacity scan. @internal */
export function countKernel(materialTypeCount: number) {
    const pair = pairFactory(materialTypeCount);
    return tgpu
        .computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const index = input.gid.x;
            const slot = input.gid.y;
            if (index >= cullLayout.$.params.instanceCount) return;
            const g = pair(index);
            if (g.pair >= cullLayout.$.params.pairCount) return;
            if (
                (cullLayout.$.instances[g.row].flags & 1) !== 0 &&
                cullLayout.$.cullVolumes[slot * CULL_STRIDE].y !== 0
            )
                return;
            if (!visible(g.mid, g.globalTransformRow, slot)) return;
            std.atomicAdd(countLayout.$.counts[slot * cullLayout.$.params.pairCount + g.pair], 1);
        })
        .$name("meshPreprocessCount");
}

const SCAN_WG = 256;

const temp = tgpu.workgroupVar(d.arrayOf(d.u32, SCAN_WG));
const carry = tgpu.workgroupVar(d.u32);

/**
 * exclusive prefix sum, one workgroup per view slot. Each slot's row is scanned in parallel: a `SCAN_WG`-wide
 * LDS Hillis-Steele scan walks the row in tiles, a `carry` threading the running offset across tiles, so the
 * slot's packedEids region starts at `slot * instanceCapacity`. Writes instanceCount + the compacted firstInstance,
 * resets the tallies for the next count pass, and leaves indexCount / firstIndex (lanes 0, 2) alone.
 * baseVertex temporarily saves the tally; scatter restores it to zero and restores instanceCount.
 * Pure LDS (no subgroup ops) — the instance pack stays inside the base feature floor, so a
 * physics-free app never needs `subgroups`. One workgroup per slot keeps the pass independent of the
 * view-slot count. Compaction is this GPU prefix-sum scan, never a CPU gather.
 * @internal
 */
export function scanKernel() {
    return tgpu.computeFn({
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
                scanLayout.$.drawArgs[idx].firstInstance =
                    slot * scanLayout.$.params.instanceCapacity + carry.$ + excl;
                // Packed mesh indices require baseVertex = 0 at draw time. Until scatter finishes,
                // the word preserves the tally while instanceCount serves as the reverse cursor.
                scanLayout.$.drawArgs[idx].baseVertex = d.i32(c);
                std.atomicStore(scanLayout.$.counts[idx], 0);
            }
            std.workgroupBarrier();
            if (local === 0) carry.$ = carry.$ + tileTotal;
            std.workgroupBarrier();

            tileBase = tileBase + SCAN_WG;
        }
    });
}

/** Scatter visible identities and MeshInstance rows into matching dense per-view draw lists. @internal */
export function scatterKernel(materialTypeCount: number) {
    const pair = pairFactory(materialTypeCount);
    return tgpu
        .computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const index = input.gid.x;
            const slot = input.gid.y;
            if (index >= cullLayout.$.params.instanceCount) return;
            const g = pair(index);
            if (g.pair >= cullLayout.$.params.pairCount) return;
            if (
                (cullLayout.$.instances[g.row].flags & 1) !== 0 &&
                cullLayout.$.cullVolumes[slot * CULL_STRIDE].y !== 0
            )
                return;
            if (!visible(g.mid, g.globalTransformRow, slot)) return;
            const idx = slot * cullLayout.$.params.pairCount + g.pair;
            const remaining = std.atomicSub(scatterLayout.$.drawArgs[idx].instanceCount, 1);
            const target = scatterLayout.$.drawArgs[idx].firstInstance + remaining - 1;
            scatterLayout.$.packedEids[target] = d.vec4u(g.eid, g.globalTransformRow, g.row + 1, 0);
            if (remaining === 1) {
                // Every survivor has already decremented the cursor. Only this invocation reads
                // the saved tally, restoring both indirect words before any render pass consumes them.
                const count = d.u32(scatterLayout.$.drawArgs[idx].baseVertex);
                scatterLayout.$.drawArgs[idx].baseVertex = 0;
                std.atomicStore(scatterLayout.$.drawArgs[idx].instanceCount, count);
            }
        })
        .$name("meshPreprocessScatter");
}
