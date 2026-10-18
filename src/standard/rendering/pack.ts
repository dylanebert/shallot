import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    CULL_FRUSTUM,
    CULL_VOLUME_FLOATS,
    CullVolumes,
    Xform,
    xformPoint,
} from "../../core/rendering";
import { MeshInstanceInput } from "./material-data";
import { DrawIndexedIndirect } from "./registry";

// Raw compute passes bind by index, so layouts pin their group indices rather than relying on resolve order.

/** dense MeshInstance row counts and view/pair dimensions, written once per changed frame @internal */
export const CullParams = d.struct({
    viewCount: d.u32,
    pairCount: d.u32,
    meshCount: d.u32,
    instanceCount: d.u32,
    instanceCapacity: d.u32,
});

/** Generation and bounds share one binding, indexed by the dense handle index. @internal */
export const MeshBounds = d.struct({ sphere: d.vec4f, handle: d.u32 });

/** shared dense inputs for count + scatter, plus mesh bounds and per-view cull volumes @internal */
export const cullLayout = tgpu
    .bindGroupLayout({
        instanceRows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
        instances: { storage: d.arrayOf(MeshInstanceInput), access: "readonly" },
        globalTransforms: { storage: d.arrayOf(Xform), access: "readonly" },
        globalTransformRows: { storage: d.arrayOf(d.u32), access: "readonly" },
        meshBounds: { storage: d.arrayOf(MeshBounds), access: "readonly" },
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

export const CULL_STRIDE = CULL_VOLUME_FLOATS / 4;

/**
 * test the instance's world bounding sphere against this slot's frustum cull volume, dispatched on the
 * leading tag word. `CULL_FRUSTUM` = a 6-plane AND (every camera, the sun, and each point/spot shadow
 * combo's depth view). An unknown tag (an unwritten slot) keeps the instance, and so does a slot past the
 * active view count (headless, or the fallback slot 0 when no camera exists) — the pack then degrades to
 * plain compaction. The radius scales by the largest |scale| axis, conservative for non-uniform scale.
 * @internal
 */
export function visibility(
    layout: typeof cullLayout | typeof import("./producer-table").producerCullLayout,
) {
    return tgpu.fn(
        [d.u32, d.u32, d.u32, d.u32],
        d.bool,
    )((mid, instanceRow, globalTransformRow, slot) => {
        "use gpu";
        if (slot >= layout.$.params.viewCount) return true;
        const xf = layout.$.globalTransforms[globalTransformRow];
        let b = d.vec4f(layout.$.meshBounds[mid].sphere);
        const override = layout.$.instances[instanceRow].cullBounds;
        if (override.w >= 0) b = d.vec4f(override);
        const center = xformPoint(xf, d.vec3f(b.x, b.y, b.z));
        const radius =
            b.w * std.max(std.abs(xf.scale.x), std.max(std.abs(xf.scale.y), std.abs(xf.scale.z)));
        const base = slot * CULL_STRIDE;
        const header = layout.$.cullVolumes[base];
        if (d.u32(header.x) !== CULL_FRUSTUM) return true;
        let i = d.u32(0);
        while (i < 6) {
            const pl = layout.$.cullVolumes[base + 1 + i];
            if (std.dot(d.vec3f(pl.x, pl.y, pl.z), center) + pl.w < -radius) return false;
            i = i + 1;
        }
        return true;
    });
}

export const visible = visibility(cullLayout);

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
            const mid = instance.mesh & 0xfffff;
            const materialType = instance.materialType;
            const alphaMode = instance.alphaMode;
            if (
                materialType >= materialTypeCount ||
                alphaMode > 6 ||
                mid >= cullLayout.$.params.meshCount ||
                cullLayout.$.meshBounds[mid].handle !== instance.mesh ||
                encodedGlobalTransform === 0
            ) {
                return Pair({
                    pair: invalidPair,
                    mid,
                    eid,
                    row,
                    globalTransformRow: 0,
                });
            }
            let alphaKey = alphaMode;
            if (alphaMode === 3 || alphaMode === 4) alphaKey = 3;
            else if (alphaMode === 5) alphaKey = 4;
            else if (alphaMode === 6) alphaKey = 5;
            return Pair({
                pair: (materialType * 6 + alphaKey) * cullLayout.$.params.meshCount + mid,
                mid,
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
            if (!visible(g.mid, g.row, g.globalTransformRow, slot)) return;
            std.atomicAdd(countLayout.$.counts[slot * cullLayout.$.params.pairCount + g.pair], 1);
        })
        .$name("meshPreprocessCount");
}

const SCAN_WG = 256;

const temp = tgpu.workgroupVar(d.arrayOf(d.u32, SCAN_WG));
const carry = tgpu.workgroupVar(d.u32);

/**
 * Exclusive prefix sum, one workgroup per view slot. Writes instanceCount and firstInstance;
 * firstInstance addresses that slot's instance-capacity-sized region in packedEids.
 * Leaves indexCount and firstIndex unchanged and clears count tallies for the next count pass.
 * baseVertex temporarily holds the tally while scatter uses
 * instanceCount as a cursor; scatter must restore both before drawing.
 * Workgroup-local scanning needs no subgroups and keeps each view's prefix sum independent. @internal
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

        // The loop condition is workgroup-uniform, so its barriers are legal.
        let tileBase = d.u32(0);
        while (tileBase < pairCount) {
            const p = tileBase + local;
            const inRange = p < pairCount;
            const idx = base + p;
            let c = d.u32(0);
            if (inRange) c = std.atomicLoad(scanLayout.$.counts[idx]);

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
            if (!visible(g.mid, g.row, g.globalTransformRow, slot)) return;
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
