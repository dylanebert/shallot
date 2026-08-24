// Scene-bounds reduction — primitive AABBs → one scene AABB, the first stage of the
// LBVH builder. The build's first pass: Morton
// normalization (Phase 3) needs the scene extent before any code is computed.
//
// Subgroup-first, the gpu.md "reduce in workgroup, then one atomic" rule in its
// subgroup form: each lane folds its grid-stride slice, `subgroupMin`/`subgroupMax`
// reduces across the subgroup, a second subgroup op folds the per-subgroup partials,
// and one lane does a single atomicMin/Max per axis to the global slot — no LDS tree
// where a subgroup op suffices, no per-thread global atomic.
//
// A subgroup-free LDS variant ships alongside it, selected by the `subgroups` flag
// (WebKit has no subgroups — Safari / WKWebView). It swaps the two subgroup reduces
// for one canonical workgroup-shared-memory tree reduce over the WG lanes; everything
// else (grid-stride fold, ordered-u32 atomics, finalize) is shared. Both produce the
// bit-identical scene AABB — min/max introduce no rounding — so they pass the same
// oracle gate, and the choice lands once at build, never per frame.
//
// WebGPU has integer atomics only, so the six axis extremes reduce as order-
// preserving u32 (`orderU32`): flip the sign bit on positives, all bits on negatives,
// which makes integer atomicMin/Max agree with float ordering across the whole range.
// A finalize pass decodes the six back to the f32 scene AABB. min/max introduce no
// rounding and the bitcast is bijective, so the result is bit-exact against the CPU
// oracle (tests/bvh/oracle.ts sceneBounds) — the Phase 2 gate, validated on the real
// GPU by the `accel` gym scenario's build gate, including degenerate (coplanar, single-prim).
//
// Designed for subgroup width 32, correct at 64 (the platform floor). The level-2
// reduce assumes the per-subgroup partials fit one subgroup (numSub <= sgsize, true
// for sgsize >= 16); software/sub-16 subgroups are out of scope.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { precompile, precompileScope } from "../../engine/runtime";
import { bitcastF32toU32, idiv } from "../../engine/utils/core";

const WG = 256; // workgroup size, both kernels
const MAX_SUB = 64; // max subgroup size on the floor — sizes the partials array so wgMin[sid] never reads OOB
const MAX_WG = 1024; // workgroup cap; the grid-stride loop folds any larger prim count
const HALF_WG = WG >> 1; // the LDS tree's first stride
const SCRATCH_U32 = 6; // ordered-u32 axis extremes: [min.xyz, max.xyz]
const F32_MAX_BITS = 0x7f7fffff; // neutral element: empty lanes don't move the extreme

/** the reduce's bindings: prim AABBs in, the six ordered-u32 extremes out, the GPU count gating the
 *  grid-stride loop. Shared by both arms — they differ only in how the WG lanes cooperate. @internal */
export const reduceLayout = tgpu.bindGroupLayout({
    prims: { storage: d.arrayOf(d.vec4f), access: "readonly" }, // 2 vec4/prim: min.xyz+pad, max.xyz+pad
    scratch: { storage: d.arrayOf(d.atomic(d.u32), SCRATCH_U32), access: "mutable" },
    countBuf: { storage: d.arrayOf(d.u32), access: "readonly" }, // [0] = GPU-driven prim count
});

/** the finalize pass's bindings: the same scratch read non-atomically, decoded into the scene AABB.
 *  @internal */
export const finalizeLayout = tgpu.bindGroupLayout({
    scratch: { storage: d.arrayOf(d.u32), access: "readonly" },
    bounds: { storage: d.arrayOf(d.vec4f, 2), access: "mutable" },
});

/**
 * IEEE float → order-preserving u32. Positive (sign 0): set the sign bit, so it sorts above every
 * negative. Negative (sign 1): flip all bits, reversing magnitude order. Bijective, so {@link
 * unorderU32} decodes it exactly — which is what lets WebGPU's integer-only atomicMin/Max agree with
 * float ordering across the whole range.
 * @example const key = orderU32(-1); // sorts below orderU32(0)
 */
export const orderU32 = tgpu.fn(
    [d.f32],
    d.u32,
)((f) => {
    "use gpu";
    const u = bitcastF32toU32(f);
    // both arms carry the same formula; JS bitwise ops are signed 32-bit, so the CPU arm needs the
    // trailing `>>> 0` to land back in u32 range (the `tgsl.ts` dual shape — the ternary folds at
    // shader-gen). The shifts are `>>>` in both arms: JS `>>` on a u32 is arithmetic and disagrees
    // with the WGSL `>>` both forms emit.
    return std.isBeingTranspiled()
        ? std.select(~u, u | 0x80000000, u >>> 31 === 0)
        : u >>> 31 === 0
          ? (u | 0x80000000) >>> 0
          : ~u >>> 0;
});

/**
 * the inverse of {@link orderU32}: an order-preserving u32 back to its f32. Exact — the encode is
 * bijective and min/max introduce no rounding, so a reduced scene AABB is bit-identical to the CPU
 * oracle's.
 * @example unorderU32(orderU32(x)) === x
 */
export const unorderU32 = tgpu.fn(
    [d.u32],
    d.f32,
)((o) => {
    "use gpu";
    return std.bitcastU32toF32(
        std.isBeingTranspiled()
            ? std.select(~o, o ^ 0x80000000, o >>> 31 === 1)
            : o >>> 31 === 1
              ? (o ^ 0x80000000) >>> 0
              : ~o >>> 0,
    );
});

/** one lane's folded slice of the prim range */
const Extremes = d.struct({ mn: d.vec3f, mx: d.vec3f }).$name("Extremes");

// grid-stride fold of this lane's slice. The neutral element is ±f32 max, so a lane with no prims
// contributes nothing to either extreme.
const fold = tgpu
    .fn(
        [d.u32, d.u32],
        Extremes,
    )((start, stride) => {
        "use gpu";
        const fmax = std.bitcastU32toF32(F32_MAX_BITS);
        let mn = d.vec3f(fmax);
        let mx = d.vec3f(-fmax);
        const count = reduceLayout.$.countBuf[0];
        let i = start;
        while (i < count) {
            mn = std.min(mn, reduceLayout.$.prims[i * 2].xyz);
            mx = std.max(mx, reduceLayout.$.prims[i * 2 + 1].xyz);
            i = i + stride;
        }
        return Extremes({ mn, mx });
    })
    .$name("foldSlice");

// one global atomic per axis per workgroup — the gpu.md "reduce in the workgroup, then ONE atomic"
// rule's publish step. Six ordered-u32 slots: [min.xyz, max.xyz].
const publish = tgpu
    .fn([Extremes])((t) => {
        "use gpu";
        std.atomicMin(reduceLayout.$.scratch[0], orderU32(t.mn.x));
        std.atomicMin(reduceLayout.$.scratch[1], orderU32(t.mn.y));
        std.atomicMin(reduceLayout.$.scratch[2], orderU32(t.mn.z));
        std.atomicMax(reduceLayout.$.scratch[3], orderU32(t.mx.x));
        std.atomicMax(reduceLayout.$.scratch[4], orderU32(t.mx.y));
        std.atomicMax(reduceLayout.$.scratch[5], orderU32(t.mx.z));
    })
    .$name("publishExtremes");

// The two arms are two entry points, not one kernel behind a flag or a slot: the subgroup arm's
// `subgroup_invocation_id` / `subgroup_size` are *entry-point parameters*, and declaring them at all
// needs `enable subgroups` — which the subgroup-free tier can't have. So the fork has to sit above the
// builtin list, and everything below it (the fold, the ordered-u32 publish) is shared verbatim.

// the subgroup path's partials: one slot per subgroup (numSub = WG/sgsize <= MAX_SUB)
const sgMin = tgpu.workgroupVar(d.arrayOf(d.vec3f, MAX_SUB));
const sgMax = tgpu.workgroupVar(d.arrayOf(d.vec3f, MAX_SUB));

const subgroupReduce = tgpu
    .computeFn({
        workgroupSize: [WG],
        in: {
            gid: d.builtin.globalInvocationId,
            lid: d.builtin.localInvocationId,
            nwg: d.builtin.numWorkgroups,
            sid: d.builtin.subgroupInvocationId,
            sgsize: d.builtin.subgroupSize,
        },
    })((input) => {
        "use gpu";
        const tid = input.lid.x;
        const t = fold(input.gid.x, input.nwg.x * WG);

        // level 1: reduce across the subgroup; lane 0 publishes the partial
        const sgid = idiv(tid, input.sgsize);
        const smin = std.subgroupMin(t.mn);
        const smax = std.subgroupMax(t.mx);
        if (input.sid === 0) {
            sgMin.$[sgid] = d.vec3f(smin);
            sgMax.$[sgid] = d.vec3f(smax);
        }
        std.workgroupBarrier();

        // level 2: every subgroup redundantly folds the numSub partials (they fit one subgroup since
        // numSub = WG/sgsize <= sgsize for sgsize >= 16). No second barrier, no LDS tree — the
        // subgroup op is the whole reduce.
        const numSub = idiv(WG, input.sgsize);
        const fmax = std.bitcastU32toF32(F32_MAX_BITS);
        const vmin = std.select(d.vec3f(fmax), sgMin.$[input.sid], input.sid < numSub);
        const vmax = std.select(d.vec3f(-fmax), sgMax.$[input.sid], input.sid < numSub);
        // both reduces run on every lane: a subgroup op inside the `tid == 0` guard is
        // non-uniform control flow, which Tint rejects outright
        const tmin = std.subgroupMin(vmin);
        const tmax = std.subgroupMax(vmax);
        if (tid === 0) publish(Extremes({ mn: tmin, mx: tmax }));
    })
    .$name("boundsReduceSubgroup");

// the LDS path's scratch: one slot per WG lane, halved down the tree
const ldsMin = tgpu.workgroupVar(d.arrayOf(d.vec3f, WG));
const ldsMax = tgpu.workgroupVar(d.arrayOf(d.vec3f, WG));

const ldsReduce = tgpu
    .computeFn({
        workgroupSize: [WG],
        in: {
            gid: d.builtin.globalInvocationId,
            lid: d.builtin.localInvocationId,
            nwg: d.builtin.numWorkgroups,
        },
    })((input) => {
        "use gpu";
        const tid = input.lid.x;
        const t = fold(input.gid.x, input.nwg.x * WG);

        // LDS tree reduce over the WG lanes: each lane seeds its slot, then a halving tree folds to
        // slot 0. The canonical no-subgroup reduce (the gpu.md rule's pre-subgroup form).
        ldsMin.$[tid] = d.vec3f(t.mn);
        ldsMax.$[tid] = d.vec3f(t.mx);
        std.workgroupBarrier();
        let s = d.u32(HALF_WG);
        while (s > 0) {
            if (tid < s) {
                ldsMin.$[tid] = std.min(ldsMin.$[tid], ldsMin.$[tid + s]);
                ldsMax.$[tid] = std.max(ldsMax.$[tid], ldsMax.$[tid + s]);
            }
            std.workgroupBarrier();
            s = s >>> 1;
        }
        if (tid === 0) publish(Extremes({ mn: ldsMin.$[0], mx: ldsMax.$[0] }));
    })
    .$name("boundsReduceLds");

// decode the six ordered-u32 extremes into the f32 scene AABB (2 vec4: min.xyz+pad, max.xyz+pad)
const finalize = tgpu
    .computeFn({ workgroupSize: [1] })(() => {
        "use gpu";
        const s = finalizeLayout.$.scratch;
        finalizeLayout.$.bounds[0] = d.vec4f(
            unorderU32(s[0]),
            unorderU32(s[1]),
            unorderU32(s[2]),
            0,
        );
        finalizeLayout.$.bounds[1] = d.vec4f(
            unorderU32(s[3]),
            unorderU32(s[4]),
            unorderU32(s[5]),
            0,
        );
    })
    .$name("boundsFinalize");

/** the emitted bounds WGSL — the device-free structural seam its test resolves. Both reduce arms plus
 *  the shared finalize, so a test can diff the arms against each other.
 *  @internal */
export function boundsWgsl(): { subgroup: string; lds: string; finalize: string } {
    return {
        subgroup: tgpu.resolve([subgroupReduce], { names: "strict" }),
        lds: tgpu.resolve([ldsReduce], { names: "strict" }),
        finalize: tgpu.resolve([finalize], { names: "strict" }),
    };
}

/** buffers {@link createBvh} threads through so the stages share one set; an omitted field is allocated internally */
export interface SceneBoundsShared {
    /** input prim AABB buffer — 2 vec4<f32> per prim */
    prims?: GPUBuffer;
    /** scene AABB output (2 × vec4<f32>) */
    bounds?: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]); the reduce gates its loop on it */
    count?: GPUBuffer;
}

/**
 * a built scene-bounds reducer sized for `maxPrims`. Write primitive AABBs into {@link SceneBounds.prims}
 * (2 × vec4<f32> per prim), record {@link SceneBounds.reduce}, submit, then read the scene AABB back from
 * {@link SceneBounds.bounds} (2 × vec4<f32>: `min.xyz+pad`, `max.xyz+pad`).
 */
export interface SceneBounds {
    /** input prim AABB buffer — fill [0, count) prims, 2 vec4 each */
    readonly prims: GPUBuffer;
    /** output scene AABB — 2 vec4<f32> after the reduction (min then max) */
    readonly bounds: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]) — write it (≤ `maxPrims`) before recording */
    readonly count: GPUBuffer;
    /** capacity the input buffer is sized for */
    readonly maxPrims: number;
    /** record the bounds reduction over [0, count) prims onto `encoder`, count read from {@link SceneBounds.count} */
    reduce(encoder: GPUCommandEncoder): void;
    destroy(): void;
}

/**
 * build a scene-bounds reducer for up to `maxPrims` primitive AABBs. Compiles the
 * reduce + finalize kernels and allocates the input / scratch / output buffers up
 * front; {@link SceneBounds.reduce} then records with no further allocation.
 *
 * `subgroups` picks the reduce kernel — the subgroup reduce (default, when the device
 * has the feature) or the subgroup-free LDS tree reduce. Both yield the bit-identical
 * scene AABB; force `false` to exercise the LDS path on a subgroup-capable device.
 *
 * @example
 * const sb = await createSceneBounds(device, 1 << 20);
 * device.queue.writeBuffer(sb.prims, 0, primAabbs);
 * device.queue.writeBuffer(sb.count, 0, new Uint32Array([count]));
 * const enc = device.createCommandEncoder();
 * sb.reduce(enc);
 * device.queue.submit([enc.finish()]);
 */
export async function createSceneBounds(
    device: GPUDevice,
    maxPrims: number,
    shared: SceneBoundsShared = {},
    subgroups: boolean = device.features.has("subgroups"),
): Promise<SceneBounds> {
    const root = Compute.root;
    const cap = Math.max(1, maxPrims);
    const owned: GPUBuffer[] = [];
    const own = (label: string, size: number, usage: number): GPUBuffer => {
        const b = device.createBuffer({ label, size, usage });
        owned.push(b);
        return b;
    };
    const StoreDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    const prims = shared.prims ?? own("bounds-prims", cap * 32, StoreDst);
    const scratch = own("bounds-scratch", SCRATCH_U32 * 4, StoreDst);
    const bounds =
        shared.bounds ?? own("bounds-out", 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const count = shared.count ?? own("bounds-count", 4, StoreDst);

    // the two arms are separate entry points (the subgroup builtins can't be declared without the
    // feature), so the pick is a branch on the pipeline, not a value inside one kernel
    const reducePipeline = subgroups
        ? root.createComputePipeline({ compute: subgroupReduce }).$name("bounds-reduce")
        : root.createComputePipeline({ compute: ldsReduce }).$name("bounds-reduce");
    const reduce = reducePipeline.with(
        root.createBindGroup(reduceLayout, { prims, scratch, countBuf: count }),
    );
    const finalizeBound = root
        .createComputePipeline({ compute: finalize })
        .$name("bounds-finalize")
        .with(root.createBindGroup(finalizeLayout, { scratch, bounds }));
    // per-instance labels — an app can build more than one BVH, and the queue rejects a duplicate label
    const scope = precompileScope("bounds");
    for (const [label, bound] of [
        ["reduce", reduce],
        ["finalize", finalizeBound],
    ] as const) {
        await precompile(`${scope}-${label}`, () => {
            return bound;
        });
    }

    // scratch reset before each reduce: min axes start above every value, max below.
    const init = new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0]);
    // dispatch is a CPU constant — the grid-stride loop folds [0, count) with count read
    // on the GPU, so the count buffer never crosses to the CPU
    const numWg = Math.min(Math.ceil(cap / WG), MAX_WG);

    return {
        prims,
        bounds,
        count,
        maxPrims,
        reduce(encoder: GPUCommandEncoder): void {
            device.queue.writeBuffer(scratch, 0, init);

            // profile under one accumulating row (no-op without ProfilePlugin → the gym measure unaffected)
            const reducePass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("bvh:bounds"),
            });
            reduce.with(reducePass).dispatchWorkgroups(numWg);
            reducePass.end();

            const finalizePass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("bvh:bounds"),
            });
            finalizeBound.with(finalizePass).dispatchWorkgroups(1);
            finalizePass.end();
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
