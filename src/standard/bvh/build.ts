// LBVH binary BVH build + refit — sorted Morton codes → BVH2, the coherence-safe
// builder. Replaces the
// single-kernel H-PLOC build (and its atomic-climb refit), which relied on
// cross-workgroup memory ordering WGSL cannot express (no device-scope fence — see
// gpu.md "Cross-workgroup ordering"). Cheaper to rebuild every frame, at the cost
// of lower SAH than PLOC — the right trade for the forest's per-frame rebuild, whose
// consumer (shadow any-hit) is the least SAH-sensitive ray.
//
// Two passes, both coherence-safe by construction:
//
// 1. Topology (Karras 2012 radix tree, reference/hip-bvh-construction TwoPassLbvh
//    `BvhBuild`). One thread per internal node: `determineRange` + `findSplit` derive
//    the node's sorted-key range and split point directly from the sorted Morton keys,
//    independently — no atomics, no climb, no cross-workgroup reads. Each thread writes
//    only its own node's two child pointers. The reference lays internal nodes at
//    [0, N−1) and leaves by sorted position; we keep the engine's node layout instead
//    (leaves at slots [0, N), leaf `j` bounds prim `j`; internal at [N, 2N−1); root the
//    last node 2N−2) by remapping Karras internal index `i` → slot `2N−2−i` and a leaf
//    at sorted position `s` → slot `payload[s]` (the prim index). So the output is the
//    same 32 B node format the traverser (traverse.ts) and the oracle expect.
//
// 2. Bounds (bottom-up, dispatch-boundary-ordered). A node's bounds = the union of its
//    children's. The unsound way (H-PLOC's build + the old refit) is a single-dispatch
//    atomic climb where a lane reads a sibling subtree's bounds written by another
//    workgroup, ordered only by an atomic flag — not visible without a device fence
//    (gpu.md "Cross-workgroup ordering"). Instead we relax: each sweep is its own
//    dispatch, a node completes only when BOTH children completed in a PRIOR sweep, so
//    every cross-node read crosses a dispatch boundary (spec-clean visibility) and a
//    node's bounds are written once, with no concurrent reader. `valid` flags are
//    double-buffered (read prior-sweep state, write this-sweep state) so a flag a peer
//    flips mid-sweep is never observed with its bounds still in flight. Each sweep climbs
//    LEVELS levels at once (a node resolves a child from deeper prior-sweep descendants when
//    the child isn't valid yet), so the fixed sweep count is ceil(worst-case height / LEVELS); sweeps past
//    convergence carry, leaving the result stable. This pass IS the refit: refit re-runs
//    leaf-init (moved prim bounds) + the relaxation, no rebuild.
//
// Node = 32B (8 u32: min.xyz, leftChild, max.xyz, rightChild), plain storage — every
// cross-node read crosses a dispatch boundary, so no atomics on the node buffer (unlike
// H-PLOC, which needed them for the within-kernel climb). Leaves: leftChild = INVALID,
// rightChild = prim index. Child pointers stay local (0-based within the BLAS); the
// traverser adds the slot base, so concatenated in-place BLASes need no rebase.
// `NODE_BASE` (count[1]) offsets only the physical read/write address for the in-place
// caster build (gpu.md binding rule 3 — folded into the count buffer's header).
//
// Validated on the real GPU by the `accel` gym scenario's build gate: per fixture, readback
// the nodes and check the oracle's invariants + SAH within tolerance + ray-vs-brute-
// force agreement; `checkForestProbe` rebuilds the live caster path pipelined over
// frozen geometry and asserts ray-invariance across reps (the coherence gate H-PLOC
// failed). Exact topology is not the gate — the oracle build is PLOC, so only the
// structural + ray + SAH-ratio checks are compared.

import tgpu, { type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { precompile, precompileScope } from "../../engine/runtime";
import { bitcastF32toU32, idiv } from "../../engine/utils/core";

const WG_INIT = 256; // leaf-init: embarrassingly parallel, one thread per node
const WG_TOPO = 128; // topology: one thread per internal node, each a small binary search
const WG_BOUNDS = 256; // relaxation sweep: one thread per node
const MAX_DISPATCH = 65535; // maxComputeWorkgroupsPerDimension floor
const NODE_U32 = 8; // u32 per 32B node
// Absolute cap on the relaxation sweep count. The Karras radix tree splits on the highest
// differing bit of the (30-bit Morton << 32 | sortedPos) key, so a root→leaf path is a
// strictly-increasing prefix-length sequence in [0, 61] → height ≤ 62. The per-builder
// `boundsSteps` derives a tighter bound from `maxPrims` (30 + ceil(log2 N)); this is the
// ceiling for any N a 30-bit-Morton build allows. 64-bit Morton (future) would raise it.
const MAX_BOUNDS_STEPS = 64;
// Bottom-up levels each relaxation sweep climbs. `resolveChain` generates the resolve0..resolve(LEVELS-1)
// chain from it; `sweepCount` derives the sweep total as ceil(height / LEVELS). 1 = the plain per-level fit (a node completes
// when both children completed a prior sweep); each extra level lets a node resolve a child from one
// level deeper of prior-sweep descendants, so the sweep (dispatch) count drops as 1/LEVELS at the cost
// of more frontier reads per sweep. Coherence-safe at any value — every bounds read still gates on a
// prior-dispatch valid flag, and a resolved child is always its true final bounds (bit-identical fit).
// Measured on Lovelace.
const LEVELS = 3;

// Node accessors over the plain nodes buffer (8 u32/node: min.xyz, leftChild, max.xyz,
// rightChild). No atomics: every cross-node read in the bounds relaxation crosses a
// dispatch boundary, and each node's words are written by exactly one thread, so plain
// storage is coherent here (the H-PLOC build needed atomics only for its within-kernel
// climb). `writeBounds` touches the 6 bound words only — child pointers are topology and
// stay put through the relaxation + a refit. NODE_BASE (count[1]) shifts the physical
// address into a sub-region of a larger shared buffer (the in-place caster concatenation);
// child pointers stored in a node stay local, so base 0 (standalone) is byte-identical.
const INVALID = 0xffffffff; // leaf sentinel in leftChild; also "no node"

/**
 * the node buffer + the control buffer — the prefix leaf-init, topology and the relaxation all share.
 * One layout for the shared prefix, so the node accessors below resolve against a single declaration and
 * the three passes cannot drift; each pass binds this group plus its own I/O.
 * @internal
 */
export const nodeLayout = tgpu.bindGroupLayout({
    nodes: { storage: d.arrayOf(d.u32), access: "mutable" },
    countBuf: { storage: d.arrayOf(d.u32), access: "readonly" }, // [0] = prim count, [1] = node-write base
});

// NODE_BASE (count[1]) shifts the physical address into a sub-region of a larger shared buffer (the
// in-place caster concatenation); child pointers stored in a node stay local, so base 0 (standalone) is
// byte-identical. Read from the control buffer once per thread rather than on every accessor call.
const nodeBase = tgpu.privateVar(d.u32, 0);

const wordOf = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((n) => {
        "use gpu";
        return (nodeBase.$ + n) * NODE_U32;
    })
    .$name("nodeWord");

const nodeMin = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )((n) => {
        "use gpu";
        const o = wordOf(n);
        return d.vec3f(
            std.bitcastU32toF32(nodeLayout.$.nodes[o]),
            std.bitcastU32toF32(nodeLayout.$.nodes[o + 1]),
            std.bitcastU32toF32(nodeLayout.$.nodes[o + 2]),
        );
    })
    .$name("nodeMin");

const nodeMax = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )((n) => {
        "use gpu";
        const o = wordOf(n);
        return d.vec3f(
            std.bitcastU32toF32(nodeLayout.$.nodes[o + 4]),
            std.bitcastU32toF32(nodeLayout.$.nodes[o + 5]),
            std.bitcastU32toF32(nodeLayout.$.nodes[o + 6]),
        );
    })
    .$name("nodeMax");

const nodeLeft = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((n) => {
        "use gpu";
        return nodeLayout.$.nodes[wordOf(n) + 3];
    })
    .$name("nodeLeft");

const nodeRight = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((n) => {
        "use gpu";
        return nodeLayout.$.nodes[wordOf(n) + 7];
    })
    .$name("nodeRight");

const writeLeaf = tgpu
    .fn([d.u32, d.vec3f, d.vec3f])((j, mn, mx) => {
        "use gpu";
        const o = wordOf(j);
        nodeLayout.$.nodes[o] = bitcastF32toU32(mn.x);
        nodeLayout.$.nodes[o + 1] = bitcastF32toU32(mn.y);
        nodeLayout.$.nodes[o + 2] = bitcastF32toU32(mn.z);
        nodeLayout.$.nodes[o + 3] = INVALID;
        nodeLayout.$.nodes[o + 4] = bitcastF32toU32(mx.x);
        nodeLayout.$.nodes[o + 5] = bitcastF32toU32(mx.y);
        nodeLayout.$.nodes[o + 6] = bitcastF32toU32(mx.z);
        nodeLayout.$.nodes[o + 7] = j;
    })
    .$name("writeLeaf");

const writeChildren = tgpu
    .fn([d.u32, d.u32, d.u32])((n, left, right) => {
        "use gpu";
        const o = wordOf(n);
        nodeLayout.$.nodes[o + 3] = left;
        nodeLayout.$.nodes[o + 7] = right;
    })
    .$name("writeChildren");

const writeBounds = tgpu
    .fn([d.u32, d.vec3f, d.vec3f])((n, mn, mx) => {
        "use gpu";
        const o = wordOf(n);
        nodeLayout.$.nodes[o] = bitcastF32toU32(mn.x);
        nodeLayout.$.nodes[o + 1] = bitcastF32toU32(mn.y);
        nodeLayout.$.nodes[o + 2] = bitcastF32toU32(mn.z);
        nodeLayout.$.nodes[o + 4] = bitcastF32toU32(mx.x);
        nodeLayout.$.nodes[o + 5] = bitcastF32toU32(mx.y);
        nodeLayout.$.nodes[o + 6] = bitcastF32toU32(mx.z);
    })
    .$name("writeBounds");

// Prepare the build's indirect dispatch args from the GPU count: one workgroup writes
// [ceil((2N−1)/WG_INIT)] for leaf-init (over every node) and [ceil((N−1)/WG_TOPO)] for the
// topology pass (over internal nodes), so both scale with the live count, not the cap. A
// separate dispatch from the indirect passes (a buffer can't be a storage-write target and
// an indirect source at once), so the writes are visible across the boundary. The relaxation
// sweeps dispatch direct (a fixed cap-sized count), so they need no indirect arg here.
/** the prepare pass's own I/O — it derives dispatch sizes, so it never touches the node buffer.
 *  @internal */
export const prepareLayout = tgpu.bindGroupLayout({
    countBuf: { storage: d.arrayOf(d.u32), access: "readonly" },
    indirect: { storage: d.arrayOf(d.u32), access: "mutable" },
});

const prepareKernel = tgpu
    .computeFn({ workgroupSize: [1] })(() => {
        "use gpu";
        const n = prepareLayout.$.countBuf[0];
        const total = 2 * n - 1;
        const nInternal = std.select(d.u32(0), n - 1, n > 1);
        prepareLayout.$.indirect[0] = idiv(total + (WG_INIT - 1), WG_INIT); // leaf-init
        prepareLayout.$.indirect[1] = 1;
        prepareLayout.$.indirect[2] = 1;
        prepareLayout.$.indirect[3] = idiv(nInternal + (WG_TOPO - 1), WG_TOPO); // topology
        prepareLayout.$.indirect[4] = 1;
        prepareLayout.$.indirect[5] = 1;
    })
    .$name("buildPrepare");

// Leaf-init: write each leaf node (slots [0, N)) from its prim AABB (leftChild = INVALID,
// rightChild = the prim index), seed valid for both flag buffers. Shared by build and refit —
// refit reuses it to rewrite leaf bounds from moved prims and reset valid without touching the
// topology. A leaf carries valid = 1 in BOTH buffers because the relaxation sweeps internal nodes
// only, and an internal node reads its leaf children's flag from whichever buffer is that sweep's
// input — both must hold 1. Internal nodes seed 0: validA is the first sweep's input (must read
// clear); validB's internal seed is don't-care (every sweep overwrites the internal validOut) but
// cleared for symmetry.
/** leaf-init's own I/O: the prim AABBs it bounds each leaf from, and both ping-pong flag buffers.
 *  @internal */
export const leafLayout = tgpu.bindGroupLayout({
    prims: { storage: d.arrayOf(d.vec4f), access: "readonly" }, // 2 vec4/prim: min.xyz+pad, max.xyz+pad
    validA: { storage: d.arrayOf(d.u32), access: "mutable" }, // sweep ping-pong A
    validB: { storage: d.arrayOf(d.u32), access: "mutable" }, // sweep ping-pong B; leaves stay 1 here too
});

const leafKernel = tgpu
    .computeFn({ workgroupSize: [WG_INIT], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        nodeBase.$ = nodeLayout.$.countBuf[1];
        const primCount = nodeLayout.$.countBuf[0];
        const total = 2 * primCount - 1;
        const j = input.gid.x;
        if (j >= total) return;
        if (j < primCount) {
            writeLeaf(j, leafLayout.$.prims[j * 2].xyz, leafLayout.$.prims[j * 2 + 1].xyz);
            leafLayout.$.validA[j] = 1;
            leafLayout.$.validB[j] = 1;
        } else {
            leafLayout.$.validA[j] = 0;
            leafLayout.$.validB[j] = 0;
        }
    })
    .$name("buildLeaf");

// Karras radix-tree topology (reference/hip-bvh-construction TwoPassLbvh `BvhBuild`). One
// thread per internal node `i`: `determineRange` finds its sorted-key range by the
// longest-common-prefix direction + binary search, `findSplit` the split position, and the
// two children are the sub-ranges' nodes. Reads only the immutable sorted keys + payload —
// no atomics, no climb. The `delta` helper is the common-prefix length of the extended keys
// (Morton << 32 | sortedPos), emulated without u64: equal codes fall back to the index bits.
// Remap to the engine layout: internal `i` → slot 2N−2−i (root i=0 → 2N−2); a leaf child at
// sorted position `s` → slot payload[s] (= prim index, the slot leaf-init filled).
/** topology's own inputs: the sorted keys it derives every range from, and the payload that maps a
 *  sorted position back to its prim slot. Both immutable through the pass. @internal */
export const topoLayout = tgpu.bindGroupLayout({
    keys: { storage: d.arrayOf(d.u32), access: "readonly" }, // sorted Morton codes, by sorted position
    payload: { storage: d.arrayOf(d.u32), access: "readonly" }, // sorted prim index, by sorted position
});

// WGSL has `firstLeadingBit`, not a count-leading-zeros; the x == 0 arm is what the subtraction can't
// express (firstLeadingBit(0) is 0xffffffff).
const clz32 = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((x) => {
        "use gpu";
        return std.select(31 - std.firstLeadingBit(x), d.u32(32), x === 0);
    })
    .$name("clz32");

// common-prefix length of the extended key (Morton << 32 | sortedPos) at sorted positions a
// (code ca, hoisted by the caller — it stays fixed across a search) and b, or -1 when b is
// outside [0, n). Equal Morton codes tie-break on the position bits (the << 32 emulation).
// Signed throughout: -1 is the out-of-range answer the two searches terminate on, so this is the one
// place in the builder where i32 is the correct type rather than a stray literal.
const delta = tgpu
    .fn(
        [d.u32, d.i32, d.i32, d.i32],
        d.i32,
    )((ca, a, b, n) => {
        "use gpu";
        if (b < 0 || b >= n) return -1;
        const cb = topoLayout.$.keys[d.u32(b)];
        if (ca === cb) return 32 + d.i32(clz32(d.u32(a) ^ d.u32(b)));
        return d.i32(clz32(ca ^ cb));
    })
    .$name("delta");

// The two searches use genuine dynamic loops, not for-loops capped at a small constant: a
// constant bound is a DXC unroll target, and an unrolled binary search issues every
// iteration's scattered keys[] load + bloats code and registers regardless of the early
// break (gpu.md "verify the unroll happened"). The dynamic loop runs only the ~log2(range)
// real iterations. Both provably terminate — lMax doubles only while in range (delta = -1
// past the ends stops it), and t halves to 0 — so no hang-guard counter is needed (adding
// one risks re-enabling the unroll).
const determineRange = tgpu
    .fn(
        [d.u32, d.i32],
        d.vec2u,
    )((idx, n) => {
        "use gpu";
        if (idx === 0) return d.vec2u(0, d.u32(n - 1));
        const i = d.i32(idx);
        const ci = topoLayout.$.keys[idx]; // the node's own code, fixed across every delta below
        const dL = delta(ci, i, i - 1, n);
        const dR = delta(ci, i, i + 1, n);
        const dir = std.select(-1, 1, dR > dL);
        const deltaMin = std.min(dL, dR);
        let lMax = d.i32(2);
        while (delta(ci, i, i + dir * lMax, n) > deltaMin) {
            lMax = lMax << 1;
        }
        let l = d.i32(0);
        let t = lMax >> 1;
        while (t > 0) {
            if (delta(ci, i, i + (l + t) * dir, n) > deltaMin) l = l + t;
            t = t >> 1;
        }
        const jdx = i + l * dir;
        if (dir < 0) return d.vec2u(d.u32(jdx), idx);
        return d.vec2u(idx, d.u32(jdx));
    })
    .$name("determineRange");

// the split is the last position whose prefix with `first` is longer than the node's own — a binary
// search that must run its body once before the stride test, so the exit condition rides a flag rather
// than the `while` head (WGSL's `loop` has no TGSL form).
const findSplit = tgpu
    .fn(
        [d.u32, d.u32, d.i32],
        d.u32,
    )((first, last, n) => {
        "use gpu";
        const firstCode = topoLayout.$.keys[first]; // fixed across the search
        const deltaNode = delta(firstCode, d.i32(first), d.i32(last), n);
        let split = d.i32(first);
        let stride = d.i32(last) - d.i32(first);
        let more = true;
        while (more) {
            stride = (stride + 1) >> 1;
            const middle = split + stride;
            if (middle < d.i32(last) && delta(firstCode, d.i32(first), middle, n) > deltaNode)
                split = middle;
            more = stride > 1;
        }
        return d.u32(split);
    })
    .$name("findSplit");

const topoKernel = tgpu
    .computeFn({ workgroupSize: [WG_TOPO], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        nodeBase.$ = nodeLayout.$.countBuf[1];
        const primCount = nodeLayout.$.countBuf[0];
        const i = input.gid.x;
        if (primCount <= 1 || i >= primCount - 1) return;
        const n = d.i32(primCount);
        const last = 2 * primCount - 2; // root slot; internal i maps to last - i
        const range = determineRange(i, n);
        const split = findSplit(range.x, range.y, n);
        const leftSlot = std.select(last - split, topoLayout.$.payload[split], split === range.x);
        const rightSlot = std.select(
            last - (split + 1),
            topoLayout.$.payload[split + 1],
            split + 1 === range.y,
        );
        writeChildren(last - i, leftSlot, rightSlot);
    })
    .$name("buildTopo");

// The relaxation's bottom-up resolver, generated for LEVELS (WGSL has no recursion, so the chain
// unrolls into fixed functions). resolveK(c) returns c's final bounds when c's subtree completed
// within K levels of a prior sweep: validIn[c] (c itself done), else the union of resolve(K-1) over
// c's two children. A sweep thread resolves each of its node's two children to depth LEVELS-1, so the
// node climbs LEVELS levels per dispatch. The coherence + bit-identity argument is above `sweepKernel`.
/** the relaxation sweep's own I/O: the double-buffered completion flags. `validIn` is the prior sweep's
 *  state (a dispatch boundary back, so visible and stable), `validOut` this sweep's. @internal */
export const sweepLayout = tgpu.bindGroupLayout({
    validIn: { storage: d.arrayOf(d.u32), access: "readonly" },
    validOut: { storage: d.arrayOf(d.u32), access: "mutable" },
});

/** a child's final bounds resolved from prior-sweep state, looking up to LEVELS-1 levels past it.
 *  `ok` is false when the child's subtree hasn't reached this thread yet (a later sweep completes it). */
const Resolved = d.struct({ ok: d.bool, mn: d.vec3f, mx: d.vec3f }).$name("Resolved");

const resolve0 = tgpu
    .fn(
        [d.u32],
        Resolved,
    )((c) => {
        "use gpu";
        if (sweepLayout.$.validIn[c] === 1)
            return Resolved({ ok: true, mn: nodeMin(c), mx: nodeMax(c) });
        return Resolved({ ok: false, mn: d.vec3f(), mx: d.vec3f() });
    })
    .$name("resolve0");

// one authored body per level, re-emitted per captured `deeper` — WGSL has no recursion, so the chain
// is what unrolls it into LEVELS fixed functions.
function deeperResolve(k: number, deeper: typeof resolve0): typeof resolve0 {
    return tgpu
        .fn(
            [d.u32],
            Resolved,
        )((c) => {
            "use gpu";
            if (sweepLayout.$.validIn[c] === 1)
                return Resolved({ ok: true, mn: nodeMin(c), mx: nodeMax(c) });
            const l = deeper(nodeLeft(c));
            const r = deeper(nodeRight(c));
            if (l.ok && r.ok)
                return Resolved({
                    ok: true,
                    mn: std.min(l.mn, r.mn),
                    mx: std.max(l.mx, r.mx),
                });
            return Resolved({ ok: false, mn: d.vec3f(), mx: d.vec3f() });
        })
        .$name(`resolve${k}`);
}

function resolveChain(levels: number): typeof resolve0 {
    let chain = resolve0;
    for (let k = 1; k < levels; k++) chain = deeperResolve(k, chain);
    return chain;
}

const resolveTop = resolveChain(LEVELS);

// Relaxation sweep: one INTERNAL node per thread, LEVELS bottom-up levels per sweep. The
// dispatch covers internal slots [N, 2N−1) only (node index `primCount + gid.x`); leaves are
// never a sweep thread, since leaf-init seeds their flag = 1 in both buffers and their bounds
// once, so a leaf child reads valid = 1 from whichever buffer is the input. That double seed is
// what makes skipping the N leaves sound. Reads `validIn` (the prior sweep's completion state — a
// dispatch boundary back, so visible and stable), writes `validOut`.
//
// Multi-level climb (the dispatch-count lever). The 1-level
// form (LEVELS=1) completes a node only when BOTH its children were valid a sweep ago, so the root
// needs one dispatch per tree level. The generated resolve0..resolve(LEVELS-1) chain (above) instead
// resolves a child from its own prior-sweep bounds OR, when it isn't valid yet, from progressively
// deeper prior-sweep descendants (down to LEVELS levels below the node), so a node climbs LEVELS
// levels per dispatch and the worst-case sweep count drops by LEVELS×. The coherence-safety
// argument is unchanged from the 1-level form, by the same two invariants:
//   (1) each node's bounds are written by exactly one thread (its own), the sweep it completes; the
//       resolver only READS, it never writes a descendant, so there is still no second writer;
//   (2) a descendant's bounds are read ONLY when that descendant carries validIn = 1, i.e. it
//       completed in a PRIOR dispatch, so its bounds were written >= 1 dispatch ago with no in-flight
//       writer. The intermediate nodeLeft/nodeRight reads down the chain touch immutable topology,
//       safe at any time. A resolved child always yields that child's *true* final bounds (union of
//       its children = what the child's own thread writes), so the finished tree is bit-identical to
//       the 1-level fit at any LEVELS.
//
// validIn/validOut ping-pong between the two flag buffers each sweep (build/refit alternate the
// bind group). The caller dispatches a fixed cap-sized count of these (a CPU constant — direct
// dispatch, no per-sweep prepare/indirect), enough to exceed ceil(worst-case height / LEVELS);
// sweeps past convergence just carry, so the result is stable. Threads past the live internal
// range early-out. A non-valid child is internal (leaves carry validIn = 1), so its child
// pointers are real topology, safe to read.
const sweepKernel = tgpu
    .computeFn({ workgroupSize: [WG_BOUNDS], in: { gid: d.builtin.globalInvocationId } })(
        (input) => {
            "use gpu";
            nodeBase.$ = nodeLayout.$.countBuf[1];
            const primCount = nodeLayout.$.countBuf[0];
            if (primCount <= 1) return; // single prim: the leaf is the root, no internal nodes
            const n = primCount + input.gid.x; // internal slot [N, 2N−1)
            if (n > 2 * primCount - 2) return; // past the root
            if (sweepLayout.$.validIn[n] === 1) {
                sweepLayout.$.validOut[n] = 1; // already done, carry
                return;
            }
            const lc = resolveTop(nodeLeft(n));
            if (!lc.ok) {
                sweepLayout.$.validOut[n] = 0; // not ready yet; a later sweep completes it
                return;
            }
            const rc = resolveTop(nodeRight(n));
            if (!rc.ok) {
                sweepLayout.$.validOut[n] = 0;
                return;
            }
            writeBounds(n, std.min(lc.mn, rc.mn), std.max(lc.mx, rc.mx));
            sweepLayout.$.validOut[n] = 1;
        },
    )
    .$name("buildSweep");

/** the emitted build WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function buildWgsl(): {
    prepare: string;
    leaf: string;
    topo: string;
    sweep: string;
} {
    const names = { names: "strict" } as const;
    return {
        prepare: tgpu.resolve([prepareKernel], names),
        leaf: tgpu.resolve([leafKernel], names),
        topo: tgpu.resolve([topoKernel], names),
        sweep: tgpu.resolve([sweepKernel], names),
    };
}

/**
 * how many relaxation sweeps a builder sized for `maxPrims` runs. The worst-case radix-tree height is
 * 30 Morton bits plus the index tiebreak's `ceil(log2 N)` (the all-equal-code case), and each sweep
 * climbs {@link LEVELS} levels, so this is `ceil(height / LEVELS)` — enough that the deepest node's
 * bounds reach the root. Sweeps past convergence just carry, so the result is stable; a fixed count is
 * what avoids the per-sweep prepare + indirect early-out (its pass count, ~2× the sweep count, was the
 * build's dominant cost).
 *
 * Under-counting is the silent failure mode — internal nodes keep unfinished bounds and the traverser
 * returns wrong hits, with nothing thrown and no fixture deep enough to notice — so the invariant
 * `sweepCount(n) * LEVELS >= 30 + ceil(log2 n)` is unit-tested rather than read.
 * @internal
 */
export function sweepCount(maxPrims: number): number {
    const cap = Math.max(1, maxPrims);
    const height = Math.min(MAX_BOUNDS_STEPS, 30 + Math.ceil(Math.log2(Math.max(2, cap))));
    return Math.ceil(height / LEVELS);
}

/** buffers {@link createBvh} threads through so the stages share one set; an omitted field is allocated internally */
export interface BuildShared {
    /** input prim AABB buffer — 2 vec4<f32> per prim, leaf-index order */
    prims?: GPUBuffer;
    /** sorted Morton codes (the sort's key output) */
    keys?: GPUBuffer;
    /** sorted prim indices (the sort's payload output) */
    payload?: GPUBuffer;
    /** BVH2 node output buffer — pass a larger shared buffer + a non-zero `count[1]` to build in place */
    nodes?: GPUBuffer;
    /** control buffer: [0] = prim count, [1] = node-write base (0 = standalone). The build reads its `2N−1` range from [0] */
    count?: GPUBuffer;
}

/**
 * a built LBVH BVH2 builder sized for `maxPrims`. Write primitive AABBs into {@link
 * Build.prims} (2 × vec4<f32> per prim, leaf-index order) and the sorted Morton output
 * into {@link Build.keys} (codes) + {@link Build.payload} (prim indices, the {@link
 * RadixSort} result), record {@link Build.build}, submit, then read the BVH2 back from
 * {@link Build.nodes} (`2N−1` nodes × 32 B; root is node `2N−2` for N≥2, else node 0).
 * For stable topology under motion, write moved AABBs and record {@link Build.refit}
 * instead: the bounds relaxation alone, topology untouched.
 */
export interface Build {
    /** input prim AABB buffer — fill [0, count) prims, 2 vec4 each (leaf-index order) */
    readonly prims: GPUBuffer;
    /** input sorted Morton codes — one u32 per prim, sorted-position order */
    readonly keys: GPUBuffer;
    /** input sorted prim indices — the sort payload, sorted-position order */
    readonly payload: GPUBuffer;
    /** output BVH2 nodes — `2N−1` nodes, 8 u32 (32 B) each */
    readonly nodes: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]) — write it (≤ `maxPrims`) before recording */
    readonly count: GPUBuffer;
    /** dispatch args the prepare pass derives from {@link Build.count}: leaf-init [0..3), topology [3..6) — for diagnostics */
    readonly indirect: GPUBuffer;
    /** capacity the buffers are sized for */
    readonly maxPrims: number;
    /** record the full build (prepare + leaf-init + topology + bounds relaxation), count read from {@link Build.count} */
    build(encoder: GPUCommandEncoder): void;
    /** record a bounds-only refit over the existing topology (leaf-init + relaxation), count read from {@link Build.count} */
    refit(encoder: GPUCommandEncoder): void;
    destroy(): void;
}

/**
 * build an LBVH BVH2 builder for up to `maxPrims` primitives. Compiles the prepare +
 * leaf-init + topology + relaxation kernels and allocates the node / flag / working
 * buffers up front; {@link Build.build} and {@link Build.refit} then record with no
 * further allocation, dispatching every pass indirectly off the GPU count.
 *
 * @example
 * const b = await createBuild(device, 1 << 20);
 * device.queue.writeBuffer(b.prims, 0, primAabbs);
 * device.queue.writeBuffer(b.keys, 0, sortedCodes);    // from createRadixSort
 * device.queue.writeBuffer(b.payload, 0, sortedPrimIds);
 * device.queue.writeBuffer(b.count, 0, new Uint32Array([count]));
 * const enc = device.createCommandEncoder();
 * b.build(enc);
 * device.queue.submit([enc.finish()]);
 */
export async function createBuild(
    device: GPUDevice,
    maxPrims: number,
    shared: BuildShared = {},
): Promise<Build> {
    const root = Compute.root;
    const cap = Math.max(1, maxPrims);
    const nodeCount = 2 * cap; // 2N−1 rounded up; the extra node is never addressed
    // every pass is one thread per node/internal-node (no grid-stride), so the worst-case
    // dispatch must fit the per-dimension limit — checked here since count lives on the GPU
    if (Math.ceil((2 * cap) / WG_BOUNDS) > MAX_DISPATCH)
        throw new Error(
            `createBuild: maxPrims ${maxPrims} exceeds the ${MAX_DISPATCH}-workgroup limit`,
        );

    const owned: GPUBuffer[] = [];
    const storage = (label: string, bytes: number, extra = 0): GPUBuffer => {
        const b = device.createBuffer({
            label,
            size: Math.max(4, bytes),
            usage: GPUBufferUsage.STORAGE | extra,
        });
        owned.push(b);
        return b;
    };

    const prims = shared.prims ?? storage("build-prims", cap * 32, GPUBufferUsage.COPY_DST);
    const keys = shared.keys ?? storage("build-keys", cap * 4, GPUBufferUsage.COPY_DST);
    const payload = shared.payload ?? storage("build-payload", cap * 4, GPUBufferUsage.COPY_DST);
    const nodes =
        shared.nodes ?? storage("build-nodes", nodeCount * NODE_U32 * 4, GPUBufferUsage.COPY_SRC);
    // control buffer: [0] = prim count, [1] = node-write base. 8 B so the build-in-place caster
    // can fold the per-mesh base into [1] without a binding (gpu.md rule 3); standalone leaves
    // [1] zero-init (base 0). Every pass reads [0] for its node range + dispatch sizing.
    const count = shared.count ?? storage("build-count", 8, GPUBufferUsage.COPY_DST);
    // double-buffered completion flags for the bounds relaxation (one u32 per node, local index)
    const validA = storage("build-valid-a", nodeCount * 4);
    const validB = storage("build-valid-b", nodeCount * 4);
    // 2 dispatch-arg triples (leaf-init, topology); INDIRECT to dispatch, COPY_SRC so a gate can read the sizes
    const indirect = storage(
        "build-indirect",
        2 * 3 * 4,
        GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
    );

    const boundsSteps = sweepCount(cap);
    // each sweep covers the internal nodes only ([N, 2N−1) ≈ half the tree); leaves never need a
    // thread (leaf-init seeds their flag in both buffers), so the dispatch is sized to cap−1, not
    // 2·cap−1. max(1, …) keeps a ≥1 dispatch for cap = 1 (no internal nodes; threads all early-out).
    const sweepWG = Math.max(1, Math.ceil((cap - 1) / WG_BOUNDS));

    // one shared node group every pass binds, plus each pass's own I/O — the node accessors resolve
    // against `nodeLayout`, so the three kernels reference the same declaration
    const nodeGroup = root.createBindGroup(nodeLayout, { nodes, countBuf: count });
    const prepare = root
        .createComputePipeline({ compute: prepareKernel })
        .$name("build-prepare")
        .with(root.createBindGroup(prepareLayout, { countBuf: count, indirect }));
    const leaf = root
        .createComputePipeline({ compute: leafKernel })
        .$name("build-leaf")
        .with(nodeGroup)
        .with(root.createBindGroup(leafLayout, { prims, validA, validB }));
    const topo = root
        .createComputePipeline({ compute: topoKernel })
        .$name("build-topo")
        .with(nodeGroup)
        .with(root.createBindGroup(topoLayout, { keys, payload }));
    // two sweep bind groups ping-pong the double-buffered flags: even sweeps read A write B,
    // odd sweeps read B write A. The first sweep reads validA (leaf-init's output).
    const sweepPipe = root.createComputePipeline({ compute: sweepKernel }).$name("build-sweep");
    const sweepBound = (vin: GPUBuffer, vout: GPUBuffer) =>
        sweepPipe
            .with(nodeGroup)
            .with(root.createBindGroup(sweepLayout, { validIn: vin, validOut: vout }));
    const sweepAB = sweepBound(validA, validB);
    const sweepBA = sweepBound(validB, validA);

    // per-instance labels — an app can build more than one BVH, and the queue rejects a duplicate label
    const scope = precompileScope("build");
    for (const [label, bound] of [
        ["prepare", prepare],
        ["leaf", leaf],
        ["topo", topo],
        ["sweep", sweepAB],
    ] as const) {
        await precompile(`${scope}-${label}`, () => {
            return bound;
        });
    }

    const dispatch = (
        encoder: GPUCommandEncoder,
        bound: TgpuComputePipeline,
        wg: number,
        span: string,
    ): void => {
        const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.(span) });
        bound.with(pass).dispatchWorkgroups(wg);
        pass.end();
    };
    const indirectPass = (
        encoder: GPUCommandEncoder,
        bound: TgpuComputePipeline,
        offset: number,
        span: string,
    ): void => {
        const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.(span) });
        bound.with(pass).dispatchWorkgroupsIndirect(indirect, offset);
        pass.end();
    };

    // the bounds relaxation, shared by build + refit: a fixed `boundsSteps` direct sweeps,
    // ping-ponging the double-buffered valid flags. Enough to exceed the worst-case height, so
    // the deepest node's bounds reach the root; later sweeps carry, leaving the result stable.
    const relax = (encoder: GPUCommandEncoder, span: string): void => {
        for (let k = 0; k < boundsSteps; k++) {
            dispatch(encoder, k % 2 === 0 ? sweepAB : sweepBA, sweepWG, span);
        }
    };

    return {
        prims,
        keys,
        payload,
        nodes,
        count,
        indirect,
        maxPrims,
        build(encoder: GPUCommandEncoder): void {
            dispatch(encoder, prepare, 1, "bvh:build");
            indirectPass(encoder, leaf, 0, "bvh:build");
            indirectPass(encoder, topo, 12, "bvh:build");
            relax(encoder, "bvh:build");
        },
        refit(encoder: GPUCommandEncoder): void {
            // topology persists in `nodes`; re-bound the leaves from the moved prims (and reset
            // the flags), then re-run the relaxation over the fixed topology
            dispatch(encoder, prepare, 1, "bvh:refit");
            indirectPass(encoder, leaf, 0, "bvh:refit");
            relax(encoder, "bvh:refit");
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
