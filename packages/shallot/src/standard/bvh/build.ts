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

import { Compute } from "../../engine";

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
// Bottom-up levels each relaxation sweep climbs. SWEEP_WGSL generates the resolve0..resolve(LEVELS-1)
// chain from it; boundsSteps = ceil(height / LEVELS). 1 = the plain per-level fit (a node completes
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
const NODE_WGSL = /* wgsl */ `
const INVALID = 0xffffffffu;        // leaf sentinel in leftChild; also "no node"
var<private> NODE_BASE: u32 = 0u;
fn nodeMin(n: u32) -> vec3<f32> {
    let o = (NODE_BASE + n) * ${NODE_U32}u;
    return vec3<f32>(bitcast<f32>(nodes[o]), bitcast<f32>(nodes[o + 1u]), bitcast<f32>(nodes[o + 2u]));
}
fn nodeMax(n: u32) -> vec3<f32> {
    let o = (NODE_BASE + n) * ${NODE_U32}u;
    return vec3<f32>(bitcast<f32>(nodes[o + 4u]), bitcast<f32>(nodes[o + 5u]), bitcast<f32>(nodes[o + 6u]));
}
fn nodeLeft(n: u32) -> u32 { return nodes[(NODE_BASE + n) * ${NODE_U32}u + 3u]; }
fn nodeRight(n: u32) -> u32 { return nodes[(NODE_BASE + n) * ${NODE_U32}u + 7u]; }
fn writeLeaf(j: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let o = (NODE_BASE + j) * ${NODE_U32}u;
    nodes[o] = bitcast<u32>(mn.x); nodes[o + 1u] = bitcast<u32>(mn.y); nodes[o + 2u] = bitcast<u32>(mn.z);
    nodes[o + 3u] = INVALID;
    nodes[o + 4u] = bitcast<u32>(mx.x); nodes[o + 5u] = bitcast<u32>(mx.y); nodes[o + 6u] = bitcast<u32>(mx.z);
    nodes[o + 7u] = j;
}
fn writeChildren(n: u32, left: u32, right: u32) {
    let o = (NODE_BASE + n) * ${NODE_U32}u;
    nodes[o + 3u] = left;
    nodes[o + 7u] = right;
}
fn writeBounds(n: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let o = (NODE_BASE + n) * ${NODE_U32}u;
    nodes[o] = bitcast<u32>(mn.x); nodes[o + 1u] = bitcast<u32>(mn.y); nodes[o + 2u] = bitcast<u32>(mn.z);
    nodes[o + 4u] = bitcast<u32>(mx.x); nodes[o + 5u] = bitcast<u32>(mx.y); nodes[o + 6u] = bitcast<u32>(mx.z);
}
`;

// Prepare the build's indirect dispatch args from the GPU count: one workgroup writes
// [ceil((2N−1)/WG_INIT)] for leaf-init (over every node) and [ceil((N−1)/WG_TOPO)] for the
// topology pass (over internal nodes), so both scale with the live count, not the cap. A
// separate dispatch from the indirect passes (a buffer can't be a storage-write target and
// an indirect source at once), so the writes are visible across the boundary. The relaxation
// sweeps dispatch direct (a fixed cap-sized count), so they need no indirect arg here.
const PREPARE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> countBuf: array<u32>;
@group(0) @binding(1) var<storage, read_write> indirect: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let n = countBuf[0];
    let total = 2u * n - 1u;
    let nInternal = select(0u, n - 1u, n > 1u);
    indirect[0] = (total + ${WG_INIT - 1}u) / ${WG_INIT}u; indirect[1] = 1u; indirect[2] = 1u;        // leaf-init
    indirect[3] = (nInternal + ${WG_TOPO - 1}u) / ${WG_TOPO}u; indirect[4] = 1u; indirect[5] = 1u;    // topology
}
`;

// Leaf-init: write each leaf node (slots [0, N)) from its prim AABB (leftChild = INVALID,
// rightChild = the prim index), seed valid for both flag buffers. Shared by build and refit —
// refit reuses it to rewrite leaf bounds from moved prims and reset valid without touching the
// topology. A leaf carries valid = 1 in BOTH buffers because the relaxation sweeps internal nodes
// only, and an internal node reads its leaf children's flag from whichever buffer is that sweep's
// input — both must hold 1. Internal nodes seed 0: validA is the first sweep's input (must read
// clear); validB's internal seed is don't-care (every sweep overwrites the internal validOut) but
// cleared for symmetry.
const LEAF_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> prims: array<vec4<f32>>;   // 2 vec4/prim: min.xyz+pad, max.xyz+pad
@group(0) @binding(1) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(2) var<storage, read_write> validA: array<u32>;  // sweep ping-pong A
@group(0) @binding(3) var<storage, read> countBuf: array<u32>;      // [0] = prim count, [1] = node-write base
@group(0) @binding(4) var<storage, read_write> validB: array<u32>;  // sweep ping-pong B; leaves stay 1 here too
${NODE_WGSL}
@compute @workgroup_size(${WG_INIT})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    NODE_BASE = countBuf[1];
    let primCount = countBuf[0];
    let total = 2u * primCount - 1u;
    let j = gid.x;
    if (j >= total) { return; }
    if (j < primCount) {
        writeLeaf(j, prims[j * 2u].xyz, prims[j * 2u + 1u].xyz);
        validA[j] = 1u;
        validB[j] = 1u;
    } else {
        validA[j] = 0u;
        validB[j] = 0u;
    }
}
`;

// Karras radix-tree topology (reference/hip-bvh-construction TwoPassLbvh `BvhBuild`). One
// thread per internal node `i`: `determineRange` finds its sorted-key range by the
// longest-common-prefix direction + binary search, `findSplit` the split position, and the
// two children are the sub-ranges' nodes. Reads only the immutable sorted keys + payload —
// no atomics, no climb. The `delta` helper is the common-prefix length of the extended keys
// (Morton << 32 | sortedPos), emulated without u64: equal codes fall back to the index bits.
// Remap to the engine layout: internal `i` → slot 2N−2−i (root i=0 → 2N−2); a leaf child at
// sorted position `s` → slot payload[s] (= prim index, the slot leaf-init filled).
const TOPO_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> keys: array<u32>;          // sorted Morton codes, by sorted position
@group(0) @binding(1) var<storage, read> payload: array<u32>;       // sorted prim index, by sorted position
@group(0) @binding(2) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(3) var<storage, read> countBuf: array<u32>;      // [0] = prim count, [1] = node-write base
${NODE_WGSL}

fn clz32(x: u32) -> u32 { return select(31u - firstLeadingBit(x), 32u, x == 0u); }

// common-prefix length of the extended key (Morton << 32 | sortedPos) at sorted positions a
// (code ca, hoisted by the caller — it stays fixed across a search) and b, or -1 when b is
// outside [0, n). Equal Morton codes tie-break on the position bits (the << 32 emulation).
fn delta(ca: u32, a: i32, b: i32, n: i32) -> i32 {
    if (b < 0 || b >= n) { return -1; }
    let cb = keys[u32(b)];
    if (ca == cb) { return 32 + i32(clz32(u32(a) ^ u32(b))); }
    return i32(clz32(ca ^ cb));
}

// The two searches use genuine dynamic loops, not for-loops capped at a small constant: a
// constant bound is a DXC unroll target, and an unrolled binary search issues every
// iteration's scattered keys[] load + bloats code and registers regardless of the early
// break (gpu.md "verify the unroll happened"). The dynamic loop runs only the ~log2(range)
// real iterations. Both provably terminate — lMax doubles only while in range (delta = -1
// past the ends stops it), and t halves to 0 — so no hang-guard counter is needed (adding
// one risks re-enabling the unroll).
fn determineRange(idx: u32, n: i32) -> vec2<u32> {
    if (idx == 0u) { return vec2<u32>(0u, u32(n - 1)); }
    let i = i32(idx);
    let ci = keys[idx];                      // the node's own code, fixed across every delta below
    let dL = delta(ci, i, i - 1, n);
    let dR = delta(ci, i, i + 1, n);
    let d = select(-1, 1, dR > dL);
    let deltaMin = min(dL, dR);
    var lMax = 2;
    loop {
        if (delta(ci, i, i + d * lMax, n) <= deltaMin) { break; }
        lMax = lMax << 1u;
    }
    var l = 0;
    var t = lMax >> 1u;
    loop {
        if (t <= 0) { break; }
        if (delta(ci, i, i + (l + t) * d, n) > deltaMin) { l += t; }
        t = t >> 1u;
    }
    let jdx = i + l * d;
    if (d < 0) { return vec2<u32>(u32(jdx), idx); }
    return vec2<u32>(idx, u32(jdx));
}

fn findSplit(first: u32, last: u32, n: i32) -> u32 {
    let firstCode = keys[first];             // fixed across the search
    let deltaNode = delta(firstCode, i32(first), i32(last), n);
    var split = i32(first);
    var stride = i32(last) - i32(first);
    loop {
        stride = (stride + 1) >> 1u;
        let middle = split + stride;
        if (middle < i32(last) && delta(firstCode, i32(first), middle, n) > deltaNode) { split = middle; }
        if (stride <= 1) { break; }
    }
    return u32(split);
}

@compute @workgroup_size(${WG_TOPO})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    NODE_BASE = countBuf[1];
    let primCount = countBuf[0];
    let i = gid.x;
    if (primCount <= 1u || i >= primCount - 1u) { return; }
    let n = i32(primCount);
    let last = 2u * primCount - 2u;            // root slot; internal i maps to last - i
    let range = determineRange(i, n);
    let split = findSplit(range.x, range.y, n);
    let leftSlot = select(last - split, payload[split], split == range.x);
    let rightSlot = select(last - (split + 1u), payload[split + 1u], split + 1u == range.y);
    writeChildren(last - i, leftSlot, rightSlot);
}
`;

// The relaxation's bottom-up resolver, generated for LEVELS (WGSL has no recursion, so the chain
// unrolls into fixed functions). resolveK(c) returns c's final bounds when c's subtree completed
// within K levels of a prior sweep: validIn[c] (c itself done), else the union of resolve(K-1) over
// c's two children. A sweep thread resolves each of its node's two children to depth LEVELS-1, so the
// node climbs LEVELS levels per dispatch. The coherence + bit-identity argument is in SWEEP_WGSL.
function resolveChain(levels: number): string {
    const fns: string[] = [];
    for (let k = 0; k < levels; k++) {
        const deeper =
            k === 0
                ? ""
                : `
    let l = resolve${k - 1}(nodeLeft(c));
    let r = resolve${k - 1}(nodeRight(c));
    if (l.ok && r.ok) { return Resolved(true, min(l.mn, r.mn), max(l.mx, r.mx)); }`;
        fns.push(`fn resolve${k}(c: u32) -> Resolved {
    if (validIn[c] == 1u) { return Resolved(true, nodeMin(c), nodeMax(c)); }${deeper}
    return Resolved(false, vec3<f32>(0.0), vec3<f32>(0.0));
}`);
    }
    return fns.join("\n");
}

// Relaxation sweep: one INTERNAL node per thread, ${LEVELS} bottom-up levels per sweep. The
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
// deeper prior-sweep descendants (down to LEVELS levels below the node), so a node climbs ${LEVELS}
// levels per dispatch and the worst-case sweep count drops by ${LEVELS}×. The coherence-safety
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
// dispatch, no per-sweep prepare/indirect), enough to exceed ceil(worst-case height / ${LEVELS});
// sweeps past convergence just carry, so the result is stable. Threads past the live internal
// range early-out. A non-valid child is internal (leaves carry validIn = 1), so its child
// pointers are real topology, safe to read.
const SWEEP_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> validIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> validOut: array<u32>;
@group(0) @binding(3) var<storage, read> countBuf: array<u32>;      // [0] = prim count, [1] = node-write base
${NODE_WGSL}

// a child's final bounds resolved from prior-sweep state, looking up to LEVELS-1 levels past it.
// ok = false when the child's subtree hasn't reached this thread yet (a later sweep completes it).
struct Resolved { ok: bool, mn: vec3<f32>, mx: vec3<f32> }
${resolveChain(LEVELS)}

@compute @workgroup_size(${WG_BOUNDS})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    NODE_BASE = countBuf[1];
    let primCount = countBuf[0];
    if (primCount <= 1u) { return; }                   // single prim: the leaf is the root, no internal nodes
    let n = primCount + gid.x;                         // internal slot [N, 2N−1)
    if (n > 2u * primCount - 2u) { return; }           // past the root
    if (validIn[n] == 1u) { validOut[n] = 1u; return; } // already done, carry
    let lc = resolve${LEVELS - 1}(nodeLeft(n));
    if (!lc.ok) { validOut[n] = 0u; return; }          // not ready yet; a later sweep completes it
    let rc = resolve${LEVELS - 1}(nodeRight(n));
    if (!rc.ok) { validOut[n] = 0u; return; }
    writeBounds(n, min(lc.mn, rc.mn), max(lc.mx, rc.mx));
    validOut[n] = 1u;
}
`;

function storageEntry(binding: number, readonly: boolean): GPUBindGroupLayoutEntry {
    return {
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: readonly ? "read-only-storage" : "storage" },
    };
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

    // The bounds relaxation runs a fixed sweep count: the worst-case radix-tree height for this
    // builder (30 Morton bits + the index tiebreak's ceil(log2 N), the all-equal-code worst case)
    // divided by the levels each sweep climbs — enough that the deepest node's bounds reach the
    // root. Sweeps past convergence just carry, so the result is stable; this avoids the per-sweep
    // prepare + indirect early-out (its pass count, ~2× the sweep count, was the build's dominant
    // cost). Direct dispatch over a cap-sized workgroup count (a CPU constant): a build sized to
    // its workload over-dispatches nothing, and a small mesh in a shared builder over-dispatches
    // threads that early-out cheaply.
    const heightBound = Math.min(MAX_BOUNDS_STEPS, 30 + Math.ceil(Math.log2(Math.max(2, cap))));
    const boundsSteps = Math.ceil(heightBound / LEVELS);
    // each sweep covers the internal nodes only ([N, 2N−1) ≈ half the tree); leaves never need a
    // thread (leaf-init seeds their flag in both buffers), so the dispatch is sized to cap−1, not
    // 2·cap−1. max(1, …) keeps a ≥1 dispatch for cap = 1 (no internal nodes; threads all early-out).
    const sweepWG = Math.max(1, Math.ceil((cap - 1) / WG_BOUNDS));

    const prepareLayout = device.createBindGroupLayout({
        label: "build-prepare",
        entries: [storageEntry(0, true), storageEntry(1, false)],
    });
    const leafLayout = device.createBindGroupLayout({
        label: "build-leaf",
        entries: [
            storageEntry(0, true),
            storageEntry(1, false),
            storageEntry(2, false),
            storageEntry(3, true),
            storageEntry(4, false),
        ],
    });
    const topoLayout = device.createBindGroupLayout({
        label: "build-topo",
        entries: [
            storageEntry(0, true),
            storageEntry(1, true),
            storageEntry(2, false),
            storageEntry(3, true),
        ],
    });
    const sweepLayout = device.createBindGroupLayout({
        label: "build-sweep",
        entries: [
            storageEntry(0, false),
            storageEntry(1, true),
            storageEntry(2, false),
            storageEntry(3, true),
        ],
    });

    const pipe = (
        label: string,
        code: string,
        layout: GPUBindGroupLayout,
    ): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
        });

    const [preparePipe, leafPipe, topoPipe, sweepPipe] = await Promise.all([
        pipe("build-prepare", PREPARE_WGSL, prepareLayout),
        pipe("build-leaf", LEAF_WGSL, leafLayout),
        pipe("build-topo", TOPO_WGSL, topoLayout),
        pipe("build-sweep", SWEEP_WGSL, sweepLayout),
    ]);

    const prepareBg = device.createBindGroup({
        layout: prepareLayout,
        entries: [
            { binding: 0, resource: { buffer: count } },
            { binding: 1, resource: { buffer: indirect } },
        ],
    });
    const leafBg = device.createBindGroup({
        layout: leafLayout,
        entries: [
            { binding: 0, resource: { buffer: prims } },
            { binding: 1, resource: { buffer: nodes } },
            { binding: 2, resource: { buffer: validA } },
            { binding: 3, resource: { buffer: count } },
            { binding: 4, resource: { buffer: validB } },
        ],
    });
    const topoBg = device.createBindGroup({
        layout: topoLayout,
        entries: [
            { binding: 0, resource: { buffer: keys } },
            { binding: 1, resource: { buffer: payload } },
            { binding: 2, resource: { buffer: nodes } },
            { binding: 3, resource: { buffer: count } },
        ],
    });
    // two sweep bind groups ping-pong the double-buffered flags: even sweeps read A write B,
    // odd sweeps read B write A. The first sweep reads validA (leaf-init's output).
    const sweepBg = (vin: GPUBuffer, vout: GPUBuffer): GPUBindGroup =>
        device.createBindGroup({
            layout: sweepLayout,
            entries: [
                { binding: 0, resource: { buffer: nodes } },
                { binding: 1, resource: { buffer: vin } },
                { binding: 2, resource: { buffer: vout } },
                { binding: 3, resource: { buffer: count } },
            ],
        });
    const sweepAB = sweepBg(validA, validB);
    const sweepBA = sweepBg(validB, validA);

    const dispatch = (
        encoder: GPUCommandEncoder,
        pipeline: GPUComputePipeline,
        bg: GPUBindGroup,
        wg: number,
        span: string,
    ): void => {
        const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.(span) });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wg);
        pass.end();
    };
    const indirectPass = (
        encoder: GPUCommandEncoder,
        pipeline: GPUComputePipeline,
        bg: GPUBindGroup,
        args: GPUBuffer,
        offset: number,
        span: string,
    ): void => {
        const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.(span) });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroupsIndirect(args, offset);
        pass.end();
    };

    // the bounds relaxation, shared by build + refit: a fixed `boundsSteps` direct sweeps,
    // ping-ponging the double-buffered valid flags. Enough to exceed the worst-case height, so
    // the deepest node's bounds reach the root; later sweeps carry, leaving the result stable.
    const relax = (encoder: GPUCommandEncoder, span: string): void => {
        for (let k = 0; k < boundsSteps; k++) {
            dispatch(encoder, sweepPipe, k % 2 === 0 ? sweepAB : sweepBA, sweepWG, span);
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
            dispatch(encoder, preparePipe, prepareBg, 1, "bvh:build");
            indirectPass(encoder, leafPipe, leafBg, indirect, 0, "bvh:build");
            indirectPass(encoder, topoPipe, topoBg, indirect, 12, "bvh:build");
            relax(encoder, "bvh:build");
        },
        refit(encoder: GPUCommandEncoder): void {
            // topology persists in `nodes`; re-bound the leaves from the moved prims (and reset
            // the flags), then re-run the relaxation over the fixed topology
            dispatch(encoder, preparePipe, prepareBg, 1, "bvh:refit");
            indirectPass(encoder, leafPipe, leafBg, indirect, 0, "bvh:refit");
            relax(encoder, "bvh:refit");
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
