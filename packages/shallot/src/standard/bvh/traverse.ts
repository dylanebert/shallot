// Software BVH2 traversal — the single-level ray-AABB query over the 32 B node format
// the builder (build.ts) emits and its bounds relaxation keeps current. WebGPU has no
// hardware ray-query, so any consumer that wants spatial queries descends the LBVH tree
// in software. This is the unopinionated query: it knows only AABBs (the builder's input
// is an AABB array), names no leaf geometry and no consumer. The physics broadphase /
// raycast and sear's RT lighting both splice it; the RT-specific layers (ray-triangle
// leaf tests, two-level BLAS/TLAS instancing) live with their consumer (shadow/), not here.
//
// The chunk reads a `nodes: array<vec4<f32>>` binding the consumer declares — node `n`
// is two sequential `vec4<f32>`: `nodes[2n]` = (min.xyz, bitcast leftChild) and
// `nodes[2n+1]` = (max.xyz, bitcast rightChild). A leaf has `leftChild == INVALID` and
// stores its primitive index in `rightChild`; its bounds are the primitive's AABB, so the
// slab test against the leaf node is the primitive hit.
//
// `bvhClosestHit` is the nearest-hit query (primary rays, the debug trace); `bvhAnyHit`
// is the occlusion query — it returns at the first hit inside the ray interval, so it
// traverses far less. Both prune a subtree whose node AABB the ray enters beyond the
// current limit, matching the CPU oracle's `nearestHitBvh` (tests/bvh/oracle.ts), the
// spec the GPU traverser is gated against.

/** bytes per BVH2 node (2 × vec4<f32>) */
export const BVH_NODE_BYTES = 32;

/** leaf sentinel in a node's `leftChild` lane; also "no primitive" */
export const BVH_INVALID = 0xffffffff;

/**
 * the BVH depth bound both traversers size against, and what a consumer's own traverser
 * uses. It is **derived, not tuned**: the LBVH key is `(Morton30 << 32 | index)` = 62 bits,
 * and a binary radix tree on distinct keys has depth ≤ its key width, so an LBVH is ≤ 62 deep
 * for *any* scene the builder can produce. 64 covers that provably: no measurement, no
 * margin, no overflow path to get wrong. (A 64-bit-Morton build would raise the key width
 * and this with it.) `bvhClosestHit` sizes its far-child stack to this; `bvhAnyHit`'s restart
 * trail covers this many levels. Either way the push/level index can't exceed it, so a hit is
 * never dropped.
 *
 * The two queries use different traversal schemes because measurement (the `accel` scenario,
 * Lovelace) splits them: occlusion (`bvhAnyHit`) is occupancy-bound — it early-exits, reads
 * few nodes — so the stackless restart trail (a 2-bit-per-level trail + a small
 * {@link BVH_SHORT_STACK}, far less per-thread state than a depth-sized stack) wins ~1.6×.
 * Closest-hit reads many nodes and pops often; the trail's per-pop O(depth) parent scan is
 * costly on a deep binary tree and it gains nothing from the smaller footprint (closest is
 * occupancy-insensitive on this desktop GPU), so it keeps a plain stack. Restart trail:
 * Vaidyanathan et al., "Wide BVH Traversal with a Short Stack", HPG 2019 / Laine 2010,
 * specialized to a binary BVH.
 */
export const BVH_TRAIL_LEVELS = 64;

/** u32 words holding the 2-bit-per-level restart trail (16 levels per word) */
const BVH_TRAIL_WORDS = Math.ceil((BVH_TRAIL_LEVELS * 2) / 32);

/**
 * short-stack capacity (power of two). A pure perf knob: an entry evicted from the bottom is
 * recovered by the trail on a later restart, so any size is correct — this only trades
 * per-thread occupancy against restart frequency (HPG 2019: ~5 entries lands within ~10% of
 * a full stack).
 */
const BVH_SHORT_STACK = 8;

/** the BVH2 root node index for `primCount` primitives (the last-allocated node) */
export function bvhRoot(primCount: number): number {
    return primCount <= 1 ? 0 : 2 * primCount - 2;
}

/**
 * WGSL form of {@link bvhRoot}: `fn bvhRoot(primCount: u32) -> u32`. Splice into a
 * consumer that reads the prim count from the GPU-driven count buffer so the trace's
 * root is computed on the GPU rather than passed as a CPU uniform.
 */
export const BVH_ROOT_WGSL = /* wgsl */ `
fn bvhRoot(primCount: u32) -> u32 { return select(0u, 2u * primCount - 2u, primCount > 1u); }
`;

/**
 * WGSL traversal chunk. Splice into a module that declares
 * `var<storage, read> nodes: array<vec4<f32>>`. Exposes `BvhHit`, `bvhClosestHit`,
 * and `bvhAnyHit`. `dir` need not be normalized (the hit distance is then in `dir`
 * lengths); pass `1.0 / dir` as `invDir`. `tMax` bounds the ray interval and seeds
 * the closest distance. Pass a finite limit (a true miss returns `prim == INVALID`
 * / `false`).
 *
 * @example
 * ```wgsl
 * @group(0) @binding(3) var<storage, read> nodes: array<vec4<f32>>;
 * // BVH_TRAVERSE_WGSL here
 * let hit = bvhClosestHit(root, ro, 1.0 / rd, 1e30);
 * if (hit.prim != 0xffffffffu) { shade(hit.t, hit.prim); }
 * ```
 */
export const BVH_TRAVERSE_WGSL = /* wgsl */ `
const BVH_INVALID = ${BVH_INVALID}u;
const BVH_MISS = 3.0e38;          // slab-miss sentinel, above any finite tMax

struct BvhHit {
    t: f32,                       // hit distance in \`dir\` lengths, clamped at 0
    prim: u32,                    // primitive index, or BVH_INVALID on a miss
    visits: u32,                  // nodes visited — the heatmap signal
};

fn bvhNodeMin(n: u32) -> vec3<f32> { return nodes[2u * n].xyz; }
fn bvhNodeMax(n: u32) -> vec3<f32> { return nodes[2u * n + 1u].xyz; }
fn bvhLeft(n: u32) -> u32 { return bitcast<u32>(nodes[2u * n].w); }
fn bvhRight(n: u32) -> u32 { return bitcast<u32>(nodes[2u * n + 1u].w); }
fn bvhIsLeaf(n: u32) -> bool { return bvhLeft(n) == BVH_INVALID; }

// Ray-AABB slab entry distance, or BVH_MISS when the interval [tEnter, tExit] does
// not overlap [0, +inf). Matches the oracle's intersectAabb; \`inv\` is 1/dir.
fn bvhSlab(n: u32, ro: vec3<f32>, inv: vec3<f32>) -> f32 {
    let t1 = (bvhNodeMin(n) - ro) * inv;
    let t2 = (bvhNodeMax(n) - ro) * inv;
    let lo = min(t1, t2);
    let hi = max(t1, t2);
    let tEnter = max(max(lo.x, lo.y), lo.z);
    let tExit = min(min(hi.x, hi.y), hi.z);
    if (tExit < max(tEnter, 0.0)) { return BVH_MISS; }
    return tEnter;
}

// --- restart-trail + short-stack state for the occlusion query bvhAnyHit (per invocation;
// it zeroes them at entry). bvhClosestHit uses a plain stack instead — see there. A full
// per-thread stack sized to tree depth spills to local memory and caps occupancy; the trail
// instead records, in 2 bits per level, how many of each node's children are entered (0, 1,
// or BVH_N = both done), so a pop can restart from the root and skip resolved subtrees. A
// small short stack of recent far children skips most restarts; an entry evicted from its
// bottom is recovered by the trail, so its size is a pure perf knob. The trail covers
// BVH_TRAIL_LEVELS, the provable LBVH depth bound — correct for any tree. Occlusion is the
// occupancy-bound query (early-exit, few node reads), so the smaller footprint wins big
// (the accel scenario: ~1.6× over a full stack on the 1M any-hit row).
const BVH_N = 2u;
const BVH_TRAIL_WORDS = ${BVH_TRAIL_WORDS}u;
const BVH_SHORT_STACK = ${BVH_SHORT_STACK}u;

var<private> bvhTrail: array<u32, ${BVH_TRAIL_WORDS}>;
var<private> bvhSS: array<u32, ${BVH_SHORT_STACK}>;       // ring of recent far ("last") children
var<private> bvhSSHead: u32;                              // ring base; advances as the bottom is evicted
var<private> bvhSSCount: u32;

fn bvhTrailGet(level: u32) -> u32 { return (bvhTrail[level >> 4u] >> ((level & 15u) * 2u)) & 3u; }
fn bvhTrailSet(level: u32, v: u32) {
    let w = level >> 4u;
    let s = (level & 15u) * 2u;
    bvhTrail[w] = (bvhTrail[w] & ~(3u << s)) | (v << s);
}
// clear every level deeper than pl — its subtrees are about to be re-walked
fn bvhTrailResetBelow(pl: u32) {
    let w0 = pl >> 4u;
    let sub = pl & 15u;
    if (sub != 15u) { bvhTrail[w0] = bvhTrail[w0] & ((1u << ((sub + 1u) * 2u)) - 1u); }
    for (var w = w0 + 1u; w < BVH_TRAIL_WORDS; w = w + 1u) { bvhTrail[w] = 0u; }
}
// highest ancestor level (scanning up) with a child still to enter, or -1 when traversal is done
fn bvhFindParent(level: u32) -> i32 {
    for (var i = i32(level) - 1; i >= 0; i = i - 1) {
        if (bvhTrailGet(u32(i)) != BVH_N) { return i; }
    }
    return -1;
}
fn bvhPushFar(far: u32) {
    if (bvhSSCount < BVH_SHORT_STACK) {
        bvhSS[(bvhSSHead + bvhSSCount) & (BVH_SHORT_STACK - 1u)] = far;
        bvhSSCount += 1u;
    } else {
        bvhSS[bvhSSHead] = far;                            // evict oldest; the trail recovers it
        bvhSSHead = (bvhSSHead + 1u) & (BVH_SHORT_STACK - 1u);
    }
}

// Nearest primitive the ray hits within [0, tMax]. A node entered at or beyond the closest
// hit so far is pruned; a leaf's slab test is its primitive hit.
//
// Explicit far-child stack, NOT the restart trail bvhAnyHit uses. Measured (the accel
// scenario, Lovelace, ×3): the restart trail regresses closest-hit ~1.5× — its per-pop
// O(depth) parent-level scan is expensive on a deep binary tree (the trail technique targets
// shallow wide BVHs), and closest-hit is occupancy-insensitive here anyway (a 64→40 stack
// shrink moved it ~3%), so it gains nothing from the trail's smaller footprint. So closest
// keeps the simple stack. The stack is sized to BVH_TRAIL_LEVELS — the *derived* provable
// depth bound (62 ≤ this), not a tuned constant — so the push guard never trips and a hit is
// never dropped. Occupancy being null on desktop, the stack might still help on an integrated
// GPU; that's the unmeasured case, not this one.
//
// Front-to-back ordered descent: both children are slab-tested, the nearer descended
// directly, the farther pushed; the nearer subtree tightens hit.t first, so the farther is
// pruned (entered beyond hit.t) far more often than an unordered push. The stack carries only
// far children (one u32/slot).
fn bvhClosestHit(root: u32, ro: vec3<f32>, inv: vec3<f32>, tMax: f32) -> BvhHit {
    var hit: BvhHit;
    hit.t = tMax;
    hit.prim = BVH_INVALID;
    hit.visits = 0u;

    var stack: array<u32, ${BVH_TRAIL_LEVELS}>;
    var sp = 0u;
    var node = root;
    loop {
        hit.visits += 1u;
        if (bvhIsLeaf(node)) {
            let t = bvhSlab(node, ro, inv);
            let d = max(t, 0.0);
            if (t < BVH_MISS && d < hit.t) {
                hit.t = d;
                hit.prim = bvhRight(node);
            }
            if (sp == 0u) { break; }
            sp -= 1u;
            node = stack[sp];
            continue;
        }
        let l = bvhLeft(node);
        let r = bvhRight(node);
        let tl = bvhSlab(l, ro, inv);
        let tr = bvhSlab(r, ro, inv);
        let okL = tl < BVH_MISS && max(tl, 0.0) < hit.t;
        let okR = tr < BVH_MISS && max(tr, 0.0) < hit.t;
        if (okL && okR) {
            let leftNear = tl <= tr;        // descend the nearer, push the farther
            if (sp < ${BVH_TRAIL_LEVELS}u) { stack[sp] = select(l, r, leftNear); sp += 1u; }
            node = select(r, l, leftNear);
        } else if (okL) {
            node = l;
        } else if (okR) {
            node = r;
        } else {
            if (sp == 0u) { break; }
            sp -= 1u;
            node = stack[sp];
        }
    }
    return hit;
}

// True if any primitive lies within (0, tMax) — the occlusion query. Returns at the first
// leaf hit. Uses the restart trail + short stack (the helpers above): tMax is fixed (no
// closest tightening), so the live-child set never shrinks and the trail count is trivially
// stable; front-to-back order is then irrelevant to correctness, kept only for code symmetry
// with the trail's general form. This is the query the stackless rewrite wins on.
fn bvhAnyHit(root: u32, ro: vec3<f32>, inv: vec3<f32>, tMax: f32) -> bool {
    for (var w = 0u; w < BVH_TRAIL_WORDS; w = w + 1u) { bvhTrail[w] = 0u; }
    bvhSSHead = 0u;
    bvhSSCount = 0u;
    var level = 0u;
    var node = root;
    loop {
        var needPop = false;
        if (bvhIsLeaf(node)) {
            let t = bvhSlab(node, ro, inv);
            if (t < BVH_MISS && max(t, 0.0) < tMax) { return true; }
            needPop = true;
        } else {
            let l = bvhLeft(node);
            let r = bvhRight(node);
            let tl = bvhSlab(l, ro, inv);
            let tr = bvhSlab(r, ro, inv);
            let okL = tl < BVH_MISS && max(tl, 0.0) < tMax;
            let okR = tr < BVH_MISS && max(tr, 0.0) < tMax;
            let leftNear = tl <= tr;
            let nearNode = select(r, l, leftNear);
            let farNode = select(l, r, leftNear);
            let bothOk = okL && okR;
            let anyOk = okL || okR;
            let c = bvhTrailGet(level);
            var nextNode = BVH_INVALID;
            var pushFar = false;
            var setLast = false;
            if (c == 0u) {
                if (bothOk) { nextNode = nearNode; pushFar = true; }
                else if (anyOk) { nextNode = nearNode; setLast = true; }
            } else if (c == 1u) {
                if (bothOk) { nextNode = farNode; setLast = true; }
            } else {
                if (bothOk) { nextNode = farNode; setLast = true; }
                else if (anyOk) { nextNode = nearNode; setLast = true; }
            }
            if (nextNode != BVH_INVALID) {
                if (pushFar) { bvhPushFar(farNode); }
                if (setLast) { bvhTrailSet(level, BVH_N); }
                node = nextNode;
                level += 1u;
            } else {
                needPop = true;
            }
        }
        if (needPop) {
            let pl = bvhFindParent(level);
            if (pl < 0) { break; }
            let plu = u32(pl);
            bvhTrailSet(plu, bvhTrailGet(plu) + 1u);
            bvhTrailResetBelow(plu);
            if (bvhSSCount == 0u) {
                node = root;
                level = 0u;
            } else {
                bvhSSCount -= 1u;
                node = bvhSS[(bvhSSHead + bvhSSCount) & (BVH_SHORT_STACK - 1u)];
                bvhTrailSet(plu, BVH_N);
                level = plu + 1u;
            }
        }
    }
    return false;
}
`;
