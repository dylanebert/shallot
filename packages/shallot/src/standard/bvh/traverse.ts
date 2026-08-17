// Software BVH2 traversal — the single-level ray-AABB query over the 32 B node format
// the builder (build.ts) emits and its bounds relaxation keeps current. WebGPU has no
// hardware ray-query, so any consumer that wants spatial queries descends the LBVH tree
// in software. This is the unopinionated query: it knows only AABBs (the builder's input
// is an AABB array), names no leaf geometry and no consumer. The physics broadphase splices
// `bvhRootWgsl` and hand-writes its own descent (its per-body nearest-K prune is not a ray
// query); the gym tracer is the traversal chunk's consumer. Consumer-specific layers —
// ray-triangle leaf tests, two-level BLAS/TLAS instancing — live with the consumer, not here.
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
//
// TGSL, with the four node accessors WGSL-bodied: they read `nodes` as a module-scope
// global the *consumer* declares by name (the relocatable contract, sear's shadow-sampler
// shape), which has no TGSL spelling until the consumer contract itself is typed. Every
// layer above them — the slab test, the restart trail, both queries — is TGSL over the
// accessors, so the arithmetic has one source. Not CPU-callable: every entry point either
// reads `nodes` or a `private` var.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { chunk, spliceNs } from "../../engine/utils/core";

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

/** slab-miss sentinel, above any finite tMax */
const BVH_MISS = 3.0e38;

/** trail code for "both children entered" */
const BVH_N = 2;

/**
 * the BVH2 root node index for `primCount` primitives (the last-allocated node). One source for
 * both sides: called on the CPU as a plain function, and spliced as WGSL by {@link bvhRootWgsl}
 * for a consumer that reads the count from the GPU-driven count buffer (so the trace's root is
 * computed on the GPU rather than passed as a CPU uniform).
 * @example const root = bvhRoot(primCount);
 */
export const bvhRoot = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((primCount) => {
        "use gpu";
        // both arms evaluate (typegpu's select is a call, not a short-circuit); the discarded arm
        // underflows at primCount 0, which is harmless on integers — no non-finite intermediate
        return std.select(0, 2 * primCount - 2, primCount > 1);
    })
    .$name("bvhRoot");

/** returns the `bvhRoot(primCount)` WGSL — splice into a consumer computing the trace root on the
 *  GPU from the count buffer. */
export const bvhRootWgsl = chunk("bvhRootWgsl", [bvhRoot], spliceNs);

/** one traversal result: hit distance, primitive, and the node-visit count (the heatmap signal) */
export const BvhHit = d
    .struct({
        /** hit distance in `dir` lengths, clamped at 0 */
        t: d.f32,
        /** primitive index, or {@link BVH_INVALID} on a miss */
        prim: d.u32,
        /** nodes visited — the heatmap signal */
        visits: d.u32,
    })
    .$name("BvhHit");

// The four node readers — WGSL-bodied, because `nodes` is a module-scope global the *consumer*
// declares by name (the relocatable contract; no TGSL spelling until that contract is typed).
const bvhNodeMin = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )(/* wgsl */ `(n: u32) -> vec3f { return nodes[2u * n].xyz; }`)
    .$name("bvhNodeMin");

const bvhNodeMax = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )(/* wgsl */ `(n: u32) -> vec3f { return nodes[2u * n + 1u].xyz; }`)
    .$name("bvhNodeMax");

const bvhLeft = tgpu
    .fn(
        [d.u32],
        d.u32,
    )(/* wgsl */ `(n: u32) -> u32 { return bitcast<u32>(nodes[2u * n].w); }`)
    .$name("bvhLeft");

const bvhRight = tgpu
    .fn(
        [d.u32],
        d.u32,
    )(/* wgsl */ `(n: u32) -> u32 { return bitcast<u32>(nodes[2u * n + 1u].w); }`)
    .$name("bvhRight");

const bvhIsLeaf = tgpu
    .fn(
        [d.u32],
        d.bool,
    )((n) => {
        "use gpu";
        return bvhLeft(n) === BVH_INVALID;
    })
    .$name("bvhIsLeaf");

// Ray-AABB slab entry distance, or BVH_MISS when the interval [tEnter, tExit] does
// not overlap [0, +inf). Matches the oracle's intersectAabb; `inv` is 1/dir.
const bvhSlab = tgpu
    .fn(
        [d.u32, d.vec3f, d.vec3f],
        d.f32,
    )((n, ro, inv) => {
        "use gpu";
        const t1 = std.mul(std.sub(bvhNodeMin(n), ro), inv);
        const t2 = std.mul(std.sub(bvhNodeMax(n), ro), inv);
        const lo = std.min(t1, t2);
        const hi = std.max(t1, t2);
        const tEnter = std.max(std.max(lo.x, lo.y), lo.z);
        const tExit = std.min(std.min(hi.x, hi.y), hi.z);
        if (tExit < std.max(tEnter, 0)) return BVH_MISS;
        return tEnter;
    })
    .$name("bvhSlab");

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
const bvhTrail = tgpu.privateVar(d.arrayOf(d.u32, BVH_TRAIL_WORDS));
/** ring of recent far ("last") children */
const bvhSS = tgpu.privateVar(d.arrayOf(d.u32, BVH_SHORT_STACK));
/** ring base; advances as the bottom is evicted */
const bvhSSHead = tgpu.privateVar(d.u32);
const bvhSSCount = tgpu.privateVar(d.u32);

const bvhTrailGet = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((level) => {
        "use gpu";
        return (bvhTrail.$[level >>> 4] >>> ((level & 15) * 2)) & 3;
    })
    .$name("bvhTrailGet");

const bvhTrailSet = tgpu
    .fn([d.u32, d.u32])((level, v) => {
        "use gpu";
        const w = level >>> 4;
        const s = (level & 15) * 2;
        // d.u32 on the mask literals is load-bearing: a bare `3`/`1` materializes i32, so at level 15
        // (s = 30) `3 << s` overflows signed and the `&` becomes a mixed-sign implicit conversion
        bvhTrail.$[w] = (bvhTrail.$[w] & ~(d.u32(3) << s)) | (v << s);
    })
    .$name("bvhTrailSet");

// clear every level deeper than pl — its subtrees are about to be re-walked
const bvhTrailResetBelow = tgpu
    .fn([d.u32])((pl) => {
        "use gpu";
        const w0 = pl >>> 4;
        const sub = pl & 15;
        if (sub !== 15) bvhTrail.$[w0] = bvhTrail.$[w0] & ((d.u32(1) << ((sub + 1) * 2)) - 1);
        let w = w0 + 1;
        while (w < BVH_TRAIL_WORDS) {
            bvhTrail.$[w] = 0;
            w = w + 1;
        }
    })
    .$name("bvhTrailResetBelow");

// highest ancestor level (scanning up) with a child still to enter, or BVH_INVALID when
// traversal is done. Unsigned throughout: the shipped WGSL scanned down from `i32(level) - 1`
// and returned -1, which is the signed-arithmetic trap; the sentinel is the
// same scan with the same visit order and the same results.
const bvhFindParent = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((level) => {
        "use gpu";
        let i = level;
        while (i > 0) {
            i = i - 1;
            if (bvhTrailGet(i) !== BVH_N) return i;
        }
        return d.u32(BVH_INVALID);
    })
    .$name("bvhFindParent");

const bvhPushFar = tgpu
    .fn([d.u32])((far) => {
        "use gpu";
        if (bvhSSCount.$ < BVH_SHORT_STACK) {
            bvhSS.$[(bvhSSHead.$ + bvhSSCount.$) & (BVH_SHORT_STACK - 1)] = far;
            bvhSSCount.$ = bvhSSCount.$ + 1;
        } else {
            bvhSS.$[bvhSSHead.$] = far; // evict oldest; the trail recovers it
            bvhSSHead.$ = (bvhSSHead.$ + 1) & (BVH_SHORT_STACK - 1);
        }
    })
    .$name("bvhPushFar");

// the closest-hit far-child stack. `private`, not a function-local `var` as the shipped WGSL had
// it: TGSL has no spelling for a local fixed-size array. Semantically identical — `sp` is zeroed
// at entry, so nothing carries across calls, and a `private` array and a function-local one both
// live in the same per-invocation storage class.
const bvhStack = tgpu.privateVar(d.arrayOf(d.u32, BVH_TRAIL_LEVELS));

/**
 * nearest primitive the ray hits within [0, tMax]. A node entered at or beyond the closest
 * hit so far is pruned; a leaf's slab test is its primitive hit. `dir` need not be normalized
 * (the hit distance is then in `dir` lengths); pass `1.0 / dir` as `inv`, and a finite `tMax`
 * (a true miss returns `prim == BVH_INVALID`).
 *
 * Explicit far-child stack, NOT the restart trail {@link bvhAnyHit} uses. Measured (the accel
 * scenario, Lovelace, ×3): the restart trail regresses closest-hit ~1.5× — its per-pop
 * O(depth) parent-level scan is expensive on a deep binary tree (the trail technique targets
 * shallow wide BVHs), and closest-hit is occupancy-insensitive here anyway (a 64→40 stack
 * shrink moved it ~3%), so it gains nothing from the trail's smaller footprint. So closest
 * keeps the simple stack. The stack is sized to {@link BVH_TRAIL_LEVELS} — the *derived*
 * provable depth bound (62 ≤ this), not a tuned constant — so the push guard never trips and a
 * hit is never dropped. Occupancy being null on desktop, the stack might still help on an
 * integrated GPU; that's the unmeasured case, not this one.
 *
 * Front-to-back ordered descent: both children are slab-tested, the nearer descended
 * directly, the farther pushed; the nearer subtree tightens hit.t first, so the farther is
 * pruned (entered beyond hit.t) far more often than an unordered push. The stack carries only
 * far children (one u32/slot).
 * @example const hit = bvhClosestHit(root, ro, 1.0 / rd, 1e30);
 */
export const bvhClosestHit = tgpu
    .fn(
        [d.u32, d.vec3f, d.vec3f, d.f32],
        BvhHit,
    )((root, ro, inv, tMax) => {
        "use gpu";
        let hitT = tMax;
        let hitPrim = d.u32(BVH_INVALID);
        let visits = d.u32(0);

        let sp = d.u32(0);
        let node = root;
        while (true) {
            visits = visits + 1;
            if (bvhIsLeaf(node)) {
                const t = bvhSlab(node, ro, inv);
                const dist = std.max(t, 0);
                if (t < BVH_MISS && dist < hitT) {
                    hitT = dist;
                    hitPrim = bvhRight(node);
                }
                if (sp === 0) break;
                sp = sp - 1;
                node = bvhStack.$[sp];
                continue;
            }
            const l = bvhLeft(node);
            const r = bvhRight(node);
            const tl = bvhSlab(l, ro, inv);
            const tr = bvhSlab(r, ro, inv);
            const okL = tl < BVH_MISS && std.max(tl, 0) < hitT;
            const okR = tr < BVH_MISS && std.max(tr, 0) < hitT;
            if (okL && okR) {
                const leftNear = tl <= tr; // descend the nearer, push the farther
                if (sp < BVH_TRAIL_LEVELS) {
                    bvhStack.$[sp] = std.select(l, r, leftNear);
                    sp = sp + 1;
                }
                node = std.select(r, l, leftNear);
            } else if (okL) {
                node = l;
            } else if (okR) {
                node = r;
            } else {
                if (sp === 0) break;
                sp = sp - 1;
                node = bvhStack.$[sp];
            }
        }
        return BvhHit({ t: hitT, prim: hitPrim, visits: visits });
    })
    .$name("bvhClosestHit");

/**
 * true if any primitive lies within (0, tMax) — the occlusion query. Returns at the first
 * leaf hit. Uses the restart trail + short stack: tMax is fixed (no closest tightening), so
 * the live-child set never shrinks and the trail count is trivially stable; front-to-back
 * order is then irrelevant to correctness, kept only for code symmetry with the trail's
 * general form. This is the query the stackless rewrite wins on.
 * @example if (bvhAnyHit(root, ro, 1.0 / rd, dist)) { shadowed(); }
 */
export const bvhAnyHit = tgpu
    .fn(
        [d.u32, d.vec3f, d.vec3f, d.f32],
        d.bool,
    )((root, ro, inv, tMax) => {
        "use gpu";
        let w = d.u32(0);
        while (w < BVH_TRAIL_WORDS) {
            bvhTrail.$[w] = 0;
            w = w + 1;
        }
        bvhSSHead.$ = 0;
        bvhSSCount.$ = 0;
        let level = d.u32(0);
        let node = root;
        while (true) {
            let needPop = false;
            if (bvhIsLeaf(node)) {
                const t = bvhSlab(node, ro, inv);
                if (t < BVH_MISS && std.max(t, 0) < tMax) return true;
                needPop = true;
            } else {
                const l = bvhLeft(node);
                const r = bvhRight(node);
                const tl = bvhSlab(l, ro, inv);
                const tr = bvhSlab(r, ro, inv);
                const okL = tl < BVH_MISS && std.max(tl, 0) < tMax;
                const okR = tr < BVH_MISS && std.max(tr, 0) < tMax;
                const leftNear = tl <= tr;
                const nearNode = std.select(r, l, leftNear);
                const farNode = std.select(l, r, leftNear);
                const bothOk = okL && okR;
                const anyOk = okL || okR;
                const c = bvhTrailGet(level);
                let nextNode = d.u32(BVH_INVALID);
                let pushFar = false;
                let setLast = false;
                if (c === 0) {
                    if (bothOk) {
                        nextNode = nearNode;
                        pushFar = true;
                    } else if (anyOk) {
                        nextNode = nearNode;
                        setLast = true;
                    }
                } else if (c === 1) {
                    if (bothOk) {
                        nextNode = farNode;
                        setLast = true;
                    }
                } else {
                    if (bothOk) {
                        nextNode = farNode;
                        setLast = true;
                    } else if (anyOk) {
                        nextNode = nearNode;
                        setLast = true;
                    }
                }
                if (nextNode !== BVH_INVALID) {
                    if (pushFar) bvhPushFar(farNode);
                    if (setLast) bvhTrailSet(level, BVH_N);
                    node = nextNode;
                    level = level + 1;
                } else {
                    needPop = true;
                }
            }
            if (needPop) {
                const pl = bvhFindParent(level);
                if (pl === BVH_INVALID) break;
                bvhTrailSet(pl, bvhTrailGet(pl) + 1);
                bvhTrailResetBelow(pl);
                if (bvhSSCount.$ === 0) {
                    node = root;
                    level = 0;
                } else {
                    bvhSSCount.$ = bvhSSCount.$ - 1;
                    node = bvhSS.$[(bvhSSHead.$ + bvhSSCount.$) & (BVH_SHORT_STACK - 1)];
                    bvhTrailSet(pl, BVH_N);
                    level = pl + 1;
                }
            }
        }
        return false;
    })
    .$name("bvhAnyHit");

/**
 * returns the traversal WGSL. Splice into a module that declares
 * `var<storage, read> nodes: array<vec4<f32>>`. Exposes exactly three definitions —
 * `struct BvhHit`, `fn bvhClosestHit`, `fn bvhAnyHit` — and **no WGSL consts**: every
 * constant folds to a literal, so compare a miss against the {@link BVH_INVALID} export
 * interpolated into your own template rather than a spliced `BVH_INVALID` name.
 *
 * `dir` need not be normalized (the hit distance is then in `dir` lengths); pass
 * `1.0 / dir` as `invDir`. `tMax` bounds the ray interval and seeds the closest distance.
 * Pass a finite limit (a true miss returns `prim == BVH_INVALID` / `false`).
 *
 * @example
 * ```wgsl
 * @group(0) @binding(3) var<storage, read> nodes: array<vec4<f32>>;
 * // bvhTraverseWgsl() here
 * let hit = bvhClosestHit(root, ro, 1.0 / rd, 1e30);
 * if (hit.prim != 0xffffffffu) { shade(hit.t, hit.prim); }
 * ```
 */
export const bvhTraverseWgsl = chunk(
    "bvhTraverseWgsl",
    [BvhHit, bvhClosestHit, bvhAnyHit],
    spliceNs,
);
