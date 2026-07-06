// CPU reference for the H-PLOC GPU BVH builder — the executable spec the GPU build
// is validated against. Test scaffolding (consumed by oracle.test.ts and the gym's
// `accel` scenario), kept out of the shipped src/ tree.
//
// It ports the NexusBVH pipeline (Benthin et al. 2024, reference/NexusBVH): scene
// bounds → 30-bit Morton codes → stable sort → locally-ordered PLOC clustering →
// binary BVH2. The GPU runs this cooperatively across a subgroup; here it runs
// sequentially so the result is deterministic and `bun test`-fast, no GPU needed.
//
// Exact topology is NOT the contract — parallel merge order is non-deterministic
// even in NexusBVH. The contract is the per-pass output (bounds, codes, sort) and,
// for the build, the invariants + SAH + ray-vs-brute-force agreement that
// {@link invariants} and {@link compareRays} check. See scratch.md "H-PLOC BVH".

import { type Prims, primMax, primMin } from "./fixtures";

/** PLOC nearest-neighbor search radius (NexusBVH SEARCH_RADIUS) */
export const SEARCH_RADIUS = 8;

/** leaf sentinel + "no node" marker (NexusBVH INVALID_IDX) */
export const INVALID = 0xffffffff;

/** floats per BVH2 node: 2 × vec4 = `[min.xyz, leftChild], [max.xyz, rightChild]` (32 B) */
export const NODE_F32 = 8;

export type Vec3 = [number, number, number];

/** axis-aligned bounds; `min`/`max` are inclusive corners */
export interface Aabb {
    min: Vec3;
    max: Vec3;
}

/**
 * binary BVH2 in the GPU output layout: a flat array of `2N−1` nodes (N=1 → 1 node),
 * each 32 B as two `vec4<f32>`. `bounds` reads the corners; `child` reads the two
 * `.w` lanes (bitcast u32). A node is a leaf when `leftChild === INVALID`, and then
 * `rightChild` is the primitive index. Internal `leftChild`/`rightChild` are node
 * indices into this same array. Root is the last-allocated node.
 */
export interface Bvh2 {
    bounds: Float32Array;
    child: Uint32Array;
    count: number;
    root: number;
    primCount: number;
}

export function isLeaf(bvh: Bvh2, node: number): boolean {
    return bvh.child[node * NODE_F32 + 3] === INVALID;
}

export function primOf(bvh: Bvh2, node: number): number {
    return bvh.child[node * NODE_F32 + 7];
}

export function left(bvh: Bvh2, node: number): number {
    return bvh.child[node * NODE_F32 + 3];
}

export function right(bvh: Bvh2, node: number): number {
    return bvh.child[node * NODE_F32 + 7];
}

/** read a node's AABB straight from a bounds array — usable mid-build, before a Bvh2 exists */
function readAabb(bounds: Float32Array, node: number): Aabb {
    const o = node * NODE_F32;
    return {
        min: [bounds[o], bounds[o + 1], bounds[o + 2]],
        max: [bounds[o + 4], bounds[o + 5], bounds[o + 6]],
    };
}

export function nodeAabb(bvh: Bvh2, node: number): Aabb {
    return readAabb(bvh.bounds, node);
}

// ---- bounds (Phase 2 reference) ----------------------------------------------

/** scene AABB = union of every primitive AABB */
export function sceneBounds(p: Prims): Aabb {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.count; i++) {
        const mn = primMin(p, i);
        const mx = primMax(p, i);
        for (let a = 0; a < 3; a++) {
            if (mn[a] < min[a]) min[a] = mn[a];
            if (mx[a] > max[a]) max[a] = mx[a];
        }
    }
    return { min, max };
}

function area(a: Aabb): number {
    const dx = a.max[0] - a.min[0];
    const dy = a.max[1] - a.min[1];
    const dz = a.max[2] - a.min[2];
    return dx * dy + dy * dz + dz * dx;
}

// ---- Morton codes (Phase 3 reference) ----------------------------------------

/**
 * spread the low 10 bits of x to every third bit (insert two zeros between bits).
 * Bit-identical to NexusBVH InterleaveBits32; all intermediates stay within 32 bits.
 */
export function interleaveBits32(x: number): number {
    let v = x & 0x3ff;
    v = (v | (v << 16)) & 0x30000ff;
    v = (v | (v << 8)) & 0x300f00f;
    v = (v | (v << 4)) & 0x30c30c3;
    v = (v | (v << 2)) & 0x9249249;
    return v >>> 0;
}

/**
 * 30-bit Morton code from a centroid already normalized to [0,1] per axis.
 * Quantizes to 10 bits/axis (×1023, NexusBVH) then interleaves x|y<<1|z<<2.
 * The ×1023 rounds through `fround` so the bin matches the f32 GPU (morton.ts).
 */
export function mortonCode(nx: number, ny: number, nz: number): number {
    const q = (t: number): number => {
        const c = t < 0 ? 0 : t > 1 ? 1 : t;
        const v = Math.floor(Math.fround(c * 1023));
        return v < 0 ? 0 : v > 1023 ? 1023 : v;
    };
    return (
        (interleaveBits32(q(nx)) |
            (interleaveBits32(q(ny)) << 1) |
            (interleaveBits32(q(nz)) << 2)) >>>
        0
    );
}

/**
 * per-primitive 30-bit Morton codes. The centroid is the AABB midpoint, normalized
 * by the scene bounds. A zero-extent axis (coplanar fixture) normalizes to 0 rather
 * than NaN — the defined-code requirement of Phase 3.
 *
 * Every float op rounds through `Math.fround`, so this models f32 arithmetic exactly
 * and stays bit-identical to the f32 GPU (morton.ts) — without it, the f64 normalize
 * divide lands ~80/1M centroids in a different quantization bin than the GPU. (CUDA's
 * NexusBVH runs f32 too, so this is the more faithful port.)
 */
export function mortonCodes(p: Prims, bounds: Aabb): Uint32Array {
    const fr = Math.fround;
    const codes = new Uint32Array(p.count);
    const ext: Vec3 = [
        fr(bounds.max[0] - bounds.min[0]),
        fr(bounds.max[1] - bounds.min[1]),
        fr(bounds.max[2] - bounds.min[2]),
    ];
    for (let i = 0; i < p.count; i++) {
        const mn = primMin(p, i);
        const mx = primMax(p, i);
        const norm = (a: number): number => {
            if (ext[a] <= 0) return 0;
            const centroid = fr(fr(mn[a] + mx[a]) * 0.5);
            return fr(fr(centroid - bounds.min[a]) / ext[a]);
        };
        codes[i] = mortonCode(norm(0), norm(1), norm(2));
    }
    return codes;
}

// ---- sort (Phase 1 reference) ------------------------------------------------

/**
 * primitive indices ordered by Morton code, stable (equal codes keep input order).
 * The GPU DeviceRadixSort of `(code, primIndex)` pairs is validated against this.
 */
export function sortMorton(codes: Uint32Array): Uint32Array {
    const idx = Array.from({ length: codes.length }, (_, i) => i);
    idx.sort((a, b) => codes[a] - codes[b] || a - b);
    return Uint32Array.from(idx);
}

// ---- build (Phase 4 reference) -----------------------------------------------

/** full pipeline: bounds → codes → sort → PLOC clustering → BVH2 */
export function build(p: Prims): Bvh2 {
    const n = p.count;
    const count = Math.max(1, 2 * n - 1);
    const buf = new ArrayBuffer(count * NODE_F32 * 4);
    const bounds = new Float32Array(buf);
    const child = new Uint32Array(buf);

    // Leaf nodes occupy slots [0, n): each bounds its prim, leftChild = INVALID,
    // rightChild = the primitive index.
    for (let i = 0; i < n; i++) {
        writeNode(bounds, child, i, { min: primMin(p, i), max: primMax(p, i) }, INVALID, i);
    }

    if (n === 1) return { bounds, child, count, root: 0, primCount: 1 };

    const sb = sceneBounds(p);
    const order = sortMorton(mortonCodes(p, sb));

    // Working clusters: node indices in Morton order. Each PLOC step merges every
    // mutual nearest-neighbor pair (by merged-AABB area) within ±SEARCH_RADIUS,
    // emitting one internal node per merge and compacting in place — order
    // preserved by writing each parent at its left member's position.
    let clusters = Array.from(order);
    let next = n;

    while (clusters.length > 1) {
        const m = clusters.length;
        const boxes = clusters.map((c) => readAabb(bounds, c));

        const nn = new Int32Array(m);
        for (let i = 0; i < m; i++) {
            let best = -1;
            let bestArea = Infinity;
            const lo = Math.max(0, i - SEARCH_RADIUS);
            const hi = Math.min(m - 1, i + SEARCH_RADIUS);
            for (let j = lo; j <= hi; j++) {
                if (j === i) continue;
                const a = area(union(boxes[i], boxes[j]));
                if (a < bestArea) {
                    bestArea = a;
                    best = j;
                }
            }
            nn[i] = best;
        }

        const merged = new Uint8Array(m);
        const parentAt = new Int32Array(m).fill(-1);
        for (let i = 0; i < m; i++) {
            const j = nn[i];
            // The globally-closest within-radius pair is always mutual, so each step
            // merges at least one pair → the loop terminates in n−1 merges.
            if (j > i && nn[j] === i) {
                const node = next++;
                writeNode(bounds, child, node, union(boxes[i], boxes[j]), clusters[i], clusters[j]);
                merged[i] = 1;
                merged[j] = 1;
                parentAt[i] = node;
            }
        }

        const out: number[] = [];
        for (let i = 0; i < m; i++) {
            if (parentAt[i] >= 0) out.push(parentAt[i]);
            else if (!merged[i]) out.push(clusters[i]);
        }
        clusters = out;
    }

    return { bounds, child, count, root: clusters[0], primCount: n };
}

function union(a: Aabb, b: Aabb): Aabb {
    return {
        min: [
            Math.min(a.min[0], b.min[0]),
            Math.min(a.min[1], b.min[1]),
            Math.min(a.min[2], b.min[2]),
        ],
        max: [
            Math.max(a.max[0], b.max[0]),
            Math.max(a.max[1], b.max[1]),
            Math.max(a.max[2], b.max[2]),
        ],
    };
}

function writeNode(
    bounds: Float32Array,
    child: Uint32Array,
    node: number,
    b: Aabb,
    leftIdx: number,
    rightIdx: number,
): void {
    const o = node * NODE_F32;
    bounds[o] = b.min[0];
    bounds[o + 1] = b.min[1];
    bounds[o + 2] = b.min[2];
    bounds[o + 4] = b.max[0];
    bounds[o + 5] = b.max[1];
    bounds[o + 6] = b.max[2];
    child[o + 3] = leftIdx;
    child[o + 7] = rightIdx;
}

function setBounds(bvh: Bvh2, node: number, b: Aabb): void {
    const o = node * NODE_F32;
    bvh.bounds[o] = b.min[0];
    bvh.bounds[o + 1] = b.min[1];
    bvh.bounds[o + 2] = b.min[2];
    bvh.bounds[o + 4] = b.max[0];
    bvh.bounds[o + 5] = b.max[1];
    bvh.bounds[o + 6] = b.max[2];
}

// ---- refit (fixed-topology bounds recompute) ---------------------------------

/**
 * recompute every node's bounds from new primitive AABBs, leaving topology
 * untouched — the executable spec for the GPU refit (the bounds relaxation in
 * build.ts). A leaf re-bounds its primitive; an internal node is the union of its
 * children. Walks the tree post-order from the root so children are bounded before
 * their parent, which holds for any topology (the LBVH build does not order node
 * indices by depth, unlike the former H-PLOC build). Mutates `bvh` in place. The
 * result is bit-exact — min/max introduce no rounding, and union is order-
 * independent — so the GPU refit, computing the same unique fixed point, matches it
 * to the bit.
 */
export function refit(bvh: Bvh2, p: Prims): void {
    refitNode(bvh, p, bvh.root);
}

function refitNode(bvh: Bvh2, p: Prims, node: number): Aabb {
    if (isLeaf(bvh, node)) {
        const pi = primOf(bvh, node);
        const b = { min: primMin(p, pi), max: primMax(p, pi) };
        setBounds(bvh, node, b);
        return b;
    }
    const u = union(refitNode(bvh, p, left(bvh, node)), refitNode(bvh, p, right(bvh, node)));
    setBounds(bvh, node, u);
    return u;
}

// ---- invariants --------------------------------------------------------------

/**
 * structural checks on a built BVH2 — empty result means it holds. Returns a list
 * of violations: node count = 2N−1, every prim a leaf exactly once, each internal
 * node's AABB equals the union of its children, root AABB equals the scene bounds.
 * AABB comparisons are exact: min/max never introduce rounding, so a correct tree
 * is bit-identical, no tolerance to tune.
 */
export function invariants(bvh: Bvh2, p: Prims): string[] {
    const errs: string[] = [];
    const n = p.count;
    const expected = Math.max(1, 2 * n - 1);
    if (bvh.count !== expected) errs.push(`node count ${bvh.count}, expected ${expected}`);

    const seen = new Uint8Array(n);
    let leaves = 0;
    const stack = [bvh.root];
    let guard = 0;
    while (stack.length > 0) {
        if (guard++ > 4 * expected) {
            errs.push("traversal did not terminate (cycle?)");
            break;
        }
        const node = stack.pop() as number;
        if (isLeaf(bvh, node)) {
            const pi = primOf(bvh, node);
            leaves++;
            if (pi >= n) errs.push(`leaf prim index ${pi} out of range`);
            else if (seen[pi]++) errs.push(`prim ${pi} appears in more than one leaf`);
        } else {
            const l = left(bvh, node);
            const r = right(bvh, node);
            const u = union(nodeAabb(bvh, l), nodeAabb(bvh, r));
            if (!aabbExact(nodeAabb(bvh, node), u))
                errs.push(`node ${node} bounds != union of children`);
            stack.push(l, r);
        }
    }
    if (leaves !== n) errs.push(`reached ${leaves} leaves, expected ${n}`);
    for (let i = 0; i < n; i++) if (!seen[i]) errs.push(`prim ${i} never reached`);

    if (!aabbExact(nodeAabb(bvh, bvh.root), sceneBounds(p)))
        errs.push("root bounds != scene bounds");
    return errs;
}

function aabbExact(a: Aabb, b: Aabb): boolean {
    for (let i = 0; i < 3; i++) {
        if (a.min[i] !== b.min[i] || a.max[i] !== b.max[i]) return false;
    }
    return true;
}

/**
 * surface area heuristic cost, `(ct·Σ area(internal) + ci·Σ area(leaf)) / area(root)`
 * with ct = ci = 1. The Phase 4 gate compares GPU vs oracle SAH within tolerance;
 * topology differences move it slightly, a broken bound blows it up.
 */
export function sah(bvh: Bvh2): number {
    const rootArea = area(nodeAabb(bvh, bvh.root));
    if (rootArea <= 0) return 0;
    let internal = 0;
    let leaf = 0;
    const stack = [bvh.root];
    while (stack.length > 0) {
        const node = stack.pop() as number;
        const a = area(nodeAabb(bvh, node));
        if (isLeaf(bvh, node)) {
            leaf += a;
        } else {
            internal += a;
            stack.push(left(bvh, node), right(bvh, node));
        }
    }
    return (internal + leaf) / rootArea;
}

// ---- ray vs brute force ------------------------------------------------------

export interface Ray {
    origin: Vec3;
    dir: Vec3;
}

export interface Hit {
    prim: number;
    t: number;
}

/** ray-AABB slab test; returns the entry distance, or null if the ray misses */
export function intersectAabb(ray: Ray, inv: Vec3, b: Aabb): number | null {
    let tmin = -Infinity;
    let tmax = Infinity;
    for (let a = 0; a < 3; a++) {
        const t1 = (b.min[a] - ray.origin[a]) * inv[a];
        const t2 = (b.max[a] - ray.origin[a]) * inv[a];
        const lo = Math.min(t1, t2);
        const hi = Math.max(t1, t2);
        if (lo > tmin) tmin = lo;
        if (hi < tmax) tmax = hi;
    }
    // interval [tmin, tmax] must overlap [0, ∞)
    if (tmax < Math.max(tmin, 0)) return null;
    return tmin;
}

export function invDir(ray: Ray): Vec3 {
    return [1 / ray.dir[0], 1 / ray.dir[1], 1 / ray.dir[2]];
}

/** nearest primitive the ray hits, scanning every prim (the ground truth) */
export function nearestHitBrute(p: Prims, ray: Ray): Hit | null {
    const inv = invDir(ray);
    let hit: Hit | null = null;
    for (let i = 0; i < p.count; i++) {
        const t = intersectAabb(ray, inv, { min: primMin(p, i), max: primMax(p, i) });
        if (t === null) continue;
        const d = Math.max(t, 0);
        if (!hit || d < hit.t) hit = { prim: i, t: d };
    }
    return hit;
}

/**
 * nearest primitive via BVH traversal, pruning any subtree whose node AABB the ray
 * misses. Agreement with {@link nearestHitBrute} proves the node bounds correctly
 * enclose their subtrees — a wrongly-tight bound would drop a prim brute force finds.
 */
export function nearestHitBvh(bvh: Bvh2, p: Prims, ray: Ray): Hit | null {
    const inv = invDir(ray);
    let hit: Hit | null = null;
    const stack = [bvh.root];
    while (stack.length > 0) {
        const node = stack.pop() as number;
        if (intersectAabb(ray, inv, nodeAabb(bvh, node)) === null) continue;
        if (isLeaf(bvh, node)) {
            const pi = primOf(bvh, node);
            const t = intersectAabb(ray, inv, { min: primMin(p, pi), max: primMax(p, pi) });
            if (t === null) continue;
            const d = Math.max(t, 0);
            if (!hit || d < hit.t) hit = { prim: pi, t: d };
        } else {
            stack.push(left(bvh, node), right(bvh, node));
        }
    }
    return hit;
}

// ---- restart-trail + short-stack traversal (the stackless GPU traverser's spec) --------
//
// Vaidyanathan, Woop & Benthin, "Wide BVH Traversal with a Short Stack" (HPG 2019),
// specialized to a binary BVH — equivalently Laine 2010's restart trail. The motivation:
// a full per-thread stack sized to the worst-case tree height spills to local memory and
// caps GPU occupancy. Instead a per-level *trail* (2 bits/level: how many of a node's
// children are already processed, or N=2 = "all entered") lets a pop restart from the root
// and skip resolved subtrees, so the only unbounded state is the trail — sized to the
// provable max height (62 for a 30-bit-Morton + index-tiebreak tree), not tuned. A small
// short stack of the most-recently-pushed far children skips most restarts; entries evicted
// from its bottom are recovered by the trail on a later restart, so the short stack size is
// a pure perf knob with no correctness role (shortCap = 0 = pure restart, the strongest
// stress). Front-to-back child order is load-bearing for closest-hit: t-pruning only culls
// the *farther* child, so a culled child always sorts last and "skip the first c" keeps
// referring to the same child after a restart with a tightened hit.
//
// Trail size: the GPU traverser sizes its trail to a fixed 64 levels, which is ≥ the
// provable LBVH depth bound — the key is `(Morton30 << 32 | index)` = 62 bits and a binary
// radix tree on distinct keys has depth ≤ key bits, so an LBVH is ≤ 62 deep for any scene
// (no tuning). This oracle builds PLOC, not LBVH, and PLOC chains on coincident geometry to
// depths well past 62, so it sizes the trail to the tree it's handed (`primCount` levels
// covers the degenerate chain); CPU memory is free. Either way the trail only has to exceed
// the tree depth — its size never changes the result, only whether it can overflow.

const TRAIL_N = 2; // binary BVH: a node's children are fully entered at count 2

/**
 * closest hit via restart-trail + short-stack traversal. The GPU closest-hit traverser uses
 * a plain stack (the trail regresses it on deep binary trees — see traverse.ts), so this is
 * not its direct spec; it's the rigorous correctness proof of the *trail mechanism* under the
 * hardest case (closest-hit's t-pruning, which the simpler any-hit trail never exercises).
 * Agreement with {@link nearestHitBrute} at any `shortCap` (including 0 = pure restart) proves
 * the trail + restart logic independent of stack depth.
 */
export function nearestHitRestart(bvh: Bvh2, p: Prims, ray: Ray, shortCap: number): Hit | null {
    const inv = invDir(ray);
    const trail = new Uint8Array(Math.max(2, bvh.primCount));
    const short: number[] = [];
    let hit: Hit | null = null;
    let level = 0;
    let node = bvh.root;

    const limit = (): number => (hit ? hit.t : Number.POSITIVE_INFINITY);
    // box-entry distance if the child is worth descending (hits the interval within the
    // closest hit so far), else +Inf — so a culled child reads as "dead", sorting last
    const entry = (child: number): number => {
        const t = intersectAabb(ray, inv, nodeAabb(bvh, child));
        if (t === null) return Number.POSITIVE_INFINITY;
        const d = Math.max(t, 0);
        return d < limit() ? d : Number.POSITIVE_INFINITY;
    };
    // advance the trail to the next pending child; returns true when traversal is done
    const pop = (): boolean => {
        let pl = -1;
        for (let i = level - 1; i >= 0; i--) {
            if (trail[i] !== TRAIL_N) {
                pl = i;
                break;
            }
        }
        if (pl < 0) return true;
        trail[pl] += 1;
        for (let i = pl + 1; i < trail.length; i++) trail[i] = 0;
        if (short.length === 0) {
            node = bvh.root;
            level = 0;
        } else {
            node = short.pop() as number; // far = last child
            trail[pl] = TRAIL_N;
            level = pl + 1;
        }
        return false;
    };

    for (;;) {
        if (isLeaf(bvh, node)) {
            const pi = primOf(bvh, node);
            const t = intersectAabb(ray, inv, { min: primMin(p, pi), max: primMax(p, pi) });
            if (t !== null) {
                const d = Math.max(t, 0);
                if (d < limit()) hit = { prim: pi, t: d };
            }
            if (pop()) break;
            continue;
        }
        const l = left(bvh, node);
        const r = right(bvh, node);
        const el = entry(l);
        const er = entry(r);
        const nearNode = el <= er ? l : r; // smaller box entry first (front-to-back)
        const farNode = el <= er ? r : l;
        const nearE = Math.min(el, er);
        const farE = Math.max(el, er);
        const c = trail[level];
        let next = -1;
        let setLast = false; // entering this node's last live child → mark the level done
        let pushFar = false; // both children live → far is pending after the near descent
        if (c === 0) {
            if (farE < Number.POSITIVE_INFINITY) {
                next = nearNode;
                pushFar = true;
            } else if (nearE < Number.POSITIVE_INFINITY) {
                next = nearNode;
                setLast = true;
            }
        } else if (c === 1) {
            if (farE < Number.POSITIVE_INFINITY) {
                next = farNode;
                setLast = true;
            }
        } else if (farE < Number.POSITIVE_INFINITY) {
            next = farNode; // c === N: re-descend into the last live child
            setLast = true;
        } else if (nearE < Number.POSITIVE_INFINITY) {
            next = nearNode;
            setLast = true;
        }
        if (next < 0) {
            if (pop()) break;
            continue;
        }
        if (pushFar) {
            if (short.length >= shortCap) short.shift(); // evict oldest; the trail recovers it
            short.push(farNode);
        }
        if (setLast) trail[level] = TRAIL_N;
        node = next;
        level += 1;
    }
    return hit;
}

/**
 * occlusion (any-hit) via the same restart-trail machinery — returns at the first primitive
 * within `(0, tMax)`. tMax is fixed (no closest tightening), so the live-child set never
 * shrinks and the trail count is trivially stable; front-to-back order is then irrelevant
 * to correctness. Spec for the GPU `bvhAnyHit`.
 */
export function anyHitRestart(bvh: Bvh2, ray: Ray, tMax: number, shortCap: number): boolean {
    const inv = invDir(ray);
    const trail = new Uint8Array(Math.max(2, bvh.primCount));
    const short: number[] = [];
    let level = 0;
    let node = bvh.root;

    const entry = (n: number): number => {
        const t = intersectAabb(ray, inv, nodeAabb(bvh, n));
        if (t === null) return Number.POSITIVE_INFINITY;
        const d = Math.max(t, 0);
        return d < tMax ? d : Number.POSITIVE_INFINITY;
    };
    const pop = (): boolean => {
        let pl = -1;
        for (let i = level - 1; i >= 0; i--) {
            if (trail[i] !== TRAIL_N) {
                pl = i;
                break;
            }
        }
        if (pl < 0) return true;
        trail[pl] += 1;
        for (let i = pl + 1; i < trail.length; i++) trail[i] = 0;
        if (short.length === 0) {
            node = bvh.root;
            level = 0;
        } else {
            node = short.pop() as number;
            trail[pl] = TRAIL_N;
            level = pl + 1;
        }
        return false;
    };

    for (;;) {
        if (isLeaf(bvh, node)) {
            if (entry(node) < Number.POSITIVE_INFINITY) return true; // leaf box = prim AABB
            if (pop()) break;
            continue;
        }
        const l = left(bvh, node);
        const r = right(bvh, node);
        const el = entry(l);
        const er = entry(r);
        const nearNode = el <= er ? l : r;
        const farNode = el <= er ? r : l;
        const nearE = Math.min(el, er);
        const farE = Math.max(el, er);
        const c = trail[level];
        let next = -1;
        let setLast = false;
        let pushFar = false;
        if (c === 0) {
            if (farE < Number.POSITIVE_INFINITY) {
                next = nearNode;
                pushFar = true;
            } else if (nearE < Number.POSITIVE_INFINITY) {
                next = nearNode;
                setLast = true;
            }
        } else if (c === 1) {
            if (farE < Number.POSITIVE_INFINITY) {
                next = farNode;
                setLast = true;
            }
        } else if (farE < Number.POSITIVE_INFINITY) {
            next = farNode;
            setLast = true;
        } else if (nearE < Number.POSITIVE_INFINITY) {
            next = nearNode;
            setLast = true;
        }
        if (next < 0) {
            if (pop()) break;
            continue;
        }
        if (pushFar) {
            if (short.length >= shortCap) short.shift();
            short.push(farNode);
        }
        if (setLast) trail[level] = TRAIL_N;
        node = next;
        level += 1;
    }
    return false;
}

/** a deterministic ray batch aimed at the scene, plus guaranteed misses */
export function rays(bounds: Aabb, n: number, seed: number): Ray[] {
    let s = seed >>> 0 || 1;
    const rand = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
    const center: Vec3 = [
        (bounds.min[0] + bounds.max[0]) * 0.5,
        (bounds.min[1] + bounds.max[1]) * 0.5,
        (bounds.min[2] + bounds.max[2]) * 0.5,
    ];
    const span = Math.max(
        bounds.max[0] - bounds.min[0],
        bounds.max[1] - bounds.min[1],
        bounds.max[2] - bounds.min[2],
        1,
    );
    const out: Ray[] = [];
    for (let i = 0; i < n; i++) {
        // origin on a shell around the scene, aimed at a random interior point
        const dir: Vec3 = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
        const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
        const r = span * 2;
        const origin: Vec3 = [
            center[0] - (dir[0] / len) * r,
            center[1] - (dir[1] / len) * r,
            center[2] - (dir[2] / len) * r,
        ];
        const target: Vec3 = [
            bounds.min[0] + rand() * (bounds.max[0] - bounds.min[0]),
            bounds.min[1] + rand() * (bounds.max[1] - bounds.min[1]),
            bounds.min[2] + rand() * (bounds.max[2] - bounds.min[2]),
        ];
        out.push({
            origin,
            dir: [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]],
        });
        // a ray pointed away from the scene — must miss in both
        out.push({
            origin,
            dir: [origin[0] - center[0], origin[1] - center[1], origin[2] - center[2]],
        });
    }
    return out;
}

/** mismatches between BVH traversal and brute force over a ray batch — empty = agree */
export function compareRays(bvh: Bvh2, p: Prims, rays: Ray[]): string[] {
    const errs: string[] = [];
    for (let i = 0; i < rays.length; i++) {
        const a = nearestHitBrute(p, rays[i]);
        const b = nearestHitBvh(bvh, p, rays[i]);
        if (a === null && b === null) continue;
        if (a === null || b === null) {
            errs.push(`ray ${i}: brute ${a ? "hit" : "miss"}, bvh ${b ? "hit" : "miss"}`);
            continue;
        }
        if (a.t !== b.t)
            errs.push(`ray ${i}: brute t=${a.t} prim=${a.prim}, bvh t=${b.t} prim=${b.prim}`);
    }
    return errs;
}

// ---- traversal-quality metrics -----------------------------------------------
//
// The hardware-independent traversal cost, the standard BVH-quality diagnostic
// (madmann91/bvh visited_nodes / visited_leaves; tray_racing per-ray cost): how many
// node boxes a ray pierces and how many leaves it reaches (each a leaf test a real
// traverser then performs), averaged over a batch. It correlates with the SAH cost the
// build optimizes, so a topology change that lowers SAH should lower these. The descent
// prunes only on an AABB miss (no t-max pruning against the running best hit), so the
// count is a pure tree-quality proxy independent of the GPU traverser's pruning — the
// number to read alongside SAH, not a prediction of the kernel's exact step count.

/**
 * deepest root→leaf path length (edges) in a built BVH2 — the number of trail levels the
 * stackless traverser touches. Used by the `accel` scenario's tree-depth gate to confirm the
 * LBVH build stays under the trail's provable coverage (62, the radix-tree key-width bound);
 * a tree deeper than that would mean a malformed build, not a traverser bug. Walks every
 * node iteratively; the both-push frontier is ≤ depth + 1 nodes, so a small fixed walk
 * stack covers any 30-bit-Morton tree.
 */
export function treeMaxDepth(bvh: Bvh2): number {
    let max = 0;
    const node = new Int32Array(256);
    const depth = new Int32Array(256);
    let sp = 0;
    node[sp] = bvh.root;
    depth[sp] = 0;
    sp++;
    let guard = 0;
    const limit = 4 * bvh.count + 8;
    while (sp > 0) {
        if (guard++ > limit) throw new Error("treeMaxDepth: walk did not terminate (cycle?)");
        sp--;
        const n = node[sp];
        const d = depth[sp];
        if (d > max) max = d;
        if (!isLeaf(bvh, n)) {
            if (sp + 2 > node.length)
                throw new Error(`treeMaxDepth: frontier exceeds ${node.length} (malformed tree?)`);
            node[sp] = left(bvh, n);
            depth[sp] = d + 1;
            sp++;
            node[sp] = right(bvh, n);
            depth[sp] = d + 1;
            sp++;
        }
    }
    return max;
}

/** mean node-pierces (steps) and leaves reached (leaf tests) per ray over a batch */
export function traceStats(bvh: Bvh2, batch: Ray[]): { avgSteps: number; avgLeafTests: number } {
    let steps = 0;
    let leafTests = 0;
    for (const ray of batch) {
        const inv = invDir(ray);
        const stack = [bvh.root];
        while (stack.length > 0) {
            const node = stack.pop() as number;
            steps++;
            if (intersectAabb(ray, inv, nodeAabb(bvh, node)) === null) continue;
            if (isLeaf(bvh, node)) {
                leafTests++;
            } else {
                stack.push(left(bvh, node), right(bvh, node));
            }
        }
    }
    const n = Math.max(1, batch.length);
    return { avgSteps: steps / n, avgLeafTests: leafTests / n };
}

/**
 * incoherent ray batch: origins scattered through the scene volume, directions uniform
 * on the sphere — no shared focus, so adjacent rays descend divergent subtrees. The
 * divergence stress the {@link rays} shell batch (origins on a shell, aimed at the
 * scene — coherent) doesn't apply: a physics query or a GI bounce traces rays like
 * these, so the coherent-vs-incoherent MRays/s + step gap reads how much the traverser's
 * cost is divergence rather than tree depth.
 */
export function incoherentRays(bounds: Aabb, n: number, seed: number): Ray[] {
    let s = seed >>> 0 || 1;
    const rand = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
    };
    const out: Ray[] = [];
    for (let i = 0; i < n; i++) {
        const origin: Vec3 = [
            bounds.min[0] + rand() * (bounds.max[0] - bounds.min[0]),
            bounds.min[1] + rand() * (bounds.max[1] - bounds.min[1]),
            bounds.min[2] + rand() * (bounds.max[2] - bounds.min[2]),
        ];
        // uniform direction on the unit sphere (z + azimuth), so no two rays share a target
        const z = rand() * 2 - 1;
        const phi = rand() * 2 * Math.PI;
        const r = Math.sqrt(Math.max(0, 1 - z * z));
        out.push({ origin, dir: [r * Math.cos(phi), r * Math.sin(phi), z] });
    }
    return out;
}
