import { describe, expect, test } from "bun:test";
import {
    allFixtures,
    clumps,
    clustered,
    coincident,
    coplanar,
    PRIM_F32,
    type Prims,
    slivers,
    twoPrim,
    uniformRandom,
} from "./fixtures";
import {
    anyHitRestart,
    type Bvh2,
    build,
    compareRays,
    incoherentRays,
    invariants,
    mortonCodes,
    nearestHitBrute,
    nearestHitRestart,
    type Ray,
    rays,
    refit,
    sah,
    sceneBounds,
    sortMorton,
    traceStats,
} from "./oracle";

// translate each prim by a deterministic per-prim offset — a fresh scene with the
// same prim count, so leaf node `i` still maps to prim `i` and refit applies. The
// per-prim (not uniform) jitter forces every node's bounds to genuinely recompute.
function move(p: Prims, seed: number): Prims {
    let s = seed >>> 0 || 1;
    const rand = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return (s / 0x100000000 - 0.5) * 6;
    };
    const data = new Float32Array(p.data);
    for (let i = 0; i < p.count; i++) {
        const dx = rand();
        const dy = rand();
        const dz = rand();
        const o = i * PRIM_F32;
        data[o] += dx;
        data[o + 1] += dy;
        data[o + 2] += dz;
        data[o + 4] += dx;
        data[o + 5] += dy;
        data[o + 6] += dz;
    }
    return { count: p.count, data };
}

// The oracle is the executable spec the GPU BVH build is validated against, so its
// own correctness is gated on the same checks the GPU will face: structural
// invariants + ray-vs-brute-force on every fixture, plus the per-pass references
// (bounds, Morton, sort) that Phases 1–3 compare GPU output against.

describe("bvh oracle — invariants per fixture", () => {
    for (const { name, prims } of allFixtures()) {
        test(name, () => {
            const bvh = build(prims);
            expect(invariants(bvh, prims)).toEqual([]);
            // SAH is finite and ≥ 1 (root alone costs 1); a broken bound blows it up.
            const cost = sah(bvh);
            expect(Number.isFinite(cost)).toBe(true);
            expect(cost).toBeGreaterThan(0);
        });
    }
});

describe("bvh oracle — ray vs brute force per fixture", () => {
    for (const { name, prims } of allFixtures()) {
        test(name, () => {
            const bvh = build(prims);
            const batch = rays(sceneBounds(prims), 128, 0xabcdef ^ prims.count);
            expect(compareRays(bvh, prims, batch)).toEqual([]);
        });
    }
});

describe("bvh oracle — refit holds invariants + rays on a moved scene", () => {
    for (const { name, prims } of allFixtures()) {
        test(name, () => {
            const bvh = build(prims);
            const moved = move(prims, 0x9e3779b9 ^ prims.count);
            refit(bvh, moved);
            // topology is unchanged, but every bound now reflects the moved prims:
            // unions hold, root equals the moved scene bounds, rays match brute force
            expect(invariants(bvh, moved)).toEqual([]);
            const batch = rays(sceneBounds(moved), 128, 0xfeed ^ moved.count);
            expect(compareRays(bvh, moved, batch)).toEqual([]);
        });
    }
});

describe("bvh oracle — scene bounds (Phase 2 reference)", () => {
    test("union of two separated prims", () => {
        const b = sceneBounds(twoPrim());
        expect(b.min).toEqual([-2, -2, -2]);
        expect(b.max).toEqual([2, 2, 2]);
    });
});

describe("bvh oracle — Morton codes (Phase 3 reference)", () => {
    test("deterministic across calls", () => {
        const prims = uniformRandom(256, 1);
        const sb = sceneBounds(prims);
        expect(Array.from(mortonCodes(prims, sb))).toEqual(Array.from(mortonCodes(prims, sb)));
    });

    test("defined on a zero-extent axis, no NaN collapse", () => {
        const prims = coplanar(256, 0x6666);
        const codes = mortonCodes(prims, sceneBounds(prims));
        // every code is a valid 30-bit integer (a NaN centroid would collapse to 0)
        for (const c of codes) {
            expect(Number.isInteger(c)).toBe(true);
            expect(c).toBeGreaterThanOrEqual(0);
            expect(c).toBeLessThan(1 << 30);
        }
        // x/y variation survives the degenerate z axis: more than one distinct code
        expect(new Set(codes).size).toBeGreaterThan(1);
    });
});

describe("bvh oracle — sort (Phase 1 reference)", () => {
    test("stable on equal codes", () => {
        // codes: 5 5 3 3 5 → order by code, ties keep original index → 2 3 0 1 4
        const order = sortMorton(Uint32Array.from([5, 5, 3, 3, 5]));
        expect(Array.from(order)).toEqual([2, 3, 0, 1, 4]);
    });
});

describe("bvh oracle — node count is 2N-1", () => {
    for (const { name, prims } of allFixtures()) {
        test(name, () => {
            expect(build(prims).count).toBe(Math.max(1, 2 * prims.count - 1));
        });
    }
});

// The traversal-quality metrics reported alongside SAH. A deeper tree
// (more prims) must pierce more node boxes per ray than a shallow one — the property
// that makes avg-steps a tree-quality proxy.
describe("bvh oracle — traversal stats", () => {
    test("steps grow with tree depth; leaf tests are finite", () => {
        const small = uniformRandom(16, 0x1234);
        const large = uniformRandom(4096, 0x1234);
        const batch = (p: Prims): ReturnType<typeof traceStats> =>
            traceStats(build(p), rays(sceneBounds(p), 128, 0xa5));
        const s = batch(small);
        const l = batch(large);
        expect(s.avgSteps).toBeGreaterThan(0);
        expect(Number.isFinite(s.avgLeafTests)).toBe(true);
        expect(l.avgSteps).toBeGreaterThan(s.avgSteps);
    });

    test("incoherent rays are unit-length", () => {
        const batch = incoherentRays(sceneBounds(uniformRandom(64, 7)), 64, 0xbeef);
        expect(batch.length).toBe(64);
        for (const r of batch) {
            const len = Math.hypot(r.dir[0], r.dir[1], r.dir[2]);
            expect(len).toBeCloseTo(1, 5);
        }
    });
});

// The restart-trail mechanism (bvhAnyHit on the GPU rides it; closest-hit there keeps a
// stack) must agree with brute force at ANY short-stack capacity — including 0 (pure
// restart), where every pop re-descends from the root. Agreement there proves the trail
// bookkeeping and restart logic independent of stack depth, the property that lets any-hit
// drop its fixed stack. Both the closest and any-hit trail forms are checked: closest is the
// harder one (t-pruning), so it stresses the trail beyond what the shipped any-hit needs.
// Both math sides are f64, so a closest hit's distance is bit-exact; the primitive may differ
// only on an exact distance tie (the same relaxation compareRays takes). The deep adversarial
// fixtures (coincident, clumps) drive the trail to its degenerate-tree depth, so shortCap 0
// exercises the heaviest restart path.
describe("bvh oracle — restart-trail traversal", () => {
    const scenes: { name: string; prims: Prims }[] = [
        ...allFixtures(),
        { name: "uniform-2048", prims: uniformRandom(2048, 0x70a5) },
        { name: "clustered-1024", prims: clustered(1024, 0x71b6) },
        { name: "slivers-512", prims: slivers(512, 0x72c7) },
        { name: "coincident-512", prims: coincident(512, 0x73d8) },
        { name: "clumps-1024", prims: clumps(64, 16, 0x74e9) },
    ];
    // 0 = pure restart (no stack at all), then sizes from under any tree depth to over it
    const caps = [0, 1, 4, 64];

    function batch(prims: Prims): Ray[] {
        const sb = sceneBounds(prims);
        return [...rays(sb, 96, 0x9a5f), ...incoherentRays(sb, 128, 0x9b6e)];
    }

    test("closest-hit matches brute force at every short-stack capacity", () => {
        for (const { name, prims } of scenes) {
            const bvh: Bvh2 = build(prims);
            const batchRays = batch(prims);
            for (const cap of caps) {
                for (let i = 0; i < batchRays.length; i++) {
                    const brute = nearestHitBrute(prims, batchRays[i]);
                    const got = nearestHitRestart(bvh, prims, batchRays[i], cap);
                    if (brute === null) {
                        expect(`${name} cap=${cap} ray=${i}: ${got ? "hit" : "miss"}`).toBe(
                            `${name} cap=${cap} ray=${i}: miss`,
                        );
                        continue;
                    }
                    // distance is the gate (prim may differ on an exact tie)
                    expect(got).not.toBeNull();
                    expect((got as { t: number }).t).toBe(brute.t);
                }
            }
        }
    });

    test("any-hit matches brute force at every short-stack capacity", () => {
        for (const { name, prims } of scenes) {
            const bvh: Bvh2 = build(prims);
            const batchRays = batch(prims);
            for (let i = 0; i < batchRays.length; i++) {
                const brute = nearestHitBrute(prims, batchRays[i]);
                // tMax bracketing the nearest hit makes some rays occlude and some not —
                // a fixed huge tMax would only ever exercise the "occluded" branch
                const tMaxes =
                    brute === null
                        ? [1e30]
                        : [brute.t * 0.5, brute.t + 1e-4, brute.t * 2 + 1, 1e30];
                for (const tMax of tMaxes) {
                    const want = brute !== null && brute.t < tMax;
                    for (const cap of caps) {
                        const got = anyHitRestart(bvh, batchRays[i], tMax, cap);
                        expect(`${name} cap=${cap} ray=${i} tMax=${tMax}: ${got}`).toBe(
                            `${name} cap=${cap} ray=${i} tMax=${tMax}: ${want}`,
                        );
                    }
                }
            }
        }
    });
});
