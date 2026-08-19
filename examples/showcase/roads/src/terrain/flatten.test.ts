import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../../../packages/shallot/tests/wgsl";
import { generateNetwork } from "../overlay/network";
import {
    buildNetworkGeometry,
    computeFalloff,
    FALLOFF_SAMPLE_SEGMENTS,
    flattenHeight,
    flattenHeightGpu,
    flattenWgsl,
    SIDE_SLOPE_LIMIT,
} from "./flatten";
import { SPACING } from "./grid";
import { mulberry32 } from "./noise";
import { MAX_GRADE, PROFILE_STEP } from "./profile";

// FALLOFF is no longer a fixed module constant (stage 8's re-derivation, `flatten.ts`'s module header) —
// these pure-math tests exercise `flattenHeight`/`flattenHeightGpu` at an arbitrary representative
// falloff, since the formula's own correctness doesn't depend on which falloff a network happened to
// produce.
const TEST_FALLOFF = 4;

// The flatten formula's own gate — the spec's Validation criterion, "Flattening — oracle: centerline
// height + monotone falloff". Pure cosine-ease math (`flatten.ts`'s module header), device-free.

describe("flattenHeight — the CPU reference", () => {
    test("at or inside the core boundary (coreDist <= 0), the target height wins outright — the centerline case", () => {
        expect(flattenHeight(100, 10, 0, TEST_FALLOFF)).toBe(10);
        expect(flattenHeight(100, 10, -0.001, TEST_FALLOFF)).toBe(10);
        expect(flattenHeight(100, 10, -50, TEST_FALLOFF)).toBe(10); // deep inside a wide road's core
    });

    test("at or past the falloff distance, fully natural — unmodified terrain", () => {
        expect(flattenHeight(100, 10, TEST_FALLOFF, TEST_FALLOFF)).toBe(100);
        expect(flattenHeight(100, 10, TEST_FALLOFF + 0.001, TEST_FALLOFF)).toBe(100);
        expect(flattenHeight(100, 10, 1000, TEST_FALLOFF)).toBe(100);
    });

    test("monotone across the falloff band, moving from target toward natural with no overshoot", () => {
        const natural = 40;
        const target = -10;
        const steps = 50;
        let prev = flattenHeight(natural, target, 0, TEST_FALLOFF);
        expect(prev).toBe(target);
        for (let i = 1; i <= steps; i++) {
            const coreDist = (i / steps) * TEST_FALLOFF;
            const h = flattenHeight(natural, target, coreDist, TEST_FALLOFF);
            expect(h).toBeGreaterThanOrEqual(prev - 1e-9); // monotone: never steps back toward target
            expect(h).toBeLessThanOrEqual(Math.max(natural, target) + 1e-9); // never overshoots either endpoint
            expect(h).toBeGreaterThanOrEqual(Math.min(natural, target) - 1e-9);
            prev = h;
        }
        expect(prev).toBeCloseTo(natural, 9);
    });

    test("monotonicity holds with natural below target too (a cutting, not a fill)", () => {
        const natural = -20;
        const target = 5;
        let prev = flattenHeight(natural, target, 0, TEST_FALLOFF);
        for (let i = 1; i <= 20; i++) {
            const coreDist = (i / 20) * TEST_FALLOFF;
            const h = flattenHeight(natural, target, coreDist, TEST_FALLOFF);
            expect(h).toBeLessThanOrEqual(prev + 1e-9); // moving downward toward natural, monotonically
            prev = h;
        }
    });

    test("natural === target collapses the whole band to one constant height", () => {
        for (const coreDist of [-5, 0, TEST_FALLOFF / 2, TEST_FALLOFF, TEST_FALLOFF * 3]) {
            expect(flattenHeight(7, 7, coreDist, TEST_FALLOFF)).toBe(7);
        }
    });
});

describe("differential oracle — flattenHeightGpu vs flattenHeight", () => {
    test("two independently written cosine-ease derivations agree over random inputs", () => {
        const rng = mulberry32(11);
        const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
        for (let i = 0; i < 200; i++) {
            const natural = rand(-40, 40);
            const target = rand(-40, 40);
            const falloff = rand(1, 20);
            const coreDist = rand(-falloff, falloff * 2);
            const cpu = flattenHeight(natural, target, coreDist, falloff);
            const gpu = flattenHeightGpu(natural, target, coreDist, falloff);
            expect(Math.abs(cpu - gpu)).toBeLessThan(1e-4); // f32 roundoff only, same geometric quantity
        }
    });
});

describe("flattenedHeightAt — WGSL structural resolve", () => {
    const wgsl = flattenWgsl();

    test("resolves the natural-height, network-geometry, and cosine-ease graph", () => {
        for (const helper of ["flattenedHeightAt", "flattenHeightGpu", "networkCore", "heightAt"]) {
            expect(wgsl).toContain(`fn ${helper}(`);
        }
        expect(wgsl.match(/var<storage/g)?.length).toBeGreaterThanOrEqual(3); // segments + polygons + polyVerts
    });

    test("integer discipline holds over the network-geometry loops", () => {
        // scoped to this module's own function (never the whole resolved graph): `heightAt`'s own
        // dependency chain (noise.ts's `perlin2`) legitimately uses i32 for its lattice hash, which
        // `generate.test.ts`'s equivalent check also scopes past by reading only the kernel body.
        integerDiscipline(body(flat(wgsl), "fn networkCore("));
    });
});

describe("flattenedHeightAt — degenerate empty network", () => {
    test("with no network buffers bound this is a JS-level concern only — networkCore's own sentinel\n     (f32-max coreDist) is what degrades an empty document to plain heightAt, pinned structurally: an\n     always-natural coreDist never satisfies coreDist <= 0 nor moves the blend off 'natural'", () => {
        // flattenHeight itself, at the sentinel coreDist networkCore returns when no primitive exists,
        // returns exactly the natural height regardless of target — the degradation this module's header
        // comment promises.
        const sentinel = 3.402823e38;
        expect(flattenHeight(12.5, 0, sentinel, TEST_FALLOFF)).toBe(12.5);
    });
});

describe("computeFalloff — the re-derived FALLOFF (stage 8), sampling floor (stage 11)", () => {
    test("floors at FALLOFF_SAMPLE_SEGMENTS * SPACING when the cut depth is zero — a transition narrower\n     than the mesh's own reconstruction can't read as a curve", () => {
        expect(computeFalloff(0)).toBe(FALLOFF_SAMPLE_SEGMENTS * SPACING);
        expect(computeFalloff(-5)).toBe(FALLOFF_SAMPLE_SEGMENTS * SPACING); // nonsensical input, still floors
    });

    test("the sampling floor is strictly wider than the old one-quad floor it replaces", () => {
        expect(FALLOFF_SAMPLE_SEGMENTS * SPACING).toBeGreaterThan(SPACING);
    });

    test("past the floor, the cosine ease's peak slope equals SIDE_SLOPE_LIMIT by construction", () => {
        for (const cutDepth of [10, 40, 100]) {
            const falloff = computeFalloff(cutDepth);
            expect(falloff).toBeGreaterThan(FALLOFF_SAMPLE_SEGMENTS * SPACING);
            // flattenHeight's ease derivative peaks at coreDist = falloff / 2 (t = 0.5): d/dCoreDist of
            // targetHeight + (natural - targetHeight) * (0.5 - 0.5 cos(pi t)), t = coreDist / falloff.
            const eps = 1e-4;
            const mid = falloff / 2;
            const h0 = flattenHeight(cutDepth, 0, mid - eps, falloff);
            const h1 = flattenHeight(cutDepth, 0, mid + eps, falloff);
            const peakSlope = (h1 - h0) / (2 * eps);
            expect(peakSlope).toBeCloseTo(SIDE_SLOPE_LIMIT, 3);
        }
    });

    test("monotone in cut depth — a deeper cut always widens the transition, never narrows it", () => {
        expect(computeFalloff(80)).toBeGreaterThan(computeFalloff(40));
        expect(computeFalloff(40)).toBeGreaterThan(computeFalloff(20));
    });

    test("scale multiplies the derived result — the live handover's own entry point", () => {
        expect(computeFalloff(0, 2)).toBe(computeFalloff(0) * 2);
        expect(computeFalloff(40, 1.5)).toBe(computeFalloff(40) * 1.5);
        expect(computeFalloff(0)).toBe(computeFalloff(0, 1)); // default scale is a no-op
    });
});

describe("FALLOFF_SAMPLE_SEGMENTS — the sampling-derived floor's own justification", () => {
    // The observable this pins: sampling flattenHeight's own ease at N+1 evenly spaced vertex positions
    // and reading the sign pattern of the piecewise-linear reconstruction's slopes. This is the property
    // FALLOFF_SAMPLE_SEGMENTS is chosen to guarantee, not the ease formula's arithmetic restated.
    function sampledSlopes(segments: number, falloff: number): number[] {
        const step = falloff / segments;
        const heights: number[] = [];
        for (let i = 0; i <= segments; i++) {
            heights.push(flattenHeight(1, 0, i * step, falloff)); // natural=1, target=0
        }
        const slopes: number[] = [];
        for (let i = 0; i < segments; i++) slopes.push(heights[i + 1] - heights[i]);
        return slopes;
    }

    test("at 2 segments the sampled slopes are identical — the ease's own point symmetry collapses the\n     reconstruction to a straight line, indistinguishable from no easing at all", () => {
        const slopes = sampledSlopes(2, TEST_FALLOFF);
        expect(slopes.length).toBe(2);
        expect(slopes[0]).toBeCloseTo(slopes[1], 10);
    });

    test("at FALLOFF_SAMPLE_SEGMENTS (4) each of the ease's two monotone arcs gets its own interior\n     sample, so both arcs individually show non-constant slope rather than just the whole curve", () => {
        const slopes = sampledSlopes(FALLOFF_SAMPLE_SEGMENTS, TEST_FALLOFF);
        expect(slopes.length).toBe(4);
        // arc 1 (segments 0-1, t in [0, 0.5]) and arc 2 (segments 2-3, t in [0.5, 1]) each split into two
        // sub-segments with different slopes — the concave arc's own curvature and the convex arc's own
        // curvature both register, not just an asymmetry between the halves.
        expect(Math.abs(slopes[1] - slopes[0])).toBeGreaterThan(1e-6); // arc 1 shows curvature on its own
        expect(Math.abs(slopes[3] - slopes[2])).toBeGreaterThan(1e-6); // arc 2 shows curvature on its own
        expect(slopes[0]).toBeCloseTo(slopes[3], 10); // point-symmetric about the midpoint, as expected
        expect(slopes[1]).toBeCloseTo(slopes[2], 10);
    });
});

describe("buildNetworkGeometry — the CPU profile-to-segment builder", () => {
    test("every polyline resamples to <= PROFILE_STEP-spaced sub-segments spanning its own endpoints", () => {
        const doc = generateNetwork(42);
        const { segments } = buildNetworkGeometry(doc, 1337, 3);
        expect(segments.length).toBeGreaterThan(doc.polylines.length); // each 2-point road subdivides
        for (const seg of segments) {
            const len = Math.hypot(seg.bx - seg.ax, seg.bz - seg.az);
            expect(len).toBeLessThanOrEqual(PROFILE_STEP + 1e-6);
        }
    });

    test("consecutive sub-segments of one road chain endpoint to endpoint, no gap", () => {
        const doc = generateNetwork(7);
        const { segments: roadSegs } = buildNetworkGeometry(doc, 1337, 0);
        // segments are emitted per-polyline in order (buildNetworkGeometry's own loop) — walk that same
        // order and check each polyline's own run chains endpoint to endpoint.
        let i = 0;
        for (const line of doc.polylines) {
            const [[ax, az], [bx, bz]] = line.points;
            const len = Math.hypot(bx - ax, bz - az);
            const n = Math.max(1, Math.ceil(len / PROFILE_STEP));
            for (let s = 0; s < n; s++) {
                const seg = roadSegs[i++];
                if (s > 0) {
                    const prev = roadSegs[i - 2];
                    expect(seg.ax).toBeCloseTo(prev.bx, 9);
                    expect(seg.az).toBeCloseTo(prev.bz, 9);
                    expect(seg.aHeight).toBeCloseTo(prev.bHeight, 9);
                }
            }
        }
    });

    test("grade never exceeds MAX_GRADE along any road's own smoothed profile, at radius 0 (grade-clamp alone)", () => {
        const doc = generateNetwork(615); // stage 6's pinned regression seed — still a good stress case
        const { segments } = buildNetworkGeometry(doc, 1337, 0);
        for (const seg of segments) {
            const len = Math.hypot(seg.bx - seg.ax, seg.bz - seg.az);
            if (len < 1e-6) continue;
            const grade = Math.abs(seg.bHeight - seg.aHeight) / len;
            expect(grade).toBeLessThanOrEqual(MAX_GRADE + 1e-6);
        }
    });

    test("cut depth is a non-negative real measurement, not a network-independent worst case", () => {
        const flat = buildNetworkGeometry({ polylines: [], polygons: [] }, 1337, 3);
        expect(flat.cutDepth).toBe(0); // no polylines, nothing to cut
        const doc = generateNetwork(42);
        const withRoads = buildNetworkGeometry(doc, 1337, 3);
        expect(withRoads.cutDepth).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(withRoads.cutDepth)).toBe(true);
    });
});
