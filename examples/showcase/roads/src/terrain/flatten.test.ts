import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../../../packages/shallot/tests/wgsl";
import { generateNetwork } from "../overlay/network";
import {
    buildNetworkGeometry,
    computeFalloff,
    flattenHeight,
    flattenHeightGpu,
    flattenWgsl,
    SIDE_SLOPE_LIMIT,
} from "./flatten";
import { SPACING } from "./grid";
import { mulberry32 } from "./noise";
import { MAX_GRADE } from "./profile";

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
        expect(wgsl.match(/var<storage/g)?.length).toBeGreaterThanOrEqual(1); // segments
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

describe("computeFalloff — the re-derived FALLOFF (stage 8)", () => {
    test("floors at SPACING when the cut depth is zero — a transition narrower than the mesh can't resolve", () => {
        expect(computeFalloff(0)).toBe(SPACING);
        expect(computeFalloff(-5)).toBe(SPACING); // a negative cut depth is nonsensical input, still floors
    });

    test("past the floor, the cosine ease's peak slope equals SIDE_SLOPE_LIMIT by construction", () => {
        for (const cutDepth of [1, 4, 10, 40]) {
            const falloff = computeFalloff(cutDepth);
            expect(falloff).toBeGreaterThan(SPACING);
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
        expect(computeFalloff(20)).toBeGreaterThan(computeFalloff(10));
        expect(computeFalloff(10)).toBeGreaterThan(computeFalloff(5));
    });
});

describe("buildNetworkGeometry — the CPU profile-to-segment builder", () => {
    test("every polyline produces exactly one chord segment spanning its own endpoints (stage 17)", () => {
        // the chord profile returns exactly two profile points per polyline (the endpoints), so
        // `buildNetworkGeometry` emits one segment per road — no resampling, no sub-segments.
        const doc = generateNetwork();
        const { segments } = buildNetworkGeometry(doc, 1337);
        expect(segments.length).toBe(doc.polylines.length);
        for (const [i, line] of doc.polylines.entries()) {
            const seg = segments[i];
            const [[ax, az], [bx, bz]] = line.points;
            expect(seg.ax).toBeCloseTo(ax, 9);
            expect(seg.az).toBeCloseTo(az, 9);
            expect(seg.bx).toBeCloseTo(bx, 9);
            expect(seg.bz).toBeCloseTo(bz, 9);
        }
    });

    test("each road's single chord segment spans its own polyline endpoints — no sub-segment chain", () => {
        // stage 17: the chord is one segment per road, so there's no chain to check — each segment's
        // endpoints ARE the polyline's endpoints. This pin verifies that directly.
        const doc = generateNetwork();
        const { segments } = buildNetworkGeometry(doc, 1337);
        expect(segments.length).toBe(doc.polylines.length);
        for (const [i, line] of doc.polylines.entries()) {
            const seg = segments[i];
            const [[ax, az], [bx, bz]] = line.points;
            expect(seg.ax).toBeCloseTo(ax, 9);
            expect(seg.az).toBeCloseTo(az, 9);
            expect(seg.bx).toBeCloseTo(bx, 9);
            expect(seg.bz).toBeCloseTo(bz, 9);
        }
    });

    test("segments are emitted road-contiguously — one segment per road, road field non-decreasing", () => {
        // The GPU mirror (`networkCore` in `flatten.ts`) groups sub-segments into their owning road
        // by detecting `seg.road` changes in the segment array — it relies on segments being emitted
        // contiguously per road. The CPU mirror (`networkCoreCpu` in `flatness.ts`) uses a Map keyed
        // by `road`, so it does not rely on contiguity — the invariant is load-bearing on one side
        // only. Stage 17: one segment per road, so the invariant is trivially true — the `road` field
        // is strictly increasing (0, 1, 2, ..., ROAD_COUNT-1). Stage 1 (`roads-interactive.md`)
        // deleted route selection: `generateNetwork` always returns the one fixed standard chord, so
        // the seed scan this arm used to run over is gone with it — there is exactly one document to
        // check.
        const doc = generateNetwork();
        const { segments } = buildNetworkGeometry(doc, 1337);
        expect(segments.length).toBe(doc.polylines.length);
        for (let i = 1; i < segments.length; i++) {
            expect(segments[i].road).toBeGreaterThanOrEqual(segments[i - 1].road);
        }
    });

    test("grade stays under MAX_GRADE along any road's chord — by measurement, not by a limiter (stage 17)", () => {
        // stage 17: the chord has no grade limiter — the grade is the natural chord grade between the
        // endpoint heights. It stays under MAX_GRADE by measurement. Stage 1 (`roads-interactive.md`)
        // replaced route selection with the one fixed standard chord, so this reads that measurement
        // on the standard chord against perm 1337's terrain.
        const doc = generateNetwork();
        const { segments } = buildNetworkGeometry(doc, 1337);
        for (const seg of segments) {
            const len = Math.hypot(seg.bx - seg.ax, seg.bz - seg.az);
            if (len < 1e-6) continue;
            const grade = Math.abs(seg.bHeight - seg.aHeight) / len;
            expect(grade).toBeLessThanOrEqual(MAX_GRADE + 1e-6);
        }
    });

    test("cut depth is a non-negative real measurement, not a network-independent worst case", () => {
        const flat = buildNetworkGeometry({ polylines: [] }, 1337);
        expect(flat.cutDepth).toBe(0); // no polylines, nothing to cut
        const doc = generateNetwork();
        const withRoads = buildNetworkGeometry(doc, 1337);
        expect(withRoads.cutDepth).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(withRoads.cutDepth)).toBe(true);
    });
});
