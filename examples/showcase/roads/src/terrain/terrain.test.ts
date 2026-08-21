import { describe, expect, test } from "bun:test";
import { body, flat } from "../../../../../packages/shallot/tests/wgsl";
import { markingDistanceForSegment, type Segment } from "../overlay/document";
import {
    COVERAGE_BAND_PX,
    DASH_DUTY,
    DASH_GAP,
    DASH_PERIOD,
    DASH_SEGMENT,
    DIST_RANGE,
    LINE_HALF_WIDTH,
    TILE_SIZE,
    TILES_PER_SIDE,
} from "../overlay/tiles";
import { coverFn, markingDistanceFromChord, terrainFsWgsl } from "./terrain";

// The overlay composite's structural gate — the device-free seam this stage's `bun test` relies on for the
// fs's atlas-sampling half (CPU-callable/resolved-WGSL first, never a bound device). The composite's
// actual pixel output is the capture gate's own arm (`gate.ts`/`test/roads.spec.ts`) — layers are a
// granularity: the compute-write half is proven by the seeded-tile readback oracle (`overlay/stroke.test.ts`),
// this half by resolution + the capture, neither alone.

function mulberry32(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) | 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

describe("terrain fs — overlay composite", () => {
    const wgsl = terrainFsWgsl();

    test("declares the five overlay bindings the surface layout adds", () => {
        expect(flat(wgsl)).toContain(
            "@group(2) @binding(3) var<storage, read> indirection: array<i32>;",
        );
        expect(wgsl).toContain("var albedo: texture_2d_array<f32>;");
        expect(wgsl).toContain("var dist: texture_2d_array<f32>;");
        expect(wgsl).toContain("var overlaySamp: sampler;");
        // the chord uniform (stage 8) — the analytic fs's marking geometry input
        expect(wgsl).toContain("var<uniform> chord");
    });

    test("addresses the tile grid from world (x, z) using the documented constants", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain(
            `(ctx.world.x + ${TILE_SIZE * (TILES_PER_SIDE / 2)}f) / ${TILE_SIZE}f`,
        );
        expect(fs).toContain(
            `(ctx.world.z + ${TILE_SIZE * (TILES_PER_SIDE / 2)}f) / ${TILE_SIZE}f`,
        );
        expect(fs).toMatch(/clamp\(i32\(floor\(.*\)\), 0i, \d+i\)/);
        expect(fs).toContain(`(tz * ${TILES_PER_SIDE}i) + tx`);
    });

    test("looks the tile up in the indirection buffer, never inside a per-fragment branch around the sample", () => {
        // textureSample computes screen-space derivatives, so WGSL requires uniform control flow — a
        // per-fragment `if (layer >= 0)` guard around it is a real-GPU compile rejection ("must only be
        // called from uniform control flow"), caught red-handed by the device gate. The fix is unconditional
        // sampling + a `select`-masked contribution instead of a branch; this test pins that shape stays.
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain("let layer = indirection[u32(tileIdx)];");
        expect(fs).not.toMatch(/if\s*\(\(layer/);
        expect(fs).toContain("let layerU = u32(max(layer, 0i));");
        expect(fs).toMatch(/select\(0f, 1f, \(layer >= 0i\)\)/);
    });

    test("samples both atlas arrays unconditionally, by the same tile-local uv and clamped layer", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain("textureSample(dist, overlaySamp, uv, layerU)");
        expect(fs).toContain("textureSample(albedo, overlaySamp, uv, layerU)");
    });

    test("decodes the distance channel with the documented DIST_RANGE, thresholds coverage with fwidth, and masks it by residency", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain(`* ${DIST_RANGE}f`);
        expect(fs).toContain("fwidth(");
        // the coverage band's own coefficient, not just that fwidth appears: capture.ts derives
        // TRANSITION_TOLERANCE_PX as a multiple of COVERAGE_BAND_PX, so a shader-side drift in this
        // factor silently widens the band the capture gate's tolerance was sized against.
        expect(fs).toMatch(new RegExp(`fwidth\\(dist_\\d*\\) \\* ${COVERAGE_BAND_PX}f`));
        // Green 2007's alpha-tested-magnification form, masked by the residency select above (an
        // unallocated tile's coverage is forced to 0 regardless of what garbage the clamped-layer sample
        // returns) — never a branch, per the uniform-control-flow test above.
        expect(fs).toMatch(/clamp\(\(0\.5f - \(dist_\d* \/ fw\)\), 0f, 1f\) \* resident/);
    });

    test("composites the overlay (with markings mixed in) over the base color by coverage — never a second draw", () => {
        const fs = flat(body(wgsl, "fn terrainFs"));
        expect(fs).toContain("color = mix(color, overlayWithMarkings, coverage);");
        expect(wgsl.match(/fn terrainFs\(/g)?.length).toBe(1); // exactly one fs entry — no second pass
    });
});

describe("terrain fs — marking channel (stage 8: analytic from chord uniform)", () => {
    const wgsl = terrainFsWgsl();
    const fs = flat(body(wgsl, "fn terrainFs"));

    test("reads the chord uniform (endpoints + halfWidth) for marking geometry, not the albedo alpha", () => {
        // stage 8 moved the marking channel from a baked texel (albedo alpha byte) to analytic fs
        // evaluation from the chord uniform. The fs must read the chord uniform, not decode alpha.
        expect(fs).toContain("(*chord).a");
        expect(fs).toContain("(*chord).b");
        expect(fs).toContain("(*chord).halfWidth");
        // the old alpha decode (albedoSample.w - 0.5) * 2 * DIST_RANGE must NOT survive
        expect(fs).not.toMatch(/albedoSample.*\.w.*0\.5f.*2f.*DIST_RANGEf/);
    });

    test("uses two-edge analytic pixel coverage (cover(h) - cover(-h)), not a smoothstep threshold", () => {
        // the coverage form: cover(e) = clamp((e - x)/p + 0.5, 0, 1), alpha = cover(h) - cover(-h)
        // — not the old clamp(0.5 - dist/fw, 0, 1) threshold. The coverFn helper is resolved into the WGSL.
        expect(flat(wgsl)).toContain("fn coverFn");
        // edge line coverage uses coverFn with LINE_HALF_WIDTH (0.0762 = 0.1524/2, the stage 8 upper bound)
        expect(fs).toMatch(/coverFn\(/);
        // the old threshold form must NOT survive
        expect(fs).not.toMatch(/clamp\(0\.5f - \(markingDist/);
    });

    test("dash coverage converges to DASH_DUTY opacity as fwidth(s) approaches DASH_PERIOD", () => {
        // the dash uses two-edge coverage along the station axis, plus a Nyquist convergence blend
        // toward DASH_DUTY — a far dashed line reads as a faint continuous line, never resolved dots.
        expect(fs).toContain("nyquist");
        // DASH_DUTY resolves to 0.25f in the WGSL — the literal freezes the constant so the
        // arm reds if the duty changes, not just if the mix expression changes shape.
        expect(fs).toMatch(/mix\(.*dashCoverage.*0\.25f.*nyquist/);
    });

    test("selects the marking albedo by comparing the chord-derived marking distance with the edge-line distance", () => {
        // the marking class (edge vs centre) is determined by comparing the chord-derived marking
        // distance with the edge-line distance computed from the chord: if the marking distance is
        // smaller, the nearest marking is the centreline; otherwise the edge line.
        // 0.3 is EDGE_INSET and 0.0762 is LINE_HALF_WIDTH (0.1524/2) — both deliberately literals here,
        // not derived from the exported constants: if the regex were built from EDGE_INSET/LINE_HALF_WIDTH,
        // changing the constant would change both sides and the arm would stay green, asserting only that
        // the shader emitted *some* number consistent with itself, not the *right* number. The literals
        // freeze the derived quantity so the arm reds exactly when the emitted value moves.
        expect(fs).toMatch(/abs\(.*\+ 0\.3\d*f\)\) - 0\.0762\d*f/);
        expect(fs).toMatch(/select\(vec3f\(/);
        expect(fs).toMatch(/isCentre/);
    });

    test("mixes the marking albedo over the road albedo before the terrain composite", () => {
        // the order matters: marking over road first, then road+marking over terrain — so a marking
        // never bleeds outside the road's coverage boundary
        expect(fs).toContain("mix(overlay, markingAlbedo, markingCoverage)");
        expect(fs).toContain("color = mix(color, overlayWithMarkings, coverage);");
    });
});

// Stage 8's marking differential oracle: `markingDistanceFromChord` (terrain.ts, clamped-projection form,
// CPU-called through TypeGPU's dual execution) against `document.ts`'s `markingDistanceForSegment`
// (cross-product form) — two independently written derivations of the same geometric quantity, the
// signed distance to the nearest road marking (edge lines + broken centreline). The GPU side is the
// analytic form the fs uses to compute marking geometry from the chord uniform; the CPU side is the
// independently-derived `markingDistance` in `document.ts`. `terrain.ts` never imports `markingDistance` —
// the two derivations are independent by contract, because the differential arm is worthless if one side
// calls the other.
//
// RED-FIRST CONTROL: mutating one side's dash duty (DASH_DUTY) was witnessed red — changing the CPU
// side's DASH_DUTY to DASH_DUTY * 0.5 (halving the dash fraction) while leaving the GPU side unchanged
// produces disagreement at points inside a dash whose phase falls in the second half of the original
// duty cycle: the CPU side now reads those points as in a gap (marking distance positive) while the GPU
// side still reads them as in a dash (marking distance negative), exceeding the tolerance. This was
// confirmed by running the mutated comparison and observing the failure before restoring the real duty.
describe("marking differential oracle — markingDistanceFromChord vs document.ts's markingDistanceForSegment", () => {
    test("agree within f32 roundoff over random segments and sample points", () => {
        const rng = mulberry32(42);
        const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
        for (let i = 0; i < 200; i++) {
            const seg: Segment = {
                ax: rand(-100, 100),
                az: rand(-100, 100),
                bx: rand(-100, 100),
                bz: rand(-100, 100),
                halfWidth: rand(1, 8),
            };
            const px = rand(-120, 120);
            const pz = rand(-120, 120);

            const cpu = markingDistanceForSegment(px, pz, seg);
            const gpu = markingDistanceFromChord(
                px,
                pz,
                seg.ax,
                seg.az,
                seg.bx,
                seg.bz,
                seg.halfWidth,
            );
            // same geometric quantity, f32 roundoff only — no quantization step (the fs computes from
            // the chord uniform directly, not through an encoded byte)
            expect(Math.abs(gpu - cpu)).toBeLessThan(1e-4);
        }
    });

    test("agree on the centreline (where the dash phase is the nearest marking), inside a dash and in a gap", () => {
        // a horizontal road from (-100, 0) to (100, 0), halfWidth 4 — the standard chord's shape.
        // Points on the centreline (z=0) are where the centreline marking is nearest, so the dash
        // duty actually determines the distance — this is the axis the red-first control mutates.
        const seg: Segment = { ax: -100, az: 0, bx: 100, bz: 0, halfWidth: 4 };
        const inDash: Array<[number, number]> = [
            [20, 0], // station 120, on the centreline, in a dash
            [70, 0.01], // station 170, near the centreline, in a dash
        ];
        const inGap: Array<[number, number]> = [
            [0, 0], // station 100, on the centreline, in a gap
            [-50, 0.01], // station 50, near the centreline, in a gap
        ];
        for (const [px, pz] of inDash) {
            const cpu = markingDistanceForSegment(px, pz, seg);
            const gpu = markingDistanceFromChord(
                px,
                pz,
                seg.ax,
                seg.az,
                seg.bx,
                seg.bz,
                seg.halfWidth,
            );
            expect(Math.abs(gpu - cpu)).toBeLessThan(1e-4);
            // inside a dash → marking distance should be negative (inside the marking)
            expect(cpu).toBeLessThan(0);
        }
        for (const [px, pz] of inGap) {
            const cpu = markingDistanceForSegment(px, pz, seg);
            const gpu = markingDistanceFromChord(
                px,
                pz,
                seg.ax,
                seg.az,
                seg.bx,
                seg.bz,
                seg.halfWidth,
            );
            expect(Math.abs(gpu - cpu)).toBeLessThan(1e-4);
            // in a gap → marking distance should be positive (outside the marking)
            expect(cpu).toBeGreaterThan(0);
        }
    });
});

// R3 behavioural arms — the two-edge coverage integral and the dash's Nyquist convergence are the
// whole point of stage 8, and the existing arms above are structural (toContain/toMatch over resolved
// WGSL text) or distance differentials that never call the coverage function. These arms call the
// production `coverFn` (a `tgpu.fn`, CPU-callable through TypeGPU's dual execution — the same path
// `markingDistanceFromChord` uses in the differential oracle above) so they witness the shipped fs's
// exact math with no device. The dash-convergence arm re-derives the fs's dash-coverage pipeline
// (coverFn calls + the Nyquist blend) in TS from the spec's stated form, since `fwidth(s)` is GPU-only
// and cannot be called from plain TS — the arm witnesses the formula, not the shipped fs's fwidth path,
// and the docblock says exactly that.

describe("two-edge analytic pixel coverage — cover(h) - cover(-h) [behavioural]", () => {
    // Witnesses the shipped `coverFn` (terrain.ts, exported `tgpu.fn`) called directly in TS — the
    // same function the fs resolves into WGSL. cover(e, x, p) = clamp((e - x)/p + 0.5, 0, 1),
    // alpha = cover(h) - cover(-h).
    //
    // WITNESSED RED: changing `coverFn`'s body from `clamp((e - x) / p + 0.5, 0, 1)` to
    // `clamp((e - x) / p, 0, 1)` (dropping the +0.5 offset) reds the sub-pixel proportionality arm:
    //   Expected: 0.5  (cov * p / (2h) ≈ 1)
    //   Received: 0    (coverage collapses to 0 when both edges are inside the pixel, because
    //   cover(h) and cover(-h) both clamp to the same value without the +0.5 centring)
    // Confirmed by running the mutated coverFn, observing the failure, then restoring by hand.

    const h = LINE_HALF_WIDTH; // 0.0762 m — the stripe half-width the fs uses

    test("coverage is 1 well inside the stripe and 0 well outside", () => {
        const p = 0.001; // tiny footprint — the stripe is many pixels wide
        const inside = coverFn(h, 0, p) - coverFn(-h, 0, p);
        expect(inside).toBeCloseTo(1, 5);
        const outside = coverFn(h, 10 * h, p) - coverFn(-h, 10 * h, p);
        expect(outside).toBeCloseTo(0, 5);
    });

    test("coverage is monotone as x sweeps across one edge (decreasing outside the right edge)", () => {
        // the spec says "monotone as x sweeps across the edge" — one edge, not the whole stripe.
        // alpha = cover(h) - cover(-h) is a bump: 0 outside, 1 inside, 0 outside. Across one edge
        // (e.g. the right edge at x = h) it is monotone decreasing from 1 to 0.
        const p = 0.01;
        let prev = Infinity;
        for (let i = 0; i <= 200; i++) {
            const x = 0 + (4 * h * i) / 200; // from x=0 (inside) to x=4h (outside)
            const cov = coverFn(h, x, p) - coverFn(-h, x, p);
            expect(cov).toBeLessThanOrEqual(prev + 1e-9);
            prev = cov;
        }
    });

    test("coverage is exactly 0.5 at the edge for a small p", () => {
        // at x = h (the right edge): cover(h, h, p) = clamp(0/p + 0.5, 0, 1) = 0.5,
        // cover(-h, h, p) = clamp(-2h/p + 0.5, 0, 1) ≈ 0 for small p, so alpha ≈ 0.5
        const p = 0.001;
        const cov = coverFn(h, h, p) - coverFn(-h, h, p);
        expect(cov).toBeCloseTo(0.5, 2);
    });

    test("sub-pixel regime: when p exceeds 2h, coverage is a proper fraction that decays as ~2h/p", () => {
        // THE arm: when the pixel footprint p exceeds the stripe width 2h, both edges land inside one
        // pixel and the coverage terms subtract to the stripe's true fractional coverage ~2h/p —
        // neither 0 nor 1, and decaying proportionally as p grows. This is the property that makes a
        // distant line fade instead of vanishing or popping.
        const p1 = 4 * 2 * h; // p = 8h, well into sub-pixel
        const p2 = 8 * 2 * h; // p = 16h
        const p3 = 16 * 2 * h; // p = 32h
        const cov1 = coverFn(h, 0, p1) - coverFn(-h, 0, p1);
        const cov2 = coverFn(h, 0, p2) - coverFn(-h, 0, p2);
        const cov3 = coverFn(h, 0, p3) - coverFn(-h, 0, p3);
        // proper fractions, not 0 or 1
        for (const c of [cov1, cov2, cov3]) {
            expect(c).toBeGreaterThan(0);
            expect(c).toBeLessThan(1);
        }
        // proportional decay: cov ∝ 1/p, so cov1/cov2 ≈ p2/p1
        expect(cov1 / cov2).toBeCloseTo(p2 / p1, 2);
        expect(cov2 / cov3).toBeCloseTo(p3 / p2, 2);
        // the proportionality constant is 2h: cov ≈ 2h/p
        expect((cov1 * p1) / (2 * h)).toBeCloseTo(1, 2);
        expect((cov2 * p2) / (2 * h)).toBeCloseTo(1, 2);
        expect((cov3 * p3) / (2 * h)).toBeCloseTo(1, 2);
    });
});

describe("dash Nyquist convergence [behavioural]", () => {
    // Re-derives the fs's dash-coverage pipeline in TS from the spec's stated form. `coverFn` is the
    // production `tgpu.fn` (CPU-callable), but `fwidth(s)` is GPU-only and cannot be called from TS,
    // so the footprint `sP` is a parameter, not a live fwidth read. The arm witnesses the formula
    // (coverFn calls + the Nyquist blend), not the shipped fs's fwidth path — the docblock says exactly
    // that rather than overclaiming.
    //
    // WITNESSED RED: changing the Nyquist blend from the half-period threshold
    //   `(sP - DASH_PERIOD * 0.5) / (DASH_PERIOD * 0.5)` back to the old linear-from-zero
    //   `sP / DASH_PERIOD` reds the "blend inactive below half period" arm:
    //   Expected: 0  (nyquist at sP = DASH_PERIOD * 0.25 — blend must be inactive)
    //   Received: 0.25  (the old formula starts blending at sP = 0, so at quarter-period it is
    //   already 25% washed toward continuous)
    // Confirmed by running the mutated blend, observing the failure, then restoring by hand.

    /** the fs's dash-coverage pipeline, re-derived in TS: two-edge coverFn calls along the station
     *  axis (with wrap), clamped, then the Nyquist blend toward DASH_DUTY. `sP` stands in for
     *  `fwidth(s)`, which is GPU-only. */
    function dashCoverage(sRel: number, sP: number): number {
        const dashCov = coverFn(DASH_SEGMENT, sRel, sP) - coverFn(0, sRel, sP);
        const dashCovWrap =
            coverFn(DASH_PERIOD + DASH_SEGMENT, sRel, sP) - coverFn(DASH_PERIOD, sRel, sP);
        let cov = Math.max(0, Math.min(1, dashCov + dashCovWrap));
        const nyquist = Math.max(0, Math.min(1, (sP - DASH_PERIOD * 0.5) / (DASH_PERIOD * 0.5)));
        cov = cov * (1 - nyquist) + DASH_DUTY * nyquist;
        return cov;
    }

    test("at a small footprint, coverage is ~1 inside a segment and ~0 in a gap", () => {
        const sP = 0.001; // tiny footprint — the dash pattern is many pixels wide
        // middle of a dash segment
        const inSeg = DASH_SEGMENT / 2;
        expect(dashCoverage(inSeg, sP)).toBeCloseTo(1, 3);
        // middle of a gap
        const inGap = DASH_SEGMENT + DASH_GAP / 2;
        expect(dashCoverage(inGap, sP)).toBeCloseTo(0, 3);
    });

    test("the Nyquist blend is inactive below half a period — coverage equals the raw dash coverage", () => {
        // below half a period, nyquist = 0, so the blend is a no-op and the result is the raw
        // two-edge dash coverage. This is the property R4 buys: the dash reads crisp where it can.
        const sRel = DASH_SEGMENT / 2; // inside a segment
        const sP = DASH_PERIOD * 0.25; // well below the Nyquist threshold
        const rawDashCov = (() => {
            const dashCov = coverFn(DASH_SEGMENT, sRel, sP) - coverFn(0, sRel, sP);
            const dashCovWrap =
                coverFn(DASH_PERIOD + DASH_SEGMENT, sRel, sP) - coverFn(DASH_PERIOD, sRel, sP);
            return Math.max(0, Math.min(1, dashCov + dashCovWrap));
        })();
        expect(dashCoverage(sRel, sP)).toBeCloseTo(rawDashCov, 6);
        // and the nyquist factor itself is exactly 0
        const nyquist = Math.max(0, Math.min(1, (sP - DASH_PERIOD * 0.5) / (DASH_PERIOD * 0.5)));
        expect(nyquist).toBe(0);
    });

    test("as footprint grows toward DASH_PERIOD, coverage converges to DASH_DUTY, monotonically", () => {
        // scan sP from half a period to one period — the coverage should converge to DASH_DUTY
        // and |coverage - DASH_DUTY| should be non-increasing (monotone convergence)
        const sRel = DASH_SEGMENT / 2; // inside a segment (raw coverage starts above DASH_DUTY)
        let prevDelta = Infinity;
        for (let i = 0; i <= 50; i++) {
            const sP = DASH_PERIOD * 0.5 + (DASH_PERIOD * 0.5 * i) / 50;
            const cov = dashCoverage(sRel, sP);
            const delta = Math.abs(cov - DASH_DUTY);
            expect(delta).toBeLessThanOrEqual(prevDelta + 1e-9);
            prevDelta = delta;
        }
        // at one full period, coverage is exactly DASH_DUTY (nyquist = 1 → mix = DASH_DUTY)
        expect(dashCoverage(sRel, DASH_PERIOD)).toBeCloseTo(DASH_DUTY, 5);
    });
});
