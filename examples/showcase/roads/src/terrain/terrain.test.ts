import { describe, expect, test } from "bun:test";
import { body, flat } from "../../../../../packages/shallot/tests/wgsl";
import { markingDistanceForSegment, type Segment } from "../overlay/document";
import { COVERAGE_BAND_PX, DIST_RANGE, TILE_SIZE, TILES_PER_SIDE } from "../overlay/tiles";
import { markingDistanceFromChord, terrainFsWgsl } from "./terrain";

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
