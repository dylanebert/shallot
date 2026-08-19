import { describe, expect, test } from "bun:test";
import { generateNetwork } from "../overlay/network";
import { makePermutation } from "./noise";
import {
    boxFilter,
    buildPolylineProfile,
    clampGrade,
    DEFAULT_SMOOTH_RADIUS,
    heightAtCpu,
    longitudinalOracle,
    MAX_GRADE,
    MAX_SMOOTH_RADIUS,
    MIN_SMOOTH_RADIUS,
    PROFILE_STEP,
    type ProfilePoint,
} from "./profile";

// The longitudinal profile pipeline's own gate — the spec's Validation criterion added by the release
// look: "sample the flattened height along each centreline at <= the vertex spacing and bound both the
// first difference (grade, against a cited road-design limit) and the second difference (jitter)." Pure
// CPU math, device-free (`terrain/profile.ts`'s module header).

describe("heightAtCpu — the CPU noise mirror", () => {
    test("agrees with the GPU heightAt at the one analytically-known point (0, 0) — zero gradient at every octave", () => {
        const perm = makePermutation(1337);
        // stroke.ts's own header comment: heightAt(0, 0) === GROUND_LEVEL regardless of seed, since (0,0)
        // sits exactly on the perlin lattice where every gradient contributes zero.
        expect(heightAtCpu(0, 0, perm)).toBeCloseTo(0, 6);
    });

    test("deterministic in seed, off-lattice — same perm gives the same height, a different seed almost certainly doesn't", () => {
        const permA = makePermutation(1337);
        const permB = makePermutation(1337);
        expect(heightAtCpu(37.5, -12.25, permA)).toBe(heightAtCpu(37.5, -12.25, permB));
        const permC = makePermutation(7);
        expect(heightAtCpu(37.5, -12.25, permA)).not.toBe(heightAtCpu(37.5, -12.25, permC));
    });
});

describe("boxFilter", () => {
    test("radius 0 is a no-op copy", () => {
        const h = [1, 5, 2, 9, 3];
        expect(boxFilter(h, 0)).toEqual(h);
    });

    test("averages a spike toward its neighbours, clamped at the sequence's own ends (no wrap/zero-pad)", () => {
        const h = [0, 0, 0, 10, 0, 0, 0];
        const smoothed = boxFilter(h, 1);
        expect(smoothed[3]).toBeCloseTo(10 / 3, 9); // (0 + 10 + 0) / 3
        expect(smoothed[0]).toBeCloseTo(0 / 2, 9); // end clamp: only [h[0], h[1]] average in, not a 3-wide window
    });

    test("a constant sequence is unchanged by any radius", () => {
        const h = new Array(10).fill(5);
        for (const r of [0, 1, 4, 8]) expect(boxFilter(h, r)).toEqual(h);
    });
});

describe("clampGrade", () => {
    test("clips a spike that violates the grade bound, example from the docstring", () => {
        expect(clampGrade([0, 10], [1], 1)).toEqual([0, 1]);
    });

    test("every adjacent pair satisfies |delta| <= maxGrade * step, for arbitrary noisy input", () => {
        const rng = (() => {
            let s = 99;
            return () => {
                s = (s * 1103515245 + 12345) & 0x7fffffff;
                return s / 0x7fffffff;
            };
        })();
        const heights = Array.from({ length: 60 }, () => rng() * 80 - 40);
        const step = 4;
        const steps = new Array(heights.length - 1).fill(step);
        const maxGrade = 0.12;
        const graded = clampGrade(heights, steps, maxGrade);
        for (let i = 1; i < graded.length; i++) {
            expect(Math.abs(graded[i] - graded[i - 1])).toBeLessThanOrEqual(maxGrade * step + 1e-9);
        }
    });

    test("leaves an already-compliant profile untouched", () => {
        const heights = [0, 0.4, 0.8, 1.2, 1.6];
        const steps = [4, 4, 4, 4];
        expect(clampGrade(heights, steps, 0.12)).toEqual(heights); // exactly at the grade limit each step
    });
});

describe("buildPolylineProfile — the straight chord (stage 17)", () => {
    test("returns exactly the two polyline endpoints, no resampling — one segment per road", () => {
        const perm = makePermutation(1337);
        const points: [number, number][] = [
            [-100, 0],
            [100, 0],
        ];
        const profile = buildPolylineProfile(points, perm, DEFAULT_SMOOTH_RADIUS);
        expect(profile).toHaveLength(2);
        expect(profile[0].x).toBeCloseTo(-100, 9);
        expect(profile[0].z).toBeCloseTo(0, 9);
        expect(profile[1].x).toBeCloseTo(100, 9);
        expect(profile[1].z).toBeCloseTo(0, 9);
    });

    test("endpoint heights are the natural terrain heights there (heightAtCpu), not smoothed or grade-limited", () => {
        const perm = makePermutation(9001);
        const points: [number, number][] = [
            [-150, -80],
            [150, 90],
        ];
        const profile = buildPolylineProfile(points, perm, DEFAULT_SMOOTH_RADIUS);
        expect(profile[0].height).toBeCloseTo(heightAtCpu(-150, -80, perm), 9);
        expect(profile[1].height).toBeCloseTo(heightAtCpu(150, 90, perm), 9);
    });

    test("the chord ignores the smoothRadius parameter — same output at any radius (plumbing retained, stage 19)", () => {
        const perm = makePermutation(2024);
        const points: [number, number][] = [
            [-110, 40],
            [110, -40],
        ];
        const atMin = buildPolylineProfile(points, perm, MIN_SMOOTH_RADIUS);
        const atMax = buildPolylineProfile(points, perm, MAX_SMOOTH_RADIUS);
        expect(atMin).toEqual(atMax);
    });
});

describe("the smoothing-strength control's own range (F9-idiom bracket-key control, boot.ts)", () => {
    test("MIN/MAX/DEFAULT are sane: MIN <= DEFAULT <= MAX, MIN non-negative", () => {
        expect(MIN_SMOOTH_RADIUS).toBeGreaterThanOrEqual(0);
        expect(DEFAULT_SMOOTH_RADIUS).toBeGreaterThanOrEqual(MIN_SMOOTH_RADIUS);
        expect(DEFAULT_SMOOTH_RADIUS).toBeLessThanOrEqual(MAX_SMOOTH_RADIUS);
    });
});

/** reconstruction of stage 6's shipped (pre-stage-8) raw-profile behaviour: `flatten.ts`'s `networkCore`
 *  used to set a segment's target to `heightAt(cx, cz)`, the bare natural height at the exact projected
 *  centreline point — no low-pass, no grade limit. On the centreline itself (`coreDist <= 0`) the ease
 *  returns the target outright, so the flattened height sampled along the centreline *is* this: natural
 *  height at each <= PROFILE_STEP-spaced point, unfiltered. Not a call into `buildPolylineProfile` (that
 *  would just be testing the new code against itself) — a from-scratch resample + sample, mirroring only
 *  the *shape* of stage 6's own target derivation. */
function rawCenterlineProfile(
    points: ReadonlyArray<readonly [number, number]>,
    perm: Uint32Array,
): ProfilePoint[] {
    const raw: { x: number; z: number }[] = [{ x: points[0][0], z: points[0][1] }];
    for (let i = 0; i < points.length - 1; i++) {
        const [ax, az] = points[i];
        const [bx, bz] = points[i + 1];
        const len = Math.hypot(bx - ax, bz - az);
        const n = Math.max(1, Math.ceil(len / PROFILE_STEP));
        for (let s = 1; s <= n; s++) {
            const t = s / n;
            raw.push({ x: ax + (bx - ax) * t, z: az + (bz - az) * t });
        }
    }
    return raw.map((p) => ({ x: p.x, z: p.z, height: heightAtCpu(p.x, p.z, perm) }));
}

describe("longitudinalOracle — the spec's own Validation criterion, proven by mutation", () => {
    test("catches a V-shaped kink (steep up, then equally steep down) — a signed-grade check, not magnitude", () => {
        // |grade| is identical on both sides of a sharp reversal, so a jitter check built from unsigned
        // grade magnitudes would read this as zero change even though the profile visibly kinks. Every
        // step here individually satisfies MAX_GRADE (0.12), so only the jitter axis can catch it.
        const step = PROFILE_STEP;
        const g = MAX_GRADE * 0.8;
        const profile: ProfilePoint[] = [
            { x: 0, z: 0, height: 0 },
            { x: step, z: 0, height: g * step },
            { x: step * 2, z: 0, height: 0 }, // reverses direction at full opposite grade
        ];
        const check = longitudinalOracle(profile);
        expect(check.gradeOk).toBe(true); // neither individual step exceeds MAX_GRADE
        expect(check.jitterOk).toBe(false); // but the grade change (2g) exceeds MAX_GRADE_BREAK
    });

    test("the chord profile passes on every road of a real generated network, several seeds", () => {
        // stage 17: the chord is one straight segment per road — grade is constant along it (so jitter is
        // zero by construction, one step) and stays under MAX_GRADE by measurement, not by a limiter.
        const perm = makePermutation(1337);
        for (const seed of [1, 42, 615, 9001]) {
            const doc = generateNetwork(seed);
            for (const line of doc.polylines) {
                const profile = buildPolylineProfile(line.points, perm, DEFAULT_SMOOTH_RADIUS);
                const check = longitudinalOracle(profile);
                expect(check.gradeOk).toBe(true);
                expect(check.jitterOk).toBe(true);
            }
        }
    });

    test("stage 6's shipped raw-profile behaviour FAILS this oracle on the same network — the mutation the spec calls for, run, not reasoned about", () => {
        const perm = makePermutation(1337);
        let checked = 0;
        let sawGradeFailure = false;
        let sawJitterFailure = false;
        for (const seed of [1, 42, 615, 9001, 271828]) {
            const doc = generateNetwork(seed);
            for (const line of doc.polylines) {
                const raw = rawCenterlineProfile(line.points, perm);
                const check = longitudinalOracle(raw);
                checked++;
                if (!check.gradeOk) sawGradeFailure = true;
                if (!check.jitterOk) sawJitterFailure = true;
                // the combined verdict (what "passes the oracle" means) fails for every road at every
                // seed — not a cherry-picked instance.
                expect(check.gradeOk && check.jitterOk).toBe(false);
            }
        }
        expect(checked).toBeGreaterThan(0);
        // both axes independently demonstrate a violation somewhere in the sweep too, proving the oracle
        // discriminates on each axis rather than one axis carrying the other.
        expect(sawGradeFailure).toBe(true);
        expect(sawJitterFailure).toBe(true);
    });
});
