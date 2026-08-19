import { describe, expect, test } from "bun:test";
import { generateNetwork } from "../overlay/network";
import { makePermutation } from "./noise";
import {
    buildPolylineProfile,
    heightAtCpu,
    longitudinalOracle,
    PROFILE_STEP,
    type ProfilePoint,
} from "./profile";

// The longitudinal profile pipeline's own gate — the spec's Validation criterion added by the release
// look: "sample the flattened height along each centreline at <= the vertex spacing and bound the first
// difference (grade, against a cited road-design limit)." Pure CPU math, device-free
// (`terrain/profile.ts`'s module header).

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

describe("buildPolylineProfile — the straight chord (stage 17)", () => {
    test("returns exactly the two polyline endpoints, no resampling — one segment per road", () => {
        const perm = makePermutation(1337);
        const points: [number, number][] = [
            [-100, 0],
            [100, 0],
        ];
        const profile = buildPolylineProfile(points, perm);
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
        const profile = buildPolylineProfile(points, perm);
        expect(profile[0].height).toBeCloseTo(heightAtCpu(-150, -80, perm), 9);
        expect(profile[1].height).toBeCloseTo(heightAtCpu(150, 90, perm), 9);
    });
});

/** reconstruction of stage 6's shipped (pre-stage-8) raw-profile behaviour: `flatten.ts`'s `networkCore`
 *  used to set a segment's target to `heightAt(cx, cz)`, the bare natural height at the exact projected
 *  centreline point — unfiltered. On the centreline itself (`coreDist <= 0`) the ease returns the target
 *  outright, so the flattened height sampled along the centreline *is* this: natural height at each
 *  <= PROFILE_STEP-spaced point, unfiltered. Not a call into `buildPolylineProfile` (that would just be
 *  testing the new code against itself) — a from-scratch sample, mirroring only the *shape* of stage 6's
 *  own target derivation. */
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
    test("the chord profile passes on every road of a real generated network, several seeds", () => {
        // stage 17: the chord is one straight segment per road — grade is constant along it and stays
        // under MAX_GRADE by measurement, not by a limiter.
        const perm = makePermutation(1337);
        for (const seed of [1, 42, 615, 9001]) {
            const doc = generateNetwork(seed);
            for (const line of doc.polylines) {
                const profile = buildPolylineProfile(line.points, perm);
                const check = longitudinalOracle(profile);
                expect(check.gradeOk).toBe(true);
            }
        }
    });

    test("stage 6's shipped raw-profile behaviour FAILS this oracle on the same network — the mutation the spec calls for, run, not reasoned about", () => {
        const perm = makePermutation(1337);
        let checked = 0;
        let sawGradeFailure = false;
        for (const seed of [1, 42, 615, 9001, 271828]) {
            const doc = generateNetwork(seed);
            for (const line of doc.polylines) {
                const raw = rawCenterlineProfile(line.points, perm);
                const check = longitudinalOracle(raw);
                checked++;
                if (!check.gradeOk) sawGradeFailure = true;
            }
        }
        expect(checked).toBeGreaterThan(0);
        // the grade axis independently demonstrates a violation somewhere in the sweep, proving the oracle
        // discriminates on the axis it checks (not trivially always-green).
        expect(sawGradeFailure).toBe(true);
    });
});
