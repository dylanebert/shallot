import { describe, expect, test } from "bun:test";
import { State, type System } from "../../engine";

// Scheduler-driven validation of the camera follow (PlayerSnapshotSystem + followPos), against the REAL
// scheduler, no GPU. A value that advances on the FIXED clock, snapshotted into prev/curr in the `fixed`
// group and read as lerp(prev, curr, fixedAlpha) in the sim group, renders at constant velocity across an
// irregular render rate. The naive alternative this replaced — reading the latest fixed value every render
// frame — steps at the fixed rate (the jitter). This is the property that makes the follow render-rate-
// independent; it would go red if the snapshot moved off the fixed group or the lerp dropped fixedAlpha.

function makeRng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
    };
}

describe("camera follow interpolation (scheduler)", () => {
    test("fixed-clock snapshot + fixedAlpha renders constant velocity across an irregular render rate", () => {
        const V = 6; // m/s — the swept pose moves at constant velocity, so ideal camera velocity is constant
        const FixedDt = 1 / 60;
        const state = new State();

        let pose = 0; // the swept pose, advances one step per fixed tick
        let prev = 0;
        let curr = 0;
        let init = false;
        const lerpCam: number[] = [];
        const naiveCam: number[] = [];
        const dts: number[] = [];

        state.addSystem({
            group: "fixed",
            update: () => {
                pose += V * FixedDt;
            },
        } satisfies System);
        // the fix (PlayerSnapshotSystem): snapshot prev/curr on the fixed clock, after the step
        state.addSystem({
            group: "fixed",
            last: true,
            update: () => {
                if (!init) {
                    prev = curr = pose;
                    init = true;
                } else {
                    prev = curr;
                    curr = pose;
                }
            },
        } satisfies System);
        // the camera reads both: the interpolated follow (followPos) + the naive latest-value read it replaces
        state.addSystem({
            group: "simulation",
            update: (s) => {
                lerpCam.push(init ? prev + (curr - prev) * s.time.fixedAlpha : 0);
                naiveCam.push(pose);
            },
        } satisfies System);

        // drive at an irregular high frame rate (the measured WSL case: ~238fps, dt cv ~12%, 0.6/8.7ms outliers)
        const rng = makeRng(1);
        for (let f = 0; f < 800; f++) {
            let dt = (1 / 238) * (1 + 0.4 * (rng() * 2 - 1));
            if (rng() < 0.05) dt = 0.0006;
            if (rng() < 0.05) dt = 0.0087;
            dts.push(dt);
            state.step(dt);
        }

        const vel = (cam: number[]) => cam.slice(1).map((x, i) => (x - cam[i]) / dts[i + 1]);
        const lv = vel(lerpCam).slice(150); // steady state, past the ring fill
        const nv = vel(naiveCam).slice(150);

        // the fix: constant velocity every frame (the pose is linear, so the lerp is exact — f64 roundoff only)
        for (const v of lv) {
            expect(v).toBeGreaterThan(0); // forward only
            expect(v).toBeCloseTo(V, 4); // constant speed at any render rate
        }
        // the naive per-frame read it replaces is NOT smooth: it holds (0 velocity) whole frames then spikes
        // a full fixed step when a tick lands — the stutter the interpolation removes
        expect(Math.min(...nv)).toBe(0);
        expect(Math.max(...nv)).toBeGreaterThan(V * 2);

        state.dispose();
    });
});
