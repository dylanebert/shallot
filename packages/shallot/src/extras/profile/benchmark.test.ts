import { describe, expect, it } from "bun:test";
import {
    distribution,
    foldIndirect,
    INDIRECT_FLOOR_US,
    indirectFloorUs,
    passStats,
} from "./benchmark";

describe("distribution", () => {
    it("summarizes a sample array", () => {
        const d = distribution([30, 10, 50, 20, 40]);
        expect(d.avg).toBe(30);
        expect(d.median).toBe(30); // sorted[floor(5/2)] = sorted[2]
        expect(d.min).toBe(10);
        expect(d.max).toBe(50);
        expect(d.p5).toBe(10); // sorted[floor(5*0.05)] = sorted[0]
        expect(d.p95).toBe(50); // sorted[floor(5*0.95)] = sorted[4]
        expect(d.p99).toBe(50); // sorted[floor(5*0.99)] = sorted[4]
        // variance = (0² + 20² + 20² + 10² + 10²)/5 = 1000/5 = 200 → stddev = √200 ≈ 14.14
        expect(d.stddev).toBeCloseTo(14.14, 2);
    });

    // p99 separates from p95/max only with enough samples: a single high outlier in 100 lands at the
    // tail of both, but a steady array keeps stddev at 0 — the even-pacing signal.
    it("reads the spike tail and the variance", () => {
        const flat = Array(100).fill(5);
        const d0 = distribution(flat);
        expect(d0.stddev).toBe(0); // perfectly even — no variance to bound
        expect(d0.p99).toBe(5);

        const spiky = [...Array(99).fill(5), 100]; // one spike in 100
        const d1 = distribution(spiky);
        expect(d1.median).toBe(5);
        expect(d1.p95).toBe(5); // sorted[floor(100*0.95)] = sorted[95] — the spike is past the 95th rank
        expect(d1.p99).toBe(100); // sorted[floor(100*0.99)] = sorted[99] — p99 catches what p95 misses
        expect(d1.max).toBe(100);
        expect(d1.stddev).toBeGreaterThan(0);
    });

    it("is order-independent", () => {
        const a = distribution([1, 2, 3, 4]);
        const b = distribution([4, 3, 2, 1]);
        expect(a).toEqual(b);
    });
});

describe("passStats — mixed fixed/variable clock", () => {
    // a render pass fires once per frame (fires == frames): its per-occurrence cost IS its per-frame
    // cost, cadence 1, classified render.
    it("classifies a per-frame pass as render", () => {
        const p = passStats(10, 100, 100, 0.25);
        expect(p.occMs).toBe(0.1);
        expect(p.perFrameMs).toBe(0.1);
        expect(p.firesPerFrame).toBe(1);
        expect(p.clock).toBe("render");
    });

    // a sim pass fires once per fixed step — here 25 steps across 100 frames (0.25 steps/frame). The
    // per-STEP cost (occMs) is what climbs toward the fixed budget; the per-FRAME amortized cost is 4×
    // smaller. Reporting occMs separately is the "don't under-report the heavy step" guarantee.
    it("classifies a per-step pass as sim and keeps the per-step cost un-amortized", () => {
        const p = passStats(4.5, 25, 100, 0.25);
        expect(p.occMs).toBe(0.18); // 4.5 / 25 — the real per-step cost
        expect(p.perFrameMs).toBe(0.045); // 4.5 / 100 — amortized, 4× smaller
        expect(p.firesPerFrame).toBe(0.25);
        expect(p.clock).toBe("sim");
        // the invariant that makes both numbers honest: amortized = per-step × cadence
        expect(p.perFrameMs).toBeCloseTo(p.occMs * p.firesPerFrame, 6);
        // a sub-1 cadence means the per-step cost strictly exceeds the amortized cost
        expect(p.occMs).toBeGreaterThan(p.perFrameMs);
    });

    // the threshold adapts to the measured step rate: the split sits midway between steps/frame and 1,
    // so a pass at the fixed cadence reads sim and a per-frame pass reads render at any headless rate.
    it("splits at the midpoint between the step rate and 1", () => {
        const steps = 0.4;
        const sim = passStats(1, 40, 100, steps); // cadence 0.40 == steps/frame
        const render = passStats(1, 100, 100, steps); // cadence 1.0
        expect(sim.clock).toBe("sim");
        expect(render.clock).toBe("render");
    });

    it("reports zero cadence when the window spans no frames", () => {
        const p = passStats(1, 1, 0, 0.25);
        expect(p.firesPerFrame).toBe(0);
        expect(p.perFrameMs).toBe(0);
        expect(p.occMs).toBe(1); // per-occurrence is still defined (time / fires)
    });

    // with no per-occurrence sample list (the default), the percentile tail is 0 — backward-compatible
    // with every existing caller that omits it.
    it("defaults the per-occurrence percentiles to 0 without samples", () => {
        const p = passStats(10, 100, 100, 0.25);
        expect(p.occP95).toBe(0);
        expect(p.occP99).toBe(0);
    });

    // the percentile tail catches a spike the window-mean occMs averages away: 19 flat frames at 0.1 ms
    // plus one 1.0 ms spike → occMs ≈ 0.145, but occP99 = the spike. The stress-suite attribution reads
    // exactly this — a pass whose p99 ≫ its mean is the one that moved.
    it("captures a per-occurrence spike the window mean hides", () => {
        const samples = [...Array(19).fill(0.1), 1.0];
        const dTime = samples.reduce((s, v) => s + v, 0);
        const p = passStats(dTime, 20, 20, 1, samples);
        expect(p.occMs).toBeCloseTo(0.145, 3); // the mean — the spike is one part in 20
        expect(p.occP99).toBe(1); // sorted[floor(20*0.99)] = sorted[19] — the spike
        expect(p.occP99).toBeGreaterThan(p.occMs * 5); // the tail dwarfs the mean
    });
});

describe("indirect-draw validation floor", () => {
    // the floor is the issued-command count × the per-draw constant — the untimed cost a future pass that
    // grows its indirect-draw count silently re-inflates fence wait with (gpu.md "WebGPU-specific traps")
    it("derives the floor from the issued draw count", () => {
        expect(indirectFloorUs(124)).toBe(124 * INDIRECT_FLOOR_US);
        expect(indirectFloorUs(0)).toBe(0);
    });

    // name→count round-trip: each frame's per-pass tally folds into the cumulative counters one fire at a
    // time, so the benchmark recovers the per-frame count as count/fires (the window-diff unit), exactly
    // like the GPU pass timers (gpuTime / gpuFires).
    it("folds each frame's per-pass tally into the cumulative counters", () => {
        const count = new Map<string, number>();
        const fires = new Map<string, number>();
        foldIndirect(
            new Map([
                ["sear:color", 124],
                ["sear:pointshadow", 124],
            ]),
            count,
            fires,
        );
        foldIndirect(new Map([["sear:color", 124]]), count, fires);

        expect(count.get("sear:color")).toBe(248);
        expect(fires.get("sear:color")).toBe(2);
        expect(count.get("sear:pointshadow")).toBe(124);
        expect(fires.get("sear:pointshadow")).toBe(1);
        expect(count.get("sear:color")! / fires.get("sear:color")!).toBe(124);
    });

    it("leaves the counters untouched for an empty frame", () => {
        const count = new Map<string, number>();
        const fires = new Map<string, number>();
        foldIndirect(new Map(), count, fires);
        expect(count.size).toBe(0);
        expect(fires.size).toBe(0);
    });
});
