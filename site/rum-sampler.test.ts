import { describe, expect, test } from "bun:test";
import {
    type FrameSamplerState,
    initialFrameSamplerState,
    type SlowFrameReport,
    sampleFrame,
} from "./rum-sampler";

// Red-first: witnessed red before `sampleFrame` reported anything (stub returning `report: null`
// unconditionally) — "reports exactly the >=50ms deltas" failed with `received length 0` against
// `expected length 2`, and "no report below threshold" passed vacuously, confirming the stub
// stubbed to the wrong side of the property before the real threshold comparison was restored.

function run(timestamps: number[]): { reports: SlowFrameReport[]; state: FrameSamplerState } {
    let state = initialFrameSamplerState;
    const reports: SlowFrameReport[] = [];
    for (const t of timestamps) {
        const result = sampleFrame(state, t);
        state = result.state;
        if (result.report) reports.push(result.report);
    }
    return { reports, state };
}

describe("sampleFrame", () => {
    test("reports exactly the >=50ms deltas", () => {
        // deltas: 16, 80 (slow), 16, 60 (slow)
        const { reports } = run([0, 16, 96, 112, 172]);
        expect(reports).toHaveLength(2);
        expect(reports[0].duration).toBe(80);
        expect(reports[1].duration).toBe(60);
    });

    test("first frame never reports (no prior timestamp)", () => {
        const { reports } = run([1000]);
        expect(reports).toHaveLength(0);
    });

    test("no report for deltas below threshold", () => {
        const { reports } = run([0, 16, 32, 48, 64]);
        expect(reports).toHaveLength(0);
    });

    test("a delta exactly at the threshold reports", () => {
        const { reports } = run([0, 50]);
        expect(reports).toHaveLength(1);
        expect(reports[0].duration).toBe(50);
    });

    test("context: startTime is the frame's own start, duration matches the delta", () => {
        const { reports } = run([100, 100 + 70]);
        expect(reports[0].startTime).toBe(100);
        expect(reports[0].duration).toBe(70);
        expect(reports[0].context.duration).toBe(70);
    });

    test("context: msSinceLoad is the frame-end timestamp", () => {
        const { reports } = run([100, 200]);
        expect(reports[0].context.msSinceLoad).toBe(200);
    });

    test("context: framesObserved counts every frame seen so far, including this one", () => {
        const { reports } = run([0, 16, 32, 112]); // 4th frame (delta 80) is slow
        expect(reports[0].context.framesObserved).toBe(4);
    });

    test("context: rollingMedianIntervalMs reflects prior steady-state pacing, not the slow delta itself", () => {
        // steady 16ms frames, then one 100ms stall
        const timestamps = [0, 16, 32, 48, 64, 164];
        const { reports } = run(timestamps);
        expect(reports).toHaveLength(1);
        expect(reports[0].context.rollingMedianIntervalMs).toBe(16);
    });

    test("context: rollingMedianIntervalMs is 0 when no prior interval exists", () => {
        const { reports } = run([0, 80]);
        expect(reports[0].context.rollingMedianIntervalMs).toBe(0);
    });
});
