import { describe, test, expect } from "bun:test";
import { valueToNorm, normToValue } from "../../../examples/workstation/src/presets";

const EPS = 1e-9;

describe("knob math: linear scale", () => {
    test("endpoints map to 0 and 1", () => {
        expect(valueToNorm(0, 0, 100, "linear")).toBeCloseTo(0, 12);
        expect(valueToNorm(100, 0, 100, "linear")).toBeCloseTo(1, 12);
    });

    test("midpoint is arithmetic mean", () => {
        expect(valueToNorm(50, 0, 100, "linear")).toBeCloseTo(0.5, 12);
    });

    test("step quantization rounds to nearest step", () => {
        // norm 0.123 over [0, 100] = 12.3, rounded to step 1 = 12
        expect(normToValue(0.123, 0, 100, "linear", 1)).toBeCloseTo(12, 12);
        // step 0.01 preserves 12.34
        expect(normToValue(0.1234, 0, 100, "linear", 0.01)).toBeCloseTo(12.34, 12);
    });
});

describe("knob math: log scale", () => {
    const min = 0.25;
    const max = 4;

    test("endpoints map to 0 and 1", () => {
        expect(valueToNorm(min, min, max, "log")).toBeCloseTo(0, 12);
        expect(valueToNorm(max, min, max, "log")).toBeCloseTo(1, 12);
    });

    test("geometric midpoint sits at norm 0.5", () => {
        // sqrt(0.25 * 4) = 1.0
        expect(valueToNorm(1, min, max, "log")).toBeCloseTo(0.5, 12);
        expect(normToValue(0.5, min, max, "log", 0.01)).toBeCloseTo(1, 12);
    });

    test("round-trip preserves value (no quantization on log)", () => {
        // bug regression: linear-step rounding on a log knob produces non-uniform jumps
        const samples = [0.25, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0];
        for (const v of samples) {
            const back = normToValue(valueToNorm(v, min, max, "log"), min, max, "log", 0.01);
            expect(back).toBeCloseTo(v, 10);
        }
    });

    test("equal norm steps produce equal multiplicative ratios", () => {
        // first-principles: a log-scale knob's defining property is that uniform
        // dragging produces uniform multiplicative change. linear-step rounding
        // breaks this. measure ratios across the full range and assert they're equal.
        const N = 32;
        const ratios: number[] = [];
        let prev = normToValue(0, min, max, "log", 0.01);
        for (let i = 1; i <= N; i++) {
            const v = normToValue(i / N, min, max, "log", 0.01);
            ratios.push(v / prev);
            prev = v;
        }
        // expected ratio: (max/min)^(1/N) = 16^(1/32) = 2^(1/8)
        const expected = Math.pow(max / min, 1 / N);
        for (const r of ratios) {
            expect(r).toBeCloseTo(expected, 10);
        }
    });

    test("monotonic across the range", () => {
        let prev = -Infinity;
        for (let i = 0; i <= 100; i++) {
            const v = normToValue(i / 100, min, max, "log", 0.01);
            expect(v).toBeGreaterThan(prev);
            prev = v;
        }
    });
});

describe("knob math: log scale, frequency range", () => {
    // filter cutoff range — verifies the fix at low frequencies where the
    // linear-step bug was most audible (1 Hz step at 20 Hz = 5% jumps)
    const min = 20;
    const max = 22050;

    test("a 1% norm change near min produces a multiplicative ratio, not a 1 Hz jump", () => {
        const v0 = normToValue(0, min, max, "log", 1);
        const v1 = normToValue(0.01, min, max, "log", 1);
        const v2 = normToValue(0.02, min, max, "log", 1);
        // multiplicative ratio v1/v0 must equal v2/v1 (log uniformity)
        const r01 = v1 / v0;
        const r12 = v2 / v1;
        expect(r12 / r01).toBeCloseTo(1, 8);
        // and the ratio itself should be > 1 (not stuck at v0 due to linear rounding)
        expect(r01).toBeGreaterThan(1 + EPS);
    });

    test("near max, ratio is the same as near min", () => {
        const v0 = normToValue(0.0, min, max, "log", 1);
        const v1 = normToValue(0.01, min, max, "log", 1);
        const v98 = normToValue(0.98, min, max, "log", 1);
        const v99 = normToValue(0.99, min, max, "log", 1);
        const rLow = v1 / v0;
        const rHigh = v99 / v98;
        expect(rLow).toBeCloseTo(rHigh, 10);
    });
});
