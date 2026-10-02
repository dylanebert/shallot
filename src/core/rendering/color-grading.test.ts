import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { gradingWeights } from "./color-grading";

test("lower crossover weighs shadows, not highlights; both crossovers are continuous", () => {
    const range = d.vec2f(0.2, 0.7);
    const weights = gradingWeights(0.15, range);
    expect(weights.x).toBeCloseTo(0.75);
    expect(weights.y).toBeCloseTo(0.25);
    expect(weights.z).toBe(0);
    for (const cutoff of [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]) {
        const left = gradingWeights(cutoff - 1e-7, range);
        const right = gradingWeights(cutoff + 1e-7, range);
        for (const lane of ["x", "y", "z"] as const)
            expect(Math.abs(left[lane] - right[lane])).toBeLessThan(2e-6);
        expect(left.x + left.y + left.z).toBeCloseTo(1);
    }
});
