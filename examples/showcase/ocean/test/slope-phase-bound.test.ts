import { expect, test } from "bun:test";
import { phaseInputPerturbationNorms } from "../src/verification/phase-bound";

test("phase-error L2 propagation uses the exact per-mode spectrum operator norm", () => {
    const h0 = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const phaseErrorL2 = 0.25;
    const actual = phaseInputPerturbationNorms(h0, 2, 2 * Math.PI, phaseErrorL2);

    // For N=2 and L=2π, |k| is one on each nonzero axis. Mode (1,1) is its own
    // negative-index partner and has norm hypot(7,8,7,8)=sqrt(226), the maximum on both axes.
    // The unnormalized 2D inverse DFT contributes its L2 operator norm N=2.
    expect(actual.x).toBeCloseTo(2 * Math.sqrt(226) * phaseErrorL2, 12);
    expect(actual.z).toBeCloseTo(2 * Math.sqrt(226) * phaseErrorL2, 12);
});

test("inflating the measured phase-error norm inflates both propagated terms linearly", () => {
    const h0 = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const measured = phaseInputPerturbationNorms(h0, 2, 2 * Math.PI, 0.25);
    const inflated = phaseInputPerturbationNorms(h0, 2, 2 * Math.PI, 0.5);
    expect(inflated.x).toBe(measured.x * 2);
    expect(inflated.z).toBe(measured.z * 2);
});
