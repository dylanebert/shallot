// Wires `checkReconstructionContinuity`/`maxGradientJump`/`syntheticField` (`reconstruction.ts`) into
// a gate — before this file they had no reader beyond the package barrel.
//
// `checkReconstructionContinuity` takes an authored `tolerance` parameter; deriving it here rather
// than picking a number is the whole point of this file. Uniform Catmull-Rom (tau=0.5) is C1 by
// algebraic identity (the two adjacent cubic pieces' derivatives match exactly AT a knot — verify by
// expanding `catmullRom1D`'s own coefficients at t=1 for one segment and t=0 for the next), so
// `maxGradientJump`'s finite-`h` reading is pure O(h) FINITE-DIFFERENCE truncation error, never a real
// discontinuity: halving `h` halves the reading. Measured exactly (`syntheticField(128)`, seed 1):
// h=2e-4 -> 4.9077e-3, h=1e-4 -> 2.4542e-3, h=5e-5 -> 1.2272e-3 — each step an exact 2x. Bilinear (C0)
// has a REAL derivative discontinuity at every texel boundary, so its reading is h-INDEPENDENT:
// 2.5842 at every one of those three steps.
//
// The tolerance is the coarse-`h` reading's own O(h) linear-convergence PREDICTION for the finer `h`
// `checkReconstructionContinuity` uses internally, plus a 5% headroom for floating-point noise in that
// prediction (not a fit — the ratio measures exactly 2x, so 5% is slack around an exact relationship,
// never a magic separation floor). This is what makes it discriminate the shipped C1 kernel from the
// C0 bracket at the SAME tolerance, with no separately authored constant for either side.
import { describe, expect, test } from "bun:test";
import {
    bicubicSample,
    bilinearSample,
    checkReconstructionContinuity,
    maxGradientJump,
    syntheticField,
} from "../src/reconstruction";

const N = 128;
const H_FINE = 1e-4; // matches `checkReconstructionContinuity`'s own default `maxGradientJump` step.
const H_COARSE = 2e-4;
const HEADROOM = 1.05; // slack around an exact O(h) relationship — see this file's header.

describe("checkReconstructionContinuity — derived tolerance from the O(h) truncation model", () => {
    const field = syntheticField(N);
    const coarseJump = maxGradientJump(bicubicSample, field, N, H_COARSE);
    const predictedFineJump = coarseJump * (H_FINE / H_COARSE);
    const tolerance = predictedFineJump * HEADROOM;

    test("bicubic (shipped, C1): the fine-h reading holds the coarse-h reading's own O(h) prediction", () => {
        const finding = checkReconstructionContinuity("bicubic", bicubicSample, N, tolerance);
        expect(finding.ok, finding.message).toBe(true);
    });

    test("RED-WITNESS — bilinear (C0) breaks the same derived tolerance", () => {
        // re-runs the guarded arm's own comparison with only the subject (the reconstruction kernel)
        // mutated to bilinearSample, against the SAME tolerance — a real discontinuity does not
        // shrink with h, so it exceeds an O(h)-derived bound at any practical step.
        const finding = checkReconstructionContinuity(
            "bilinear (RED-WITNESS)",
            bilinearSample,
            N,
            tolerance,
        );
        expect(finding.ok, finding.message).toBe(false);
    });

    test("the O(h) relationship itself: bicubic's jump scales linearly with h, bilinear's does not", () => {
        const bicubicFine = maxGradientJump(bicubicSample, field, N, H_FINE);
        const bicubicCoarse = maxGradientJump(bicubicSample, field, N, H_COARSE);
        expect(bicubicFine / bicubicCoarse).toBeCloseTo(H_FINE / H_COARSE, 3);

        const bilinearFine = maxGradientJump(bilinearSample, field, N, H_FINE);
        const bilinearCoarse = maxGradientJump(bilinearSample, field, N, H_COARSE);
        expect(bilinearFine / bilinearCoarse).toBeCloseTo(1, 6);
    });
});
