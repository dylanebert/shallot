// The mode-placement oracle checks the transform's realized frequency against its declared labels.
//
// Leg (a) — mode-placement: does a single labelled mode's realized energy land on its label, not its
// Nyquist image? The spike ran this at one even N (64) and one odd N (81); the odd-N case dies with
// power-of-two N (I1's Locked decision — every shipped cascade N is now a power of two). Both shipped
// cascade N (64, 128) are exercised here instead — still two distinct N, still a real cross-N check.
//
// Leg (b) — the parity witness (`../src/parity-witness.ts`) checks the realized raster against its
// declared mode labels. Production (`kIndex`) must read a small disagreement. The centered-order
// fixture is run in both the original defective shape (centered placement and prediction) and an
// independently mismatched prediction; both must read a large disagreement so the witness cannot
// pass merely because one callback drives every side of the comparison.
import { describe, expect, test } from "bun:test";
import { idft2, updateH } from "../src/cpu-reference";
import { lag1AutocorrParityWitness } from "../src/parity-witness";
import { CASCADE_CONFIGS, kIndex } from "../src/spectrum";

const PI = Math.PI;

function centeredLabelPreFix(i: number, N: number): number {
    return i - N / 2;
}

function worldForTexel(x: number, N: number, L: number): number {
    return ((x + 0.5) / N - 0.5) * L;
}

type LabelFn = (i: number, N: number) => number;

function nearestIndexForLabel(labelFn: LabelFn, N: number, dk: number, target: number): number {
    let best = 0;
    let bestDiff = Number.POSITIVE_INFINITY;
    for (let i = 0; i < N; i++) {
        const diff = Math.abs(labelFn(i, N) * dk - target);
        if (diff < bestDiff) {
            bestDiff = diff;
            best = i;
        }
    }
    return best;
}

function driveSingleMode(labelFn: LabelFn, N: number, L: number, target: number): Float32Array {
    const dk = (2 * PI) / L;
    const idx = nearestIndexForLabel(labelFn, N, dk, target);
    const h0 = new Float32Array(N * N * 2);
    h0[idx * 2] = 1; // real=1, imag=0
    const H = updateH(h0, N, L, 0);
    const height = idft2(H, N);
    const row = new Float32Array(N);
    for (let x = 0; x < N; x++) row[x] = height[(0 * N + x) * 2]; // real part, row y=0
    return row;
}

function projectOnto(raster: Float32Array, N: number, L: number, k: number): number {
    let re = 0;
    let im = 0;
    for (let x = 0; x < N; x++) {
        const w = worldForTexel(x, N, L);
        const phase = -k * w;
        re += raster[x] * Math.cos(phase);
        im += raster[x] * Math.sin(phase);
    }
    return Math.sqrt(re * re + im * im) / N;
}

describe("mode-placement oracle — leg (a): both shipped power-of-two N", () => {
    for (const N of [64, 128]) {
        test(`N=${N}: production energy lands on the label, not its Nyquist image`, () => {
            const L = N === 64 ? 80 : 31; // the shipped cascades' own L (spectrum.ts's CASCADE_CONFIGS)
            const dk = (2 * PI) / L;
            const kNyquist = (N / 2) * dk;
            const target = 3 * dk;

            const prodRaster = driveSingleMode(kIndex, N, L, target);
            const onLabel = projectOnto(prodRaster, N, L, target);
            const onImage = Math.max(
                projectOnto(prodRaster, N, L, target + kNyquist),
                projectOnto(prodRaster, N, L, target - kNyquist),
            );
            expect(onLabel).toBeGreaterThan(onImage * 4);
        });

        test(`N=${N}: red-witness — the pre-fix centered label lands energy on the Nyquist image instead`, () => {
            const L = N === 64 ? 80 : 31;
            const dk = (2 * PI) / L;
            const kNyquist = (N / 2) * dk;
            const target = 3 * dk;

            const fixtureRaster = driveSingleMode(centeredLabelPreFix, N, L, target);
            const onLabel = projectOnto(fixtureRaster, N, L, target);
            const onImage = Math.max(
                projectOnto(fixtureRaster, N, L, target + kNyquist),
                projectOnto(fixtureRaster, N, L, target - kNyquist),
            );
            expect(onImage).toBeGreaterThan(onLabel);
        });
    }
});

describe("mode-placement oracle — leg (b): the lag-1 autocorrelation parity witness (I1's replacement for the odd-N leg)", () => {
    for (const cfg of CASCADE_CONFIGS) {
        test(`cascade N=${cfg.N}: production labeling (kIndex) predicts the realized field's own autocorrelation`, () => {
            const reading = lag1AutocorrParityWitness(cfg, kIndex, kIndex);
            // production is the identity case of its own derivation (Wiener-Khinchin, exact up to
            // f32/f64 rounding through updateH+idft2) — a tight bound is the honest one, not a fit.
            expect(reading.relDiff).toBeLessThan(0.01);
        });

        test(`cascade N=${cfg.N}: RED-WITNESS — production placement against centered prediction exposes the checkerboard class`, () => {
            const reading = lag1AutocorrParityWitness(cfg, kIndex, centeredLabelPreFix);
            expect(reading.relDiff).toBeGreaterThan(0.5);
        });
    }
});
