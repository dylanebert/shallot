// I1 — the mode-placement oracle's replacement parity witness. The spike's mode-labeling oracle had
// two legs: (a) a single labelled mode's realized energy lands on its label, not its Nyquist image,
// checked at one even and one odd N; (b) cross-instrument agreement between the spectral Jacobian and
// finite-difference-on-reconstruction. Leg (a)'s odd-N case dies at power-of-two N (I1's Locked
// decision) — this file is the stated replacement, never a silent narrowing: it is the one instrument
// that still catches the checkerboard class the spec's Residue names ("A spectrum labelled in
// centered order and transformed with unshifted phase emerges modulated by a full-amplitude
// `(-1)^(x+y)` Nyquist checkerboard that no label-space instrument can see. The cheap detector is a
// lag-1 autocorrelation against the oversampling ratio the declared band implies.").
//
// Derivation, not a fit: for ANY unshifted-phase FFT (`kIndex`'s own labeling — `k=(i<=N/2?i:i-N)*dk`
// is exactly the frequency that phase convention realizes, a pure algebraic identity of periodicity,
// e^{i2πx(k-N)/N} = e^{i2πxk/N} for integer x), the discrete Wiener-Khinchin theorem gives the
// realized spatial field's lag-1 autocorrelation EXACTLY as the energy-weighted average of
// `cos(k(x)·dx)` over every mode, weighted by that mode's own realized power `|H(k)|²` — no free
// parameter, no round number, computed once from the SAME `H(k,t)` array the realized spatial field
// (`idft2(h, N)`) comes from. `measuredLag1AutocorrX` (the real, transform-computed autocorrelation of
// the actual spatial field) and the closed-form spectral-domain sum `lag1AutocorrParityWitness`
// computes inline must agree to numeric precision whenever the SAME labeling convention drives both
// the spectral mask and the transform's own phase — which is exactly the invariant the checkerboard
// defect breaks: a centered-order label used to build `h0`/weight `predicted` while the transform
// still realizes `kIndex`'s own unshifted phase makes `predicted` sum the WRONG per-bin cosine against
// each bin's REAL realized frequency, producing a large, discriminating disagreement (the red-witness
// below).

import { idft2, updateH } from "./cpu-reference";
import { type CascadeConfig, kIndex, philips } from "./spectrum";

export type LabelFn = (i: number, N: number) => number;

/** the pre-S3c centered-order label — a literal fixture, kept ONLY to red-witness this file's own
 *  detector (production never reads this function; every real kernel uses `kIndex`). */
export function centeredLabelPreFix(i: number, N: number): number {
    return i - N / 2;
}

/** `generateH0` parameterized over the label convention used to place a mode's energy — production
 *  always calls this with `kIndex` (equivalent to `generateH0` itself); `centeredLabelPreFix` is the
 *  red-witness fixture. */
function generateH0WithLabel(cfg: CascadeConfig, labelFn: LabelFn): Float32Array {
    const N = cfg.N;
    const dk = (2 * Math.PI) / cfg.L;
    const h0 = new Float32Array(N * N * 2);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const kx = labelFn(x, N) * dk;
            const kz = labelFn(y, N) * dk;
            const idx = (y * N + x) * 2;
            const kMag = Math.sqrt(kx * kx + kz * kz);
            if (kMag < cfg.kLo || kMag > cfg.kHi) continue;
            const p = philips(kx, kz, cfg);
            const sqrtP = Math.sqrt(Math.max(p, 0)) / Math.SQRT2;
            // deterministic (not the hash-based draw) — this fixture only needs a repeatable,
            // non-degenerate field, not a physically-drawn one.
            const seed = (x * 928371 + y * 121933) >>> 0;
            const xiR = (seed % 2000) / 1000 - 1;
            const xiI = ((seed >>> 8) % 2000) / 1000 - 1;
            h0[idx] = xiR * sqrtP;
            h0[idx + 1] = xiI * sqrtP;
        }
    }
    return h0;
}

/** the realized spatial field's own lag-1 autocorrelation along x, averaged over every row —
 *  measured directly from `idft2`'s real output, no label anywhere in this function. */
export function measuredLag1AutocorrX(height: Float64Array | Float32Array, N: number): number {
    let mean = 0;
    for (let i = 0; i < N * N; i++) mean += height[i];
    mean /= N * N;
    let num = 0;
    let den = 0;
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const a = height[y * N + x] - mean;
            const b = height[y * N + ((x + 1) % N)] - mean;
            num += a * b;
            den += a * a;
        }
    }
    return den > 0 ? num / den : 0;
}

export interface ParityWitnessReading {
    predicted: number;
    measured: number;
    relDiff: number;
}

/**
 * The parity witness: builds `h0` with `labelFn` (production: `kIndex`), evolves it through the REAL
 * production `updateH` (always `kIndex` — the transform's own phase never changes) at `t=0`,
 * transforms it through the REAL production `idft2`, then compares `measuredLag1AutocorrX` (the
 * actual spatial autocorrelation) against a closed-form spectral prediction weighted by `labelFn`.
 * `labelFn = kIndex` (the default) is what production ships; passing `centeredLabelPreFix` red-
 * witnesses the detector against the exact defect class the FFT swap could have reintroduced.
 */
export function lag1AutocorrParityWitness(
    cfg: CascadeConfig,
    labelFn: LabelFn = kIndex,
): ParityWitnessReading {
    const N = cfg.N;
    const L = cfg.L;
    const dx = L / N;
    const h0 = generateH0WithLabel(cfg, labelFn);
    const h = updateH(h0, N, L, 0); // production kernel — always kIndex internally
    const height = idft2(h, N); // production transform — the one true phase convention

    const realHeight = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) realHeight[i] = height[i * 2];
    const measured = measuredLag1AutocorrX(realHeight, N);

    // predicted: energy-weighted cos(k(x)·dx) over h's own realized power |H(k)|², k assigned via
    // labelFn — the closed-form Wiener-Khinchin sum this file's docblock derives.
    const dk = (2 * Math.PI) / L;
    let num = 0;
    let den = 0;
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const idx = y * N + x;
            const energy = h[idx * 2] * h[idx * 2] + h[idx * 2 + 1] * h[idx * 2 + 1];
            if (energy <= 0) continue;
            const kx = labelFn(x, N) * dk;
            num += energy * Math.cos(kx * dx);
            den += energy;
        }
    }
    const predicted = den > 0 ? num / den : 0;
    const denom = Math.max(Math.abs(predicted), Math.abs(measured), 1e-9);
    const relDiff = Math.abs(predicted - measured) / denom;
    return { predicted, measured, relDiff };
}
