// The parity witness checks that declared mode labels agree with the frequency realized by the
// unshifted inverse transform. It compares a raster measurement with an energy-weighted
// Wiener-Khinchin prediction. The H0 placement label and prediction label are separate inputs so a
// mismatched pair is observable rather than being made equal by one callback.

import { idft2, updateH } from "./cpu-reference";
import { type CascadeConfig, generateH0, type LabelFn } from "./spectrum";

export type { LabelFn } from "./spectrum";

/** centered mode labels used only as the deliberate checkerboard red-witness prediction. */
export function centeredLabelPreFix(i: number, N: number): number {
    return i - N / 2;
}

/** the realized spatial field's own lag-1 autocorrelation along x, averaged over every row. */
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

/** measures the realized lag-1 autocorrelation against a prediction using an independent label.
 * Both labels are required so a caller cannot accidentally use one callback for the measured and
 * predicted sides, making the witness vacuous. */
export function lag1AutocorrParityWitness(
    cfg: CascadeConfig,
    h0Label: LabelFn,
    predictLabel: LabelFn,
): ParityWitnessReading {
    const N = cfg.N;
    const L = cfg.L;
    const dx = L / N;
    const h0 = generateH0(cfg, h0Label);
    const h = updateH(h0, N, L, 0);
    const height = idft2(h, N);

    const realHeight = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) realHeight[i] = height[i * 2];
    const measured = measuredLag1AutocorrX(realHeight, N);

    // The closed-form Wiener-Khinchin prediction uses a label independent from H0 placement.
    const dk = (2 * Math.PI) / L;
    let num = 0;
    let den = 0;
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const idx = y * N + x;
            const energy = h[idx * 2] * h[idx * 2] + h[idx * 2 + 1] * h[idx * 2 + 1];
            if (energy <= 0) continue;
            const kx = predictLabel(x, N) * dk;
            num += energy * Math.cos(kx * dx);
            den += energy;
        }
    }
    const predicted = den > 0 ? num / den : 0;
    const denom = Math.max(Math.abs(predicted), Math.abs(measured), 1e-9);
    const relDiff = Math.abs(predicted - measured) / denom;
    return { predicted, measured, relDiff };
}
