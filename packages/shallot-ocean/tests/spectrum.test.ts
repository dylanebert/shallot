import { describe, expect, test } from "bun:test";
import {
    CASCADE_CONFIGS,
    declaredBandVariance,
    deriveFoldBand,
    generateH0,
    SEA_STATE,
    spectralCellAmplitude,
    spectrumNormalization,
    unifiedSpectrum,
    whitecapFraction,
} from "../src/spectrum";

describe("physical spectrum normalization", () => {
    test("restores the spectral-cell factor: the same density scales with Δk", () => {
        const kx = 0.4;
        const kz = 0.2;
        const cfg80 = { ...CASCADE_CONFIGS[0], L: 80, kLo: 0.01, kHi: 2 };
        const cfg40 = { ...cfg80, L: 40 };
        const a80 = spectralCellAmplitude(cfg80, kx, kz);
        const a40 = spectralCellAmplitude(cfg40, kx, kz);
        expect(a80 / a40).toBeCloseTo(0.5, 5);
    });

    test("declared-band variance is N-invariant for the same physical patch", () => {
        const low = { ...CASCADE_CONFIGS[0], N: 64, L: 80 };
        const high = { ...low, N: 128 };
        // Both resolutions contain every mode in the declared band because their Nyquist limits are
        // above kHi; summing the normalized density is therefore exactly N-independent.
        expect(declaredBandVariance(low)).toBeCloseTo(declaredBandVariance(high), 12);
        expect(spectrumNormalization([low])).toBeGreaterThan(0);
        expect(generateH0(low).length).toBe(low.N * low.N * 2);
    });

    test("one shared wind and sea-state target are explicit", () => {
        expect(SEA_STATE.significantWaveHeight).toBe(3);
        expect(SEA_STATE.windSpeed).toBe(15);
        expect(CASCADE_CONFIGS.every((config) => !("windSpeed" in config))).toBe(true);
        expect(CASCADE_CONFIGS.every((config) => !("amplitude" in config))).toBe(true);
        expect(CASCADE_CONFIGS.every((config) => !("lambda" in config))).toBe(true);
    });
});

describe("Elfouhaily unified directional spectrum", () => {
    test("is finite and positive on both gravity and short-wave bands", () => {
        expect(unifiedSpectrum(0.2, 0.1)).toBeGreaterThan(0);
        expect(unifiedSpectrum(8, 3)).toBeGreaterThan(0);
        expect(unifiedSpectrum(0, 0)).toBe(0);
    });

    test("high-k energy is supplied by the unified tail, not a Phillips amplitude", () => {
        const low = unifiedSpectrum(0.3, 0);
        const high = unifiedSpectrum(8, 0);
        expect(Number.isFinite(high)).toBe(true);
        expect(high).toBeGreaterThan(0);
        expect(low).toBeGreaterThan(high);
    });
});

describe("composed-field whitecap fold derivation", () => {
    test("anchor is the published wind coverage and λ is strictly below the fold ceiling", () => {
        const band = deriveFoldBand();
        expect(band.whitecapAnchor).toBeCloseTo(whitecapFraction(SEA_STATE.windSpeed), 12);
        expect(band.whitecapAnchor).toBeGreaterThan(0);
        expect(band.lambda).toBeGreaterThan(0);
        expect(band.lambda).toBeLessThan(band.lambdaCeiling);
        expect(band.lambda).toBeCloseTo(SEA_STATE.lambda, 12);
        expect(band.lambda).toBeGreaterThan(0.1);
        expect(band.lambda).toBeLessThan(10);
    });
});
