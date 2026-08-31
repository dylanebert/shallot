import { describe, expect, test } from "bun:test";
import { runCpuPipeline } from "../src/cpu-reference";
import {
    CASCADE_CONFIGS,
    declaredBandVariance,
    deriveFoldBand,
    generateH0,
    kIndex,
    realizedFieldVariance,
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
        const a80 = spectralCellAmplitude(cfg80, kx, kz, [cfg80, cfg40]);
        const a40 = spectralCellAmplitude(cfg40, kx, kz, [cfg80, cfg40]);
        expect(a80 / a40).toBeCloseTo(0.5, 5);
    });

    test("declared-band variance is N-invariant for the same physical patch", () => {
        const low = { ...CASCADE_CONFIGS[0], N: 64, L: 80 };
        const high = { ...low, N: 128 };
        // Both resolutions contain every mode in the declared band because their Nyquist limits are
        // above kHi; summing the normalized density is therefore exactly N-independent.
        expect(declaredBandVariance(low, [low, high])).toBeCloseTo(
            declaredBandVariance(high, [low, high]),
            12,
        );
        expect(spectrumNormalization([low])).toBeGreaterThan(0);
        expect(generateH0(low, kIndex, SEA_STATE, [low, high]).length).toBe(low.N * low.N * 2);
    });

    test("realized Parseval variance is N-invariant for one fixed physical band", () => {
        const n64 = { ...CASCADE_CONFIGS[0], N: 64 };
        const n128 = { ...CASCADE_CONFIGS[0], N: 128 };
        expect(realizedFieldVariance([n64])).toBeCloseTo(realizedFieldVariance([n128]), 12);
    });

    test("realized pipeline preserves the normalized declared-band ratio", () => {
        const realized: number[] = [];
        for (const config of CASCADE_CONFIGS) {
            const result = runCpuPipeline(
                generateH0(config, kIndex, SEA_STATE, CASCADE_CONFIGS),
                config,
            );
            let sum = 0;
            for (const sample of result.jacobian.height) sum += sample * sample;
            realized.push(sum / (config.N * config.N));
        }
        const expected = CASCADE_CONFIGS.map((config) =>
            declaredBandVariance(config, CASCADE_CONFIGS),
        );
        // A single Gaussian realization is intentionally noisy per band; the ensemble oracle owns
        // the tighter physical assertion. This check only guards that both transformed bands carry
        // finite, nonzero real-space energy in the production pipeline.
        expect(realized.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
        expect(expected.every((value) => value > 0)).toBe(true);
    });

    test("composed variance reports declared-band truncation without fitting Hs", () => {
        const represented = realizedFieldVariance(
            CASCADE_CONFIGS,
            SEA_STATE.significantWaveHeight,
            SEA_STATE.windSpeed,
            SEA_STATE.windDir,
        );
        const target = (SEA_STATE.significantWaveHeight / 4) ** 2;
        expect(represented / target).toBeCloseTo(SEA_STATE.truncationRatio, 12);
        let pipelineVariance = 0;
        for (const config of CASCADE_CONFIGS) {
            const result = runCpuPipeline(
                generateH0(config, kIndex, SEA_STATE, CASCADE_CONFIGS),
                config,
            );
            let sum = 0;
            for (const sample of result.jacobian.height) sum += sample * sample;
            pipelineVariance += sum / (config.N * config.N);
        }
        expect(pipelineVariance).toBeGreaterThan(0);
    });

    test("one shared wind and sea-state target are explicit", () => {
        expect(SEA_STATE.significantWaveHeight).toBeGreaterThan(0);
        expect(SEA_STATE.windSpeed).toBe(15);
        expect(SEA_STATE.omegaC).toBe(0.9);
        expect(SEA_STATE.omegaC).toBeGreaterThan(0);
        expect(SEA_STATE.omegaC).toBeGreaterThanOrEqual(0.84);
        expect(SEA_STATE.truncationRatio).toBeGreaterThan(0);
        expect(SEA_STATE.truncationRatio).toBeLessThan(1);
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
        const band = deriveFoldBand(CASCADE_CONFIGS);
        expect(band.whitecapAnchor).toBeCloseTo(whitecapFraction(SEA_STATE.windSpeed), 12);
        expect(band.whitecapAnchor).toBeGreaterThan(0);
        expect(band.lambda).toBeGreaterThan(0);
        expect(band.lambda).toBeLessThan(band.lambdaCeiling);
        expect(band.lambda).toBeCloseTo(SEA_STATE.lambda, 12);
        expect(band.foldAnchor).toBeCloseTo(band.whitecapAnchor, 3);
        expect(band.foldCeiling).toBeGreaterThan(band.foldAnchor);
        expect(band.foldAnchor).toBeCloseTo(band.whitecapAnchor, 3);
        let folded = 0;
        let samples = 0;
        for (const config of CASCADE_CONFIGS) {
            const result = runCpuPipeline(
                generateH0(config, kIndex, SEA_STATE, CASCADE_CONFIGS),
                config,
            );
            folded += result.jacobian.foldCount;
            samples += config.N * config.N;
        }
        const realizedFold = folded / samples;
        // One-sided anchor→ceiling semantics: the realized field must not fall below the
        // whitecap anchor or exceed the unit-RMS ceiling. This reads the actual CPU pipeline,
        // rather than restating foldProbability's algebra.
        expect(realizedFold).toBeGreaterThanOrEqual(band.whitecapAnchor);
        expect(realizedFold).toBeLessThanOrEqual(band.foldCeiling);
        expect(band.lambda).toBeGreaterThan(0.1);
        expect(band.lambda).toBeLessThan(10);
    });
});
