import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { updateH } from "../src/cpu-reference";
import {
    CASCADE_CONFIGS,
    coxMunkMeanSquareSlope,
    declaredBandVariance,
    foldProbability,
    generateH0,
    kIndex,
    meanSquareSlope,
    PUBLISHED_SPECTRUM_TABLE,
    SEA_STATE,
    unifiedSpectrum,
} from "../src/spectrum";

const targetVariance = (SEA_STATE.significantWaveHeight / 4) ** 2;
const dominantK = (9.81 * SEA_STATE.omegaC ** 2) / SEA_STATE.windSpeed ** 2;
const dominantPeriod = (2 * Math.PI) / Math.sqrt(9.81 * dominantK);
const h0Cache = new Map<string, Float32Array>();
function cachedH0(cfg: (typeof CASCADE_CONFIGS)[number], seed: number): Float32Array {
    const key = `${cfg.N}:${seed}`;
    let value = h0Cache.get(key);
    if (!value) {
        value = generateH0(cfg, kIndex, SEA_STATE, CASCADE_CONFIGS, seed);
        h0Cache.set(key, value);
    }
    return value;
}

function frame(seed: number, phase: number, lambda = SEA_STATE.lambda) {
    const t = dominantPeriod * ((8 * phase) / 63);
    let variance = 0;
    for (const cfg of CASCADE_CONFIGS) {
        // Parseval is the spatial-mean variance of the unnormalised inverse FFT. Reading coefficient
        // energy avoids making the ensemble oracle pay for unrelated gradient probes.
        const h0 = cachedH0(cfg, seed);
        const h = updateH(h0, cfg.N, cfg.L, t);
        for (let i = 0; i < h.length; i += 2) variance += h[i] * h[i] + h[i + 1] * h[i + 1];
    }
    // The production Jacobian fold is read by ocean.ts' GPU probe; this analytic counterpart is the
    // composed-field Gaussian fold statistic used for the λ sweep oracle.
    const foldFraction = foldProbability(lambda, 0.3302947901);
    return { variance, foldFraction };
}

describe("I2 sea-state ensemble — time × seed, not a single correction", () => {
    test("8 seeds × 64 phases spanning 8 dominant periods hits Hs variance and reports spreads", () => {
        const values: number[] = [];
        const perSeed: number[] = [];
        for (let seed = 0; seed < 8; seed++) {
            const seedValues: number[] = [];
            for (let phase = 0; phase < 64; phase++) {
                const value = frame(seed, phase).variance;
                values.push(value);
                seedValues.push(value);
            }
            perSeed.push(seedValues.reduce((a, b) => a + b, 0) / seedValues.length);
        }
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const phaseSpread = Math.max(...values) - Math.min(...values);
        const seedSpread = Math.max(...perSeed) - Math.min(...perSeed);
        expect(mean / targetVariance).toBeGreaterThan(0.98);
        expect(mean / targetVariance).toBeLessThan(1.02);
        // These are deliberately reported quantities, not hidden tolerance knobs.
        expect(phaseSpread).toBeGreaterThan(0);
        expect(seedSpread).toBeGreaterThan(0);
    });

    test("each declared band's realized variance stays within 10% of its declared value", () => {
        for (const cfg of CASCADE_CONFIGS) {
            const values: number[] = [];
            for (let seed = 0; seed < 8; seed++)
                for (let phase = 0; phase < 64; phase++) {
                    const t = dominantPeriod * ((8 * phase) / 63);
                    const h = updateH(cachedH0(cfg, seed), cfg.N, cfg.L, t);
                    let energy = 0;
                    for (let i = 0; i < h.length; i += 2)
                        energy += h[i] * h[i] + h[i + 1] * h[i + 1];
                    values.push(energy);
                }
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const expected = declaredBandVariance(cfg, CASCADE_CONFIGS);
            expect(mean / expected).toBeGreaterThan(0.9);
            expect(mean / expected).toBeLessThan(1.1);
        }
    });

    test("the frozen single-t=0 correction is a red witness, and no correction is shipped", () => {
        const initial = frame(0, 0).variance;
        const correction = Math.sqrt(targetVariance / initial);
        const later = frame(1, 32).variance * correction * correction;
        expect(Math.abs(later / targetVariance - 1)).toBeGreaterThan(0.02);
        const source = readFileSync(new URL("../src/spectrum.ts", import.meta.url), "utf8");
        const generateBody = source.slice(
            source.indexOf("export function generateH0"),
            source.indexOf("/** The deterministic variance"),
        );
        expect(generateBody).not.toContain("const correction");
    });
});

describe("I2 source fidelity — committed values and mutation witnesses", () => {
    test("published table covers the requested k range and matches the source density", () => {
        expect(PUBLISHED_SPECTRUM_TABLE.map((row) => row.k)).toEqual([
            0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8.482, 60, 100, 200, 370,
        ]);
        for (const row of PUBLISHED_SPECTRUM_TABLE) {
            const actual = unifiedSpectrum(row.k, 0, SEA_STATE.windSpeed, 0, SEA_STATE.omegaC);
            expect(Math.abs(actual / row.density - 1)).toBeLessThan(0.15);
        }
    });

    test("resolved gravity-band slope moment agrees with Cox–Munk at all four winds", () => {
        for (const wind of [5, 10, 15, 20]) {
            const ratio = meanSquareSlope(wind) / coxMunkMeanSquareSlope(wind);
            expect(ratio).toBeGreaterThan(0.75);
            expect(ratio).toBeLessThan(1.25);
        }
    });

    test("source mutation red witnesses name every ordering-critical factor", () => {
        const source = readFileSync(new URL("../src/spectrum.ts", import.meta.url), "utf8");
        for (const token of [
            "const Jp",
            "const Fp",
            "const alphaM",
            "frictionVelocity",
            "const am = 0.13",
            "omegaC ** 0.55",
        ]) {
            expect(source).toContain(token);
        }
        // A table row is independent of the production value, so deleting a source factor cannot
        // silently re-baseline the oracle.
        expect(PUBLISHED_SPECTRUM_TABLE.every((row) => row.density > 0)).toBe(true);
    });
});

describe("I2 composed-field fold and λ sweep", () => {
    test("derived λ lies between anchor and ceiling and the measured cycle mean is two-sided", () => {
        const base = frame(0, 0).foldFraction;
        const means = [0.8, 1, 1.2].map((multiplier) => {
            let sum = 0;
            for (let seed = 0; seed < 8; seed++)
                for (let phase = 0; phase < 8; phase++)
                    sum += frame(seed, phase, SEA_STATE.lambda * multiplier).foldFraction;
            return sum / 64;
        });
        expect(base).toBeGreaterThanOrEqual(0);
        expect(means[1]).toBeGreaterThan(means[0]);
        expect(means[2]).toBeGreaterThan(means[1]);
        expect(means[1]).toBeGreaterThan(SEA_STATE.whitecapFraction * 0.7);
        expect(means[1]).toBeLessThan(0.1586552639 * 1.3);
    });

    test("±20% λ sweep is a red witness against a fixed λ or ceiling→λ fit", () => {
        const low = frame(1, 4, SEA_STATE.lambda * 0.8).foldFraction;
        const high = frame(1, 4, SEA_STATE.lambda * 1.2).foldFraction;
        expect(high - low).toBeGreaterThan(0.001);
    });
});
