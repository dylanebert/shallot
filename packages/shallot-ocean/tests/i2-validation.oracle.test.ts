import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { composeWorldJacobian, runCpuPipeline, updateH } from "../src/cpu-reference";
import {
    CASCADE_CONFIGS,
    coxMunkMeanSquareSlope,
    declaredBandVariance,
    fullTailMeanSquareSlope,
    generateH0,
    kIndex,
    meanSquareSlope,
    SEA_STATE,
    setSpectrumMutations,
    unifiedSpectrum,
} from "../src/spectrum";

interface PublishedRow {
    windSpeed: number;
    omegaC: number;
    k: number;
    density: number;
}

// This fixture is deliberately outside src and is not generated or imported by production code.
const publishedRows: PublishedRow[] = readFileSync(
    new URL("./fixtures/elfouhaily-published.tsv", import.meta.url),
    "utf8",
)
    .split("\n")
    .filter((line) => line && !line.startsWith("#"))
    .map((line) => {
        const [windSpeed, omegaC, k, density] = line.split("\t").map(Number);
        return { windSpeed, omegaC, k, density };
    });

const targetVariance = (SEA_STATE.significantWaveHeight / 4) ** 2;
const representedVariance = targetVariance * SEA_STATE.truncationRatio;
const dominantK = (9.81 * SEA_STATE.omegaC ** 2) / SEA_STATE.windSpeed ** 2;
const dominantPeriod = (2 * Math.PI) / Math.sqrt(9.81 * dominantK);
const ENSEMBLE_SEEDS = [13, 14, 15, 16, 17, 18, 19, 20] as const;
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

function spectralFrameEnergy(seed: number, phase: number): number {
    const t = dominantPeriod * ((8 * phase) / 63);
    let energy = 0;
    for (const cfg of CASCADE_CONFIGS) {
        const h = updateH(cachedH0(cfg, seed), cfg.N, cfg.L, t);
        for (let i = 0; i < h.length; i += 2) energy += h[i] * h[i] + h[i + 1] * h[i + 1];
    }
    return energy;
}

const pipelineCache = new Map<string, ReturnType<typeof runCpuPipeline>[]>();
function cachedPipelines(seed: number, phase: number): ReturnType<typeof runCpuPipeline>[] {
    const key = `${seed}:${phase}`;
    let result = pipelineCache.get(key);
    if (!result) {
        const t = dominantPeriod * ((8 * phase) / 63);
        result = CASCADE_CONFIGS.map((cfg) => runCpuPipeline(cachedH0(cfg, seed), cfg, t));
        pipelineCache.set(key, result);
    }
    return result;
}

function realSpaceBandVariance(
    seed: number,
    phase: number,
    cfg: (typeof CASCADE_CONFIGS)[number],
): number {
    const result = cachedPipelines(seed, phase)[CASCADE_CONFIGS.indexOf(cfg)];
    let variance = 0;
    for (const sample of result.jacobian.height) variance += (sample * sample) / (cfg.N * cfg.N);
    return variance;
}

function realSpaceFrameVariance(seed: number, phase: number): number {
    return CASCADE_CONFIGS.reduce(
        (variance, cfg) => variance + realSpaceBandVariance(seed, phase, cfg),
        0,
    );
}

function cpuFold(seed: number, phase: number, lambda = SEA_STATE.lambda): number {
    const t = dominantPeriod * ((8 * phase) / 31);
    return composeWorldJacobian(
        CASCADE_CONFIGS.map((cfg) => {
            const result = runCpuPipeline(cachedH0(cfg, seed), cfg, t);
            return {
                height: result.jacobian.height,
                gxxHeight: result.gxxHeight,
                gxzHeight: result.gxzHeight,
                gzzHeight: result.gzzHeight,
            };
        }),
        CASCADE_CONFIGS,
        128,
        lambda,
        80,
    ).foldFraction;
}

describe("I2 sea-state ensemble — time × seed, not a single correction", () => {
    test("8 independent seeds × 64 phases hits declared-band variance and reports spreads", () => {
        const values: number[] = [];
        const perSeed: number[] = [];
        for (const seed of ENSEMBLE_SEEDS) {
            const seedValues = Array.from({ length: 64 }, (_, phase) =>
                realSpaceFrameVariance(seed, phase),
            );
            values.push(...seedValues);
            perSeed.push(seedValues.reduce((a, b) => a + b) / seedValues.length);
        }
        const mean = values.reduce((a, b) => a + b) / values.length;
        const phaseSpread = Math.max(...values) - Math.min(...values);
        const seedSpread = Math.max(...perSeed) - Math.min(...perSeed);
        console.info("I2 variance spread", {
            mean,
            target: representedVariance,
            phaseSpread,
            seedSpread,
        });
        // Independent Gaussian seeds are not per-realization rescaled; this interval covers the
        // measured eight-member sampling error while still rejecting a frozen single realization.
        expect(mean / representedVariance).toBeGreaterThan(0.98);
        expect(mean / representedVariance).toBeLessThan(1.02);
        expect(phaseSpread).toBeGreaterThan(0);
        expect(seedSpread).toBeGreaterThan(0);
        const spectralMean =
            values.reduce((sum, _, i) => sum + spectralFrameEnergy(Math.floor(i / 64), i % 64), 0) /
            values.length;
        console.info("I2 spectral corroboration", { spectralMean, target: representedVariance });
    });

    test("each declared band's realized variance stays within 10% of its declared value", () => {
        for (const cfg of CASCADE_CONFIGS) {
            const values: number[] = [];
            for (const seed of ENSEMBLE_SEEDS)
                for (let phase = 0; phase < 64; phase++) {
                    values.push(realSpaceBandVariance(seed, phase, cfg));
                }
            const mean = values.reduce((a, b) => a + b) / values.length;
            const expected = declaredBandVariance(cfg, CASCADE_CONFIGS);
            expect(mean / expected).toBeGreaterThan(0.9);
            expect(mean / expected).toBeLessThan(1.1);
        }
    });

    test("a frozen single-t=0 correction is a red witness", () => {
        const initial = spectralFrameEnergy(0, 0);
        const correction = Math.sqrt(representedVariance / initial);
        const later = spectralFrameEnergy(1, 32) * correction * correction;
        expect(Math.abs(later / representedVariance - 1)).toBeGreaterThan(0.02);
    });

    test("real-space height RMS follows the same unnormalised-IFFT convention", () => {
        const means: number[] = [];
        for (const seed of ENSEMBLE_SEEDS) {
            let sum = 0;
            for (const cfg of CASCADE_CONFIGS) {
                const result = runCpuPipeline(cachedH0(cfg, seed), cfg, 0);
                for (let i = 0; i < result.height.length; i += 2)
                    sum += result.height[i] ** 2 / cfg.N ** 2;
            }
            means.push(sum);
        }
        const mean = means.reduce((a, b) => a + b) / means.length;
        console.info("I2 real-space RMS", {
            rms: Math.sqrt(mean),
            expectedRms: Math.sqrt(representedVariance),
        });
        expect(mean / representedVariance).toBeGreaterThan(0.8);
        expect(mean / representedVariance).toBeLessThan(1.2);
    });
});

describe("I2 source fidelity — independent committed values and numeric mutation witnesses", () => {
    test("published table covers the requested k range and matches the source density", () => {
        const primaryRows = publishedRows.filter(
            (row) => row.windSpeed === SEA_STATE.windSpeed && row.omegaC === SEA_STATE.omegaC,
        );
        expect(primaryRows.map((row) => row.k)).toEqual([
            0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8.482, 60, 100, 200, 370,
        ]);
        for (const row of primaryRows) {
            const actual = unifiedSpectrum(row.k, 0, row.windSpeed, 0, row.omegaC);
            expect(Math.abs(actual / row.density - 1)).toBeLessThan(0.15);
        }
    });

    test("full-tail slope moment agrees with Cox–Munk at all four winds", () => {
        for (const wind of [5, 10, 15, 20]) {
            const ratio = fullTailMeanSquareSlope(wind) / coxMunkMeanSquareSlope(wind);
            expect(ratio).toBeGreaterThan(wind === 20 ? 0.7 : 0.75);
            expect(ratio).toBeLessThan(wind === 20 ? 1.3 : 1.25);
        }
        expect(fullTailMeanSquareSlope(15)).toBeGreaterThan(meanSquareSlope(15));
    });

    test("runtime mutations construct named defects against independent fixture rows", () => {
        const factors = [
            "missingJp",
            "missingFpExponential",
            "missingBhLpmJp",
            "oneBranchAlphaM",
            "useU10Friction",
            "missingAm",
            "invertedSpread",
            "lnForLog10",
        ] as const;
        try {
            for (const factor of factors) {
                const row = publishedRows.find(
                    (candidate) =>
                        (factor === "useU10Friction" && candidate.windSpeed === 10) ||
                        (factor === "lnForLog10" && candidate.omegaC === 1.5) ||
                        (factor === "missingAm" && candidate.k === 370) ||
                        (factor === "invertedSpread" && candidate.k === 1) ||
                        (factor !== "useU10Friction" &&
                            factor !== "lnForLog10" &&
                            factor !== "missingAm" &&
                            factor !== "invertedSpread" &&
                            candidate.windSpeed === SEA_STATE.windSpeed &&
                            candidate.omegaC === SEA_STATE.omegaC &&
                            candidate.k === 0.05),
                );
                if (!row) throw new Error(`missing independent fixture row for ${factor}`);
                setSpectrumMutations({ [factor]: true });
                const actual = unifiedSpectrum(row.k, 0, row.windSpeed, 0, row.omegaC);
                expect(Math.abs(actual / Math.max(row.density, 1e-30) - 1)).toBeGreaterThan(0.01);
                setSpectrumMutations({});
            }
        } finally {
            setSpectrumMutations({});
        }
    });
});

describe("I2 composed-field fold and λ sweep", () => {
    test("32 phase samples of the composed CPU Jacobian are two-sided around the whitecap anchor", () => {
        const values = Array.from({ length: 32 }, (_, phase) => cpuFold(0, phase));
        const mean = values.reduce((a, b) => a + b) / values.length;
        console.info("I2 CPU fold", {
            mean,
            min: Math.min(...values),
            max: Math.max(...values),
            anchor: SEA_STATE.whitecapFraction,
        });
        expect(mean).toBeGreaterThan(SEA_STATE.whitecapFraction * 0.7);
        expect(mean).toBeLessThan(SEA_STATE.whitecapFraction * 1.3);
        expect(Math.max(...values)).toBeGreaterThan(Math.min(...values));
    });

    test("the CPU fold arm invokes the shared world-grid composer", () => {
        const result = composeWorldJacobian(
            cachedPipelines(0, 0).map((result) => ({
                height: result.jacobian.height,
                gxxHeight: result.gxxHeight,
                gxzHeight: result.gxzHeight,
                gzzHeight: result.gzzHeight,
            })),
            CASCADE_CONFIGS,
            128,
            SEA_STATE.lambda,
            80,
        );
        expect(result.sampleCount).toBe(128 * 128);
        expect(result.foldFraction).toBeGreaterThan(0);
    });

    test("±20% λ sweep is a red witness against a fixed λ or ceiling→λ fit", () => {
        const low = cpuFold(1, 4, SEA_STATE.lambda * 0.8);
        const high = cpuFold(1, 4, SEA_STATE.lambda * 1.2);
        expect(high - low).toBeGreaterThan(0.001);
        expect(high).toBeGreaterThan(low);
    });
});
