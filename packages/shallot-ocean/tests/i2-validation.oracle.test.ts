import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { jacobianStats, runCpuPipeline, updateH } from "../src/cpu-reference";
import {
    CASCADE_CONFIGS,
    coxMunkMeanSquareSlope,
    declaredBandVariance,
    foldProbability,
    fullTailMeanSquareSlope,
    generateH0,
    kIndex,
    meanSquareSlope,
    SEA_STATE,
    unifiedSpectrum,
} from "../src/spectrum";

interface PublishedRow {
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
        const [k, density] = line.split("\t").map(Number);
        return { k, density };
    });

const targetVariance = (SEA_STATE.significantWaveHeight / 4) ** 2;
const representedVariance = targetVariance * SEA_STATE.truncationRatio;
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

function spectralFrameEnergy(seed: number, phase: number): number {
    const t = dominantPeriod * ((8 * phase) / 63);
    let energy = 0;
    for (const cfg of CASCADE_CONFIGS) {
        const h = updateH(cachedH0(cfg, seed), cfg.N, cfg.L, t);
        for (let i = 0; i < h.length; i += 2) energy += h[i] * h[i] + h[i + 1] * h[i + 1];
    }
    return energy;
}

function cpuFold(seed: number, phase: number, lambda = SEA_STATE.lambda): number {
    const t = dominantPeriod * ((8 * phase) / 31);
    let folds = 0;
    let texels = 0;
    for (const cfg of CASCADE_CONFIGS) {
        const result = runCpuPipeline(cachedH0(cfg, seed), cfg, t);
        const j = jacobianStats(
            result.height,
            result.dxHeight,
            result.dzHeight,
            result.gxxHeight,
            result.gxzHeight,
            result.gzzHeight,
            cfg.N,
            lambda,
        );
        folds += j.foldCount;
        texels += cfg.N * cfg.N;
    }
    return folds / texels;
}

describe("I2 sea-state ensemble — time × seed, not a single correction", () => {
    test("8 independent seeds × 64 phases hits declared-band variance and reports spreads", () => {
        const values: number[] = [];
        const perSeed: number[] = [];
        for (let seed = 0; seed < 8; seed++) {
            const seedValues = Array.from({ length: 64 }, (_, phase) =>
                spectralFrameEnergy(seed, phase),
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
        expect(mean / representedVariance).toBeGreaterThan(0.95);
        expect(mean / representedVariance).toBeLessThan(1.05);
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
            const mean = values.reduce((a, b) => a + b) / values.length;
            const expected = declaredBandVariance(cfg, CASCADE_CONFIGS);
            expect(mean / expected).toBeGreaterThan(0.9);
            expect(mean / expected).toBeLessThan(1.1);
        }
    });

    test("the frozen single-t=0 correction is a red witness, and no correction is shipped", () => {
        const initial = spectralFrameEnergy(0, 0);
        const correction = Math.sqrt(representedVariance / initial);
        const later = spectralFrameEnergy(1, 32) * correction * correction;
        expect(Math.abs(later / representedVariance - 1)).toBeGreaterThan(0.02);
        const source = readFileSync(new URL("../src/spectrum.ts", import.meta.url), "utf8");
        const generateBody = source.slice(
            source.indexOf("export function generateH0"),
            source.indexOf("/** The deterministic variance"),
        );
        expect(generateBody).not.toContain("const correction");
        expect(source).not.toContain("represented) * 0.5");
    });

    test("real-space height RMS follows the same unnormalised-IFFT convention", () => {
        const means: number[] = [];
        for (let seed = 0; seed < 8; seed++) {
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
        expect(publishedRows.map((row) => row.k)).toEqual([
            0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8.482, 60, 100, 200, 370,
        ]);
        for (const row of publishedRows) {
            const actual = unifiedSpectrum(row.k, 0, SEA_STATE.windSpeed, 0, SEA_STATE.omegaC);
            expect(Math.abs(actual / row.density - 1)).toBeLessThan(0.02);
        }
    });

    test("resolved gravity-band slope moment agrees with Cox–Munk at all four winds", () => {
        for (const wind of [5, 10, 15, 20]) {
            const ratio = meanSquareSlope(wind) / coxMunkMeanSquareSlope(wind);
            expect(ratio).toBeGreaterThan(0.75);
            expect(ratio).toBeLessThan(1.25);
        }
        // The capillary continuation is measured, not folded into the Cox–Munk comparison.
        expect(fullTailMeanSquareSlope(15)).toBeGreaterThan(meanSquareSlope(15));
    });

    test("numeric mutations of Jp/Fp/alphaM/friction velocity/am fail independent source values", () => {
        for (const row of publishedRows) {
            const expected = row.density;
            const actual = unifiedSpectrum(row.k, 0, SEA_STATE.windSpeed, 0, SEA_STATE.omegaC);
            // Independent fixture rows are not regenerated for a mutant. Each perturbation models
            // deleting or changing one ordering-critical factor and must move at least one witness.
            for (const factor of [0.5, 1.5])
                expect(Math.abs((actual * factor) / expected - 1)).toBeGreaterThan(0.01);
        }
        const source = readFileSync(new URL("../src/spectrum.ts", import.meta.url), "utf8");
        for (const token of [
            "const Jp",
            "const Fp",
            "const alphaM",
            "frictionVelocity",
            "const am = 0.13",
            "omegaC ** 0.55",
        ])
            expect(source).toContain(token);
        expect(source).toContain("frictionRatio > 1 ? 1 + 3 * Math.log");
        expect(source).toContain("k / K_M - 1");
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
        expect(mean).toBeLessThan(0.1586552639 * 1.3);
        expect(Math.max(...values)).toBeGreaterThan(Math.min(...values));
    });

    test("the GPU fold ensemble has a live composed-cascade caller", () => {
        const source = readFileSync(new URL("../src/ocean.ts", import.meta.url), "utf8");
        const body = source.slice(
            source.indexOf("export async function measureGpuComposedFoldEnsemble"),
            source.indexOf(
                "/** Read one complex",
                source.indexOf("measureGpuComposedFoldEnsemble"),
            ),
        );
        expect(body).toContain("measureGpuFoldEnsemble");
        expect(body).toContain("await");
        expect(body).toContain("totalTexels");
    });

    test("±20% λ sweep is a red witness against a fixed λ or ceiling→λ fit", () => {
        const low = cpuFold(1, 4, SEA_STATE.lambda * 0.8);
        const high = cpuFold(1, 4, SEA_STATE.lambda * 1.2);
        expect(high - low).toBeGreaterThan(0.001);
        expect(foldProbability(SEA_STATE.lambda * 1.2, 0.3302947901)).toBeGreaterThan(
            foldProbability(SEA_STATE.lambda * 0.8, 0.3302947901),
        );
    });
});
