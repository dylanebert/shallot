import { describe, expect, test } from "bun:test";
import {
    CASCADE_CONFIGS,
    type CascadeConfig,
    declaredBandVariance,
    directionalDensity,
    G,
    generateH0,
    kIndex,
    type RealizationMutation,
    SEA_STATE,
} from "../src/spectrum";

const SEEDS = Array.from({ length: 200 }, (_, seed) => seed);
const MUTATION_SEEDS = SEEDS.slice(0, 48);
const dominantK = (G * SEA_STATE.omegaC ** 2) / SEA_STATE.windSpeed ** 2;
const dominantPeriod = (2 * Math.PI) / Math.sqrt(G * dominantK);
const PHASE_PERIODS = Array.from({ length: 9 }, (_, period) => period);
const PHASES = PHASE_PERIODS.map((period) => period * dominantPeriod);
const CV_TOLERANCE = 0.15;

interface Prediction {
    target: number;
    cv: number;
    interval: number;
}

interface EnsembleReading {
    mean: number;
    cv: number;
    relativeError: number;
}

function prediction(sampleSize = SEEDS.length): Prediction {
    let target = 0;
    let variance = 0;
    for (const config of CASCADE_CONFIGS) {
        const dk = (2 * Math.PI) / config.L;
        for (let y = 0; y < config.N; y++) {
            for (let x = 0; x < config.N; x++) {
                const negativeX = (config.N - x) % config.N;
                const negativeY = (config.N - y) % config.N;
                if (y * config.N + x > negativeY * config.N + negativeX) continue;
                const kx = kIndex(x, config.N) * dk;
                const kz = kIndex(y, config.N) * dk;
                const k = Math.hypot(kx, kz);
                if (k < config.kLo || k > config.kHi) continue;
                const oppositeKx = kIndex(negativeX, config.N) * dk;
                const oppositeKz = kIndex(negativeY, config.N) * dk;
                const cell =
                    directionalDensity(
                        kx,
                        kz,
                        SEA_STATE.windSpeed,
                        SEA_STATE.windDir,
                        SEA_STATE.omegaC,
                    ) *
                    dk *
                    dk;
                const oppositeCell =
                    directionalDensity(
                        oppositeKx,
                        oppositeKz,
                        SEA_STATE.windSpeed,
                        SEA_STATE.windDir,
                        SEA_STATE.omegaC,
                    ) *
                    dk *
                    dk;
                const omega = Math.sqrt(G * k);
                let phaseReal = 0;
                let phaseImaginary = 0;
                for (const time of PHASES) {
                    phaseReal += Math.cos(2 * omega * time);
                    phaseImaginary += Math.sin(2 * omega * time);
                }
                phaseReal /= PHASES.length;
                phaseImaginary /= PHASES.length;
                const phaseCoherence = phaseReal * phaseReal + phaseImaginary * phaseImaginary;
                target += cell + oppositeCell;
                // For the phase-averaged pair statistic, independent circular Gaussians contribute
                // cell² + oppositeCell² plus the surviving cross-term covariance.
                variance +=
                    cell * cell +
                    oppositeCell * oppositeCell +
                    2 * phaseCoherence * cell * oppositeCell;
            }
        }
    }
    const cv = Math.sqrt(variance) / target;
    return { target, cv, interval: (1.96 * cv) / Math.sqrt(sampleSize) };
}

const realizationStatisticCache = new Map<string, number>();
function phaseAveragedVariance(
    config: CascadeConfig,
    seed: number,
    mutation: RealizationMutation = {},
): number {
    const mutationKey = mutation.omitCellArea
        ? "omit-cell-area"
        : mutation.rescalePerRealization
          ? "rescale"
          : "control";
    const key = `${mutationKey}:${config.N}:${config.L}:${seed}`;
    const cached = realizationStatisticCache.get(key);
    if (cached !== undefined) return cached;
    const h0 = generateH0(config, seed, kIndex, SEA_STATE, mutation);
    const dk = (2 * Math.PI) / config.L;
    let result = 0;
    // Parseval turns the declared real-grid statistic into pair energies. Averaging exp(i2ωt)
    // over the declared schedule first avoids nine redundant transforms of the same realization.
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const index = y * config.N + x;
            const negativeX = (config.N - x) % config.N;
            const negativeY = (config.N - y) % config.N;
            const negativeIndex = negativeY * config.N + negativeX;
            if (index >= negativeIndex) continue;
            const k = Math.hypot(kIndex(x, config.N) * dk, kIndex(y, config.N) * dk);
            if (k < config.kLo || k > config.kHi) continue;
            const omega = Math.sqrt(G * k);
            let phaseReal = 0;
            let phaseImaginary = 0;
            for (const time of PHASES) {
                phaseReal += Math.cos(2 * omega * time);
                phaseImaginary += Math.sin(2 * omega * time);
            }
            phaseReal /= PHASES.length;
            phaseImaginary /= PHASES.length;
            const ar = h0[index * 2];
            const ai = h0[index * 2 + 1];
            const br = h0[negativeIndex * 2];
            const bi = h0[negativeIndex * 2 + 1];
            const productReal = ar * br - ai * bi;
            const productImaginary = ar * bi + ai * br;
            result +=
                2 *
                (ar * ar +
                    ai * ai +
                    br * br +
                    bi * bi +
                    2 * (productReal * phaseReal - productImaginary * phaseImaginary));
        }
    }
    realizationStatisticCache.set(key, result);
    return result;
}

function mean(values: number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleCv(values: number[]): number {
    const average = mean(values);
    const variance =
        values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1);
    return Math.sqrt(variance) / average;
}

function ensembleReading(mutation: RealizationMutation = {}, seeds = SEEDS): EnsembleReading {
    const target = prediction(seeds.length).target;
    const perSeed = seeds.map((seed) =>
        CASCADE_CONFIGS.reduce(
            (sum, config) => sum + phaseAveragedVariance(config, seed, mutation),
            0,
        ),
    );
    const measured = mean(perSeed);
    return {
        mean: measured,
        cv: sampleCv(perSeed),
        relativeError: measured / target - 1,
    };
}

function failsEnsembleCriterion(reading: EnsembleReading, expected: Prediction): boolean {
    return (
        Math.abs(reading.relativeError) >= expected.interval ||
        Math.abs(reading.cv / expected.cv - 1) >= CV_TOLERANCE
    );
}

describe("seeded physical realization", () => {
    test("200 explicit seeds on the declared nine-phase schedule match represented variance", () => {
        const expected = prediction();
        const measured = ensembleReading();
        console.info("Elfouhaily realization ensemble", {
            seeds: SEEDS.length,
            phasePeriods: PHASE_PERIODS,
            dominantPeriod,
            target: expected.target,
            measured: measured.mean,
            cvPred: expected.cv,
            measuredCv: measured.cv,
            relativeError: measured.relativeError,
            interval: expected.interval,
        });
        expect(Math.abs(measured.relativeError)).toBeLessThan(expected.interval);
        expect(Math.abs(measured.cv / expected.cv - 1)).toBeLessThan(CV_TOLERANCE);
    });

    test("each declared band realizes its density within ten percent", () => {
        for (const config of CASCADE_CONFIGS) {
            const target = declaredBandVariance(config);
            const perSeed = SEEDS.map((seed) => phaseAveragedVariance(config, seed));
            const measured = mean(perSeed);
            console.info("Elfouhaily realized band", {
                N: config.N,
                band: [config.kLo, config.kHi],
                target,
                measured,
            });
            expect(Math.abs(measured / target - 1)).toBeLessThan(0.1);
        }
    });

    test("production-path per-realization rescaling reds the ensemble criterion", () => {
        const expected = prediction(MUTATION_SEEDS.length);
        const measured = ensembleReading({ rescalePerRealization: true }, MUTATION_SEEDS);
        console.info("Elfouhaily realization mutation", {
            mutation: "rescalePerRealization",
            seeds: MUTATION_SEEDS.length,
            ...measured,
            cvPred: expected.cv,
            interval: expected.interval,
        });
        expect(failsEnsembleCriterion(measured, expected)).toBe(true);
    });

    test("production-path cell-area removal reds the ensemble criterion", () => {
        const expected = prediction(MUTATION_SEEDS.length);
        const measured = ensembleReading({ omitCellArea: true }, MUTATION_SEEDS);
        console.info("Elfouhaily realization mutation", {
            mutation: "omitCellArea",
            seeds: MUTATION_SEEDS.length,
            ...measured,
            cvPred: expected.cv,
            interval: expected.interval,
        });
        expect(failsEnsembleCriterion(measured, expected)).toBe(true);
    });

    test("declared-band density and draws stay N-invariant", () => {
        for (const config of CASCADE_CONFIGS) {
            const denser = { ...config, N: config.N * 2 };
            expect(declaredBandVariance(denser)).toBeCloseTo(declaredBandVariance(config), 12);
            const base = generateH0(config, 17);
            const high = generateH0(denser, 17);
            const dk = (2 * Math.PI) / config.L;
            for (let y = 0; y < config.N; y++) {
                for (let x = 0; x < config.N; x++) {
                    const labelX = kIndex(x, config.N);
                    const labelY = kIndex(y, config.N);
                    const highX = labelX >= 0 ? labelX : denser.N + labelX;
                    const highY = labelY >= 0 ? labelY : denser.N + labelY;
                    const k = Math.hypot(labelX * dk, labelY * dk);
                    if (k < config.kLo || k > config.kHi) continue;
                    const baseIndex = (y * config.N + x) * 2;
                    const highIndex = (highY * denser.N + highX) * 2;
                    expect(high[highIndex]).toBe(base[baseIndex]);
                    expect(high[highIndex + 1]).toBe(base[baseIndex + 1]);
                }
            }
        }
    });
});
