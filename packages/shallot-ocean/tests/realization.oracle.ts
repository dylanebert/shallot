import { describe, expect, test } from "bun:test";
import { idft2, updateH } from "../src/cpu-reference";
import {
    CASCADE_CONFIGS,
    type CascadeConfig,
    declaredBandVariance,
    directionalDensity,
    G,
    generateH0,
    kIndex,
    SEA_STATE,
} from "../src/spectrum";

const SEEDS = Array.from({ length: 200 }, (_, seed) => seed);
const dominantK = (G * SEA_STATE.omegaC ** 2) / SEA_STATE.windSpeed ** 2;
const dominantPeriod = (2 * Math.PI) / Math.sqrt(G * dominantK);
const PHASES = [0, 8 * dominantPeriod];

function modeWeights(config: CascadeConfig): number[] {
    const dk = (2 * Math.PI) / config.L;
    const weights: number[] = [];
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
            const cellArea = dk * dk;
            weights.push(
                (directionalDensity(
                    kx,
                    kz,
                    SEA_STATE.windSpeed,
                    SEA_STATE.windDir,
                    SEA_STATE.omegaC,
                ) +
                    directionalDensity(
                        oppositeKx,
                        oppositeKz,
                        SEA_STATE.windSpeed,
                        SEA_STATE.windDir,
                        SEA_STATE.omegaC,
                    )) *
                    cellArea,
            );
        }
    }
    return weights;
}

const fieldVarianceCache = new Map<string, number>();
function fieldVariance(config: CascadeConfig, seed: number, time: number): number {
    const key = `${config.N}:${config.L}:${seed}:${time}`;
    const cached = fieldVarianceCache.get(key);
    if (cached !== undefined) return cached;
    const h = updateH(generateH0(config, seed), config.N, config.L, time);
    const field = idft2(h, config.N);
    let mean = 0;
    for (let i = 0; i < config.N * config.N; i++) mean += field[i * 2];
    mean /= config.N * config.N;
    let variance = 0;
    for (let i = 0; i < config.N * config.N; i++) {
        const centered = field[i * 2] - mean;
        variance += centered * centered;
    }
    const result = variance / (config.N * config.N);
    fieldVarianceCache.set(key, result);
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

describe("seeded physical realization", () => {
    test("200 explicit seeds across eight dominant periods match represented variance", () => {
        const allWeights = CASCADE_CONFIGS.flatMap(modeWeights);
        const weightSum = allWeights.reduce((sum, weight) => sum + weight, 0);
        const cvPred =
            Math.sqrt(allWeights.reduce((sum, weight) => sum + weight * weight, 0)) / weightSum;
        const target = CASCADE_CONFIGS.reduce(
            (sum, config) => sum + declaredBandVariance(config),
            0,
        );
        const perSeed = SEEDS.map((seed) =>
            mean(
                PHASES.map((time) =>
                    CASCADE_CONFIGS.reduce(
                        (sum, config) => sum + fieldVariance(config, seed, time),
                        0,
                    ),
                ),
            ),
        );
        const measured = mean(perSeed);
        const measuredCv = sampleCv(perSeed);
        const relativeError = measured / target - 1;
        const interval = (1.96 * cvPred) / Math.sqrt(SEEDS.length);
        console.info("Elfouhaily realization ensemble", {
            seeds: SEEDS.length,
            dominantPeriods: 8,
            phases: PHASES.length,
            target,
            measured,
            cvPred,
            measuredCv,
            relativeError,
            interval,
        });
        expect(Math.abs(relativeError)).toBeLessThan(interval);
        expect(Math.abs(measuredCv / cvPred - 1)).toBeLessThan(0.15);
    });

    test("each declared band realizes its density within ten percent", () => {
        for (const config of CASCADE_CONFIGS) {
            const target = declaredBandVariance(config);
            const perSeed = SEEDS.map((seed) =>
                mean(PHASES.map((time) => fieldVariance(config, seed, time))),
            );
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

    test("per-realization rescaling destroys the predicted ensemble spread", () => {
        const config = CASCADE_CONFIGS[0];
        const target = declaredBandVariance(config);
        const raw = SEEDS.map((seed) => fieldVariance(config, seed, 0));
        const rescaled = raw.map((value) => value * (target / value));
        expect(sampleCv(raw)).toBeGreaterThan(0);
        expect(sampleCv(rescaled)).toBeLessThan(1e-12);
    });

    test("removing Fourier-cell area breaks the declared-band target", () => {
        for (const config of CASCADE_CONFIGS) {
            const dk = (2 * Math.PI) / config.L;
            const correct = mean(SEEDS.slice(0, 16).map((seed) => fieldVariance(config, seed, 0)));
            const withoutCellArea = correct / (dk * dk);
            const target = declaredBandVariance(config);
            expect(Math.abs(withoutCellArea / target - 1)).toBeGreaterThan(0.1);
        }
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
