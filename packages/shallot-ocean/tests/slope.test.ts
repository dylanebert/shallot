import { describe, expect, test } from "bun:test";
import { ifft2 } from "../src/fft";
import {
    composedSlopePsd,
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    slopeSpectra,
} from "../src/slope";
import { CASCADE_CONFIGS, directionalDensity, generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

function independentSlopeMoment(config: (typeof SLOPE_CASCADE_CONFIGS)[number]): number {
    const radialSteps = 2048;
    const angularSteps = 512;
    const dLog = Math.log(config.kHi / config.kLo) / radialSteps;
    const dTheta = (2 * Math.PI) / angularSteps;
    let sum = 0;
    for (let i = 0; i < radialSteps; i++) {
        const k = config.kLo * Math.exp((i + 0.5) * dLog);
        for (let j = 0; j < angularSteps; j++) {
            const theta = (j + 0.5) * dTheta;
            sum +=
                directionalDensity(k * Math.cos(theta), k * Math.sin(theta)) *
                k ** 3 *
                dLog *
                dTheta;
        }
    }
    return sum;
}

function projection(field: Float32Array, N: number, L: number, k: number): number {
    let re = 0;
    let im = 0;
    for (let x = 0; x < N; x++) {
        const world = ((x + 0.5) / N - 0.5) * L;
        const phase = -k * world;
        re += field[x * 2] * Math.cos(phase) - field[x * 2 + 1] * Math.sin(phase);
        im += field[x * 2] * Math.sin(phase) + field[x * 2 + 1] * Math.cos(phase);
    }
    return Math.hypot(re, im) / N;
}

describe("capillary slope cascade", () => {
    test("declares the short-gravity band with zero displacement contribution", () => {
        expect(config.N).toBe(256);
        expect(config.L).toBe(13);
        expect(config.kLo).toBeCloseTo(8.482300164692441, 12);
        expect(config.kHi).toBe(60);
        expect(config.lambda).toBe(0);
        expect((config.N / 2) * ((2 * Math.PI) / config.L)).toBeGreaterThan(config.kHi);
    });

    test("mode placement lands a labelled slope mode on its label", () => {
        const target = (20 * (2 * Math.PI)) / config.L;
        const index = 20;
        const h = new Float32Array(config.N * config.N * 2);
        h[index * 2] = 1;
        const { x } = slopeSpectra(h, config);
        const field = ifft2(x, config.N);
        const onLabel = projection(field, config.N, config.L, target);
        const image = projection(
            field,
            config.N,
            config.L,
            target + (config.N / 2) * ((2 * Math.PI) / config.L),
        );
        expect(onLabel).toBeGreaterThan(image * 4);
    });

    test("N invariance preserves the declared mode and composed slope PSD", () => {
        const denser = { ...config, N: config.N * 2 };
        const referenceMoment = independentSlopeMoment(config);
        const measuredMoment = composedSlopePsd(config);
        expect(measuredMoment).toBeGreaterThan(0);
        expect(Math.abs(measuredMoment / referenceMoment - 1)).toBeLessThan(0.005);
        expect(composedSlopePsd(denser)).toBeCloseTo(measuredMoment, 12);
        const base = generateH0(config, 17);
        const high = generateH0(denser, 17);
        const dk = (2 * Math.PI) / config.L;
        for (let y = 0; y < config.N; y++) {
            for (let x = 0; x < config.N; x++) {
                const kx = kIndex(x, config.N) * dk;
                const kz = kIndex(y, config.N) * dk;
                if (Math.hypot(kx, kz) < config.kLo || Math.hypot(kx, kz) > config.kHi) continue;
                const hiX = kIndex(x, config.N) >= 0 ? x : denser.N + kIndex(x, config.N);
                const hiY = kIndex(y, config.N) >= 0 ? y : denser.N + kIndex(y, config.N);
                const a = (y * config.N + x) * 2;
                const b = (hiY * denser.N + hiX) * 2;
                expect(high[b]).toBe(base[a]);
                expect(high[b + 1]).toBe(base[a + 1]);
            }
        }
    });

    test("mip reduction publishes measured residual slope variance", () => {
        const level0 = new Float32Array([1, 0, 1, 0, 3, 0, 9, 0, 1, 2, 5, 0, 3, 2, 13, 0]);
        const mip = reduceSlopeMip(level0, 2);
        expect(mip[0]).toBeCloseTo(2, 6);
        expect(mip[1]).toBeCloseTo(1, 6);
        expect(mip[2]).toBeCloseTo(7, 6);
        expect(mip[3]).toBeCloseTo(2, 6);
        expect(mip[3]).toBeGreaterThan(0);
    });

    test("restricted slope moment rejects a missing k factor", () => {
        const expected = composedSlopePsd(config);
        const missingK = composedSlopePsd(config, { kMinusThreeDensity: true });
        expect(expected).toBeGreaterThan(0);
        expect(Math.abs(missingK / expected - 1)).toBeGreaterThan(0.25);
    });

    test("CPU reference emits only slope fields and no displacement field", () => {
        const result = runSlopeCpuPipeline(generateH0(config, 0), config);
        expect(result.xField.length).toBe(config.N * config.N * 2);
        expect(result.zField.length).toBe(config.N * config.N * 2);
        expect(config.lambda).toBe(0);
    });

    test("field-vs-mesh agreement stays band-limited to displacement cascades", () => {
        const meshSupportedHi = Math.max(...CASCADE_CONFIGS.map((cascade) => cascade.kHi));
        expect(config.kLo).toBeGreaterThanOrEqual(meshSupportedHi);
        expect(CASCADE_CONFIGS.every((cascade) => cascade.lambda !== 0)).toBe(true);
        // A slope cascade has no height/displacement output, so the mesh and displacement field
        // continue to share exactly the two supported bands while shading receives the tail separately.
        expect(config.lambda).toBe(0);
    });
});
