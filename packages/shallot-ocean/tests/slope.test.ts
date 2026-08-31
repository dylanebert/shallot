import { describe, expect, test } from "bun:test";
import { ifft2 } from "../src/fft";
import {
    composedSlopePsd,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeSpectra,
} from "../src/slope";
import { CASCADE_CONFIGS, declaredBandVariance, generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

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
        expect(config.kLo).toBeCloseTo(8.482300164692441, 12);
        expect(config.kHi).toBe(60);
        expect(config.lambda).toBe(0);
        expect(SLOPE_MIP_LEVELS).toBe(Math.log2(config.N) + 1);
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
        expect(composedSlopePsd(config)).toBeCloseTo(declaredBandVariance(config), 12);
        expect(composedSlopePsd(denser)).toBeCloseTo(declaredBandVariance(denser), 12);
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
