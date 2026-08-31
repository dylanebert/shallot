import { describe, expect, test } from "bun:test";
import { runCpuPipeline } from "../src/cpu-reference";
import { ifft2 } from "../src/fft";
import {
    composedSlopePsd,
    rasterSlopeMoment,
    realizedSlopeMss,
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
                k ** 4 *
                dLog *
                dTheta;
        }
    }
    return sum;
}

function rmsReal(field: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < field.length; i += 2) sum += field[i] ** 2;
    return Math.sqrt(sum / (field.length / 2));
}

function meshReconstruct(field: ArrayLike<number>, N: number, u: number, v: number): number {
    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const fx = u - x0;
    const fy = v - y0;
    const at = (x: number, y: number) => field[((y + N) % N) * N + ((x + N) % N)];
    return (
        at(x0, y0) * (1 - fx) * (1 - fy) +
        at(x0 + 1, y0) * fx * (1 - fy) +
        at(x0, y0 + 1) * (1 - fx) * fy +
        at(x0 + 1, y0 + 1) * fx * fy
    );
}

function directFieldAt(spectrum: Float32Array, N: number, u: number, v: number): number {
    let value = 0;
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const phase = (2 * Math.PI * (x * u + y * v)) / N;
            value +=
                spectrum[(y * N + x) * 2] * Math.cos(phase) -
                spectrum[(y * N + x) * 2 + 1] * Math.sin(phase);
        }
    }
    return value;
}

function realGrid(field: Float32Array): Float64Array {
    const out = new Float64Array(field.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = field[i * 2];
    return out;
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

    test("realized slope magnitude observes i*k rather than i*kHat", () => {
        const index = 20;
        const h = new Float32Array(config.N * config.N * 2);
        h[index * 2] = 1;
        const height = ifft2(h, config.N);
        const slope = ifft2(slopeSpectra(h, config).x, config.N);
        const targetK = index * ((2 * Math.PI) / config.L);
        expect(rmsReal(slope) / rmsReal(height)).toBeCloseTo(targetK, 5);
        expect(rmsReal(slope) / rmsReal(height)).toBeGreaterThan(1);
    });

    test("N invariance preserves the declared mode and rasterized slope moment", () => {
        const denser = { ...config, N: config.N * 2 };
        const referenceMoment = independentSlopeMoment(config);
        const measuredMoment = composedSlopePsd(config);
        const rasterMoment = rasterSlopeMoment(generateH0(config, 17), config);
        const denserRasterMoment = rasterSlopeMoment(generateH0(denser, 17), denser);
        expect(measuredMoment).toBeGreaterThan(0);
        expect(Math.abs(measuredMoment / referenceMoment - 1)).toBeLessThan(0.005);
        expect(rasterMoment).toBeGreaterThan(0);
        expect(denserRasterMoment).toBeGreaterThan(0);
        expect(Math.abs(rasterMoment / measuredMoment - 1)).toBeLessThan(0.25);
        expect(Math.abs(denserRasterMoment / measuredMoment - 1)).toBeLessThan(0.25);
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

    test("restricted slope moment ties to realized mean-square slope", () => {
        const expected = composedSlopePsd(config);
        const realized = realizedSlopeMss(runSlopeCpuPipeline(generateH0(config, 0), config));
        expect(expected).toBeGreaterThan(0);
        expect(realized).toBeGreaterThan(0);
        expect(Math.abs(realized / expected - 1)).toBeLessThan(0.25);
    });

    test("missing gradient k red-witnesses both integral and spectra paths", () => {
        const h = generateH0(config, 0);
        const correct = runSlopeCpuPipeline(h, config);
        const missingResult = runSlopeCpuPipeline(h, config, 0, { missingGradientK: true });
        const correctEnergy = correct.x.reduce((sum, value) => sum + value * value, 0);
        const missingEnergy = missingResult.x.reduce((sum, value) => sum + value * value, 0);
        expect(Math.abs(missingEnergy / correctEnergy - 1)).toBeGreaterThan(0.25);
        const expected = composedSlopePsd(config);
        const missingMoment = composedSlopePsd(config, { missingGradientK: true });
        expect(Math.abs(missingMoment / expected - 1)).toBeGreaterThan(0.25);
    });

    test("CPU reference emits only slope fields and no displacement field", () => {
        const result = runSlopeCpuPipeline(generateH0(config, 0), config);
        expect(result.xField.length).toBe(config.N * config.N * 2);
        expect(result.zField.length).toBe(config.N * config.N * 2);
        expect(config.lambda).toBe(0);
    });

    test("field-vs-mesh instrument agrees on reconstructed displacement samples", () => {
        const base = { ...CASCADE_CONFIGS[0], N: 256 };
        const result = runCpuPipeline(generateH0(base, 17), base, 0.37);
        const heightMesh = realGrid(result.height);
        const dxMesh = realGrid(result.dxHeight).map((value) => base.lambda * value);
        const dzMesh = realGrid(result.dzHeight).map((value) => base.lambda * value);
        for (const [u, v] of [
            [1.01, 2.01],
            [4.04, 7.04],
            [11.99, 13.99],
            [15.01, 0.01],
        ]) {
            expect(
                Math.abs(
                    meshReconstruct(heightMesh, base.N, u, v) -
                        directFieldAt(result.h, base.N, u, v),
                ),
            ).toBeLessThan(0.5);
            expect(
                Math.abs(
                    meshReconstruct(dxMesh, base.N, u, v) -
                        base.lambda * directFieldAt(result.dxSpec, base.N, u, v),
                ),
            ).toBeLessThan(0.5);
            expect(
                Math.abs(
                    meshReconstruct(dzMesh, base.N, u, v) -
                        base.lambda * directFieldAt(result.dzSpec, base.N, u, v),
                ),
            ).toBeLessThan(0.5);
        }
        const meshSupportedHi = Math.max(...CASCADE_CONFIGS.map((cascade) => cascade.kHi));
        expect(config.kLo).toBeGreaterThanOrEqual(meshSupportedHi);
    });

    test("zero displacement remains a separate structural slope arm", () => {
        expect(config.lambda).toBe(0);
        expect(CASCADE_CONFIGS.every((cascade) => cascade.lambda !== 0)).toBe(true);
    });
});
