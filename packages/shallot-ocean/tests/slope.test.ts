import { describe, expect, test } from "bun:test";
import { runCpuPipeline } from "../src/cpu-reference";
import { ifft2 } from "../src/fft";
import {
    assertSlopeOnly,
    composedSlopePsd,
    realizedSlopeMss,
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeMipSize,
    slopeMomentAgreementTolerance,
    slopeMomentSamplingError,
    slopeSpectra,
} from "../src/slope";
import { CASCADE_CONFIGS, generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

function rmsReal(field: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < field.length; i += 2) sum += field[i] ** 2;
    return Math.sqrt(sum / (field.length / 2));
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

    test("mip reduction publishes measured residual slope variance", () => {
        const level0 = new Float32Array([1, 0, 1, 0, 3, 0, 9, 0, 1, 2, 5, 0, 3, 2, 13, 0]);
        const mip = reduceSlopeMip(level0, 2);
        expect(mip[0]).toBeCloseTo(2, 6);
        expect(mip[1]).toBeCloseTo(1, 6);
        expect(mip[2]).toBeCloseTo(7, 6);
        expect(mip[3]).toBeCloseTo(2, 6);
        expect(mip[3]).toBeGreaterThan(0);
    });

    test("CPU reduction constructs every mip level", () => {
        const base = new Float32Array(config.N * config.N * 4);
        for (let i = 0; i < base.length; i += 4) {
            base[i] = 1;
            base[i + 1] = -1;
            base[i + 2] = 2;
        }
        const levels: Float32Array[] = [base];
        for (let mip = 1; mip < SLOPE_MIP_LEVELS; mip++) {
            levels.push(reduceSlopeMip(levels[mip - 1], slopeMipSize(config, mip - 1)));
            const size = slopeMipSize(config, mip);
            expect(levels[mip].length).toBe(size * size * 4);
        }
        expect(levels.at(-1)?.length).toBe(4);
        expect(levels.at(-1)?.[3]).toBe(0);
    });

    test("restricted slope moment ties to realized mean-square slope", () => {
        const expected = composedSlopePsd(config);
        const realized = realizedSlopeMss(runSlopeCpuPipeline(generateH0(config, 0), config));
        const samplingError = slopeMomentSamplingError(config);
        const tolerance = slopeMomentAgreementTolerance(config);
        console.log(`slope moment sampling error=${samplingError}, agreement bound=${tolerance}`);
        expect(expected).toBeGreaterThan(0);
        expect(realized).toBeGreaterThan(0);
        expect(Math.abs(realized / expected - 1)).toBeLessThan(tolerance);
    });

    test("missing gradient k red-witnesses the composed-coverage arm", () => {
        // Re-runs the guarded arm's own assertion ("restricted slope moment ties to realized
        // mean-square slope", above) with only the realized subject mutated — the expectation
        // (`composedSlopePsd`) and the shared bound (`slopeMomentAgreementTolerance`) are the
        // same ones that arm already uses, never a separately authored floor.
        const h = generateH0(config, 0);
        const expected = composedSlopePsd(config);
        const bound = slopeMomentAgreementTolerance(config);
        const greenRealized = realizedSlopeMss(runSlopeCpuPipeline(h, config));
        const mutatedRealized = realizedSlopeMss(
            runSlopeCpuPipeline(h, config, 0, { missingGradientK: true }),
        );
        const greenDeviation = Math.abs(greenRealized / expected - 1);
        const mutatedDeviation = Math.abs(mutatedRealized / expected - 1);
        console.log(
            `dropped-gradient reach: green=${greenDeviation}, mutated=${mutatedDeviation}, ` +
                `bound=${bound}, reach=${(mutatedDeviation / bound).toFixed(1)}x`,
        );
        expect(greenDeviation).toBeLessThan(bound);
        expect(mutatedDeviation).toBeGreaterThan(bound);
    });

    test("CPU reference emits only slope fields and no displacement field", () => {
        const result = runSlopeCpuPipeline(generateH0(config, 0), config);
        expect(result.xField.length).toBe(config.N * config.N * 2);
        expect(result.zField.length).toBe(config.N * config.N * 2);
        expect(config.lambda).toBe(0);
    });

    test("production slope state enumerates only slope resources", async () => {
        const source = await Bun.file(new URL("../src/slope.ts", import.meta.url)).text();
        const state = source.match(/interface SlopeState \{([\s\S]*?)\n\}/)?.[1] ?? "";
        const fields = [...state.matchAll(/^ {4}(\w+):/gm)].map((match) => match[1]);
        expect(fields).toEqual([
            "config",
            "n",
            "h0",
            "x",
            "z",
            "xTemp",
            "zTemp",
            "params",
            "mipParams",
            "texture",
            "slope",
            "post",
            "mip",
            "fftRow",
            "fftCol",
            "slopeGroup",
            "postGroup",
            "xRowGroup",
            "xColGroup",
            "zRowGroup",
            "zColGroup",
            "mipGroups",
        ]);
        expect(SLOPE_MIP_LEVELS).toBe(Math.log2(config.N) + 1);
    });

    test("displacement spectra stop at the slope-band boundary", () => {
        const boundary = 8.4823;
        for (const cascade of CASCADE_CONFIGS) {
            const result = runCpuPipeline(generateH0(cascade, 17), cascade, 0.37);
            for (let y = 0; y < cascade.N; y++) {
                for (let x = 0; x < cascade.N; x++) {
                    const k = Math.hypot(
                        kIndex(x, cascade.N) * ((2 * Math.PI) / cascade.L),
                        kIndex(y, cascade.N) * ((2 * Math.PI) / cascade.L),
                    );
                    if (k <= boundary) continue;
                    const i = (y * cascade.N + x) * 2;
                    expect(result.h[i] === 0).toBe(true);
                    expect(result.h[i + 1] === 0).toBe(true);
                    expect(result.dxSpec[i] === 0).toBe(true);
                    expect(result.dxSpec[i + 1] === 0).toBe(true);
                    expect(result.dzSpec[i] === 0).toBe(true);
                    expect(result.dzSpec[i + 1] === 0).toBe(true);
                }
            }
        }
    });

    test("the displacement-coupling arm red-witnesses a nonzero slope lambda", () => {
        expect(() => assertSlopeOnly({ ...config, lambda: 1 })).toThrow(/must not couple/);
        expect(() => assertSlopeOnly(config)).not.toThrow();
        expect(CASCADE_CONFIGS.every((cascade) => cascade.lambda !== 0)).toBe(true);
    });
});
