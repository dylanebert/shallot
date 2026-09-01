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
    slopeMipAgreement,
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

    test("publication enumeration covers the repo-wide slope reader boundary", async () => {
        const shallotRoot = new URL("../../../", import.meta.url).pathname;
        const productionFiles: string[] = [];
        for (const pattern of ["packages/**/*.ts", "examples/**/*.ts"]) {
            for await (const path of new Bun.Glob(pattern).scan({ cwd: shallotRoot })) {
                if (
                    !path.endsWith(".test.ts") &&
                    !path.endsWith(".oracle.ts") &&
                    !path.endsWith(".tier.ts")
                )
                    productionFiles.push(path);
            }
        }
        const publications: string[] = [];
        const publicationPattern = /Compute\.textures\.set\(\s*([`"'])([^`"']*)\1/g;
        for (const path of productionFiles) {
            const text = await Bun.file(new URL(path, `file://${shallotRoot}/`)).text();
            for (const match of text.matchAll(publicationPattern)) {
                if (match[2].startsWith("slope")) publications.push(match[2]);
            }
        }
        expect(productionFiles.length).toBeGreaterThan(0);
        expect(publications.length).toBeGreaterThan(0);
        expect(publications.every((name) => name.startsWith("slope"))).toBe(true);

        const readsSlopeForDisplacement = (text: string): boolean =>
            /(?:getSlopeTexture\s*\(|Compute\.textures\.get\(\s*[`"']slope)/.test(text) &&
            /\b(?:displace|height|jacob(?:ian)?|vertex|mesh)\b/i.test(text);
        expect(
            readsSlopeForDisplacement("const texture = getSlopeTexture(); use(displace(texture));"),
        ).toBe(true);

        const slopeName = publications.find((name) => !name.includes("${"));
        const directReader = slopeName
            ? new RegExp(`Compute\\.textures\\.get\\(\\s*["']${slopeName}["']`)
            : /$a/;
        for (const path of productionFiles) {
            if (path === "packages/shallot-ocean/src/slope.ts") continue;
            const text = await Bun.file(new URL(path, `file://${shallotRoot}/`)).text();
            expect(readsSlopeForDisplacement(text)).toBe(false);
            expect(directReader.test(text)).toBe(false);
        }
    });

    test("mip agreement rejects missing, truncated, and non-finite per-level payloads", () => {
        const expected = [new Float32Array(16).fill(1), new Float32Array([0, 0, 7.25, 0.5])];
        expected[0][3] = 0;
        expected[0][7] = 0;
        expected[0][11] = 0;
        expected[0][15] = 0;
        const valid = expected.map((level) => new Float32Array(level));
        expect(slopeMipAgreement(valid, expected, 0).pass).toBe(true);
        expect(slopeMipAgreement([valid[0]], expected, 0).pass).toBe(false);
        expect(slopeMipAgreement([valid[0], new Float32Array([1.5])], expected, 0).pass).toBe(
            false,
        );
        const nonFinite = valid.map((level) => new Float32Array(level));
        nonFinite[1][0] = Number.NaN;
        expect(slopeMipAgreement(nonFinite, expected, 0).pass).toBe(false);
    });
});
