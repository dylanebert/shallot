import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { State } from "@dylanebert/shallot";
import { Surfaces } from "@dylanebert/shallot/sear/core";
import * as d from "typegpu/data";
import { catmullRom1D as reference } from "../src/ocean/reconstruction";
import { composedSlopePsd, slopeMipSize } from "../src/ocean/slope";
import { f16NextDown, f16NextUp, f16Round } from "../src/ocean/slope-seam";
import { CASCADE_CONFIGS, SLOPE_CASCADE_CONFIGS } from "../src/ocean/spectrum";
import {
    AERIAL_DENSITY,
    aerialFade,
    BECKMANN_VARIANCE_FLOOR,
    beckmannSunRadiance,
    declaredFoamStrength,
    meanFresnel,
    registerOceanSurface,
    surfaceCatmullRom1D,
    surfaceCatmullRomDerivative1D,
    surfaceWrapIndex,
    troughFoam,
} from "../src/ocean/surface";
import {
    catmullRom1D as vertexCatmullRom1D,
    wrapIndex as vertexWrapIndex,
} from "../src/ocean/vertex-displacement";
import { oceanSurfaceRegistered } from "../src/verification/ocean-shading";

const controls = [1.25, -0.75, 2.5, 4.125];

describe("ocean surface device claim", () => {
    test("the registration half follows registerOceanSurface", () => {
        Surfaces.clear();
        const state = new State();
        try {
            expect(oceanSurfaceRegistered()).toBe(false);
            registerOceanSurface(state);
            expect(oceanSurfaceRegistered()).toBe(true);
        } finally {
            state.dispose();
            Surfaces.clear();
        }
    });
});

describe("radial aerial perspective", () => {
    test("uses the exponential distance law", () => {
        expect(AERIAL_DENSITY).toBeGreaterThan(0);
        for (const distance of [0, 1, 18, 100, 1_000]) {
            expect(aerialFade(AERIAL_DENSITY, distance)).toBeCloseTo(
                1 - Math.exp(-AERIAL_DENSITY * distance),
                6,
            );
        }
    });

    test("starts clear and approaches the horizontal sky sample monotonically", () => {
        expect(aerialFade(AERIAL_DENSITY, 0)).toBe(0);
        const distances = [1, 10, 100, 10_000];
        const weights = distances.map((distance) => aerialFade(AERIAL_DENSITY, distance));
        for (let i = 1; i < weights.length; i++) expect(weights[i]).toBeGreaterThan(weights[i - 1]);
        expect(weights.at(-1)).toBeLessThan(1);
        expect(weights.at(-1)).toBeGreaterThan(0.95);
    });
});

describe("fragment trough foam", () => {
    test("is black-pointed outside compressed troughs and wind-stretched within them", () => {
        expect(declaredFoamStrength()).toBeGreaterThan(0);
        expect(declaredFoamStrength()).toBeLessThan(0.2);
        expect(troughFoam(1, -1, d.vec2f(10, 20))).toBe(0);
        expect(troughFoam(0.4, 1, d.vec2f(10, 20))).toBe(0);
        const along = Array.from({ length: 80 }, (_, i) => troughFoam(0.35, -2, d.vec2f(i, 0)));
        const across = Array.from({ length: 80 }, (_, i) => troughFoam(0.35, -2, d.vec2f(0, i)));
        const transitions = (values: number[]) =>
            values.slice(1).filter((value, i) => value > 0 !== (values[i] ?? 0) > 0).length;
        expect(transitions(along)).toBeGreaterThan(transitions(across));
        expect(along.some((value) => value > 0)).toBe(true);
    });
});

describe("variance-driven Beckmann sun glitter", () => {
    test("derives the variance floor from f32 precision", () => {
        expect(BECKMANN_VARIANCE_FLOOR).toBe(Math.sqrt(2 ** -23));
        expect(BECKMANN_VARIANCE_FLOOR).toBeGreaterThan(0);
    });

    test("hemisphere-integrated reflected energy stays at most one", () => {
        const integrate = (sigma: number, stepsTheta: number) => {
            const stepsPhi = stepsTheta * 2;
            const dTheta = (Math.PI * 0.5) / stepsTheta;
            const dPhi = (Math.PI * 2) / stepsPhi;
            let integral = 0;
            const view = d.vec3f(0, 1, 0);
            for (let ti = 0; ti < stepsTheta; ti++) {
                const theta = (ti + 0.5) * dTheta;
                const sinTheta = Math.sin(theta);
                const cosTheta = Math.cos(theta);
                for (let pi = 0; pi < stepsPhi; pi++) {
                    const phi = (pi + 0.5) * dPhi;
                    const light = d.vec3f(
                        sinTheta * Math.cos(phi),
                        cosTheta,
                        sinTheta * Math.sin(phi),
                    );
                    integral +=
                        beckmannSunRadiance(d.vec3f(0, 1, 0), view, light, sigma * sigma) *
                        sinTheta *
                        dTheta *
                        dPhi;
                }
            }
            return integral;
        };
        for (const sigma of [0.035, 0.12, 0.3]) {
            let steps = 15;
            let prior = integrate(sigma, steps);
            let integral = prior;
            do {
                steps *= 2;
                integral = integrate(sigma, steps);
                if (Math.abs(integral - prior) <= 2 ** -10) break;
                prior = integral;
            } while (steps < 240);
            console.log(`beckmann hemisphere sigma=${sigma} steps=${steps} integral=${integral}`);
            expect(Math.abs(integral - prior)).toBeLessThanOrEqual(2 ** -10);
            expect(integral).toBeLessThanOrEqual(1);
            expect(integral).toBeGreaterThan(0);
        }
    });
});

describe("mean ocean Fresnel", () => {
    test("recovers the Schlick factor at zero slope deviation", () => {
        for (const cosine of [0, 0.2, 0.5, 0.8, 1]) {
            expect(meanFresnel(cosine, 0)).toBeCloseTo((1 - cosine) ** 5, 6);
        }
    });

    test("decreases as slope deviation widens the reflected sky", () => {
        for (const cosine of [0.05, 0.2]) {
            const smooth = meanFresnel(cosine, 0.08);
            const rough = meanFresnel(cosine, 0.24);
            expect(rough).toBeLessThan(smooth);
        }
    });
});

function closedDerivative(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
    const c = -0.5 * p0 + 0.5 * p2;
    return c + t * (2 * b + 3 * t * a);
}

describe("ocean fragment Catmull-Rom closed forms", () => {
    test("basis-vector recovery agrees with reconstruction and the vertex formula", () => {
        for (const t of [0, 0.13, 0.5, 0.87, 1]) {
            for (let basis = 0; basis < 4; basis++) {
                const p = [0, 0, 0, 0];
                p[basis] = 1;
                const vectors = p.map((value) => d.vec4f(value));
                const value = surfaceCatmullRom1D(
                    vectors[0],
                    vectors[1],
                    vectors[2],
                    vectors[3],
                    t,
                );
                expect(value.x).toBeCloseTo(reference(p[0], p[1], p[2], p[3], t), 6);
                const derivative = surfaceCatmullRomDerivative1D(
                    vectors[0],
                    vectors[1],
                    vectors[2],
                    vectors[3],
                    t,
                );
                expect(derivative.x).toBeCloseTo(closedDerivative(p[0], p[1], p[2], p[3], t), 6);
            }
        }
    });

    test("consolidated kernels stay in lockstep with the retired vertex-layout declarations", () => {
        for (const t of [0, 0.19, 0.5, 0.83, 1]) {
            const vectors = controls.map((value) => d.vec4f(value));
            expect(surfaceCatmullRom1D(vectors[0], vectors[1], vectors[2], vectors[3], t)).toEqual(
                vertexCatmullRom1D(vectors[0], vectors[1], vectors[2], vectors[3], t),
            );
        }
        for (let i = -17; i <= 17; i++) expect(surfaceWrapIndex(i, 8)).toBe(vertexWrapIndex(i, 8));
    });

    test("closed derivative responds to every control point", () => {
        for (const t of [0.17, 0.41, 0.73]) {
            const vectors = controls.map((value) => d.vec4f(value));
            const actual = surfaceCatmullRomDerivative1D(
                vectors[0],
                vectors[1],
                vectors[2],
                vectors[3],
                t,
            );
            expect(actual.x).toBeCloseTo(
                closedDerivative(controls[0], controls[1], controls[2], controls[3], t),
                6,
            );
        }
    });

    test("records per-cascade k-cos response, f16 slope floor, and live MSS partition", () => {
        const rows: string[] = [];
        const t = 0.37;
        const derivativeWeights = [
            closedDerivative(1, 0, 0, 0, t),
            closedDerivative(0, 1, 0, 0, t),
            closedDerivative(0, 0, 1, 0, t),
            closedDerivative(0, 0, 0, 1, t),
        ];
        for (const config of CASCADE_CONFIGS) {
            const spacing = config.L / config.N;
            const k = config.kHi;
            const x = t * spacing;
            const samples = [-1, 0, 1, 2].map((offset) => Math.sin(k * offset * spacing));
            const vectors = samples.map((value) => d.vec4f(value));
            const derivative =
                surfaceCatmullRomDerivative1D(vectors[0], vectors[1], vectors[2], vectors[3], t).x /
                spacing;
            const expectedDerivative =
                closedDerivative(samples[0], samples[1], samples[2], samples[3], t) / spacing;
            const truth = k * Math.cos(k * x);
            const response = derivative / truth;
            const expectedResponse = expectedDerivative / truth;
            const rounded = samples.map(f16Round);
            const quantizedVectors = rounded.map((value) => d.vec4f(value));
            const quantizedDerivative =
                surfaceCatmullRomDerivative1D(
                    quantizedVectors[0],
                    quantizedVectors[1],
                    quantizedVectors[2],
                    quantizedVectors[3],
                    t,
                ).x / spacing;
            const noiseFloor =
                rounded.reduce((sum, value, index) => {
                    const quantum = Math.max(f16NextUp(value) - value, value - f16NextDown(value));
                    return sum + Math.abs(derivativeWeights[index]) * quantum;
                }, 0) / spacing;
            rows.push(
                `L=${config.L} spacing=${spacing} response=${response} f16SlopeNoiseFloor=${noiseFloor}`,
            );
            expect(derivative).toBeCloseTo(expectedDerivative, 6);
            expect(response).toBeCloseTo(expectedResponse, 6);
            expect(Math.abs(quantizedDerivative - derivative)).toBeLessThanOrEqual(noiseFloor);
            expect(noiseFloor).toBeGreaterThan(0);
        }
        const slopeConfig = SLOPE_CASCADE_CONFIGS[0];
        const displacementMss = CASCADE_CONFIGS.reduce(
            (sum, config) => sum + composedSlopePsd(config),
            0,
        );
        const slopeMss = composedSlopePsd(slopeConfig);
        const slopeShare = slopeMss / (slopeMss + displacementMss);
        console.log(`${rows.join("; ")}; slopeMssShare=${slopeShare}`);
        expect(slopeMipSize(slopeConfig, 0)).toBe(slopeConfig.N);
        expect(slopeShare).toBeGreaterThan(0);
        expect(slopeShare).toBeLessThan(1);
    });

    test("package-wide source and prose contain none of the retired fragment mechanisms", () => {
        const root = join(import.meta.dir, "..");
        const files = [join(root, "src"), join(root, "test")];
        const retired = ["fold" + "T", "foot" + "print", "FS" + "_EPS", "LOD" + "_CAP"];
        const walk = (path: string): string[] =>
            readdirSync(path, { withFileTypes: true }).flatMap((entry) => {
                const child = join(path, entry.name);
                return entry.isDirectory() ? walk(child) : [child];
            });
        const hits = files.flatMap(walk).flatMap((path) => {
            const text = readFileSync(path, "utf8");
            return retired.filter((term) => text.includes(term)).map((term) => `${path}:${term}`);
        });
        expect(hits).toEqual([]);
    });
});
