import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import * as d from "typegpu/data";
import { surfaceWgsl } from "../../shallot/src/standard/sear/pipelines";
import { catmullRom1D as reference } from "../src/reconstruction";
import { composedSlopePsd, slopeMipSize } from "../src/slope";
import { CASCADE_CONFIGS, SLOPE_CASCADE_CONFIGS } from "../src/spectrum";
import {
    oceanSurfaceFs,
    oceanSurfaceLayout,
    oceanSurfaceVaryings,
    oceanSurfaceVs,
    surfaceCatmullRom1D,
    surfaceCatmullRomDerivative1D,
    surfaceWrapIndex,
} from "../src/surface";
import {
    catmullRom1D as vertexCatmullRom1D,
    wrapIndex as vertexWrapIndex,
} from "../src/vertex-displacement";

const controls = [1.25, -0.75, 2.5, 4.125];

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

    test("resolved fragment uses hardware gradients and the residual channel without retired mechanisms", () => {
        const wgsl = surfaceWgsl({
            name: "ocean-test",
            layout: oceanSurfaceLayout,
            varyings: oceanSurfaceVaryings,
            vs: oceanSurfaceVs,
            fs: oceanSurfaceFs,
        });
        expect(wgsl).toContain("dpdx");
        expect(wgsl).toContain("dpdy");
        expect(wgsl).toContain("textureSampleGrad");
        expect(wgsl).toContain("slope.w");
    });

    test("records per-cascade k-cos response, f16 slope floor, and live MSS partition", () => {
        const rows: string[] = [];
        for (const config of CASCADE_CONFIGS) {
            const spacing = config.L / config.N;
            const k = config.kHi;
            const t = 0.37;
            const x = t * spacing;
            const samples = [-1, 0, 1, 2].map((offset) => Math.sin(k * offset * spacing));
            const derivative =
                closedDerivative(samples[0], samples[1], samples[2], samples[3], t) / spacing;
            const truth = k * Math.cos(k * x);
            rows.push(`L=${config.L} spacing=${spacing} response=${derivative / truth}`);
            expect(Number.isFinite(derivative / truth)).toBe(true);
        }
        const slopeConfig = SLOPE_CASCADE_CONFIGS[0];
        const slopeSpacing = slopeConfig.L / slopeMipSize(slopeConfig, 0);
        const probe = 1 / slopeSpacing;
        const widened = Number(new Float16Array([probe])[0]);
        const noiseFloor = Math.abs(widened - probe);
        const displacementMss = CASCADE_CONFIGS.reduce(
            (sum, config) => sum + composedSlopePsd(config),
            0,
        );
        const slopeMss = composedSlopePsd(slopeConfig);
        const slopeShare = slopeMss / (slopeMss + displacementMss);
        console.log(
            `${rows.join("; ")}; f16SlopeNoiseFloor=${noiseFloor}; slopeMssShare=${slopeShare}`,
        );
        expect(slopeMipSize(slopeConfig, 0)).toBe(slopeConfig.N);
        expect(Number.isFinite(noiseFloor) && noiseFloor >= 0).toBe(true);
        expect(slopeShare).toBeGreaterThan(0);
        expect(slopeShare).toBeLessThan(1);
    });

    test("package-wide source and prose contain none of the retired fragment mechanisms", () => {
        const root = join(import.meta.dir, "..");
        const files = [join(root, "src"), join(root, "tests")];
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
