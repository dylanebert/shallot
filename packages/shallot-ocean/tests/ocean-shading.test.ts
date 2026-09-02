import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { surfaceWgsl } from "../../shallot/src/standard/sear/pipelines";
import { catmullRom1D as reference } from "../src/reconstruction";
import {
    oceanSurfaceFs,
    oceanSurfaceLayout,
    oceanSurfaceVaryings,
    oceanSurfaceVs,
    surfaceCatmullRom1D,
    surfaceCatmullRomDerivative1D,
} from "../src/surface";

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
        for (const retired of ["foldT", "footprint", "FS_EPS", "LOD_CAP"]) {
            expect(wgsl).not.toContain(retired);
        }
    });
});
