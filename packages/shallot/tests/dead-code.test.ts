import { describe, test, expect } from "bun:test";
import { findDeadFunctions } from "./helpers/wgsl";
import { solverWGSL, coloringWGSL } from "../src/standard/physics/solver.wgsl";
import { compilePresentShader } from "../src/standard/render/present";
import { compileRasterShader, compileSkyShader } from "../src/standard/raster/forward";

// Surface API utilities: available for user surface shaders, not called by default surfaces
const SURFACE_API = new Set([
    "toOKLab",
    "fromOKLab",
    "darkTone",
    "lightTone",
    "toWorldSpace",
    "toObjectSpace",
    "hash2",
    "simplex2",
    "fbm2",
    "value2d",
    "dispatchVertexTransform",
]);

const filterSurfaceAPI = (dead: string[]) =>
    dead.filter((f) => !SURFACE_API.has(f) && !f.startsWith("userVertexTransform_"));

describe("shader dead code", () => {
    test("physics solver", () => {
        const { dead } = findDeadFunctions(solverWGSL);
        expect(dead).toEqual([]);
    });

    test("physics coloring", () => {
        const { dead } = findDeadFunctions(coloringWGSL);
        expect(dead).toEqual([]);
    });

    test("present", () => {
        const { dead } = findDeadFunctions(compilePresentShader());
        expect(dead).toEqual([]);
    });

    test("raster forward (shadows)", () => {
        const shader = compileRasterShader([{}], true);
        const { dead } = findDeadFunctions(shader);
        expect(filterSurfaceAPI(dead)).toEqual([]);
    });

    test("raster forward (no shadows)", () => {
        const shader = compileRasterShader([{}], false);
        const { dead } = findDeadFunctions(shader);
        expect(filterSurfaceAPI(dead)).toEqual([]);
    });

    test("raster sky", () => {
        const shader = compileSkyShader();
        const { dead } = findDeadFunctions(shader);
        expect(filterSurfaceAPI(dead)).toEqual([]);
    });
});
