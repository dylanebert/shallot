import { describe, expect, test } from "bun:test";
import { brush, march } from "./edit";
import { chunkSlot, DENSITY, DIM, get, ISO, solid, TOTAL_CELLS } from "./grid";
import { single, slab, solidChunk } from "./patterns";

// world center of cell `c` on any axis: cell c spans world [c − DIM/2, c+1 − DIM/2] (VOXEL = 1), so its
// center is c + 0.5 − DIM/2. The DDA inverts this mapping, so the tests address cells by their world center.
const HALF = DIM.x / 2; // DIM is cubic (SLOTS·CHUNK on each axis)
const center = (c: number) => c + 0.5 - HALF;

describe("march — cursor pick (Amanatides–Woo)", () => {
    test("hits the solid cell, reporting the entry face's air cell + world distance", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        // a +x ray from far outside the grid, aimed down the row of cell (·,50,60)
        const hit = march(data, [-200, center(50), center(60)], [1, 0, 0], 1000);
        expect(hit).not.toBeNull();
        expect(hit?.cell).toEqual([40, 50, 60]);
        expect(hit?.place).toEqual([39, 50, 60]); // the air cell across the −x entry face
        // entry face of cell 40 is at world x = 40 − HALF = −88, so distance = −88 − (−200) = 112
        expect(hit?.distance).toBeCloseTo(112, 5);
    });

    test("picks the FIRST solid cell along the ray, not a farther one", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        single(data, 60, 50, 60); // farther along +x
        const hit = march(data, [-200, center(50), center(60)], [1, 0, 0], 1000);
        expect(hit?.cell).toEqual([40, 50, 60]);
    });

    test("a downward ray rests on the top face of a ground slab", () => {
        const data = new Float32Array(TOTAL_CELLS);
        slab(data, 3); // solid for y < 3
        // origin inside the grid (world y 100 ∈ [−128, 128]), aimed straight down at cell (128,·,128)
        const hit = march(data, [center(128), 100, center(128)], [0, -1, 0], 1000);
        expect(hit?.cell).toEqual([128, 2, 128]); // top solid layer
        expect(hit?.place).toEqual([128, 3, 128]); // the air cell above it (+y entry face)
        // top face of cell 2 is world y = 3 − HALF = −125, so distance = 100 − (−125) = 225
        expect(hit?.distance).toBeCloseTo(225, 5);
    });

    test("returns null when the ray clears the grid without hitting a solid cell", () => {
        const data = new Float32Array(TOTAL_CELLS); // all air
        expect(march(data, [-200, center(50), center(60)], [1, 0, 0], 1000)).toBeNull();
    });

    test("returns null when the grid is behind the ray (AABB clip rejects it)", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60);
        // aimed −x, away from the grid that sits in front along +x
        expect(march(data, [-200, center(50), center(60)], [-1, 0, 0], 1000)).toBeNull();
    });

    test("respects maxDist — a hit beyond the reach is not returned", () => {
        const data = new Float32Array(TOTAL_CELLS);
        single(data, 40, 50, 60); // entry face at distance 112
        expect(march(data, [-200, center(50), center(60)], [1, 0, 0], 100)).toBeNull();
        expect(march(data, [-200, center(50), center(60)], [1, 0, 0], 120)?.cell).toEqual([
            40, 50, 60,
        ]);
    });
});

describe("brush — additive density (add-weight sculpt)", () => {
    test("a sub-threshold dab accumulates weight but flips no cell (no instant sphere)", () => {
        const data = new Float32Array(TOTAL_CELLS);
        const touched = brush(data, 40, 50, 60, 4, ISO - 0.1); // center gains < ISO
        expect(get(data, 40, 50, 60)).toBeGreaterThan(0); // weight accumulated
        expect(solid(data, 40, 50, 60)).toBe(false); // but not yet across the threshold
        expect(touched.size).toBe(0); // no occupancy flip → nothing to re-mesh
    });

    test("repeated dabs accumulate across ISO — the surface grows continuously", () => {
        const data = new Float32Array(TOTAL_CELLS);
        brush(data, 40, 50, 60, 4, 0.3); // 0.3 at center: below ISO
        expect(solid(data, 40, 50, 60)).toBe(false);
        const touched = brush(data, 40, 50, 60, 4, 0.3); // 0.6 at center: crosses ISO
        expect(solid(data, 40, 50, 60)).toBe(true);
        expect(touched.has(chunkSlot(40, 50, 60))).toBe(true); // the crossing frame re-meshes
    });

    test("a full-strength add grows center-first (the falloff leaves the rim unset)", () => {
        const data = new Float32Array(TOTAL_CELLS);
        brush(data, 40, 50, 60, 8, DENSITY);
        expect(solid(data, 40, 50, 60)).toBe(true); // center crosses
        expect(solid(data, 47, 50, 60)).toBe(false); // dist 7 of radius 8: falloff < ISO → still air
    });

    test("carve removes weight center-first, leaving the rim solid", () => {
        const data = new Float32Array(TOTAL_CELLS);
        solidChunk(data, 1, 1, 1); // cells [32, 64) solid
        const touched = brush(data, 48, 48, 48, 4, -DENSITY);
        expect(solid(data, 48, 48, 48)).toBe(false); // center carved
        expect(solid(data, 48, 48, 52)).toBe(true); // dist 4: falloff small → still solid
        expect(touched.has(chunkSlot(48, 48, 48))).toBe(true);
    });

    test("a carve across a chunk seam marks both chunks for re-upload", () => {
        const data = new Float32Array(TOTAL_CELLS);
        solidChunk(data, 0, 1, 1); // x [0, 32)
        solidChunk(data, 1, 1, 1); // x [32, 64)
        const touched = brush(data, 31, 48, 48, 3, -DENSITY); // centered on the x = 32 seam
        expect(touched.has(chunkSlot(31, 48, 48))).toBe(true); // chunk on the −x side
        expect(touched.has(chunkSlot(32, 48, 48))).toBe(true); // chunk on the +x side
        expect(touched.size).toBeGreaterThanOrEqual(2);
    });

    test("carving air changes nothing (clamped at 0) — no chunks touched", () => {
        const data = new Float32Array(TOTAL_CELLS);
        expect(brush(data, 48, 48, 48, 3, -DENSITY).size).toBe(0);
    });

    test("clamps to the grid at a corner without writing out of bounds", () => {
        const data = new Float32Array(TOTAL_CELLS);
        const touched = brush(data, 0, 0, 0, 4, DENSITY);
        expect(solid(data, 0, 0, 0)).toBe(true);
        expect(touched.has(chunkSlot(0, 0, 0))).toBe(true);
    });
});
