import { describe, expect, test } from "bun:test";
import {
    CELLS,
    gridIndices,
    HALF,
    INDEX_COUNT,
    SPACING,
    VERTEX_COUNT,
    VERTS,
    vertexIndex,
    worldX,
    worldZ,
} from "./grid";

// The mesh-topology oracle: the fixed W×H grid's vertex/index counts and index list are pure functions of
// `CELLS`, derivable by hand for a small grid and checked against `gridIndices()` here, then relied on
// (not re-derived) for the full-size counts. `generate.test.ts`'s WGSL structural check pins the GPU
// kernel writes at the same `vertexIndex` addressing this file pins on the CPU side.

describe("grid dimensions", () => {
    test("vertex/index counts derive from CELLS (fence-post + two triangles per quad)", () => {
        expect(VERTS).toBe(CELLS + 1);
        expect(VERTEX_COUNT).toBe(VERTS * VERTS);
        expect(INDEX_COUNT).toBe(CELLS * CELLS * 6);
        expect(HALF).toBe(CELLS / 2);
    });

    test("the grid centres exactly on the world origin", () => {
        expect(worldX(0)).toBe(-HALF * SPACING);
        expect(worldX(VERTS - 1)).toBe(HALF * SPACING);
        expect(worldX(HALF)).toBe(0);
        expect(worldZ(HALF)).toBe(0);
    });
});

describe("vertexIndex addressing", () => {
    test("row-major over x within z, spanning exactly [0, VERTEX_COUNT)", () => {
        expect(vertexIndex(0, 0)).toBe(0);
        expect(vertexIndex(1, 0)).toBe(1);
        expect(vertexIndex(0, 1)).toBe(VERTS);
        expect(vertexIndex(VERTS - 1, VERTS - 1)).toBe(VERTEX_COUNT - 1);
    });
});

describe("gridIndices", () => {
    test("a 1×1 cell grid emits one quad, CCW-up-facing, from its four corners", () => {
        // corners: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1) — a single quad, verts = 2
        const idx = gridIndices(1);
        expect(idx.length).toBe(6);
        expect(Array.from(idx)).toEqual([0, 2, 1, 1, 2, 3]);
    });

    test("a 2×2 cell grid emits four quads over a 3×3 vertex fence-post, index count = 24", () => {
        const cells = 2;
        const idx = gridIndices(cells);
        expect(idx.length).toBe(cells * cells * 6);

        // hand-derive the same four quads by the documented corner formula and compare directly —
        // an independent derivation of the same rule, not a call into gridIndices itself.
        const verts = cells + 1;
        const expected: number[] = [];
        for (let iz = 0; iz < cells; iz++) {
            for (let ix = 0; ix < cells; ix++) {
                const i0 = iz * verts + ix;
                const i1 = i0 + 1;
                const i2 = i0 + verts;
                const i3 = i2 + 1;
                expected.push(i0, i2, i1, i1, i2, i3);
            }
        }
        expect(Array.from(idx)).toEqual(expected);

        // every index stays within the fence-post vertex count, and each of the 9 vertices is referenced
        // by at least one triangle (no orphan corner).
        const seen = new Set<number>();
        for (const i of idx) {
            expect(i).toBeGreaterThanOrEqual(0);
            expect(i).toBeLessThan(verts * verts);
            seen.add(i);
        }
        expect(seen.size).toBe(verts * verts);
    });

    test("the full-size grid (CELLS) matches the documented counts", () => {
        expect(gridIndices().length).toBe(INDEX_COUNT);
    });
});
