import { describe, expect, test } from "bun:test";
import { WORLD_HALF } from "../terrain/grid";
import {
    DIST_RANGE,
    decodeDist,
    dirtyTiles,
    encodeDist,
    TILE_COUNT,
    TILE_RES,
    TILE_SIZE,
    TILES_PER_SIDE,
    texelOffset,
    tileId,
    tileOf,
} from "./tiles";

describe("tile addressing", () => {
    test("tileId is row-major over tx within tz, matching TILES_PER_SIDE", () => {
        expect(tileId(0, 0)).toBe(0);
        expect(tileId(1, 0)).toBe(1);
        expect(tileId(0, 1)).toBe(TILES_PER_SIDE);
        expect(tileId(TILES_PER_SIDE - 1, TILES_PER_SIDE - 1)).toBe(TILE_COUNT - 1);
    });

    test("tileOf maps the world origin to the grid's centre tile", () => {
        // the grid centres on the origin (grid.ts), so (0, 0) sits at the boundary between the two centre
        // tiles on each axis — tileOf floors, landing on the tile whose min corner is the origin.
        expect(tileOf(0, 0)).toEqual([TILES_PER_SIDE / 2, TILES_PER_SIDE / 2]);
    });

    test("tileOf clamps the far edge into the last tile column, not one past it", () => {
        expect(tileOf(WORLD_HALF, WORLD_HALF)).toEqual([TILES_PER_SIDE - 1, TILES_PER_SIDE - 1]);
        expect(tileOf(-WORLD_HALF, -WORLD_HALF)).toEqual([0, 0]);
    });
});

describe("dirtyTiles — the dirty-set oracle", () => {
    // an independent hand-derivation of the analytic tile set, walking the full grid by its own formula
    // rather than calling dirtyTiles' own tileOf/tileId — an agreement check between two things one author
    // wrote from one document tests the document's self-consistency, never its truth: this derives the set
    // a different way (world-space overlap of each tile's known footprint) so the two
    // can actually disagree if either has a bug.
    function handDerivedDirtySet(minX: number, minZ: number, maxX: number, maxZ: number): number[] {
        const out: number[] = [];
        for (let tz = 0; tz < TILES_PER_SIDE; tz++) {
            const tileMinZ = tz * TILE_SIZE - WORLD_HALF;
            const tileMaxZ = tileMinZ + TILE_SIZE;
            if (tileMaxZ <= minZ || tileMinZ >= maxZ) continue;
            for (let tx = 0; tx < TILES_PER_SIDE; tx++) {
                const tileMinX = tx * TILE_SIZE - WORLD_HALF;
                const tileMaxX = tileMinX + TILE_SIZE;
                if (tileMaxX <= minX || tileMinX >= maxX) continue;
                out.push(tz * TILES_PER_SIDE + tx);
            }
        }
        return out;
    }

    test("a rect straddling the origin returns exactly the four tiles meeting there", () => {
        const rect = { minX: -1, minZ: -1, maxX: 1, maxZ: 1 };
        const got = dirtyTiles(rect).sort((a, b) => a - b);
        const want = handDerivedDirtySet(rect.minX, rect.minZ, rect.maxX, rect.maxZ).sort(
            (a, b) => a - b,
        );
        expect(got).toEqual(want);
        expect(got.length).toBe(4);
    });

    test("a rect fully inside one tile returns exactly that tile", () => {
        const rect = { minX: 10, minZ: 10, maxX: 12, maxZ: 12 };
        const got = dirtyTiles(rect);
        const want = handDerivedDirtySet(rect.minX, rect.minZ, rect.maxX, rect.maxZ);
        expect(got).toEqual(want);
        expect(got.length).toBe(1);
    });

    test("a rect spanning a 3×2 tile block matches the hand-derived set, clamped at the grid edge", () => {
        // deliberately overruns past the -Z/-X edge to exercise the clamp arm too
        const rect = {
            minX: -WORLD_HALF - 50,
            minZ: -WORLD_HALF - 50,
            maxX: -WORLD_HALF + 100,
            maxZ: -WORLD_HALF + 70,
        };
        const got = dirtyTiles(rect).sort((a, b) => a - b);
        const want = handDerivedDirtySet(rect.minX, rect.minZ, rect.maxX, rect.maxZ).sort(
            (a, b) => a - b,
        );
        expect(got).toEqual(want);
    });

    test("redrawing only touches the tiles the edit rect overlaps — a disjoint rect returns a disjoint set", () => {
        const near = dirtyTiles({ minX: -5, minZ: -5, maxX: 5, maxZ: 5 });
        const far = dirtyTiles({ minX: 200, minZ: 200, maxX: 205, maxZ: 205 });
        for (const id of far) expect(near).not.toContain(id);
    });
});

describe("distance codec round trip (the CPU↔GPU byte boundary)", () => {
    test("encode → decode recovers the original value within one quantization step", () => {
        const step = (2 * DIST_RANGE) / 255;
        for (const metres of [-1, -0.5, -0.1, 0, 0.1, 0.37, 0.999, 1]) {
            const round = decodeDist(encodeDist(metres));
            expect(Math.abs(round - metres)).toBeLessThanOrEqual(step / 2 + 1e-9);
        }
    });

    test("clamps outside ±DIST_RANGE to the saturated bytes", () => {
        expect(encodeDist(5)).toBe(255);
        expect(encodeDist(-5)).toBe(0);
    });

    test("zero distance encodes to the midpoint byte", () => {
        expect(encodeDist(0)).toBe(Math.round(0.5 * 255));
    });
});

describe("texelOffset — an independent stride derivation", () => {
    test("matches a hand-walked row-major byte count for a handful of texels", () => {
        // independent of texelOffset's own (y * TILE_RES + x) * bpt formula: walk texels in the same
        // row-major order one at a time, incrementing a running byte counter, and compare.
        const bpt = 4;
        const probes: Array<[number, number]> = [
            [0, 0],
            [1, 0],
            [0, 1],
            [TILE_RES - 1, 0],
            [0, TILE_RES - 1],
            [TILE_RES - 1, TILE_RES - 1],
        ];
        for (const [px, py] of probes) {
            let counted = 0;
            outer: for (let y = 0; y < TILE_RES; y++) {
                for (let x = 0; x < TILE_RES; x++) {
                    if (x === px && y === py) break outer;
                    counted += bpt;
                }
            }
            expect(texelOffset(px, py, bpt)).toBe(counted);
        }
    });
});
