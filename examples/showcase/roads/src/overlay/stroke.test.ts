import { describe, expect, test } from "bun:test";
import {
    OFF_ROAD_POINT,
    ON_ROAD_POINT,
    packStrokeTile,
    ROAD_ALBEDO,
    STROKE_HALF_WIDTH,
    strokeDistance,
    strokeRect,
} from "./stroke";
import { DIST_RANGE, decodeDist, dirtyTiles, TEXEL_SIZE, tileId, tileOrigin } from "./tiles";

describe("strokeDistance", () => {
    test("is negative at the centreline and positive well outside the road", () => {
        expect(strokeDistance(0, 0)).toBeLessThan(0);
        expect(strokeDistance(0, 50)).toBeGreaterThan(0);
    });

    test("is exactly -STROKE_HALF_WIDTH... 0 at the edge, by construction", () => {
        // at x=0 the segment's closest point is (0,0), so distance to the road edge is |z| - halfWidth
        expect(strokeDistance(0, STROKE_HALF_WIDTH)).toBeCloseTo(0, 10);
        expect(strokeDistance(0, STROKE_HALF_WIDTH + 1)).toBeCloseTo(1, 10);
    });

    test("ON_ROAD_POINT and OFF_ROAD_POINT classify as their names say", () => {
        expect(strokeDistance(...ON_ROAD_POINT)).toBeLessThan(0);
        expect(strokeDistance(...OFF_ROAD_POINT)).toBeGreaterThan(0);
    });
});

describe("strokeRect → dirtyTiles", () => {
    test("the stroke's dirty set is the pair of tile rows straddling Z=0, none elsewhere", () => {
        // the stroke's ±4 m width straddles the tz=7/tz=8 boundary at Z=0 (tile z ranges are
        // [tz*64-512, tz*64-512+64), so tz=7 covers [-64,0) and tz=8 covers [0,64)) — exactly two rows, not
        // one, since the road is centred exactly on a tile edge.
        const ids = dirtyTiles(strokeRect());
        expect(ids.length).toBeGreaterThan(0);
        const rows = new Set(ids.map((id) => Math.floor(id / 16)));
        expect(rows).toEqual(new Set([7, 8]));
    });

    test("the dirty column span matches the stroke's ±150 m length", () => {
        const ids = dirtyTiles(strokeRect());
        const cols = ids.map((id) => id % 16);
        // ±150 m from the origin (world half 512) → world x in [362, 662) maps to tx in [5, 10]
        expect(Math.min(...cols)).toBe(5);
        expect(Math.max(...cols)).toBe(10);
    });
});

describe("packStrokeTile — the seeded-tile readback oracle", () => {
    test("packs a known pattern into the tile straddling the stroke's centre, read back byte-exact", () => {
        // the tile whose corner sits at the world origin — squarely under the stroke's centreline
        // (STROKE_Z=0) and well inside its ±150 m length.
        const tx = 8;
        const tz = 8;
        expect(dirtyTiles(strokeRect())).toContain(tileId(tx, tz));
        const { albedo, dist } = packStrokeTile(tx, tz);

        // independent re-derivation: recompute the expected texel content at a handful of coordinates by
        // hand-walking the tile's own world origin and texel size, never calling packStrokeTile's internal
        // texelOffset — an agreement check against the same formula would only prove self-consistency, so
        // this reads the origin from tileOrigin (a different module function) and computes the byte offset
        // with an inline formula.
        const [ox, oz] = tileOrigin(tx, tz);
        const probes: Array<[number, number]> = [
            [0, 0],
            [256, 256], // tile centre
            [511, 0],
            [0, 511],
        ];
        for (const [px, py] of probes) {
            const worldX = ox + (px + 0.5) * TEXEL_SIZE;
            const worldZ = oz + (py + 0.5) * TEXEL_SIZE;
            // clamped the same way encodeDist saturates — this test's own independent clamp, not a call
            // into encodeDist, since the far probes here land well outside ±DIST_RANGE
            const expectedDist = Math.max(
                -DIST_RANGE,
                Math.min(DIST_RANGE, strokeDistance(worldX, worldZ)),
            );

            const distByteOffset = py * 512 + px; // r8unorm: 1 byte/texel, independent stride formula
            const gotDist = decodeDist(dist[distByteOffset]);
            const step = (2 * DIST_RANGE) / 255; // quantization step
            expect(Math.abs(gotDist - expectedDist)).toBeLessThanOrEqual(step / 2 + 1e-6);

            const albedoByteOffset = (py * 512 + px) * 4;
            expect(albedo[albedoByteOffset]).toBe(Math.round(ROAD_ALBEDO[0] * 255));
            expect(albedo[albedoByteOffset + 1]).toBe(Math.round(ROAD_ALBEDO[1] * 255));
            expect(albedo[albedoByteOffset + 2]).toBe(Math.round(ROAD_ALBEDO[2] * 255));
            expect(albedo[albedoByteOffset + 3]).toBe(255);
        }
    });

    test("a tile far from the stroke encodes uniformly-outside distance", () => {
        const { dist } = packStrokeTile(0, 15); // far corner, no overlap with the centre-row stroke
        // every texel should decode to (near-)saturated outside distance — a broad sample, not exhaustive
        for (let i = 0; i < dist.length; i += 997) {
            expect(decodeDist(dist[i])).toBeGreaterThan(0.5);
        }
    });
});
