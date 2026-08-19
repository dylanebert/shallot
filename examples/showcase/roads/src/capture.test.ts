import { describe, expect, test } from "bun:test";
import { encodePos } from "@dylanebert/shallot/utils/core";
import * as d from "typegpu/data";
import { meshHeightAt, TRANSITION_TOLERANCE_PX } from "./capture";
import { TERRAIN_QUANT } from "./terrain/generate";
import { HALF, SPACING, VERTEX_COUNT, vertexIndex, worldX, worldZ } from "./terrain/grid";

describe("TRANSITION_TOLERANCE_PX", () => {
    test("is a small positive pixel bound, not a fraction or a huge slack", () => {
        // the mechanism argument (capture.ts's docstring) puts the formula's own band at ~0.5 px; the
        // constant should sit within a small integer multiple of that, not an order of magnitude off in
        // either direction (too tight would fail on ordinary filtering slop, too loose stops catching a
        // genuinely soft/misregistered edge — the exact defect this gate exists to find).
        expect(TRANSITION_TOLERANCE_PX).toBeGreaterThan(0);
        expect(TRANSITION_TOLERANCE_PX).toBeLessThanOrEqual(4);
    });
});

// meshHeightAt's own oracle: a synthetic raw vertex stream (one grid.ts quad's worth of real values,
// everything else left at the quantizer's zero point) — device-free, since encodePos/decodePos are plain
// callable functions outside a dispatched kernel (the same pattern flatten.test.ts's flattenHeightGpu
// differential relies on).
function rawWithQuad(
    ix0: number,
    iz0: number,
    h00: number,
    h10: number,
    h01: number,
    h11: number,
): Uint32Array {
    const raw = new Uint32Array(VERTEX_COUNT * 4);
    const set = (ix: number, iz: number, y: number) => {
        const w = encodePos(d.vec3f(worldX(ix), y, worldZ(iz)), 0, TERRAIN_QUANT);
        const idx = vertexIndex(ix, iz);
        raw[idx * 4] = w.x;
        raw[idx * 4 + 1] = w.y;
    };
    set(ix0, iz0, h00);
    set(ix0 + 1, iz0, h10);
    set(ix0, iz0 + 1, h01);
    set(ix0 + 1, iz0 + 1, h11);
    return raw;
}

// quantization step (posScale.y / 65535, TERRAIN_QUANT's own AABB) — the tolerance every assertion below
// is derived against, not a value picked to make the test pass.
const QUANT_TOL = TERRAIN_QUANT.posScale.y / 65535;

describe("meshHeightAt — the real rendered surface, not nearest-vertex", () => {
    const ix0 = HALF; // an arbitrary interior column, away from the grid's own edges
    const iz0 = HALF;
    const x0 = worldX(ix0);
    const z0 = worldZ(iz0);

    test("at a vertex itself, reads that vertex's own height exactly (within quantization)", () => {
        const raw = rawWithQuad(ix0, iz0, 5, 8, 2, 11);
        expect(meshHeightAt(raw, x0, z0)).toBeCloseTo(5, 2);
    });

    test("a flat quad reads flat everywhere inside it, both triangles", () => {
        const raw = rawWithQuad(ix0, iz0, 3, 3, 3, 3);
        expect(meshHeightAt(raw, x0 + SPACING * 0.2, z0 + SPACING * 0.2)).toBeCloseTo(3, 2); // lower-left tri
        expect(meshHeightAt(raw, x0 + SPACING * 0.8, z0 + SPACING * 0.8)).toBeCloseTo(3, 2); // upper-right tri
    });

    test("interpolates linearly within the lower-left triangle (i0, i2, i1)", () => {
        const raw = rawWithQuad(ix0, iz0, 0, 10, 0, 10);
        // (tx, tz) = (0.3, 0.3): inside the i0/i2/i1 triangle (tx+tz <= 1). h00=0, h10=10, h01=0, so this
        // quad's height only varies along x — the plane through those three points gives 0.3 * 10 = 3.
        const h = meshHeightAt(raw, x0 + SPACING * 0.3, z0 + SPACING * 0.3);
        expect(h).toBeCloseTo(3, 1);
    });

    test("interpolates linearly within the upper-right triangle (i1, i2, i3)", () => {
        const raw = rawWithQuad(ix0, iz0, 0, 10, 0, 10);
        // (tx, tz) = (0.9, 0.9): tx+tz > 1, the i1/i2/i3 triangle, defined by h10=10, h01=0, h11=10 — that
        // plane is h = 10*tx regardless of tz (h10 and h11 agree at tx=1; h01 is 0 at tx=0), so (0.9, 0.9)
        // reads 9, not the h00/h10/h01 plane's own reading at the same (tx, tz) — a different triangle.
        const h = meshHeightAt(raw, x0 + SPACING * 0.9, z0 + SPACING * 0.9);
        expect(h).toBeCloseTo(9, 1);
    });

    test("mutation: swapping the triangle split (as if grid.ts's winding flipped) changes the reading — the test discriminates the split, not just the corner values", () => {
        const raw = rawWithQuad(ix0, iz0, 0, 0, 0, 10); // only h11 is non-zero
        const inLowerLeft = meshHeightAt(raw, x0 + SPACING * 0.2, z0 + SPACING * 0.2);
        const inUpperRight = meshHeightAt(raw, x0 + SPACING * 0.8, z0 + SPACING * 0.8);
        expect(inLowerLeft).toBeCloseTo(0, 1); // h11 doesn't reach the lower-left triangle at all
        expect(inUpperRight).toBeGreaterThan(QUANT_TOL * 10); // h11 does reach the upper-right triangle
    });
});
