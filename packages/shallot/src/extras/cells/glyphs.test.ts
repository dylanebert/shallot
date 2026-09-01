import { describe, expect, test } from "bun:test";
import type { GlyphAtlas } from "../text/core";
import { glyphUvRect, glyphUvTable, MISSING_GLYPH_UV } from "./glyphs";
import { CELL_GLYPH_COUNT, cellGlyphChar } from "./ramp";

// glyphUvRect/glyphUvTable are pure over `atlas.glyphs` (a Map<string, GlyphMetrics>) — a hand-built
// duck-typed atlas reaches them with no real font/device, the same shape `text/atlas.test.ts`'s
// `fakeAtlas` uses for `computeGlyphMetrics`.
function fakeAtlas(entries: Record<string, [number, number, number, number]>): GlyphAtlas {
    const glyphs = new Map<string, { u0: number; v0: number; u1: number; v1: number }>();
    for (const [char, [u0, v0, u1, v1]] of Object.entries(entries)) {
        glyphs.set(char, { u0, v0, u1, v1 });
    }
    return { glyphs } as unknown as GlyphAtlas;
}

describe("glyphUvRect", () => {
    test("reads a resident glyph's uv rect off the atlas", () => {
        const char = cellGlyphChar(0);
        const atlas = fakeAtlas({ [char]: [0.1, 0.2, 0.3, 0.4] });
        expect(glyphUvRect(atlas, 0)).toEqual([0.1, 0.2, 0.3, 0.4]);
    });

    test("returns the zero-area sentinel when the font has no outline for the glyph", () => {
        const atlas = fakeAtlas({});
        expect(glyphUvRect(atlas, 0)).toEqual(MISSING_GLYPH_UV);
        // the sentinel's own contract — `draw.ts`'s fragment stage keys on u1 <= u0
        expect(MISSING_GLYPH_UV[2]).toBeLessThanOrEqual(MISSING_GLYPH_UV[0]);
    });

    test("a real glyph's rect never collides with the sentinel (u1 > u0)", () => {
        // any real atlas-packed glyph has a positive-area SDF cell (`atlas.ts`'s SDF_SIZE), so this is a
        // structural property of the contract, not a specific atlas's data
        const char = cellGlyphChar(0);
        const atlas = fakeAtlas({ [char]: [0, 0, 0.05, 0.05] });
        const [u0, , u1] = glyphUvRect(atlas, 0);
        expect(u1).toBeGreaterThan(u0);
    });
});

describe("glyphUvTable", () => {
    test("packs every ramp index in order, missing glyphs as the sentinel", () => {
        // resident only at index 0 and the last index — every one in between must read the sentinel
        const first = cellGlyphChar(0);
        const last = cellGlyphChar(CELL_GLYPH_COUNT - 1);
        const atlas = fakeAtlas({
            [first]: [0, 0, 0.1, 0.1],
            [last]: [0.5, 0.5, 0.6, 0.6],
        });
        const table = glyphUvTable(atlas);
        expect(table.length).toBe(CELL_GLYPH_COUNT * 4);
        // Float32Array storage — a Math.fround round-trip through the reference values, not a precision bug
        expect(Array.from(table.slice(0, 4))).toEqual(Array.from(Float32Array.of(0, 0, 0.1, 0.1)));
        expect(Array.from(table.slice((CELL_GLYPH_COUNT - 1) * 4, CELL_GLYPH_COUNT * 4))).toEqual(
            Array.from(Float32Array.of(0.5, 0.5, 0.6, 0.6)),
        );
        if (CELL_GLYPH_COUNT > 2) {
            expect(Array.from(table.slice(4, 8))).toEqual([...MISSING_GLYPH_UV]);
        }
    });
});
