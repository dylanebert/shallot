import { describe, expect, test } from "bun:test";
import {
    CELL_DIRECTIONAL_GLYPHS,
    CELL_FILL_GLYPHS,
    CELL_GLYPH_COUNT,
    cellGlyphChar,
    cellGlyphString,
} from "./ramp";

// N4 (shallot-tui S1's second repair round): `cellGlyphChar`, `cellGlyphString`, `CELL_FILL_GLYPHS`, and
// the `CELL_GLYPH_COUNT` boundary had zero test references repo-wide before this file — the one property
// S3 is told to build against (fill indices below the boundary, directional above, in bucket order) had
// no arm, and neither did the range throws `cellGlyphChar` names in its own docblock. This is a
// `review:stage` first-instance-of-a-pattern stage, so it's exhaustive over the
// whole index space rather than a handful of examples: the space is small (~100 entries), and a
// boundary-only sample can't see an off-by-one anywhere except at the two edges it happened to pick.
describe("cellGlyphChar / CELL_GLYPH_COUNT boundary (S3's contract, ramp.ts's module doc)", () => {
    test("CELL_GLYPH_COUNT is exactly the fill ramp plus the directional set, both non-empty", () => {
        expect(CELL_FILL_GLYPHS.length).toBeGreaterThan(0);
        expect(CELL_DIRECTIONAL_GLYPHS.length).toBeGreaterThan(0);
        expect(CELL_GLYPH_COUNT).toBe(CELL_FILL_GLYPHS.length + CELL_DIRECTIONAL_GLYPHS.length);
    });

    test("every index below CELL_FILL_GLYPHS.length reads the fill ramp in order", () => {
        for (let i = 0; i < CELL_FILL_GLYPHS.length; i++) {
            expect(cellGlyphChar(i)).toBe(CELL_FILL_GLYPHS[i]);
        }
    });

    test("every index at or above CELL_FILL_GLYPHS.length reads the directional set, in bucket order", () => {
        for (let i = CELL_FILL_GLYPHS.length; i < CELL_GLYPH_COUNT; i++) {
            expect(cellGlyphChar(i)).toBe(CELL_DIRECTIONAL_GLYPHS[i - CELL_FILL_GLYPHS.length]);
        }
    });

    test("the boundary itself: last fill index and first directional index are adjacent, and the first directional glyph isn't duplicated into the fill ramp (full disjointness pinned in ramp-table.test.ts)", () => {
        const lastFill = CELL_FILL_GLYPHS.length - 1;
        const firstDirectional = CELL_FILL_GLYPHS.length;
        expect(cellGlyphChar(lastFill)).toBe(CELL_FILL_GLYPHS[CELL_FILL_GLYPHS.length - 1]);
        expect(cellGlyphChar(firstDirectional)).toBe(CELL_DIRECTIONAL_GLYPHS[0]);
        expect(CELL_FILL_GLYPHS).not.toContain(CELL_DIRECTIONAL_GLYPHS[0]);
    });

    test("out-of-range and non-integer indices throw — the range check `cellGlyphChar`'s own guard names", () => {
        expect(() => cellGlyphChar(-1)).toThrow();
        expect(() => cellGlyphChar(CELL_GLYPH_COUNT)).toThrow();
        expect(() => cellGlyphChar(CELL_GLYPH_COUNT + 1)).toThrow();
        expect(() => cellGlyphChar(0.5)).toThrow();
        expect(() => cellGlyphChar(Number.NaN)).toThrow();
    });

    test("index 0 and CELL_GLYPH_COUNT - 1 (the admissible boundary's own endpoints) don't throw", () => {
        expect(() => cellGlyphChar(0)).not.toThrow();
        expect(() => cellGlyphChar(CELL_GLYPH_COUNT - 1)).not.toThrow();
    });
});

describe("cellGlyphString (the atlas batch-warm argument)", () => {
    test("is exactly CELL_FILL_GLYPHS then CELL_DIRECTIONAL_GLYPHS, concatenated", () => {
        expect(cellGlyphString()).toBe(
            CELL_FILL_GLYPHS.join("") + CELL_DIRECTIONAL_GLYPHS.join(""),
        );
    });

    test("has exactly CELL_GLYPH_COUNT characters — every ramp entry is a single code unit", () => {
        expect(cellGlyphString().length).toBe(CELL_GLYPH_COUNT);
    });

    test("round-trips against cellGlyphChar at every index — this is what a live atlas relies on to key its uv-rect table by cellGlyphChar(i) after warming with cellGlyphString()", () => {
        const s = cellGlyphString();
        for (let i = 0; i < CELL_GLYPH_COUNT; i++) {
            expect(s[i]).toBe(cellGlyphChar(i));
        }
    });
});
