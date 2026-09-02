import { describe, expect, test } from "bun:test";
import { computeRampTable, glyphCoverage, loadBrandFont } from "../../../scripts/generate-ramp";
import { CELL_DIRECTIONAL_GLYPHS, CELL_FILL_EXCLUDED_GLYPHS } from "./ramp";
import { RAMP_TABLE } from "./ramp-table";

// The reproduction arm the Locked decision requires (`specs/shallot-tui.md`'s glyph-selection
// addendum): "a derived table nobody can re-derive is a hand-authored guess with extra steps" — this
// proves `ramp-table.ts` (the committed data) is exactly what `generate-ramp.ts`'s own pure
// `computeRampTable` produces off the same brand font, right now, not just at the moment it was
// generated.
describe("RAMP_TABLE reproduces from generate-ramp.ts's own computeRampTable", () => {
    test("the committed table is identical to a fresh computation off the brand font", () => {
        const fresh = computeRampTable(loadBrandFont());
        expect(RAMP_TABLE).toEqual(fresh);
    });

    // N5 (shallot-tui S1's second repair round): this used to check `RAMP_TABLE[i].coverage` — the
    // *committed* field — against itself, which can only ever red together with the reproduction test
    // above (any RAMP_TABLE row order that broke monotonicity would already fail `toEqual(fresh)`, since
    // `computeRampTable` is sorted by construction and `fresh` would then differ from RAMP_TABLE in
    // exactly the positions that broke sort order). It never independently discriminated anything. This
    // version re-measures each row's coverage straight off the font with `glyphCoverage` — bypassing both
    // the committed `.coverage` field and `computeRampTable`'s own comparator — so it still reds if the
    // *font* or the *character set* changes the true ordering, even with the reproduction test disabled.
    test("is sorted ascending by independently re-measured ink coverage", () => {
        expect(RAMP_TABLE.length).toBeGreaterThan(0);
        const font = loadBrandFont();
        for (let i = 1; i < RAMP_TABLE.length; i++) {
            const prev = glyphCoverage(font, RAMP_TABLE[i - 1].char);
            const curr = glyphCoverage(font, RAMP_TABLE[i].char);
            expect(curr).toBeGreaterThanOrEqual(prev);
        }
    });

    test("excludes every directional glyph — the two sets are disjoint", () => {
        for (const g of CELL_DIRECTIONAL_GLYPHS) {
            expect(RAMP_TABLE.some((row) => row.char === g)).toBe(false);
        }
    });

    // the s3r fill-treatment amendment's own regression guard: "the fill role emits angular glyphs only,
    // never curved ones" (`specs/shallot-tui.md`) — pinned against `CELL_FILL_EXCLUDED_GLYPHS` (`ramp.ts`'s
    // own curated set) rather than a hardcoded literal, so a widened exclusion in `ramp.ts` is what this
    // arm proves reached the generated table, not a copy of today's four characters.
    test("excludes every curated curved glyph — a regenerated ramp cannot reintroduce a parenthesis or brace into the fill role", () => {
        expect(CELL_FILL_EXCLUDED_GLYPHS.length).toBeGreaterThan(0);
        for (const g of CELL_FILL_EXCLUDED_GLYPHS) {
            expect(RAMP_TABLE.some((row) => row.char === g)).toBe(false);
        }
    });

    test("space carries zero measured coverage and sorts first", () => {
        expect(RAMP_TABLE[0]).toEqual({ char: " ", coverage: 0 });
    });

    test("a visually dense glyph outranks a visually sparse one — the ordering carries real shape signal, not an accident of code point", () => {
        const coverageOf = (char: string) => RAMP_TABLE.find((row) => row.char === char)?.coverage;
        // "." is nearly a single dot; "@" is one of the densest printable-ASCII glyphs in most fonts.
        expect(coverageOf(".")).toBeLessThan(coverageOf("@") ?? Number.POSITIVE_INFINITY);
    });
});

describe("glyphCoverage (generate-ramp.ts)", () => {
    const font = loadBrandFont();

    test("a glyph with no outline (space) measures zero coverage", () => {
        expect(glyphCoverage(font, " ")).toBe(0);
    });

    test("a code point outside the cmap resolves through the font's own .notdef fallback, never throws", () => {
        // `font.ts`'s `getGlyphId` falls back to glyph id 0 (.notdef) for an unmapped code point — this
        // asserts glyphCoverage doesn't throw on that path and returns a finite, non-negative number,
        // rather than asserting a specific value that depends on whether this font's own .notdef glyph
        // happens to carry an outline (this one does).
        const coverage = glyphCoverage(font, "\u{1F600}");
        expect(Number.isFinite(coverage)).toBe(true);
        expect(coverage).toBeGreaterThanOrEqual(0);
    });

    test("a denser glyph measures more coverage than a sparser one, independent of the committed table", () => {
        // proves the measurement itself discriminates shape, not just that the committed sort is stable —
        // this calls glyphCoverage directly rather than reading RAMP_TABLE.
        expect(glyphCoverage(font, "@")).toBeGreaterThan(glyphCoverage(font, "."));
    });
});
