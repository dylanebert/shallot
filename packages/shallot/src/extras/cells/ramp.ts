import { RAMP_TABLE } from "./ramp-table";

// The cell glyph ramp: the stable index → character mapping `Cell.glyph` indexes into, shared by every
// glyph-index consumer — the fill kernel's own test pattern (`grid.ts`'s `glyphCount` wrap), S3's
// instanced draw (a GPU-side uv-rect table indexed the same way, contract below — not built here), and
// S4's terminal encoder (glyph index → the literal character it writes to the pipe).
//
// Glyph selection is structure-first (Locked decision, `specs/shallot-tui.md`'s glyph-selection
// addendum, grounded against Xu/Zhang/Wong SIGGRAPH/TOG 2010): S3's detector picks a directional glyph
// where a strong edge exists and falls back to a coverage-ordered fill ramp elsewhere. That splits this
// module's index space into two differently-selected halves, each ordered by the rule that selects it:
//
//   - `CELL_FILL_GLYPHS` (indices `0 .. CELL_FILL_GLYPHS.length - 1`) — every printable-ASCII candidate
//     minus the directional set, ordered by *measured ink coverage* ascending (least ink first), not by
//     code point. A character set in code-point order is not a ramp — it carries no shape signal a
//     tone-based selector can walk. The ordering is generated data, not hand-authored: rendered and
//     measured by `scripts/generate-ramp.ts`, committed in `ramp-table.ts`, and reproduced by
//     `ramp-table.test.ts` against the same brand font (`assets/font.ttf`) the generator reads.
//   - `CELL_DIRECTIONAL_GLYPHS` (appended after the fill glyphs) — a small curated set, one glyph per
//     quantized edge-angle bucket (0°, 45°, 90°, 135° — the standard four-bucket split inherited from
//     Canny non-max suppression), kept separate from the fill ramp because they're selected by edge
//     angle, not by coverage. S3's detector is what runs the Sobel/DoG selection; this file only names
//     the set and where it lives in the index space.
//
// Both halves still share one property with the code-point ramp they replace: a glyph index is a stable,
// committed enumeration, so it survives a font swap or an atlas rebuild unchanged — a coverage-derived
// table is exactly as stable as a code-point one, only its *ordering rule* differs.

/** the small curated directional set (Locked decision, `specs/shallot-tui.md`) — one glyph per quantized
 *  edge-angle bucket, 0°/45°/90°/135°, the shape S3's structure-based selector consumes. Kept separate
 *  from {@link CELL_FILL_GLYPHS} (excluded from its coverage measurement, `scripts/generate-ramp.ts`)
 *  because a directional glyph is selected by edge angle, never by ink coverage. */
export const CELL_DIRECTIONAL_GLYPHS: readonly string[] = ["-", "/", "|", "\\"] as const;

/** the coverage-ordered fill ramp — every printable-ASCII candidate minus {@link CELL_DIRECTIONAL_GLYPHS},
 *  ascending by measured ink coverage (least ink first). Generated data (`scripts/generate-ramp.ts` /
 *  `ramp-table.ts`), not hand-authored. */
export const CELL_FILL_GLYPHS: readonly string[] = RAMP_TABLE.map((entry) => entry.char);

/** the glyph ramp's total length — fill glyphs plus the directional set — every {@link Cell.glyph} index
 *  a producer writes must be `< CELL_GLYPH_COUNT`. */
export const CELL_GLYPH_COUNT = CELL_FILL_GLYPHS.length + CELL_DIRECTIONAL_GLYPHS.length;

/** the character at glyph index `i` (`0 <= i < CELL_GLYPH_COUNT`) — the ramp's CPU-side inverse, what a
 *  terminal encoder (S4) writes for a decoded {@link Cell.glyph}, and how S3 keys its uv-rect table into
 *  the shared text atlas (`atlas.glyphs.get(cellGlyphChar(i))`, see the contract below). Indices
 *  `< CELL_FILL_GLYPHS.length` read the coverage-ordered fill ramp; the remainder read
 *  {@link CELL_DIRECTIONAL_GLYPHS}, in bucket order.
 *  @example cellGlyphChar(0); // the lowest-coverage fill glyph */
export function cellGlyphChar(i: number): string {
    if (i < 0 || i >= CELL_GLYPH_COUNT || !Number.isInteger(i))
        throw new Error(`[cells] cellGlyphChar: index out of range, got ${i}`);
    return i < CELL_FILL_GLYPHS.length
        ? CELL_FILL_GLYPHS[i]
        : CELL_DIRECTIONAL_GLYPHS[i - CELL_FILL_GLYPHS.length];
}

/** the whole ramp as one string, index-ordered (fill glyphs, then the directional set) — the batch-warm
 *  argument a live atlas ensures every ramp glyph with in one call (`ensureString(atlas,
 *  cellGlyphString())`, mirroring `extras/text`'s own `ASCII_CACHE` warm string). */
export function cellGlyphString(): string {
    return CELL_FILL_GLYPHS.join("") + CELL_DIRECTIONAL_GLYPHS.join("");
}

// The GPU-side half of this contract — the uv-rect table S3's instanced draw reads per glyph index —
// is defined here and built there, not here: it needs a *live* `GlyphAtlas` (a real GPU device + a
// loaded `Font`, `extras/text/atlas.ts`), which today only exists once a plugin's `initialize` creates
// one (`TextPlugin`'s own `_atlases`), and S1 ships no cells-side plugin to own that lifecycle. The
// contract a producer of that table fulfills:
//
//   1. Ensure every ramp glyph is resident once: `ensureString(atlas, cellGlyphString())`.
//   2. For `i` in `0 .. CELL_GLYPH_COUNT - 1`, read `atlas.glyphs.get(cellGlyphChar(i))` (a
//      `GlyphMetrics`, `atlas.ts`) and pack its `u0, v0, u1, v1` into lane `i` of a
//      `d.arrayOf(d.vec4f, CELL_GLYPH_COUNT)` storage buffer — the same shelf-packed uv rect
//      `computeGlyphMetrics` already produces for `extras/text`'s own instanced glyph quads, reused
//      rather than re-derived.
//   3. The instanced draw's vertex/fragment stage indexes that buffer by `cells[i].glyph`, exactly the
//      way `extras/text/index.ts`'s `typedTextSurface` indexes `textGlyphs` by instance id today.
//
// A glyph absent from the loaded font (`computeGlyphMetrics` returns `null`, e.g. an unsupported code
// point) is S3's own concern to define a fallback for — out of this file's contract, which only fixes
// the index ↔ character mapping every consumer shares.
