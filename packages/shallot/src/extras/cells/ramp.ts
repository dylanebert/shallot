// The cell glyph ramp: the stable index → character mapping `Cell.glyph` indexes into, shared by every
// glyph-index consumer — the fill kernel's own test pattern (`grid.ts`'s `glyphCount` wrap), S3's
// instanced draw (a GPU-side uv-rect table indexed the same way, contract below — not built here), and
// S4's terminal encoder (glyph index → the literal character it writes to the pipe). Printable ASCII,
// `0x20` (space) through `0x7e` (`~`), 95 characters, ordered by code point: index 0 is always the
// space and index 94 always `~`, so a glyph index survives a font swap or an atlas rebuild unchanged —
// `createCellGrid`'s own `@example` already assumed the count (95) this file now names.
const RAMP_START = 0x20; // " "
const RAMP_END = 0x7e; // "~"

/** the glyph ramp's length — every {@link Cell.glyph} index a producer writes must be
 *  `< CELL_GLYPH_COUNT`. */
export const CELL_GLYPH_COUNT = RAMP_END - RAMP_START + 1;

/** the character at glyph index `i` (`0 <= i < CELL_GLYPH_COUNT`) — the ramp's CPU-side inverse, what a
 *  terminal encoder (S4) writes for a decoded {@link Cell.glyph}, and how S3 keys its uv-rect table into
 *  the shared text atlas (`atlas.glyphs.get(cellGlyphChar(i))`, see the contract below).
 *  @example cellGlyphChar(0); // " " */
export function cellGlyphChar(i: number): string {
    if (i < 0 || i >= CELL_GLYPH_COUNT || !Number.isInteger(i))
        throw new Error(`[cells] cellGlyphChar: index out of range, got ${i}`);
    return String.fromCharCode(RAMP_START + i);
}

/** the whole ramp as one string, index-ordered — the batch-warm argument a live atlas ensures every
 *  ramp glyph with in one call (`ensureString(atlas, cellGlyphString())`, mirroring `extras/text`'s own
 *  `ASCII_CACHE` warm string).
 *  @example cellGlyphString().length; // 95 */
export function cellGlyphString(): string {
    let s = "";
    for (let i = 0; i < CELL_GLYPH_COUNT; i++) s += cellGlyphChar(i);
    return s;
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
