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
//     quantized *edge-tangent* angle bucket (0°=`-`, 45°=`/`, 90°=`|`, 135°=`\` — the angle the drawn
//     glyph visually runs along), kept separate from the fill ramp because they're selected by edge
//     angle, not by coverage. This is deliberately **not** the same angle Canny non-max suppression
//     buckets: Canny's four buckets are *gradient* angles, and a gradient is perpendicular to the edge
//     it belongs to, so indexing this array by a raw gradient bucket rotates every glyph 90° from the
//     edge it's meant to draw (gradient 0° ↔ tangent 90°, gradient 45° ↔ tangent 135°, and the reverse
//     pairs) — S3's detector must convert a computed gradient bucket to its perpendicular tangent bucket
//     before indexing here. The four-bucket *count* and *spacing* (0°/45°/90°/135°) are still standard
//     practice inherited from Canny's non-max suppression; only the angle each bucket names differs. S3's
//     detector is what runs the Sobel/DoG selection; this file only names the set, its tangent-angle
//     convention, and where it lives in the index space.
//
// Both halves still share one property with the code-point ramp they replace: a glyph index is a stable,
// committed enumeration, so it survives a font swap or an atlas rebuild unchanged — a coverage-derived
// table is exactly as stable as a code-point one, only its *ordering rule* differs.

/** the small curated directional set (Locked decision, `specs/shallot-tui.md`) — one glyph per quantized
 *  edge-*tangent* angle bucket, 0°(`-`)/45°(`/`)/90°(`|`)/135°(`\`), the shape S3's structure-based
 *  selector consumes. This is the angle the glyph visually runs along, **not** the gradient angle Canny
 *  non-max suppression buckets (a gradient is perpendicular to its edge's tangent — module doc above has
 *  the derivation) — a caller indexing by a raw gradient bucket must rotate it 90° to the perpendicular
 *  tangent bucket first, or every glyph renders turned 90° from the edge it represents. Kept separate
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

// The GPU-side half of this contract — the uv-rect + size tables the instanced draw reads per glyph
// index — is `glyphs.ts`'s `buildGlyphUvTable` / `buildGlyphSizeTable`, against a live `GlyphAtlas`
// (`text/core`'s `createGlyphAtlas` + `ensureString`, the same shelf-packed atlas `extras/text`'s own
// instanced glyph quads use — reused rather than re-derived) that `CellsPlugin` (`./index.ts`) owns the
// lifecycle of. The contract:
//
//   1. Ensure every ramp glyph is resident once: `ensureString(atlas, cellGlyphString())`.
//   2. For `i` in `0 .. CELL_GLYPH_COUNT - 1`, read `atlas.glyphs.get(cellGlyphChar(i))` (a
//      `GlyphMetrics`) and pack its `u0, v0, u1, v1` into lane `i` of a
//      `d.arrayOf(d.vec4f, CELL_GLYPH_COUNT)` storage buffer, and its `glyphWidth, glyphHeight`
//      (em-normalized, clamped to at most 1) into lane `i` of a `d.arrayOf(d.vec2f, CELL_GLYPH_COUNT)`
//      sibling — the glyph's own measured footprint.
//   3. The instanced draw's vertex stage indexes both buffers by `cells[i].glyph`: the quad geometry
//      always covers the whole cell (`draw.ts`'s `cellVertex`), and a fragment maps into the glyph's own
//      isotropically-scaled footprint within it (`cellFootprintPx` + `glyphFootprintT`, `draw.ts`) —
//      size-proportional placement, so a glyph occupies the cell in proportion to its own measured size
//      rather than every glyph's padded SDF tile stretching across the same cell footprint regardless of
//      extent, and without re-stretching that footprint by the cell's own (non-square) aspect ratio.
//      Outside the footprint, the fragment stage renders the cell's own background with no glyph ink.
//
// A glyph absent from the loaded font (`computeGlyphMetrics` returns `null`) packs the zero-area uv
// sentinel `(0,0,0,0)` (`u1 <= u0`, `MISSING_GLYPH_UV`) and the identity-footprint size sentinel `(1,1)`
// (`MISSING_GLYPH_SIZE`) — the draw's fragment stage treats the zero-area uv as "no glyph" and renders
// the cell's background alone, never sampling atlas texel `(0,0)`, which a real glyph could legitimately
// occupy; the size sentinel is inert there since no ink is ever sampled for a missing glyph.
