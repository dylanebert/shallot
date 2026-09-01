// The glyph ramp's GPU-side half (`ramp.ts`'s module doc names the contract): a per-glyph uv-rect table
// against a live SDF atlas (`text/core`), indexed the same way `Cell.glyph` indexes `cellGlyphChar` —
// what the instanced draw's vertex stage reads to place each cell's glyph quad on the shared atlas
// texture. Built once per atlas rebuild (a font load, or a device re-adopt), not per frame.

import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Compute } from "../../engine";
import { ensureString, type GlyphAtlas } from "../text/core";
import { CELL_GLYPH_COUNT, cellGlyphChar, cellGlyphString } from "./ramp";

/** the zero-area uv-rect sentinel a font-absent glyph packs (`ramp.ts`'s module doc) — `u1 <= u0`, so the
 *  draw's fragment stage can tell "no glyph" apart from a real atlas region without a second flag. */
export const MISSING_GLYPH_UV: readonly [number, number, number, number] = [0, 0, 0, 0];

/** one glyph's `[u0, v0, u1, v1]`, from a resident {@link GlyphAtlas} entry, or {@link MISSING_GLYPH_UV}
 *  when the font has no outline for it (`computeGlyphMetrics` returns `null` — `ramp.ts`'s documented
 *  fallback). Pure and device-free: reads only `atlas.glyphs`, so a hand-built fake atlas exercises it
 *  with no GPU (`glyphs.test.ts`).
 *  @example const rect = glyphUvRect(atlas, 0); // the lowest-coverage fill glyph's atlas rect */
export function glyphUvRect(
    atlas: GlyphAtlas,
    index: number,
): readonly [number, number, number, number] {
    const metrics = atlas.glyphs.get(cellGlyphChar(index));
    if (!metrics) return MISSING_GLYPH_UV;
    return [metrics.u0, metrics.v0, metrics.u1, metrics.v1];
}

/** every ramp glyph's uv rect, index-ordered — the flat data {@link buildGlyphUvTable} writes to the GPU
 *  buffer, split out so the pure derivation is testable with no device (`glyphs.test.ts`).
 *  @example const table = glyphUvTable(atlas); // table.length === CELL_GLYPH_COUNT * 4 */
export function glyphUvTable(atlas: GlyphAtlas): Float32Array {
    const out = new Float32Array(CELL_GLYPH_COUNT * 4);
    for (let i = 0; i < CELL_GLYPH_COUNT; i++) {
        const [u0, v0, u1, v1] = glyphUvRect(atlas, i);
        out[i * 4] = u0;
        out[i * 4 + 1] = v0;
        out[i * 4 + 2] = u1;
        out[i * 4 + 3] = v1;
    }
    return out;
}

/** the storage buffer type the draw pass's `glyphUv` binding reads: one `vec4f` rect per ramp index. */
export type GlyphUvBuffer = TgpuBuffer<d.WgslArray<typeof d.vec4f>> & StorageFlag;

/** the glyph-size sentinel a font-absent glyph packs — the identity footprint, `(1, 1)`, harmless since a
 *  missing glyph never samples ink regardless of its footprint size ({@link MISSING_GLYPH_UV}'s
 *  zero-area gate, read at `draw.ts`'s `cellVertex`'s `has`). */
export const MISSING_GLYPH_SIZE: readonly [number, number] = [1, 1];

/** one glyph's own measured em-normalized footprint, `[glyphWidth, glyphHeight]` off a resident
 *  {@link GlyphAtlas} entry (`text/atlas.ts`'s `computeGlyphMetrics` — the padded-bounds box divided by
 *  `unitsPerEm`), each clamped to at most 1 so a wide glyph's quad never overflows into a neighboring
 *  cell. {@link MISSING_GLYPH_SIZE} when the font has no outline for it, mirroring {@link glyphUvRect}'s
 *  fallback. This is the measurement `draw.ts`'s vertex stage scales the glyph's footprint by
 *  (`cellFootprintPx`, isotropically — both axes share one scale so the cell's own aspect never
 *  re-stretches the glyph's true shape, `specs/shallot-tui.md`'s s3r item 9) — without it every glyph's
 *  own tightly-cropped SDF tile stretches across the whole cell regardless of true size, destroying the
 *  coverage-ordered ramp's monotone progression at the point of use (`specs/shallot-tui.md`'s s3r item 8).
 *  Facade ink (rule 3, the fill-treatment amendment) is raised at the *ink* level instead of here —
 *  `draw.ts`'s `INK_DILATE_FRACTION` — since an isotropic footprint scale was tried first and measured
 *  wrong: it inflates a small-em-size glyph (which the ramp's own low-coverage band is disproportionately
 *  made of) more than an already-near-1-em one, inverting `assertMonoRamp`'s own ordering rather than
 *  preserving it (own docblock in `draw.ts` has the measurement).
 *  @example const size = glyphSizeRect(atlas, 0); // the lowest-coverage fill glyph's own footprint */
export function glyphSizeRect(atlas: GlyphAtlas, index: number): readonly [number, number] {
    const metrics = atlas.glyphs.get(cellGlyphChar(index));
    if (!metrics) return MISSING_GLYPH_SIZE;
    return [Math.min(1, metrics.glyphWidth), Math.min(1, metrics.glyphHeight)];
}

/** every ramp glyph's size, index-ordered — the flat data {@link buildGlyphSizeTable} writes to the GPU
 *  buffer, split out so the pure derivation is testable with no device (`glyphs.test.ts`).
 *  @example const table = glyphSizeTable(atlas); // table.length === CELL_GLYPH_COUNT * 2 */
export function glyphSizeTable(atlas: GlyphAtlas): Float32Array {
    const out = new Float32Array(CELL_GLYPH_COUNT * 2);
    for (let i = 0; i < CELL_GLYPH_COUNT; i++) {
        const [w, h] = glyphSizeRect(atlas, i);
        out[i * 2] = w;
        out[i * 2 + 1] = h;
    }
    return out;
}

/** the storage buffer type the draw pass's `glyphSize` binding reads: one `vec2f` footprint per ramp
 *  index. */
export type GlyphSizeBuffer = TgpuBuffer<d.WgslArray<typeof d.vec2f>> & StorageFlag;

/**
 * ensure every ramp glyph is resident in `atlas` (`ensureString`, warming the SDF atlas texture), then
 * build + upload the uv-rect table as a fresh GPU storage buffer. Call once per atlas rebuild — a font
 * load, or a device re-adopt (`CellsPlugin.warm`) — never per frame.
 *
 * @example const glyphUv = buildGlyphUvTable(atlas);
 */
export function buildGlyphUvTable(atlas: GlyphAtlas): GlyphUvBuffer {
    ensureString(atlas, cellGlyphString());
    const table = glyphUvTable(atlas);
    const buffer = Compute.root
        .createBuffer(d.arrayOf(d.vec4f, CELL_GLYPH_COUNT))
        .$usage("storage")
        .$name("cells-glyph-uv");
    Compute.device.queue.writeBuffer(Compute.root.unwrap(buffer), 0, table);
    return buffer;
}

/**
 * ensure every ramp glyph is resident in `atlas` (`ensureString`, warming the SDF atlas texture — a
 * no-op for a glyph {@link buildGlyphUvTable} already warmed), then build + upload the glyph-size table
 * ({@link glyphSizeTable}) as a fresh GPU storage buffer. Call once per atlas rebuild alongside
 * {@link buildGlyphUvTable} — a font load, or a device re-adopt (`CellsPlugin.warm`) — never per frame.
 *
 * @example const glyphSize = buildGlyphSizeTable(atlas);
 */
export function buildGlyphSizeTable(atlas: GlyphAtlas): GlyphSizeBuffer {
    ensureString(atlas, cellGlyphString());
    const table = glyphSizeTable(atlas);
    const buffer = Compute.root
        .createBuffer(d.arrayOf(d.vec2f, CELL_GLYPH_COUNT))
        .$usage("storage")
        .$name("cells-glyph-size");
    Compute.device.queue.writeBuffer(Compute.root.unwrap(buffer), 0, table);
    return buffer;
}
