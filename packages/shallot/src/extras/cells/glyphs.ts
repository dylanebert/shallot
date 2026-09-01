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
