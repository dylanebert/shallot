// Shared shapes for the encoder. This module has no engine dependency and no I/O — the whole
// package is engine-agnostic by construction (`scripts/check-tui-boundary.ts` is the mechanism
// that keeps it that way, not this comment).

/** A packed 24-bit color. Components are 0-255; the encoder quantizes down for lower tiers. */
export interface RGB {
    readonly r: number;
    readonly g: number;
    readonly b: number;
}

/**
 * One terminal cell: a single display column carrying one glyph plus optional foreground and
 * background color. `null` means "the terminal's own default" (no SGR color code emitted for
 * that channel) — distinct from black, which is an explicit RGB(0,0,0).
 *
 * `glyph` is exactly one display column. Multi-column glyphs (wide CJK, emoji) are out of scope
 * for this unit; a producer feeding one must pad the following cell itself.
 */
export interface Cell {
    readonly glyph: string;
    readonly fg: RGB | null;
    readonly bg: RGB | null;
}

/** A row-major character-cell grid: `cells[y][x]`, `cells.length === height`, each row `width` long. */
export interface Grid {
    readonly width: number;
    readonly height: number;
    readonly cells: readonly (readonly Cell[])[];
}

/** Structural equality for two nullable RGB values (both null, or all three components equal). */
export function rgbEqual(a: RGB | null, b: RGB | null): boolean {
    if (a === null || b === null) return a === b;
    return a.r === b.r && a.g === b.g && a.b === b.b;
}

/** Structural equality for two cells: same glyph and same fg/bg. */
export function cellEqual(a: Cell, b: Cell): boolean {
    return a.glyph === b.glyph && rgbEqual(a.fg, b.fg) && rgbEqual(a.bg, b.bg);
}

/** Builds a `width`x`height` grid, every cell produced by `fill(x, y)`. */
export function makeGrid(
    width: number,
    height: number,
    fill: (x: number, y: number) => Cell,
): Grid {
    const cells: Cell[][] = [];
    for (let y = 0; y < height; y++) {
        const row: Cell[] = [];
        for (let x = 0; x < width; x++) row.push(fill(x, y));
        cells.push(row);
    }
    return { width, height, cells };
}

/** The blank cell every tier renders as a single space with no color — the default fill. */
export const BLANK_CELL: Cell = { glyph: " ", fg: null, bg: null };
