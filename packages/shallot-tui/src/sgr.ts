// SGR (Select Graphic Rendition) encoding and run coalescing. A "run" is a maximal sequence of
// cells sharing one fg/bg pair — coalescing means one SGR prefix covers the whole run instead of
// one per cell, which is the difference between the spec's ~9 (ansi256) / ~19 (truecolor)
// bytes-per-*changed-cell* estimate and paying that cost per cell regardless of runs.

import type { Tier } from "./color-support";
import { SGR_RESET } from "./cursor";
import type { Cell } from "./types";
import { rgbEqual } from "./types";

/** Quantizes an 8-bit channel to the xterm 256-color cube's 6 steps (0, 51, 102, 153, 204, 255). */
export function cube6(channel: number): number {
    return Math.max(0, Math.min(5, Math.round((channel / 255) * 5)));
}

/** Maps an RGB color to its nearest xterm 256-color palette index in the 6x6x6 cube (16-231). */
export function ansi256FromRgb(r: number, g: number, b: number): number {
    return 16 + 36 * cube6(r) + 6 * cube6(g) + cube6(b);
}

/** Two cells share a run iff their fg and bg are structurally equal — glyph is irrelevant to style. */
export function sameStyle(a: Cell, b: Cell): boolean {
    return rgbEqual(a.fg, b.fg) && rgbEqual(a.bg, b.bg);
}

/**
 * The SGR escape that establishes `cell`'s style at the given tier, or `""` for a colorless tier
 * (`plain` / `glyph`) or a cell with no explicit fg/bg (terminal defaults, nothing to set).
 */
export function sgrPrefix(cell: Cell, tier: Tier): string {
    if (tier === "plain" || tier === "glyph") return "";
    const params: string[] = [];
    if (tier === "truecolor") {
        if (cell.fg) params.push(`38;2;${cell.fg.r};${cell.fg.g};${cell.fg.b}`);
        if (cell.bg) params.push(`48;2;${cell.bg.r};${cell.bg.g};${cell.bg.b}`);
    } else {
        if (cell.fg) params.push(`38;5;${ansi256FromRgb(cell.fg.r, cell.fg.g, cell.fg.b)}`);
        if (cell.bg) params.push(`48;5;${ansi256FromRgb(cell.bg.r, cell.bg.g, cell.bg.b)}`);
    }
    if (params.length === 0) return SGR_RESET;
    return `\x1b[${params.join(";")}m`;
}

/**
 * Encodes a contiguous sequence of cells (already known to be worth emitting — the diff or the
 * full-repaint path decides that), coalescing consecutive same-style cells under one SGR prefix.
 * `plain`/`glyph` tiers carry no SGR at all: this degrades to a plain glyph concatenation.
 */
export function encodeRun(cells: readonly Cell[], tier: Tier): string {
    if (tier === "plain" || tier === "glyph") {
        return cells.map((c) => c.glyph).join("");
    }
    let out = "";
    let i = 0;
    while (i < cells.length) {
        let j = i + 1;
        while (j < cells.length && sameStyle(cells[j], cells[i])) j++;
        out += sgrPrefix(cells[i], tier);
        for (let k = i; k < j; k++) out += cells[k].glyph;
        i = j;
    }
    return out;
}
