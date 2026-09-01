// A minimal terminal model — the independent reader criterion 1's round-trip test needs: it
// parses the bytes `Encoder` emits (CUP, SGR, plain glyphs, the resize repaint's clear+home) back
// into a `Grid`, playing the part of a real terminal rather than re-deriving the encoder's own
// logic. Test-only: it lives beside the arm that uses it, not in `src/`, the same shape
// `packages/shallot-ocean/tests/elfouhaily-independent.ts` already uses in this repo.

import type { Tier } from "../src/color-support";
import { cube6 } from "../src/sgr";
import type { Cell, Grid, RGB } from "../src/types";
import { makeGrid } from "../src/types";

const CUBE_STEP = [0, 51, 102, 153, 204, 255] as const;

function blankCell(): Cell {
    return { glyph: " ", fg: null, bg: null };
}

/** Everything a terminal tracks between writes: the visible grid, the cursor, and the SGR state. */
export class TerminalModel {
    private _width: number;
    private _height: number;
    private _cells: Cell[][];
    private _row = 0;
    private _col = 0;
    private _fg: RGB | null = null;
    private _bg: RGB | null = null;

    constructor(width: number, height: number) {
        this._width = width;
        this._height = height;
        this._cells = TerminalModel.blankRows(width, height);
    }

    private static blankRows(width: number, height: number): Cell[][] {
        const rows: Cell[][] = [];
        for (let y = 0; y < height; y++) rows.push(Array.from({ length: width }, blankCell));
        return rows;
    }

    /**
     * Models an out-of-band terminal-window resize: a real terminal's dimensions come from the
     * window manager, never from the byte stream, so the test drives this directly at the point
     * in the frame sequence it knows a resize occurred — exactly the information a real terminal
     * has and the encoder does not.
     */
    resize(width: number, height: number): void {
        this._width = width;
        this._height = height;
        this._cells = TerminalModel.blankRows(width, height);
    }

    /** Feeds `bytes` (one `Encoder.encode()` return value, or several concatenated) to the model. */
    write(bytes: string): void {
        let i = 0;
        while (i < bytes.length) {
            if (bytes[i] === "\x1b" && bytes[i + 1] === "[") {
                let j = i + 2;
                while (j < bytes.length && !/[A-Za-z]/.test(bytes[j])) j++;
                const params = bytes.slice(i + 2, j);
                const final = bytes[j];
                this.applyCsi(params, final);
                i = j + 1;
            } else {
                this.writeChar(bytes[i]);
                i++;
            }
        }
    }

    private writeChar(ch: string): void {
        if (ch === "\n") {
            this._row++;
            this._col = 0;
            return;
        }
        if (
            this._row >= 0 &&
            this._row < this._height &&
            this._col >= 0 &&
            this._col < this._width
        ) {
            this._cells[this._row][this._col] = { glyph: ch, fg: this._fg, bg: this._bg };
            this._col++;
        }
    }

    private applyCsi(params: string, final: string): void {
        if (final === "H") {
            const [r, c] = params.split(";").map((p) => Number.parseInt(p, 10) || 1);
            this._row = r - 1;
            this._col = c - 1;
            return;
        }
        if (final === "J") {
            // "2J" is the only form the encoder emits (full-screen erase) — clear content, leave
            // cursor and style untouched, matching real ED behavior.
            this._cells = TerminalModel.blankRows(this._width, this._height);
            return;
        }
        if (final === "m") {
            this.applySgr(params);
            return;
        }
        // "h"/"l" mode toggles (alt screen, cursor visibility) carry no grid-visible state here —
        // a real terminal's alt-screen buffer and this model's `cells` play the same role, so
        // there is nothing further to track.
    }

    private applySgr(params: string): void {
        const codes = params.split(";").filter((p) => p !== "");
        if (codes.length === 0 || codes[0] === "0") {
            this._fg = null;
            this._bg = null;
            return;
        }
        let i = 0;
        while (i < codes.length) {
            const code = codes[i];
            if (code === "38" || code === "48") {
                const isFg = code === "38";
                const mode = codes[i + 1];
                if (mode === "2") {
                    const rgb: RGB = {
                        r: Number(codes[i + 2]),
                        g: Number(codes[i + 3]),
                        b: Number(codes[i + 4]),
                    };
                    if (isFg) this._fg = rgb;
                    else this._bg = rgb;
                    i += 5;
                } else if (mode === "5") {
                    const rgb = dequantizeAnsi256(Number(codes[i + 2]));
                    if (isFg) this._fg = rgb;
                    else this._bg = rgb;
                    i += 3;
                } else {
                    i += 1;
                }
            } else if (code === "39") {
                // explicit "reset foreground to terminal default" — `sgr.ts`'s `sgrPrefix` emits
                // this for a `null` fg so a preceding run's color can't bleed onto this cell.
                this._fg = null;
                i += 1;
            } else if (code === "49") {
                // the background counterpart of "39" above.
                this._bg = null;
                i += 1;
            } else {
                i += 1;
            }
        }
    }

    /** A snapshot `Grid` of the model's current visible state. */
    grid(): Grid {
        return makeGrid(this._width, this._height, (x, y) => this._cells[y][x]);
    }
}

/** Inverse of `ansi256FromRgb`'s 6x6x6 cube for the index range the encoder ever produces (16-231). */
function dequantizeAnsi256(index: number): RGB {
    const i = index - 16;
    const r = Math.floor(i / 36);
    const g = Math.floor((i % 36) / 6);
    const b = i % 6;
    return { r: CUBE_STEP[r], g: CUBE_STEP[g], b: CUBE_STEP[b] };
}

/**
 * Projects a source grid onto what the given tier can actually carry, so a round-trip equality
 * check compares like with like: `glyph` drops color entirely (the tier never emits it), and
 * `ansi256` quantizes each channel through the same 6-step cube the encoder itself quantizes
 * through — a genuine information loss the ladder accepts, not a test artifact to work around.
 */
export function projectForTier(grid: Grid, tier: Tier): Grid {
    if (tier === "plain" || tier === "glyph") {
        return makeGrid(grid.width, grid.height, (x, y) => ({
            glyph: grid.cells[y][x].glyph,
            fg: null,
            bg: null,
        }));
    }
    if (tier === "truecolor") return grid;
    const quantize = (c: RGB | null): RGB | null =>
        c === null
            ? null
            : { r: CUBE_STEP[cube6(c.r)], g: CUBE_STEP[cube6(c.g)], b: CUBE_STEP[cube6(c.b)] };
    return makeGrid(grid.width, grid.height, (x, y) => {
        const cell = grid.cells[y][x];
        return { glyph: cell.glyph, fg: quantize(cell.fg), bg: quantize(cell.bg) };
    });
}
