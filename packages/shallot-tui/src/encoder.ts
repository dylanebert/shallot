// The encoder: cells to bytes across the four portability tiers. Stateful across calls (it
// remembers the last-painted grid and cursor position to diff against), which is the whole point
// — a caller feeds it one grid per frame and gets back only the bytes that frame needs.

import type { Tier } from "./color-support";
import { CLEAR_SCREEN, CURSOR_HOME, cursorTo, SGR_RESET } from "./cursor";
import { diffRuns } from "./diff";
import { encodeRun } from "./sgr";
import type { Grid } from "./types";

/** Renders a full grid as tier-`plain` text: one row per line, no escapes, no cursor state at all. */
function renderPlain(grid: Grid): string {
    let out = "";
    for (const row of grid.cells) {
        for (const cell of row) out += cell.glyph;
        out += "\n";
    }
    return out;
}

/**
 * Turns a sequence of cell grids into the terminal bytes that reproduce them, one frame at a
 * time. Diffed after the first frame: an unchanged frame costs zero bytes, a frame with a few
 * changed cells costs one `cursorTo` plus one coalesced SGR run per changed run
 * (`diff.ts` + `sgr.ts`), and a resize (or the first frame) costs a full repaint.
 *
 * `plain` tier is stateless by design — every call is a full text dump, appropriate for a
 * non-tty sink with no cursor to address and no reason to diff.
 *
 * **Two distinct notions of "resize."** `diffRuns` treats a change in the *grid's own* width or
 * height as a resize and repaints in full — but a real terminal reflows its own screen buffer on
 * an out-of-band window resize regardless of what the grid producer feeds it. A fixed-size grid
 * producer (the common case — a terminal command that always renders at the terminal's current
 * columns/rows, only occasionally changing) never sees that as a grid-dimension change, so
 * `diffRuns` alone never knows to repaint: it keeps diffing against a `_prev` that no longer
 * describes what's actually on screen, and the mismatch never self-corrects. `invalidate()` is
 * the seam for that case — call it from a real resize notification (`resize.ts`'s `onResize`) and
 * the next `encode()` call repaints in full, exactly like the first frame ever encoded.
 *
 * **`encode()` never stores the caller's `Grid` by reference (N8).** A producer that reuses one
 * grid buffer per frame — writing new cells into the same object rather than allocating a fresh
 * `Grid` each call — is the natural shape for a GPU readback loop. If `_prev` aliased that object,
 * a later in-place mutation would also mutate `_prev`, so the next `diffRuns(_prev, curr)` would
 * compare the grid to itself and emit zero bytes forever. `encode()` snapshots the row array
 * (`cloneGrid` below) before storing it, so the caller's own object is free to mutate between
 * calls.
 */
function cloneGrid(grid: Grid): Grid {
    return { width: grid.width, height: grid.height, cells: grid.cells.map((row) => row.slice()) };
}

export class Encoder {
    private _prev: Grid | null = null;
    private _cursorRow = -1;
    private _cursorCol = -1;

    constructor(private readonly _tier: Tier) {}

    /**
     * Forces the next `encode()` call to be an unconditional full repaint, regardless of whether
     * the grid's own dimensions changed — the seam for an out-of-band terminal resize (see the
     * class docblock). The caller owns deciding *when* a physical resize happened (typically by
     * subscribing to `resize.ts`'s `onResize`); this method is how it tells the encoder.
     */
    invalidate(): void {
        this._prev = null;
    }

    /** Encodes `grid` against the previously encoded grid (or as a full frame, on the first call,
     * or on the first call after `invalidate()`). */
    encode(grid: Grid): string {
        if (this._tier === "plain") {
            this._prev = cloneGrid(grid);
            return renderPlain(grid);
        }

        const runsOrResize = diffRuns(this._prev, grid);
        let out = "";
        let wroteColor = false;

        if (runsOrResize === "resize") {
            out += CLEAR_SCREEN + CURSOR_HOME;
            for (let y = 0; y < grid.height; y++) {
                out += cursorTo(y, 0);
                const rendered = encodeRun(grid.cells[y], this._tier);
                out += rendered;
                if (this._tier !== "glyph" && grid.width > 0) wroteColor = true;
            }
            this._cursorRow = grid.height - 1;
            this._cursorCol = grid.width;
        } else {
            for (const run of runsOrResize) {
                if (this._cursorRow !== run.row || this._cursorCol !== run.col) {
                    out += cursorTo(run.row, run.col);
                }
                out += encodeRun(run.cells, this._tier);
                if (this._tier !== "glyph" && run.cells.length > 0) wroteColor = true;
                this._cursorRow = run.row;
                this._cursorCol = run.col + run.cells.length;
            }
        }

        if (wroteColor) out += SGR_RESET;
        this._prev = cloneGrid(grid);
        return out;
    }
}
