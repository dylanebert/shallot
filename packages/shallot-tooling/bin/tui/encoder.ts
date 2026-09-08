// The encoder: cells to bytes across the four portability tiers. Stateful across calls (it
// remembers the last-painted grid and cursor position to diff against), which is the whole point
// — a caller feeds it one grid per frame and gets back only the bytes that frame needs.

import type { Tier } from "./color-support";
import { CLEAR_SCREEN, CURSOR_HOME, cursorTo, SGR_RESET } from "./cursor";
import { diffRuns } from "./diff";
import { encodeRun } from "./sgr";
import type { Cell, Grid, RGB } from "./types";

/** Renders a full grid as tier-`plain` text: one row per line, no escapes, no cursor state at all. */
function renderPlain(grid: Grid): string {
    let out = "";
    for (const row of grid.cells) {
        for (const cell of row) out += cell.glyph;
        out += "\n";
    }
    return out;
}

/** Deep-clones one nullable `RGB` — `Cell`'s `fg`/`bg` fields are typed `readonly` but nothing at
 *  runtime stops a caller from mutating the object those fields point at, so a value-level clone (not
 *  a reference copy) is what actually keeps `_prev` independent of the caller's object. */
function cloneRGB(rgb: RGB | null): RGB | null {
    return rgb === null ? null : { r: rgb.r, g: rgb.g, b: rgb.b };
}

/** Deep-clones one `Cell`, including its `fg`/`bg` `RGB` values — see {@link cloneRGB}. */
function cloneCell(cell: Cell): Cell {
    return { glyph: cell.glyph, fg: cloneRGB(cell.fg), bg: cloneRGB(cell.bg) };
}

/** Deep-clones a `Grid`: every row, every `Cell`, and every `RGB` value it carries — backs the
 *  `Encoder` class's own "never stores the caller's `Grid` by reference" guarantee below at every
 *  depth, not just the top-level `cells` array. A shallow `row.slice()` clone would still share `Cell`
 *  (and `RGB`) object references with the caller: a producer that reuses `Cell` objects across frames
 *  and mutates their fields in place, rather than allocating fresh ones, would alias `_prev` at the
 *  cell level, and `diffRuns`' structural comparison would then see `_prev` and the current grid as
 *  identical by construction (they're the same objects) and emit zero bytes forever. Cheap at this
 *  grid's fixed size (80x24) — a few thousand small object allocations per frame, not a hot loop this
 *  package needs to avoid. */
function cloneGrid(grid: Grid): Grid {
    return {
        width: grid.width,
        height: grid.height,
        cells: grid.cells.map((row) => row.map(cloneCell)),
    };
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
 * **`encode()` never stores the caller's `Grid` by reference, nor any `Cell` or `RGB` value it
 * carries (N8).** A producer that reuses one grid buffer per frame — mutating the cells or colors
 * of an existing `Grid` in place rather than allocating a fresh one each call — is the natural
 * shape for a GPU readback loop. If `_prev` aliased any of those objects, a later in-place
 * mutation would also mutate `_prev`, so the next `diffRuns(_prev, curr)` would compare the grid
 * to itself and emit zero bytes forever. `encode()` deep-clones the grid (`cloneGrid` below: every
 * row, `Cell`, and `RGB` value) before storing it, so the caller's own objects, at any depth, are
 * free to mutate between calls.
 */
export class Encoder {
    private _prev: Grid | null = null;
    private _cursorRow = -1;
    private _cursorCol = -1;

    constructor(
        private readonly _tier: Tier,
        private _origin: { row: number; col: number } = { row: 0, col: 0 },
    ) {}

    /** Move the grid within the terminal and force a repaint at the new origin. */
    place(origin: { row: number; col: number }): void {
        this._origin = origin;
        this.invalidate();
    }

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
        // `plain` never reads `_prev` — `_tier` is readonly, so once an Encoder is constructed at
        // `plain` this branch is the only one it ever takes, and `renderPlain` never diffs. Storing a
        // clone here would be a per-frame allocation with no consumer, the tier documented above as
        // "stateless by design" holding actual diff state.
        if (this._tier === "plain") {
            return renderPlain(grid);
        }

        const runsOrResize = diffRuns(this._prev, grid);
        let out = "";
        let wroteColor = false;

        if (runsOrResize === "resize") {
            out += CLEAR_SCREEN + CURSOR_HOME;
            for (let y = 0; y < grid.height; y++) {
                out += cursorTo(y + this._origin.row, this._origin.col);
                const rendered = encodeRun(grid.cells[y], this._tier);
                out += rendered;
                if (this._tier !== "glyph" && grid.width > 0) wroteColor = true;
            }
            this._cursorRow = grid.height - 1 + this._origin.row;
            this._cursorCol = grid.width + this._origin.col;
        } else {
            for (const run of runsOrResize) {
                const row = run.row + this._origin.row;
                const col = run.col + this._origin.col;
                if (this._cursorRow !== row || this._cursorCol !== col) {
                    out += cursorTo(row, col);
                }
                out += encodeRun(run.cells, this._tier);
                if (this._tier !== "glyph" && run.cells.length > 0) wroteColor = true;
                this._cursorRow = row;
                this._cursorCol = col + run.cells.length;
            }
        }

        if (wroteColor) out += SGR_RESET;
        this._prev = cloneGrid(grid);
        return out;
    }
}
