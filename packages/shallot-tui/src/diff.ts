// Frame diffing: the encoder's bandwidth lever (`shallot-tui.md`'s measurement table — the
// constraint that remains is bytes to the terminal). Turns two grids into the minimal set of
// changed horizontal runs, or the sentinel `"resize"` when dimensions moved and nothing but a
// full repaint is coherent.

import type { Cell, Grid } from "./types";
import { cellEqual } from "./types";

/** A maximal horizontal run of changed cells on one row, starting at `(row, col)`. */
export interface ChangedRun {
    readonly row: number;
    readonly col: number;
    readonly cells: readonly Cell[];
}

/**
 * Diffs `curr` against `prev`. Returns `"resize"` when `prev` is `null` or the dimensions moved —
 * a full repaint is the only coherent action, since a stale cursor position from the old size is
 * meaningless against the new one. Otherwise returns every maximal run of cells that changed,
 * scanning row-major so each run is contiguous and cursor-adjacent (the shape `encoder.ts` needs
 * to decide whether a `cursorTo` is owed before it).
 */
export function diffRuns(prev: Grid | null, curr: Grid): readonly ChangedRun[] | "resize" {
    if (prev === null || prev.width !== curr.width || prev.height !== curr.height) return "resize";
    const runs: ChangedRun[] = [];
    for (let y = 0; y < curr.height; y++) {
        const prevRow = prev.cells[y];
        const currRow = curr.cells[y];
        let runStart = -1;
        let runCells: Cell[] = [];
        for (let x = 0; x < curr.width; x++) {
            if (!cellEqual(prevRow[x], currRow[x])) {
                if (runStart === -1) runStart = x;
                runCells.push(currRow[x]);
            } else if (runStart !== -1) {
                runs.push({ row: y, col: runStart, cells: runCells });
                runStart = -1;
                runCells = [];
            }
        }
        if (runStart !== -1) runs.push({ row: y, col: runStart, cells: runCells });
    }
    return runs;
}
