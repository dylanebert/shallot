import { type Mirror, MirrorPlugin, mirror, type Plugin, run } from "@dylanebert/shallot";
import {
    CELL_BYTES,
    CELL_GLYPH_COUNT,
    createCellGrid,
    fillCellGrid,
    packCell,
    unpackCell,
} from "@dylanebert/shallot/cells/core";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { vec4f } from "typegpu/data";
import { type Check, register, type Scenario, settle } from "../gym";

// The `shallot-tui` S1 cell-grid producer's real-GPU coverage: `fillCellGrid` had zero callers anywhere
// in the tree before this scenario (a review finding, S1's own repair) — a compute-only diagnostic in
// the `gpu-diagnostic.ts` shape (`noRender: true`, no framed scene), the tier this belongs in per
// `testing.md`: "GPU behavior — bun bench... compute-kernel output (Mirror readback)". It dispatches the
// fill pass against a real device and differentials the readback against the CPU `packCell` reference
// (device-side `pack4x8unorm` vs. `packCell`'s CPU arm — the residual half-up/half-to-even rounding seam
// `cell.test.ts` cannot see with no device), closing the exemption `examples/gym/src/scenarios/
// timeouts.ts` carried for this module.
const COLS = 10;
const ROWS = 6;

let grid: ReturnType<typeof createCellGrid> | null = null;
let gridMirror: Mirror | null = null;

const CellsPlugin: Plugin = {
    name: "GymCells",
    initialize() {
        grid = createCellGrid(COLS, ROWS, CELL_GLYPH_COUNT);
        fillCellGrid(grid);
        gridMirror = mirror(grid.buffer);
    },
};

// the CPU reference for one cell, mirroring `grid.ts`'s fill kernel formula exactly (glyph ramp wrap,
// the diagonal fg/bg gradient) — the same differential shape `cell.test.ts` runs, but against a real
// dispatched buffer instead of a hand-built one.
function expectedCell(x: number, y: number) {
    const glyph = (x + y) % CELL_GLYPH_COUNT;
    const u = x / COLS;
    const v = y / ROWS;
    const packed = packCell(glyph, vec4f(u, v, 1 - u, 1), vec4f(1 - u, 1 - v, v, 1));
    const buf = new ArrayBuffer(CELL_BYTES);
    new Uint32Array(buf).set([packed.x, packed.y, packed.z]);
    return unpackCell(buf, 0);
}

async function assertFillDispatch(): Promise<Check> {
    if (!gridMirror) return { name: "cells fill dispatch", pass: false, detail: "no grid mirror" };
    await settle(gridMirror);
    const snap = gridMirror.snapshot;
    if (!snap) return { name: "cells fill dispatch", pass: false, detail: "no grid snapshot" };

    let mismatches = 0;
    let first = "";
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const i = y * COLS + x;
            const actual = unpackCell(snap.bytes, i);
            const wanted = expectedCell(x, y);
            const same =
                actual.glyph === wanted.glyph &&
                actual.fg.every((c, k) => c === wanted.fg[k]) &&
                actual.bg.every((c, k) => c === wanted.bg[k]);
            if (!same) {
                mismatches++;
                if (!first) {
                    first = `(${x},${y}) glyph ${actual.glyph}/${wanted.glyph} fg [${actual.fg}]/[${wanted.fg}] bg [${actual.bg}]/[${wanted.bg}]`;
                }
            }
        }
    }
    return {
        name: "cells fill dispatch matches the CPU packCell reference",
        pass: mismatches === 0,
        detail:
            mismatches === 0
                ? `${COLS * ROWS} cells bit-identical to the CPU reference`
                : `${mismatches}/${COLS * ROWS} mismatched; first (actual/expected): ${first}`,
    };
}

const scenario: Scenario = {
    name: "cells",
    noRender: true,
    async build(_canvas) {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [ProfilePlugin, MirrorPlugin, CellsPlugin],
        });
        return {
            state,
            dispose() {
                grid = null;
                gridMirror = null;
                dispose();
            },
        };
    },
    async assert(): Promise<Check[]> {
        return [await assertFillDispatch()];
    },
    live(): string {
        return `${COLS}x${ROWS} cell grid, compute-only (no framed scene)`;
    },
};

register(scenario);
