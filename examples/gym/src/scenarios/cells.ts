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
// timeouts.ts` carried for this module. `grid.ts`'s fill kernel sweeps fg/bg alpha across the unorm8
// lattice's own midpoints (`(x + 0.5) / 255`, `(y + 0.5) / 255` — exact `k + 0.5` for every x/y in this
// grid), so this scenario's fixture actually *constructs* the seam rather than merely being where it
// would live if it were reachable — and `assertFillDispatch`'s own comparison (below) tolerates exactly
// the +/-1 divergence that construction is expected to produce on the alpha lane, nowhere else.
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
    // mirrors grid.ts's own lattice-midpoint sweep exactly — see this file's module doc.
    const alphaFg = (x + 0.5) / 255;
    const alphaBg = (y + 0.5) / 255;
    const packed = packCell(glyph, vec4f(u, v, 1 - u, alphaFg), vec4f(1 - u, 1 - v, v, alphaBg));
    const buf = new ArrayBuffer(CELL_BYTES);
    new Uint32Array(buf).set([packed.x, packed.y, packed.z]);
    return unpackCell(buf, 0);
}

// fg/bg alpha (index 3) is the only channel deliberately swept onto the unorm8 lattice's own midpoints
// (`expectedCell`'s alphaFg/alphaBg, mirroring `grid.ts`'s fill kernel). At exactly that boundary, the
// real device's `pack4x8unorm` intrinsic is free to break the tie the opposite way from the CPU
// reference's `Math.round` (half-to-even vs. half-up) — a legitimate ±1 divergence, and the seam this
// scenario exists to construct, not a wiring defect. Every other channel (glyph, rgb) is never engineered
// onto a tie, so it stays bit-exact: a ±1 drift there, or anything wider than ±1 on alpha, is still a
// mismatch — this scenario measured 0 non-alpha divergences and >0 alpha-only ±1 divergences on the
// hardware this repair was verified against (`bun bench --scenario cells`, RTX 4090 lovelace bridge).
function sameByte(actual: number, expected: number, isAlphaLane: boolean): boolean {
    if (actual === expected) return true;
    return isAlphaLane && Math.abs(actual - expected) === 1;
}

async function assertFillDispatch(): Promise<Check> {
    if (!gridMirror) return { name: "cells fill dispatch", pass: false, detail: "no grid mirror" };
    await settle(gridMirror);
    const snap = gridMirror.snapshot;
    if (!snap) return { name: "cells fill dispatch", pass: false, detail: "no grid snapshot" };

    let mismatches = 0;
    let tieDivergences = 0;
    let first = "";
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const i = y * COLS + x;
            const actual = unpackCell(snap.bytes, i);
            const wanted = expectedCell(x, y);
            const fgOk = actual.fg.every((c, k) => sameByte(c, wanted.fg[k], k === 3));
            const bgOk = actual.bg.every((c, k) => sameByte(c, wanted.bg[k], k === 3));
            if (actual.fg[3] !== wanted.fg[3]) tieDivergences++;
            if (actual.bg[3] !== wanted.bg[3]) tieDivergences++;
            const same = actual.glyph === wanted.glyph && fgOk && bgOk;
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
                ? `${COLS * ROWS} cells match (glyph + rgb bit-exact; ${tieDivergences} alpha-lane ±1 tie divergences tolerated — the constructed rounding seam)`
                : `${mismatches}/${COLS * ROWS} mismatched beyond the tolerated ±1 alpha tie; first (actual/expected): ${first}`,
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
