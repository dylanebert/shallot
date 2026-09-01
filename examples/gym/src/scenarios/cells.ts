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
// fill pass against a real device and differentials the readback against a *predicted* reference (below)
// rather than the CPU `packCell` reference alone — closing the exemption `examples/gym/src/scenarios/
// timeouts.ts` carried for this module.
//
// Predict the tie, don't tolerate it (second S1 repair round). `grid.ts`'s fill kernel sweeps fg/bg
// alpha across `1 - (x + 0.5)/255`, `1 - (y + 0.5)/255` — mathematically exact unorm8-lattice ties
// (`k + 0.5` for k = 254 - x, resp. 254 - y) — the one input class that can expose a half-up (CPU
// `Math.round`, `engine/utils/tgsl.ts`'s `unorm8`, which multiplies the f32-rounded input by 255 in f64)
// vs. half-to-even (WGSL's `pack4x8unorm`, spec'd `round(clamp(e,0,1)*255)` with ties-to-even —
// spec-mandated at the round step itself, though the upstream division carries only a spec'd ULP
// tolerance, not a correctly-rounded guarantee) divergence. `predictedAlphaByte` below reproduces the
// spec'd computation directly (f32 arithmetic through the same steps `grid.ts`'s kernel performs, then
// ties-to-even), so this scenario predicts the byte a real device resolves rather than admitting
// whichever way it happened to fall — replacing the first repair's `±1` tolerance, which any systematic
// off-by-one in the alpha path (a wrong shift, a clamp, a transfer applied where it shouldn't be) would
// also have passed. Every non-alpha lane (glyph, rgb) is never engineered onto a tie, so it stays
// bit-exact against production `packCell` unchanged.
const COLS = 10;
const ROWS = 6;

// this fixture's own even/odd tie geometry, fixed by COLS/ROWS: fg alpha's predicted byte diverges from
// the naive half-up reference at every ODD x in [0, COLS) (5 of 10 — `1 - e` shifts the tie's floor
// parity relative to the un-inverted `k = x` construction), each affecting all ROWS rows; bg alpha
// diverges at every odd y in [0, ROWS) (3 of 6), each affecting all COLS columns. 5*6 + 3*10 = 60 — a
// static property of this JS reference (not the device), so it's asserted as an exact pin below rather
// than a >0 non-vacuity floor.
const EXPECTED_TIE_DIVERGENCES = 60;

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

// A local, spec-derived reference for WGSL's `pack4x8unorm`: `round(clamp(e, 0, 1) * 255)` with
// ties-to-even (the WGSL spec text, not a hardware quirk). Reproduced here using the same f32-precision
// arithmetic `grid.ts`'s kernel performs for one alpha channel (`1 - (idx + 0.5) / 255`, all f32 ops) —
// `f32()` after every operation matches what a WGSL `f32` variable actually holds at each step, rather
// than JS's default double precision, which is exactly the mismatch B1/B3's review found (the CPU
// `unorm8` reference multiplies by 255 in f64, the intrinsic multiplies in f32). Measured bit-exact
// against a real device (`bun bench --scenario cells`, nvidia/lovelace) for all 120 alpha samples this
// fixture produces.
const f32 = Math.fround;

function alphaF32(idx: number): number {
    const numerator = f32(f32(idx) + f32(0.5));
    const divided = f32(numerator / f32(255));
    // grid.ts packs `one - e`, not `e` — see this file's module doc (visible-byte inversion).
    return f32(1 - divided);
}

// round-half-to-even at f32 precision. `scaled` is always an exact f32 value here (from `alphaF32`
// scaled by 255, itself an f32 multiply), so its fractional part is exactly representable in f64 with no
// further rounding error — the `frac === 0.5` tie check below is exact, not approximate.
function roundTiesToEven(scaled: number): number {
    const floor = Math.floor(scaled);
    const frac = scaled - floor;
    if (frac < 0.5) return floor;
    if (frac > 0.5) return floor + 1;
    return floor % 2 === 0 ? floor : floor + 1;
}

/** the real device's predicted `pack4x8unorm` byte for one alpha channel — this repair's core fix,
 *  replacing the ±1 tolerance the first repair shipped. */
function predictedAlphaByte(idx: number): number {
    const e = alphaF32(idx);
    const clamped = Math.min(1, Math.max(0, e));
    const scaled = f32(clamped * 255);
    return roundTiesToEven(scaled) & 0xff;
}

// the CPU reference for one cell, mirroring `grid.ts`'s fill kernel formula exactly (glyph ramp wrap,
// the diagonal fg/bg gradient) — the same differential shape `cell.test.ts` runs, but against a real
// dispatched buffer instead of a hand-built one. Glyph and rgb come from production `packCell` unchanged
// (never engineered onto a tie, so `unorm8`'s CPU-vs-GPU f64-vs-f32 seam never bites them). Alpha comes
// from `predictedAlphaByte` — the spec-correct reference — not from `packCell`'s own CPU `unorm8` arm,
// which is `naiveAlphaFg`/`naiveAlphaBg` below: computed only to derive the predicted-divergence count,
// never used as the assertion target.
function expectedCell(x: number, y: number) {
    const glyph = (x + y) % CELL_GLYPH_COUNT;
    const u = x / COLS;
    const v = y / ROWS;
    const alphaFg = 1 - (x + 0.5) / 255;
    const alphaBg = 1 - (y + 0.5) / 255;
    const packed = packCell(glyph, vec4f(u, v, 1 - u, alphaFg), vec4f(1 - u, 1 - v, v, alphaBg));
    const buf = new ArrayBuffer(CELL_BYTES);
    new Uint32Array(buf).set([packed.x, packed.y, packed.z]);
    const naive = unpackCell(buf, 0);
    const predictedFgAlpha = predictedAlphaByte(x);
    const predictedBgAlpha = predictedAlphaByte(y);
    return {
        glyph: naive.glyph,
        fg: [naive.fg[0], naive.fg[1], naive.fg[2], predictedFgAlpha] as [
            number,
            number,
            number,
            number,
        ],
        bg: [naive.bg[0], naive.bg[1], naive.bg[2], predictedBgAlpha] as [
            number,
            number,
            number,
            number,
        ],
        naiveFgAlpha: naive.fg[3],
        naiveBgAlpha: naive.bg[3],
    };
}

async function assertFillDispatch(): Promise<Check> {
    if (!gridMirror) return { name: "cells fill dispatch", pass: false, detail: "no grid mirror" };
    await settle(gridMirror);
    const snap = gridMirror.snapshot;
    if (!snap) return { name: "cells fill dispatch", pass: false, detail: "no grid snapshot" };

    let mismatches = 0;
    let predictedDivergences = 0;
    let first = "";
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const i = y * COLS + x;
            const actual = unpackCell(snap.bytes, i);
            const wanted = expectedCell(x, y);
            if (wanted.fg[3] !== wanted.naiveFgAlpha) predictedDivergences++;
            if (wanted.bg[3] !== wanted.naiveBgAlpha) predictedDivergences++;
            const fgOk = actual.fg.every((c, k) => c === wanted.fg[k]);
            const bgOk = actual.bg.every((c, k) => c === wanted.bg[k]);
            const same = actual.glyph === wanted.glyph && fgOk && bgOk;
            if (!same) {
                mismatches++;
                if (!first) {
                    first = `(${x},${y}) glyph ${actual.glyph}/${wanted.glyph} fg [${actual.fg}]/[${wanted.fg}] bg [${actual.bg}]/[${wanted.bg}]`;
                }
            }
        }
    }
    // Non-vacuity: `predictedDivergences` is a fixed property of this JS reference alone (never the
    // device), so a fixture edit that silently stops constructing the seam (a COLS/ROWS change, a
    // pattern rewrite) reds here instead of the assertion below reading a vacuous 0-divergence green.
    if (predictedDivergences !== EXPECTED_TIE_DIVERGENCES) {
        return {
            name: "cells fill dispatch matches the predicted-tie reference",
            pass: false,
            detail: `predicted-tie reference constructed ${predictedDivergences} alpha divergences from the naive half-up reference, expected exactly ${EXPECTED_TIE_DIVERGENCES} — the fixture no longer constructs the rounding seam this scenario exists to exercise`,
        };
    }
    return {
        name: "cells fill dispatch matches the predicted-tie reference",
        pass: mismatches === 0,
        detail:
            mismatches === 0
                ? `${COLS * ROWS} cells bit-exact on all 4 lanes (glyph, fg, bg incl. alpha, no tolerance); ${predictedDivergences}/${COLS * ROWS * 2} alpha samples predicted (ties-to-even) to diverge from the naive half-up CPU reference, all matched bit-exact against the device`
                : `${mismatches}/${COLS * ROWS} cells mismatched against the predicted-tie reference; first (actual/expected): ${first}`,
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
