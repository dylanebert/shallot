import { Compute, type Mirror, MirrorPlugin, mirror, type Plugin, run } from "@dylanebert/shallot";
import {
    CELL_AT,
    CELL_BYTES,
    CELL_FILL_GLYPHS,
    CELL_GLYPH_COUNT,
    createCellGrid,
    dispatchSelect,
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

// this fixture's own tie-divergence set, measured directly rather than derived from a parity law: fg
// alpha's predicted byte diverges from the naive half-up reference at x = 1, 3, 5, 7, 9 (every odd x in
// [0, COLS), 5 of 10), each affecting all ROWS rows; bg alpha diverges at y = 1, 3, 5 (every odd y in
// [0, ROWS), 3 of 6), each affecting all COLS columns. 5*6 + 3*10 = 60 — a static property of this JS
// reference (not the device), so it's asserted as an exact pin below rather than a >0 non-vacuity floor.
//
// The odd/even split above is NOT a parity law and doesn't generalize: `naiveAlphaByte` (`engine/utils/
// tgsl.ts`'s `unorm8`) computes alpha in f64 and applies a single `Math.fround` at the very end, which
// does not land exactly on the tie the multi-step f32 arithmetic (`alphaF32` below) constructs — the two
// paths disagree by ordinary floating-point rounding along two different computation routes, not by a
// half-up-vs-half-to-even split at a shared tie. That the disagreement lines up with x/y parity is a
// property of this fixture's small range (COLS=10, ROWS=6): re-running the same predicted-vs-naive
// comparison for idx in [0, 255) breaks the "diverges iff odd" rule 96 times out of 255 samples — so this
// is a measured constant for this fixture, checked directly against the two production formulas above,
// not a structural pattern that would hold at other grid sizes.
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
// from `predictedAlphaByte` — which assumes correctly-rounded f32 division at each step (true of this
// JS reference's f64-divide-then-`fround` construction), where WGSL's real division only guarantees a
// spec'd ULP tolerance — so a green assertFillDispatch below is a reading about the adapter it actually
// ran on (`grid.ts`'s own module doc states this honestly), never a universal spec guarantee. Naive alpha
// (`naive.fg[3]`/`naive.bg[3]`, from `packCell`'s own CPU `unorm8` arm) is computed only to derive the
// predicted-divergence count, never used as the assertion target.
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
            name: "cells fill dispatch matches the predicted-tie reference (adapter-dependent — a correctly-rounded-division reading, not a WGSL guarantee)",
            pass: false,
            detail: `predicted-tie reference constructed ${predictedDivergences} alpha divergences from the naive half-up reference, expected exactly ${EXPECTED_TIE_DIVERGENCES} — the fixture no longer constructs the rounding seam this scenario exists to exercise`,
        };
    }
    return {
        name: "cells fill dispatch matches the predicted-tie reference (adapter-dependent — a correctly-rounded-division reading, not a WGSL guarantee)",
        pass: mismatches === 0,
        detail:
            mismatches === 0
                ? `${COLS * ROWS} cells bit-exact on all 4 lanes (glyph, fg, bg incl. alpha, no tolerance); ${predictedDivergences}/${COLS * ROWS * 2} alpha samples predicted (ties-to-even) to diverge from the naive half-up CPU reference, all matched bit-exact against the device`
                : `${mismatches}/${COLS * ROWS} cells mismatched against the predicted-tie reference; first (actual/expected): ${first} — WGSL division carries only a spec'd ULP tolerance, not a correctly-rounded guarantee, so a red here on an adapter other than the one this reference was measured against (nvidia/lovelace) is a portability reading, not necessarily a code defect — diagnose against the adapter name before treating it as a regression`,
    };
}

// S3's real content producer (`select.ts`'s two compute passes) against a synthetic HDR source texture —
// deliberately not a rendered 3D scene: `recordSelect`/`dispatchSelect` bind against any float-sampleable
// 2D texture view (`select.ts`'s own contract), so a hand-authored pattern gives a deterministic, bounded
// "one scene at one tick" input with a known edge and a known flat region, at a fraction of a real
// camera/light/mesh scene's setup cost and with zero cross-frame nondeterminism to control for. A vertical
// step edge at the grid's horizontal midpoint (columns 0-3 bright, 4-7 dark) — SELECT_COLS/SELECT_ROWS
// deliberately small (8×4) so the whole readback is cheap to walk cell-by-cell.
const SELECT_COLS = 8;
const SELECT_ROWS = 4;
const SELECT_BLOCK = 4; // device px per cell edge
const SELECT_SRC_W = SELECT_COLS * SELECT_BLOCK;
const SELECT_SRC_H = SELECT_ROWS * SELECT_BLOCK;
// bright/dark post-tonemap luma: reinhard(2.0) ≈ 0.667, reinhard(0.05) ≈ 0.0476 — a Sobel magnitude of
// 4×|0.667−0.0476| ≈ 2.48 at the boundary columns, well over EDGE_MAGNITUDE_THRESHOLD (0.4), and exactly
// 0 at every column away from the boundary (uniform neighborhoods, `select.ts`'s own derivation)
const SELECT_BRIGHT = 2.0;
const SELECT_DARK = 0.05;

let selectGrid: ReturnType<typeof createCellGrid> | null = null;
let selectMirror: Mirror | null = null;
let selectSourceTexture: GPUTexture | null = null;

const SelectPlugin: Plugin = {
    name: "GymCellsSelect",
    initialize() {
        const device = Compute.device;
        selectSourceTexture = device.createTexture({
            label: "cells-select-src",
            size: { width: SELECT_SRC_W, height: SELECT_SRC_H },
            format: "rgba32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const data = new Float32Array(SELECT_SRC_W * SELECT_SRC_H * 4);
        for (let y = 0; y < SELECT_SRC_H; y++) {
            for (let x = 0; x < SELECT_SRC_W; x++) {
                const v = x < SELECT_SRC_W / 2 ? SELECT_BRIGHT : SELECT_DARK;
                const o = (y * SELECT_SRC_W + x) * 4;
                data[o] = v;
                data[o + 1] = v;
                data[o + 2] = v;
                data[o + 3] = 1;
            }
        }
        device.queue.writeTexture(
            { texture: selectSourceTexture },
            data,
            { bytesPerRow: SELECT_SRC_W * 16 },
            { width: SELECT_SRC_W, height: SELECT_SRC_H },
        );
        selectGrid = createCellGrid(SELECT_COLS, SELECT_ROWS, CELL_GLYPH_COUNT);
        dispatchSelect(
            selectGrid.buffer,
            SELECT_COLS,
            SELECT_ROWS,
            selectSourceTexture.createView(),
            SELECT_SRC_W,
            SELECT_SRC_H,
        );
        selectMirror = mirror(selectGrid.buffer);
    },
};

// criterion 4's cross-target differential: decode the *same* readback bytes two structurally independent
// ways — `unpackCell` (a Uint32Array view + bit-shift, what `packages/shallot-tui`'s future terminal-side
// bridge calls) against a raw Uint8Array walk assembling the glyph word by hand (what a GPU vertex fetch
// reading the identical struct layout does, one level of abstraction lower). One producer buffer, one
// scene tick, two independent readers — the strongest arm in the unit, per the spec's own naming: if the
// web draw path and the terminal encoder's input could ever diverge on what one cell means, this is where
// it would show, since nothing here is shared code between the two decode functions.
function decodeByteView(bytes: ArrayBuffer, index: number) {
    const u8 = new Uint8Array(bytes, index * CELL_BYTES, CELL_BYTES);
    const glyphOff = CELL_AT.glyph * 4;
    const fgOff = CELL_AT.fg * 4;
    const bgOff = CELL_AT.bg * 4;
    const glyph =
        (u8[glyphOff] |
            (u8[glyphOff + 1] << 8) |
            (u8[glyphOff + 2] << 16) |
            (u8[glyphOff + 3] << 24)) >>>
        0;
    const fg: [number, number, number, number] = [
        u8[fgOff],
        u8[fgOff + 1],
        u8[fgOff + 2],
        u8[fgOff + 3],
    ];
    const bg: [number, number, number, number] = [
        u8[bgOff],
        u8[bgOff + 1],
        u8[bgOff + 2],
        u8[bgOff + 3],
    ];
    return { glyph, fg, bg };
}

async function assertSelectDispatch(): Promise<Check> {
    const name =
        "select dispatch: web-draw-shaped decode == terminal-encoder-shaped decode, one buffer";
    if (!selectMirror) return { name, pass: false, detail: "no select grid mirror" };
    await settle(selectMirror);
    const snap = selectMirror.snapshot;
    if (!snap) return { name, pass: false, detail: "no select grid snapshot" };

    let mismatches = 0;
    let first = "";
    let directionalAtBoundary = 0;
    let fillAwayFromBoundary = 0;
    for (let y = 0; y < SELECT_ROWS; y++) {
        for (let x = 0; x < SELECT_COLS; x++) {
            const i = y * SELECT_COLS + x;
            const a = unpackCell(snap.bytes, i);
            const b = decodeByteView(snap.bytes, i);
            const same =
                a.glyph === b.glyph &&
                a.fg.every((c, k) => c === b.fg[k]) &&
                a.bg.every((c, k) => c === b.bg[k]);
            if (!same) {
                mismatches++;
                if (!first) {
                    first = `(${x},${y}) unpackCell glyph ${a.glyph} fg [${a.fg}] bg [${a.bg}] / byteView glyph ${b.glyph} fg [${b.fg}] bg [${b.bg}]`;
                }
            }
            const directional = a.glyph >= CELL_FILL_GLYPHS.length;
            if ((x === 3 || x === 4) && directional) directionalAtBoundary++;
            if ((x === 0 || x === SELECT_COLS - 1) && !directional) fillAwayFromBoundary++;
        }
    }

    if (mismatches > 0) {
        return {
            name,
            pass: false,
            detail: `${mismatches}/${SELECT_COLS * SELECT_ROWS} cells disagreed between the two decode paths; first: ${first}`,
        };
    }
    // non-vacuity: both selection branches (edge → directional, flat → fill) must actually have fired,
    // or the decode-agreement check above is vacuously comparing an all-fill (or all-directional) buffer
    const wantBoundary = SELECT_ROWS * 2; // both boundary columns (3, 4), every row
    const wantAway = SELECT_ROWS * 2; // both far columns (0, last), every row
    if (directionalAtBoundary !== wantBoundary || fillAwayFromBoundary !== wantAway) {
        return {
            name,
            pass: false,
            detail: `branch coverage missing — directional-at-boundary ${directionalAtBoundary}/${wantBoundary}, fill-away-from-boundary ${fillAwayFromBoundary}/${wantAway}; the synthetic edge didn't reach both selection branches`,
        };
    }
    return {
        name,
        pass: true,
        detail: `${SELECT_COLS * SELECT_ROWS} cells' unpackCell and byte-view decodes agreed bit-exact; the edge fired a directional glyph at both boundary columns and a fill glyph at both far columns every row`,
    };
}

const scenario: Scenario = {
    name: "cells",
    noRender: true,
    async build(_canvas) {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [ProfilePlugin, MirrorPlugin, CellsPlugin, SelectPlugin],
        });
        return {
            state,
            dispose() {
                grid = null;
                gridMirror = null;
                selectGrid = null;
                selectMirror = null;
                selectSourceTexture?.destroy();
                selectSourceTexture = null;
                dispose();
            },
        };
    },
    async assert(): Promise<Check[]> {
        return [await assertFillDispatch(), await assertSelectDispatch()];
    },
    live(): string {
        return `${COLS}x${ROWS} cell grid, compute-only (no framed scene)`;
    },
};

register(scenario);
