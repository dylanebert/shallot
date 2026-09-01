import { Compute, type Mirror, MirrorPlugin, mirror, type Plugin, run } from "@dylanebert/shallot";
import {
    CELL_AT,
    CELL_BYTES,
    CELL_FILL_GLYPHS,
    CELL_GLYPH_COUNT,
    CELL_U32S,
    createCellGrid,
    dispatchSelect,
    drawCells,
    fillCellGrid,
    MISSING_GLYPH_UV,
    packCell,
    resetDrawPipeline,
    unpackCell,
} from "@dylanebert/shallot/cells/core";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { Render } from "@dylanebert/shallot/render/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { vec4f } from "typegpu/data";
import * as std from "typegpu/std";
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

// select's own non-vacuity check: both selection branches (edge → directional, flat → fill) actually
// fire against a real dispatch, over the same synthetic step-edge `SelectPlugin` built above. This used
// to also compare `unpackCell` against a hand-rolled byte-walk decode of the same buffer as if that were
// criterion 4's differential — it wasn't: both decoders import `CELL_AT` from `cell.ts`, so a field
// reorder moves both offsets together and the "structurally independent" claim in the module comment
// that used to sit above them was false. Criterion 4's real differential is `assertDrawDispatch` below,
// which drives an actual GPU dispatch of `draw.ts`'s own render pipeline.
async function assertSelectDispatch(): Promise<Check> {
    const name = "select dispatch: a real edge fires both the directional and fill branches";
    if (!selectMirror) return { name, pass: false, detail: "no select grid mirror" };
    await settle(selectMirror);
    const snap = selectMirror.snapshot;
    if (!snap) return { name, pass: false, detail: "no select grid snapshot" };

    let directionalAtBoundary = 0;
    let fillAwayFromBoundary = 0;
    // the s3r item 8 regression guard: a genuinely flat (zero-gradient), near-zero-luma region must
    // never select a directional glyph — the reported defect (a uniform "|" over an unlit background)
    // was diagnosed live against the real recipe scene and refuted at the selection layer: the flat
    // background's own cells read back glyph index 2 (`'`, a legitimate near-blank fill glyph, per
    // `ramp-table.ts`), never a directional index. `magnitude` is an exact 0 for a uniform 3×3
    // neighborhood by algebraic cancellation (`(1·L+2·L+1·L) - (1·L+2·L+1·L) = 0` for any `L`, no
    // floating-point rounding involved), so the fill branch is the only branch a flat field can ever
    // reach — this arm pins that a low-luma flat region (`SELECT_DARK`, the far column below) lands on a
    // low fill index specifically, not merely a non-directional one, since a low luma landing high in
    // the ramp would be a different (real) selection bug this check would otherwise miss.
    let farDarkLowIndex = true;
    const wantLowIndexBound = 10; // SELECT_DARK's luma (~0.0476 post-tonemap) rounds to index ~4
    for (let y = 0; y < SELECT_ROWS; y++) {
        for (let x = 0; x < SELECT_COLS; x++) {
            const i = y * SELECT_COLS + x;
            const a = unpackCell(snap.bytes, i);
            const directional = a.glyph >= CELL_FILL_GLYPHS.length;
            if ((x === 3 || x === 4) && directional) directionalAtBoundary++;
            if ((x === 0 || x === SELECT_COLS - 1) && !directional) fillAwayFromBoundary++;
            if (x === SELECT_COLS - 1 && a.glyph >= wantLowIndexBound) farDarkLowIndex = false;
        }
    }

    const wantBoundary = SELECT_ROWS * 2; // both boundary columns (3, 4), every row
    const wantAway = SELECT_ROWS * 2; // both far columns (0, last), every row
    if (directionalAtBoundary !== wantBoundary || fillAwayFromBoundary !== wantAway) {
        return {
            name,
            pass: false,
            detail: `branch coverage missing — directional-at-boundary ${directionalAtBoundary}/${wantBoundary}, fill-away-from-boundary ${fillAwayFromBoundary}/${wantAway}; the synthetic edge didn't reach both selection branches`,
        };
    }
    if (!farDarkLowIndex) {
        return {
            name,
            pass: false,
            detail: `the flat, near-zero-luma far column (SELECT_DARK) selected a fill glyph at or past index ${wantLowIndexBound} — a flat dark region should land near the blank end of the ramp`,
        };
    }
    return {
        name,
        pass: true,
        detail: `${SELECT_COLS * SELECT_ROWS} cells read back; the edge fired a directional glyph at both boundary columns and a fill glyph at both far columns every row`,
    };
}

// criterion 4's real differential: drive `draw.ts`'s own render pipeline — the actual vertex stage that
// indexes `drawLayout.$.cells[iid]` and the actual fragment stage that decodes `cell.fg`/`cell.bg` via
// `unpack4x8unorm` — against one hand-packed cell buffer neither `draw.ts` nor this arm's assertion
// constructed together, then compare the *rendered pixels* against `packCell`'s own inputs. This is what
// the retired `decodeByteView` comparison claimed to be and wasn't: that comparison read the same words
// through two decoders sharing `CELL_AT`, so a bug in `draw.ts`'s own body (its fg/bg swapped, or its
// `cell.glyph` read from the wrong lane) could never show there — both readers would still agree with
// each other, just wrongly. Driving the real WGSL fragment stage is the only way to see `draw.ts`'s own
// logic rather than a second transcription of `cell.ts`'s layout.
//
// Two cells, two colors, no font: cell 0's glyph indexes a zero-area (`MISSING_GLYPH_UV`) rect, so its
// fragment never draws ink and its rendered pixel is pure `bg`; cell 1's glyph indexes a full `(0,0,1,1)`
// rect sampled against a solid-255 synthetic "atlas" texture, which `sdfToSignedDistance` decodes as
// maximally inside (`text/glyph.ts`'s own derivation) — so cell 1's *whole* quad saturates to alpha 1 and
// its rendered pixel is pure `fg`. Swap `fg`/`bg` in `draw.ts`'s fragment stage and cell 0 renders `fg`'s
// color while cell 1 renders `bg`'s — this arm reds on exactly that mutation.
const DRAW_COLS = 2;
const DRAW_ROWS = 1;
const DRAW_CELL_PX = 16; // device px per cell edge, clear of any AA fringe at the sampled center
const DRAW_VIEW_W = DRAW_COLS * DRAW_CELL_PX;
const DRAW_VIEW_H = DRAW_ROWS * DRAW_CELL_PX;
const DRAW_TARGET_FORMAT: GPUTextureFormat = "rgba8unorm";
// two linear colors, comfortably mid-range — the sRGB transfer function's slope is steepest near 0,
// where packCell's sRGB encode + this target's own unorm store would amplify 8-bit quantization noise.
const DRAW_FG = vec4f(0.8, 0.2, 0.1, 1);
const DRAW_BG = vec4f(0.1, 0.6, 0.9, 1);
// two 8-bit quantization steps (packCell's sRGB pack, this target's unorm store) against a 0.7-wide gap
// between fg and bg on every channel — comfortably inside a fg/bg swap, comfortably outside round-off.
const DRAW_TOLERANCE = 0.03;

const probeLayout = tgpu.bindGroupLayout({
    src: {
        texture: d.texture2d(d.f32),
        sampleType: "unfilterable-float",
        visibility: ["compute"],
    },
    out: {
        storage: (n: number) => d.arrayOf(d.vec4f, n),
        access: "mutable",
        visibility: ["compute"],
    },
});

// samples the center of each of the two cells drawn into the draw target — reads through no code this
// unit authored beyond the bind group itself, so its own value is a plain GPU texture read.
const probeKernel = tgpu.computeFn({
    workgroupSize: [1],
    in: { gid: d.builtin.globalInvocationId },
})(() => {
    "use gpu";
    probeLayout.$.out[0] = std.textureLoad(
        probeLayout.$.src,
        d.vec2u(DRAW_CELL_PX / 2, DRAW_VIEW_H / 2),
        0,
    );
    probeLayout.$.out[1] = std.textureLoad(
        probeLayout.$.src,
        d.vec2u(DRAW_CELL_PX + DRAW_CELL_PX / 2, DRAW_VIEW_H / 2),
        0,
    );
});

let drawTarget: GPUTexture | null = null;
let drawAtlas: GPUTexture | null = null;
let drawSampler: GPUSampler | null = null;
let drawProbeOut: ReturnType<typeof allocDrawProbeOut> | null = null;
let drawProbeMirror: Mirror | null = null;

function allocDrawGlyphUv() {
    return Compute.root
        .createBuffer(d.arrayOf(d.vec4f, 2))
        .$usage("storage")
        .$name("cells-gym-draw-glyphuv");
}

function allocDrawProbeOut() {
    return Compute.root
        .createBuffer(d.arrayOf(d.vec4f, 2))
        .$usage("storage")
        .$name("cells-gym-draw-probe-out");
}

// place packCell's own (glyph, fgPacked, bgPacked) triple at Cell's *current* field offsets (`CELL_AT`,
// the same source `unpackCell` and `draw.ts`'s struct read both derive from) rather than assuming a word
// order — so this fixture stays correct under a field reorder and this arm's only sensitivity is to a
// logic bug in what `draw.ts` does with those words, not to how `cell.ts` orders them.
function writeDrawCells(buffer: GPUBuffer): void {
    const words = new Uint32Array(DRAW_COLS * DRAW_ROWS * CELL_U32S);
    const place = (cellIndex: number, packed: { x: number; y: number; z: number }) => {
        const base = cellIndex * CELL_U32S;
        words[base + CELL_AT.glyph] = packed.x;
        words[base + CELL_AT.fg] = packed.y;
        words[base + CELL_AT.bg] = packed.z;
    };
    place(0, packCell(0, DRAW_FG, DRAW_BG)); // glyph 0 -> MISSING_GLYPH_UV: blank, renders bg
    place(1, packCell(1, DRAW_FG, DRAW_BG)); // glyph 1 -> full rect on the solid atlas: renders fg
    Compute.device.queue.writeBuffer(buffer, 0, words);
}

const DrawPlugin: Plugin = {
    name: "GymCellsDraw",
    initialize() {
        const device = Compute.device;
        resetDrawPipeline();
        Render.format = DRAW_TARGET_FORMAT;

        const drawGrid = createCellGrid(DRAW_COLS, DRAW_ROWS, 2);
        writeDrawCells(Compute.root.unwrap(drawGrid.buffer));

        const drawGlyphUv = allocDrawGlyphUv();
        device.queue.writeBuffer(
            Compute.root.unwrap(drawGlyphUv),
            0,
            new Float32Array([...MISSING_GLYPH_UV, 0, 0, 1, 1]),
        );

        drawAtlas = device.createTexture({
            label: "cells-gym-draw-atlas",
            size: { width: 4, height: 4 },
            format: "r8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        // 255 everywhere: sdfToSignedDistance(1, maxDim) = -maxDim (deep inside), so cell 1's whole quad
        // reads alpha = 1 regardless of filtering — no real font, no real SDF generator, just the
        // deep-inside sentinel value the fragment stage's smoothstep saturates on.
        device.queue.writeTexture(
            { texture: drawAtlas },
            new Uint8Array(16).fill(255),
            { bytesPerRow: 4 },
            { width: 4, height: 4 },
        );
        drawSampler = device.createSampler({
            label: "cells-gym-draw",
            magFilter: "linear",
            minFilter: "linear",
        });

        drawTarget = device.createTexture({
            label: "cells-gym-draw-target",
            size: { width: DRAW_VIEW_W, height: DRAW_VIEW_H },
            format: DRAW_TARGET_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        const encoder = device.createCommandEncoder({ label: "cells-gym-draw" });
        drawCells(
            encoder,
            drawTarget.createView(),
            drawGrid.buffer,
            drawGlyphUv,
            drawAtlas.createView(),
            drawSampler,
            DRAW_COLS,
            DRAW_ROWS,
            DRAW_VIEW_W,
            DRAW_VIEW_H,
        );

        drawProbeOut = allocDrawProbeOut();
        const probeGroup = Compute.root.createBindGroup(probeLayout, {
            src: drawTarget.createView(),
            out: drawProbeOut,
        });
        const probePipeline = Compute.root
            .createComputePipeline({ compute: probeKernel })
            .$name("cells-gym-draw-probe");
        const probePass = encoder.beginComputePass({ label: "cells-gym-draw-probe" });
        probePipeline.with(probeGroup).with(probePass).dispatchWorkgroups(1);
        probePass.end();

        device.queue.submit([encoder.finish()]);
        drawProbeMirror = mirror(drawProbeOut);
    },
};

async function assertDrawDispatch(): Promise<Check> {
    const name =
        "draw dispatch: draw.ts's own vertex/fragment stage reads cell.fg/cell.bg correctly";
    if (!drawProbeMirror) return { name, pass: false, detail: "no draw probe mirror" };
    await settle(drawProbeMirror);
    const snap = drawProbeMirror.snapshot;
    if (!snap) return { name, pass: false, detail: "no draw probe snapshot" };

    const out = new Float32Array(snap.bytes);
    const blank: [number, number, number] = [out[0], out[1], out[2]];
    const ink: [number, number, number] = [out[4], out[5], out[6]];
    const wantBlank: [number, number, number] = [DRAW_BG.x, DRAW_BG.y, DRAW_BG.z];
    const wantInk: [number, number, number] = [DRAW_FG.x, DRAW_FG.y, DRAW_FG.z];
    const close = (a: [number, number, number], b: [number, number, number]) =>
        Math.abs(a[0] - b[0]) < DRAW_TOLERANCE &&
        Math.abs(a[1] - b[1]) < DRAW_TOLERANCE &&
        Math.abs(a[2] - b[2]) < DRAW_TOLERANCE;
    const blankOk = close(blank, wantBlank);
    const inkOk = close(ink, wantInk);

    return {
        name,
        pass: blankOk && inkOk,
        detail:
            blankOk && inkOk
                ? `blank cell rendered [${blank.map((c) => c.toFixed(3))}] ~ bg [${wantBlank}], ink cell rendered [${ink.map((c) => c.toFixed(3))}] ~ fg [${wantInk}] — read through draw.ts's real bind-group accessors`
                : `blank cell rendered [${blank.map((c) => c.toFixed(3))}] (want ~bg [${wantBlank}]), ink cell rendered [${ink.map((c) => c.toFixed(3))}] (want ~fg [${wantInk}]) — draw.ts's own read of cell.fg/cell.bg disagrees with what packCell packed`,
    };
}

const scenario: Scenario = {
    name: "cells",
    noRender: true,
    async build(_canvas) {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [ProfilePlugin, MirrorPlugin, CellsPlugin, SelectPlugin, DrawPlugin],
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
                drawTarget?.destroy();
                drawTarget = null;
                drawAtlas?.destroy();
                drawAtlas = null;
                drawSampler = null;
                drawProbeOut = null;
                drawProbeMirror = null;
                resetDrawPipeline();
                dispose();
            },
        };
    },
    async assert(): Promise<Check[]> {
        return [
            await assertFillDispatch(),
            await assertSelectDispatch(),
            await assertDrawDispatch(),
        ];
    },
    live(): string {
        return `${COLS}x${ROWS} cell grid, compute-only (no framed scene)`;
    },
};

register(scenario);
