// SGR (Select Graphic Rendition) encoding and run coalescing. A "run" is a maximal sequence of
// cells sharing one fg/bg pair — coalescing means one SGR prefix covers the whole run instead of
// one per cell, which is the difference between the spec's ~9 (ansi256) / ~19 (truecolor)
// bytes-per-*changed-cell* estimate and paying that cost per cell regardless of runs.

import type { Tier } from "./color-support";
import type { Cell } from "./types";
import { rgbEqual } from "./types";

/**
 * The xterm 256-color 6x6x6 cube's real per-channel levels — **not** evenly spaced. This is the
 * palette every real terminal actually renders; a formula that guesses evenly-spaced steps (as a
 * prior version of this file did) picks the visibly wrong cube corner across a wide input range
 * (B2) — e.g. r=30 was mapped to the corner that renders 95 (error 65) when 0 (error 30) is
 * nearer, and every dark channel value from 26 upward was sent to mid-grey on a real terminal.
 */
const CUBE_LEVELS = [0, 95, 135, 175, 215, 255] as const;

/** 256-color palette indices 232-255: a 24-step grayscale ramp, `8 + 10*step`. Finer-grained near
 * grey than the color cube (whose steps are 40-95 apart), so a near-grey RGB often quantizes
 * better here than to any cube corner. */
const GREY_RAMP_COUNT = 24;
const greyRampLevel = (step: number): number => 8 + 10 * step;

/** Index (0-5) of the nearest level in `levels` to `value`, by real distance — never a
 * divide-and-round guess at spacing. */
function nearestLevelIndex(value: number, levels: readonly number[]): number {
    let best = 0;
    let bestDist = Math.abs(value - levels[0]);
    for (let i = 1; i < levels.length; i++) {
        const dist = Math.abs(value - levels[i]);
        if (dist < bestDist) {
            bestDist = dist;
            best = i;
        }
    }
    return best;
}

/** Quantizes an 8-bit channel to its nearest xterm 256-color cube step (0-5), by real distance
 * against `CUBE_LEVELS`. */
export function cube6(channel: number): number {
    return nearestLevelIndex(channel, CUBE_LEVELS);
}

const squaredDistance = (r: number, g: number, b: number, cr: number, cg: number, cb: number) =>
    (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;

/**
 * Maps an RGB color to its nearest xterm 256-color palette index — the 6x6x6 color cube
 * (16-231) or the 24-step grayscale ramp (232-255), whichever is a real nearest neighbour by
 * squared distance. A saturated color always wins the cube (the ramp is achromatic, so its
 * distance to a saturated color is large); a near-grey color often quantizes more accurately on
 * the ramp, whose steps are ~10 apart versus the cube's 40-95.
 */
export function ansi256FromRgb(r: number, g: number, b: number): number {
    const cr = cube6(r);
    const cg = cube6(g);
    const cb = cube6(b);
    const cubeIndex = 16 + 36 * cr + 6 * cg + cb;
    const cubeDist = squaredDistance(r, g, b, CUBE_LEVELS[cr], CUBE_LEVELS[cg], CUBE_LEVELS[cb]);

    const avg = Math.round((r + g + b) / 3);
    const greySteps = Array.from({ length: GREY_RAMP_COUNT }, (_, i) => greyRampLevel(i));
    const greyStep = nearestLevelIndex(avg, greySteps);
    const greyValue = greySteps[greyStep];
    const greyDist = squaredDistance(r, g, b, greyValue, greyValue, greyValue);

    return greyDist < cubeDist ? 232 + greyStep : cubeIndex;
}

/** Two cells share a run iff their fg and bg are structurally equal — glyph is irrelevant to style. */
export function sameStyle(a: Cell, b: Cell): boolean {
    return rgbEqual(a.fg, b.fg) && rgbEqual(a.bg, b.bg);
}

/**
 * The SGR escape that establishes `cell`'s style at the given tier, or `""` for a colorless tier
 * (`plain` / `glyph`). SGR is sticky — a terminal keeps whatever fg/bg a prior run set until
 * something changes it — so every color-tier prefix declares *both* channels explicitly, never
 * omitting one: an explicit color code for a set channel, or the "reset to terminal default" code
 * (`39` for fg, `49` for bg) for a `null` one. `types.ts`'s `Cell` contract makes `null` mean "the
 * terminal's own default," distinct from black — a prefix that silently omitted a null channel
 * would let a preceding run's color bleed onto a cell that asked for the default, which is a
 * contract violation, not a harmless omission.
 */
export function sgrPrefix(cell: Cell, tier: Tier): string {
    if (tier === "plain" || tier === "glyph") return "";
    const { fg, bg } = cell;
    const params =
        tier === "truecolor"
            ? [
                  fg ? `38;2;${fg.r};${fg.g};${fg.b}` : "39",
                  bg ? `48;2;${bg.r};${bg.g};${bg.b}` : "49",
              ]
            : [
                  fg ? `38;5;${ansi256FromRgb(fg.r, fg.g, fg.b)}` : "39",
                  bg ? `48;5;${ansi256FromRgb(bg.r, bg.g, bg.b)}` : "49",
              ];
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
