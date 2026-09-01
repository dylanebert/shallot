// The encoder round-trips at every tier.
//
// Encodes a known multi-frame sequence (a full first paint, a sparse diffed update touching
// non-adjacent cells across different rows, a resize, another sparse diffed update, then a
// repeat of the last frame with no changes at all) and parses the emitted bytes back through
// `TerminalModel`, a minimal terminal reader independent of the encoder's own logic. Every
// intermediate frame's reconstructed grid must equal the source grid (projected onto what the
// tier can carry — `terminal-model.ts`'s `projectForTier`), and the repeated final frame must
// encode to zero bytes.
//
// Two-sided, structurally: the "the two-sided proof itself" describe block below runs the same
// shape of assertion against two deliberately-broken miniature encoders — one with diff
// suppression disabled, one with cursor addressing removed for diffed runs — and asserts each one
// fails exactly where the property it lacks predicts. That's the two-sidedness the criterion
// requires, wired as a permanent, always-run check rather than a hand-reverted-and-observed
// one-time proof whose witness doesn't survive in the tree.

import { describe, expect, test } from "bun:test";
import type { Tier } from "../src/color-support";
import { CLEAR_SCREEN, CURSOR_HOME, cursorTo, SGR_RESET } from "../src/cursor";
import { diffRuns } from "../src/diff";
import { Encoder } from "../src/encoder";
import { encodeRun } from "../src/sgr";
import type { Cell, Grid } from "../src/types";
import { makeGrid } from "../src/types";
import { projectForTier, TerminalModel } from "./terminal-model";

const COLOR_A = { r: 37, g: 140, b: 201 };
const COLOR_B = { r: 220, g: 20, b: 60 };

function cell(glyph: string, tier: Tier, variant: "a" | "b" | null): Cell {
    const useColor = tier === "ansi256" || tier === "truecolor";
    const color = variant === null ? null : variant === "a" ? COLOR_A : COLOR_B;
    return { glyph, fg: useColor ? color : null, bg: null };
}

// Frame A: 6x3, a full initial paint alternating color A/B by (x+y) parity.
function frameA(tier: Tier): Grid {
    return makeGrid(6, 3, (x, y) => cell(".", tier, (x + y) % 2 === 0 ? "a" : "b"));
}

// Frame B: sparse diffed update against A — three non-adjacent changed cells across two
// different rows, chosen so that (row 0, col 5) and (row 2, col 2) are nowhere near where the
// cursor sits after a full repaint of frame A (which ends at row 2, col 6) or after any prior
// run in this frame — exactly the shape that exposes a missing `cursorTo`.
function frameB(tier: Tier): Grid {
    const g = frameA(tier);
    const rows = g.cells.map((row) => row.slice());
    rows[0][0] = cell("X", tier, "b");
    rows[0][5] = cell("Y", tier, "a");
    rows[2][2] = cell("Z", tier, "b");
    return { width: g.width, height: g.height, cells: rows };
}

// Frame C: a resize to 4x2 with fresh content — the encoder's resize path (full repaint).
function frameC(tier: Tier): Grid {
    return makeGrid(4, 2, (x) => cell("o", tier, x < 2 ? "a" : "b"));
}

// Frame D: sparse diffed update against C, again non-adjacent across rows.
function frameD(tier: Tier): Grid {
    const g = frameC(tier);
    const rows = g.cells.map((row) => row.slice());
    rows[0][0] = cell("P", tier, "b");
    rows[1][3] = cell("Q", tier, "a");
    return { width: g.width, height: g.height, cells: rows };
}

describe("round trip — encode then parse back through TerminalModel", () => {
    for (const tier of ["glyph", "ansi256", "truecolor"] as const) {
        test(`${tier}: every frame in a diffed multi-frame sequence (incl. a resize) round-trips`, () => {
            const encoder = new Encoder(tier);
            const model = new TerminalModel(6, 3);

            const a = frameA(tier);
            model.write(encoder.encode(a));
            expect(model.grid()).toEqual(projectForTier(a, tier));

            const b = frameB(tier);
            model.write(encoder.encode(b));
            expect(model.grid()).toEqual(projectForTier(b, tier));

            // the terminal's own window resize — out-of-band from the byte stream, same as a real
            // terminal (`terminal-model.ts`'s `resize` docblock).
            model.resize(4, 2);
            const c = frameC(tier);
            model.write(encoder.encode(c));
            expect(model.grid()).toEqual(projectForTier(c, tier));

            const d = frameD(tier);
            model.write(encoder.encode(d));
            expect(model.grid()).toEqual(projectForTier(d, tier));

            // repeat of D with no changes — the diff-suppression proof: an unchanged frame must
            // cost zero bytes, not merely reconstruct correctly (a full repaint of identical
            // content also reconstructs correctly, which is exactly why this needs its own
            // assertion rather than riding on grid equality).
            const encoded = encoder.encode(d);
            expect(encoded.length).toBe(0);
            model.write(encoded);
            expect(model.grid()).toEqual(projectForTier(d, tier));
        });
    }

    // N1: `plain` is a tier too — the spec names four, and only three were ever exercised here.
    // Unlike the cursor-addressed tiers, `plain` is stateless by design (`encoder.ts`'s
    // `renderPlain`): every call is a full text dump with no cursor addressing and no diffing, so
    // each frame is checked against its own fresh `TerminalModel` — the faithful model of how a
    // real non-tty consumer reads it (one complete snapshot at a time), not an artifact of the
    // test being unable to track cursor-addressed state that `plain` never emits in the first
    // place.
    test("plain: every frame decodes independently, in full, with no cursor state carried across frames", () => {
        const encoder = new Encoder("plain");
        for (const grid of [frameA("plain"), frameB("plain"), frameC("plain"), frameD("plain")]) {
            const model = new TerminalModel(grid.width, grid.height);
            model.write(encoder.encode(grid));
            expect(model.grid()).toEqual(projectForTier(grid, "plain"));
        }
    });
});

describe("Encoder.invalidate — the out-of-band-resize seam (B4)", () => {
    test("forces a full repaint on the next encode even though the grid's own dimensions did not change", () => {
        const encoder = new Encoder("truecolor");
        const model = new TerminalModel(6, 3);

        const a = frameA("truecolor");
        model.write(encoder.encode(a));
        expect(model.grid()).toEqual(projectForTier(a, "truecolor"));

        // simulate the terminal's own window resize reflowing the physical screen — same width
        // and height from the grid producer's point of view, so `diffRuns` alone would still read
        // this as an ordinary same-size diffable frame and never know to repaint. A real terminal
        // really did clear/reflow though, which this models by wiping the TerminalModel directly
        // — the "out-of-band, independent of the byte stream" shape, just without a dimension
        // change this time (contrast `TerminalModel.resize`, used above for the dimension-change
        // case `diffRuns` already handles on its own).
        model.write(CLEAR_SCREEN + CURSOR_HOME);

        encoder.invalidate();
        const b = frameB("truecolor");
        model.write(encoder.encode(b));
        expect(model.grid()).toEqual(projectForTier(b, "truecolor"));
    });

    test("omitting invalidate() after the same out-of-band reflow fails to reconstruct — proving the seam is load-bearing, not decorative", () => {
        const encoder = new Encoder("truecolor");
        const model = new TerminalModel(6, 3);
        model.write(encoder.encode(frameA("truecolor")));
        model.write(CLEAR_SCREEN + CURSOR_HOME); // the same out-of-band reflow, no invalidate() this time

        const b = frameB("truecolor");
        model.write(encoder.encode(b)); // the encoder still thinks _prev is frame A — diffs sparsely
        expect(model.grid()).not.toEqual(projectForTier(b, "truecolor"));
    });
});

// N7 — B2's actual failure scenario: a default-fg cell following a colored run. Every color-tier
// fixture above builds cells with non-null fg, so `39` (reset fg to default) is asserted only as
// a string in sgr.test.ts and never round-tripped end-to-end. A mixed grid with a colored run
// immediately followed by a default-fg cell exercises the bleed guard the way a real diff would.
describe("round trip — a default-fg cell adjacent to a colored run round-trips the bleed guard (N7)", () => {
    for (const tier of ["ansi256", "truecolor"] as const) {
        test(`${tier}: a colored run followed by a default-fg cell does not bleed the prior color`, () => {
            const encoder = new Encoder(tier);
            const model = new TerminalModel(3, 1);
            const colored: Cell = { glyph: "R", fg: COLOR_A, bg: null };
            const defaultFg: Cell = { glyph: " ", fg: null, bg: null };
            const grid: Grid = { width: 3, height: 1, cells: [[colored, colored, defaultFg]] };

            model.write(encoder.encode(grid));
            expect(model.grid()).toEqual(projectForTier(grid, tier));

            // and diffed: change only the default-fg cell in a later frame, adjacent to a
            // still-colored run the encoder does not re-touch.
            const changed: Cell = { glyph: "!", fg: null, bg: null };
            const grid2: Grid = { width: 3, height: 1, cells: [[colored, colored, changed]] };
            model.write(encoder.encode(grid2));
            expect(model.grid()).toEqual(projectForTier(grid2, tier));
        });
    }
});

// N8 — `Encoder` must not alias the caller's `Grid` by reference. A producer that reuses one grid
// buffer per frame (mutating the same outer array in place rather than allocating fresh each
// call) is the natural shape for a GPU readback loop; if `_prev` aliased that object, the next
// diff would compare the grid to itself and emit zero bytes forever.
describe("Encoder does not alias the caller's grid (N8)", () => {
    test("mutating the caller's row array in place after encode() does not corrupt the next diff", () => {
        const encoder = new Encoder("truecolor");
        const model = new TerminalModel(3, 1);
        const cellA: Cell = { glyph: "A", fg: COLOR_A, bg: null };
        const cellB: Cell = { glyph: "B", fg: COLOR_B, bg: null };
        const rows: Cell[][] = [[cellA, cellA, cellA]];
        const grid: Grid = { width: 3, height: 1, cells: rows };

        model.write(encoder.encode(grid));
        expect(model.grid()).toEqual(projectForTier(grid, "truecolor"));

        // the producer reuses the same `Grid` object and outer `cells` array, replacing a row in
        // place rather than allocating a fresh `Grid` — if `Encoder` stored `grid` by reference,
        // this mutation would also mutate `_prev`, so the next diff would compare the grid to
        // itself and never see a change.
        rows[0] = [cellB, cellB, cellB];
        model.write(encoder.encode(grid));
        expect(model.grid()).toEqual(projectForTier(grid, "truecolor"));
    });
});

// N3 — the two-sided proof itself, wired structurally rather than demonstrated by hand. Each
// mutant below is a deliberately-broken miniature of `Encoder`, built from the same low-level
// primitives (`diffRuns`, `encodeRun`, `cursorTo`) so the only difference from the real encoder is
// the one property under test. Running the same style of assertion against each mutant proves the
// round-trip test's two assertions actually discriminate.
describe("the two-sided proof itself — the assertions above actually discriminate", () => {
    /** Mutant: diff suppression disabled. Every frame is an unconditional full repaint, never a
     * diff against a remembered previous grid — still content-correct (a full repaint of
     * identical content reconstructs fine), which is exactly why only a byte-count assertion,
     * not a grid-equality one, catches it. */
    class NoDiffSuppressionEncoder {
        constructor(private readonly _tier: Tier) {}
        encode(grid: Grid): string {
            let out = CLEAR_SCREEN + CURSOR_HOME;
            for (let y = 0; y < grid.height; y++) {
                out += cursorTo(y, 0);
                out += encodeRun(grid.cells[y], this._tier);
            }
            return out + SGR_RESET;
        }
    }

    /** Mutant: cursor addressing removed for diffed runs. The initial full paint is unaffected
     * (it still calls `cursorTo` per row, matching the real encoder), but a diffed frame's
     * changed runs are written with no repositioning at all — the one line the real `Encoder`
     * has that this mutant doesn't. */
    class NoCursorAddressingEncoder {
        private _prev: Grid | null = null;
        constructor(private readonly _tier: Tier) {}
        encode(grid: Grid): string {
            const runsOrResize = diffRuns(this._prev, grid);
            let out = "";
            if (runsOrResize === "resize") {
                out += CLEAR_SCREEN + CURSOR_HOME;
                for (let y = 0; y < grid.height; y++) {
                    out += cursorTo(y, 0);
                    out += encodeRun(grid.cells[y], this._tier);
                }
            } else {
                for (const run of runsOrResize) out += encodeRun(run.cells, this._tier);
            }
            this._prev = grid;
            return out + SGR_RESET;
        }
    }

    test("disabling diff suppression still reconstructs correctly but never costs zero bytes on a repeated frame", () => {
        const mutant = new NoDiffSuppressionEncoder("truecolor");
        const d = frameD("truecolor");
        mutant.encode(d);
        const repeat = mutant.encode(d);

        const model = new TerminalModel(4, 2);
        model.write(repeat);
        expect(model.grid()).toEqual(projectForTier(d, "truecolor")); // content-correct...
        expect(repeat.length).toBeGreaterThan(0); // ...but not zero bytes, which the real assertion requires
    });

    test("removing cursor addressing for diffed runs loses content — the cursor is never repositioned off where the initial full paint left it", () => {
        const mutant = new NoCursorAddressingEncoder("truecolor");
        const model = new TerminalModel(6, 3);

        model.write(mutant.encode(frameA("truecolor")));
        expect(model.grid()).toEqual(projectForTier(frameA("truecolor"), "truecolor")); // the full paint alone is unaffected

        model.write(mutant.encode(frameB("truecolor")));
        // frameB's diffed runs were never repositioned, and the full paint above left the cursor
        // exactly one column past the last cell (matching the real encoder's own bookkeeping) —
        // every byte this frame writes lands out of bounds and is silently dropped, so the model
        // still reads back frame A's content, not frame B's.
        expect(model.grid()).not.toEqual(projectForTier(frameB("truecolor"), "truecolor"));
        expect(model.grid()).toEqual(projectForTier(frameA("truecolor"), "truecolor"));
    });
});
