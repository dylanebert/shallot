// Criterion 1 — "The encoder round-trips at every tier" (shallot-tui spec, Validation #1).
//
// Encodes a known multi-frame sequence (a full first paint, a sparse diffed update touching
// non-adjacent cells across different rows, a resize, another sparse diffed update, then a
// repeat of the last frame with no changes at all) and parses the emitted bytes back through
// `TerminalModel`, a minimal terminal reader independent of the encoder's own logic. Every
// intermediate frame's reconstructed grid must equal the source grid (projected onto what the
// tier can carry — `terminal-model.ts`'s `projectForTier`), and the repeated final frame must
// encode to zero bytes.
//
// Those two assertions are deliberately the two-sided proof the spec requires:
//   - the zero-bytes-on-repeat assertion reds if diff suppression is disabled (a full repaint
//     every frame is still content-correct, so only a byte-budget assertion catches it);
//   - the per-frame equality assertion reds if cursor addressing is removed (glyphs land wherever
//     the cursor was last left, not where the diff says they belong, once two changed runs are
//     non-adjacent across rows).
// Both reds are demonstrated by hand (temporarily reverting `src/encoder.ts`, run, observe,
// revert) rather than wired as a permanent flag — the property under test is the shipped
// encoder's actual behavior, not a parallel "what if" code path.

import { describe, expect, test } from "bun:test";
import type { Tier } from "../src/color-support";
import { Encoder } from "../src/encoder";
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
});
