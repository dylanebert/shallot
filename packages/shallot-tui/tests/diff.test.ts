import { describe, expect, test } from "bun:test";
import { diffRuns } from "../src/diff";
import type { Cell } from "../src/types";
import { makeGrid } from "../src/types";

const blank = (): Cell => ({ glyph: " ", fg: null, bg: null });

describe("diffRuns", () => {
    test("a null previous grid always reads as a resize (first-frame full paint)", () => {
        const grid = makeGrid(3, 2, blank);
        expect(diffRuns(null, grid)).toBe("resize");
    });

    test("a dimension change reads as a resize even with identical content otherwise", () => {
        const a = makeGrid(3, 2, blank);
        const b = makeGrid(4, 2, blank);
        expect(diffRuns(a, b)).toBe("resize");
    });

    test("two identical same-size grids diff to zero runs", () => {
        const a = makeGrid(4, 3, blank);
        const b = makeGrid(4, 3, blank);
        expect(diffRuns(a, b)).toEqual([]);
    });

    test("a single changed cell produces one run of length 1 at its position", () => {
        const a = makeGrid(4, 3, blank);
        const b = makeGrid(4, 3, (x, y) =>
            x === 2 && y === 1 ? { glyph: "!", fg: null, bg: null } : blank(),
        );
        const runs = diffRuns(a, b);
        expect(runs).not.toBe("resize");
        expect(runs).toEqual([{ row: 1, col: 2, cells: [{ glyph: "!", fg: null, bg: null }] }]);
    });

    test("adjacent changed cells on one row coalesce into a single run", () => {
        const a = makeGrid(5, 1, blank);
        const b = makeGrid(5, 1, (x) =>
            x >= 1 && x <= 3 ? { glyph: "x", fg: null, bg: null } : blank(),
        );
        const runs = diffRuns(a, b);
        expect(runs).not.toBe("resize");
        expect(runs).toHaveLength(1);
        if (runs === "resize") throw new Error("unreachable");
        expect(runs[0]).toEqual({
            row: 0,
            col: 1,
            cells: [
                { glyph: "x", fg: null, bg: null },
                { glyph: "x", fg: null, bg: null },
                { glyph: "x", fg: null, bg: null },
            ],
        });
    });

    test("non-adjacent changed cells on one row produce separate runs", () => {
        const a = makeGrid(5, 1, blank);
        const b = makeGrid(5, 1, (x) =>
            x === 0 || x === 4 ? { glyph: "x", fg: null, bg: null } : blank(),
        );
        const runs = diffRuns(a, b);
        expect(runs).not.toBe("resize");
        if (runs === "resize") throw new Error("unreachable");
        expect(runs).toHaveLength(2);
        expect(runs[0].col).toBe(0);
        expect(runs[1].col).toBe(4);
    });

    test("a fg-only or bg-only change (same glyph) is still detected as changed", () => {
        const a = makeGrid(1, 1, () => ({ glyph: "x", fg: null, bg: null }));
        const b = makeGrid(1, 1, () => ({ glyph: "x", fg: { r: 1, g: 2, b: 3 }, bg: null }));
        const runs = diffRuns(a, b);
        expect(runs).not.toBe("resize");
        if (runs === "resize") throw new Error("unreachable");
        expect(runs).toHaveLength(1);
    });
});
