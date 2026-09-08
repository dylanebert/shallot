import { describe, expect, test } from "bun:test";
import {
    DARK,
    END_TICK,
    fromBlocks,
    HIT_TICK,
    lockup,
    MARK,
    splashFrame,
    toAnsi,
    toCells,
    toSvg,
    toText,
    word,
} from "./mark";

describe("mark", () => {
    test("half blocks round-trip through the bitmap", () => {
        for (const rows of Object.values(MARK)) {
            expect(toText(toCells(fromBlocks(rows)))).toBe(rows.join("\n"));
        }
    });

    test("the canonical mark is 12 by 14 with a one-row tip", () => {
        const grid = fromBlocks(MARK.m);
        expect(grid.length).toBe(14);
        expect(grid[0]?.length).toBe(12);
        expect(grid[0]?.filter(Boolean).length).toBe(0);
        expect(grid[1]?.filter(Boolean).length).toBe(2);
    });

    test("the wordmark is 36 by 7 in ink", () => {
        const grid = word();
        expect(grid.length).toBe(7);
        expect(grid[0]?.length).toBe(36);
        expect(new Set(grid.flat().filter(Boolean))).toEqual(new Set(["ink"]));
    });

    test("svg emits one rect per pixel at the given scale", () => {
        const grid = fromBlocks(MARK.s);
        const pixels = grid.flat().filter(Boolean).length;
        const svg = toSvg(grid, DARK, 3);
        expect(svg.match(/<rect /g)?.length).toBe(pixels);
        expect(svg).toContain('width="30" height="36"');
    });

    test("ansi resets at every row end", () => {
        for (const line of toAnsi(toCells(lockup())).split("\n")) {
            if (line.includes("\x1b[38")) expect(line.endsWith("\x1b[0m")).toBe(true);
        }
    });
});

describe("splash", () => {
    test("ends on the lockup with no cursor", () => {
        expect(splashFrame(END_TICK + 1)).toEqual(lockup());
        expect(splashFrame(END_TICK + 30)).toEqual(lockup());
    });

    test("the mark is complete at the hit and the name is not", () => {
        const at = splashFrame(HIT_TICK);
        const mark = fromBlocks(MARK.m);
        mark.forEach((row, y) => {
            row.forEach((tone, x) => {
                if (tone) expect(at[y]?.[x]).toBeTruthy();
            });
        });
        expect(at.flat().filter((t) => t === "ink").length).toBe(0);
    });

    test("pixels only ever switch on, never off, before the hit", () => {
        let lit = 0;
        for (let t = 0; t <= HIT_TICK; t++) {
            const now = splashFrame(t)
                .flat()
                .filter((v) => v === "gold" || v === "dim").length;
            expect(now).toBeGreaterThanOrEqual(lit);
            lit = now;
        }
    });

    test("no frame carries a colour outside the three tones", () => {
        for (let t = 0; t <= END_TICK + 1; t++) {
            for (const v of splashFrame(t).flat())
                expect([null, "gold", "dim", "ink"]).toContain(v);
        }
    });
});
