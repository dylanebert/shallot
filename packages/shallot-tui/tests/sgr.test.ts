import { describe, expect, test } from "bun:test";
import { ansi256FromRgb, cube6, encodeRun, sameStyle, sgrPrefix } from "../src/sgr";
import type { Cell } from "../src/types";

const RED: Cell = { glyph: "R", fg: { r: 255, g: 0, b: 0 }, bg: null };
const RED2: Cell = { glyph: "r", fg: { r: 255, g: 0, b: 0 }, bg: null };
const BLUE: Cell = { glyph: "B", fg: { r: 0, g: 0, b: 255 }, bg: null };
const PLAIN: Cell = { glyph: "x", fg: null, bg: null };

describe("cube6 / ansi256FromRgb", () => {
    test("quantizes the 6-step cube exactly at its own step values", () => {
        expect(cube6(0)).toBe(0);
        expect(cube6(51)).toBe(1);
        expect(cube6(102)).toBe(2);
        expect(cube6(153)).toBe(3);
        expect(cube6(204)).toBe(4);
        expect(cube6(255)).toBe(5);
    });

    test("maps pure red/green/blue to the expected cube indices (base 16, r*36 + g*6 + b)", () => {
        expect(ansi256FromRgb(255, 0, 0)).toBe(16 + 36 * 5);
        expect(ansi256FromRgb(0, 255, 0)).toBe(16 + 6 * 5);
        expect(ansi256FromRgb(0, 0, 255)).toBe(16 + 5);
        expect(ansi256FromRgb(0, 0, 0)).toBe(16);
    });

    test("hand-computed (rgb -> index) pairs beyond the four pure corners — the general case, not just axis-aligned primaries and black", () => {
        // 128/255*5 = 2.5098 -> round 3; 64/255*5 = 1.2549 -> round 1; 32/255*5 = 0.6275 -> round 1
        expect(ansi256FromRgb(128, 64, 32)).toBe(16 + 36 * 3 + 6 * 1 + 1); // 131
        // 200/255*5 = 3.9216 -> round 4; 150/255*5 = 2.9412 -> round 3; 100/255*5 = 1.9608 -> round 2
        expect(ansi256FromRgb(200, 150, 100)).toBe(16 + 36 * 4 + 6 * 3 + 2); // 180
        // pure white — every channel quantizes to the top step, the corner opposite pure black
        expect(ansi256FromRgb(255, 255, 255)).toBe(16 + 36 * 5 + 6 * 5 + 5); // 231
        // the rounding boundary itself: 25/255*5 = 0.4902 rounds down, 26/255*5 = 0.5098 rounds up
        expect(cube6(25)).toBe(0);
        expect(cube6(26)).toBe(1);
    });
});

describe("sgrPrefix", () => {
    test("plain and glyph tiers never emit an escape, regardless of color", () => {
        expect(sgrPrefix(RED, "plain")).toBe("");
        expect(sgrPrefix(RED, "glyph")).toBe("");
    });

    test("a colorless cell in a color tier explicitly resets both channels to default (39;49), never a bare reset", () => {
        expect(sgrPrefix(PLAIN, "ansi256")).toBe("\x1b[39;49m");
        expect(sgrPrefix(PLAIN, "truecolor")).toBe("\x1b[39;49m");
    });

    test("truecolor emits exact 38;2;r;g;b / 48;2;r;g;b params", () => {
        const styled: Cell = { glyph: "x", fg: { r: 10, g: 20, b: 30 }, bg: { r: 1, g: 2, b: 3 } };
        expect(sgrPrefix(styled, "truecolor")).toBe("\x1b[38;2;10;20;30;48;2;1;2;3m");
    });

    test("ansi256 emits 38;5;N / 48;5;N against the cube index", () => {
        expect(sgrPrefix(RED, "ansi256")).toBe(`\x1b[38;5;${ansi256FromRgb(255, 0, 0)};49m`);
    });

    test("a set fg with a null bg still explicitly resets the bg to default — B2: an unset channel must never inherit a prior run's color", () => {
        expect(sgrPrefix(RED, "truecolor")).toBe(`\x1b[38;2;255;0;0;49m`);
        const bgOnly: Cell = { glyph: "x", fg: null, bg: { r: 9, g: 9, b: 9 } };
        expect(sgrPrefix(bgOnly, "truecolor")).toBe(`\x1b[39;48;2;9;9;9m`);
    });
});

describe("sameStyle", () => {
    test("two cells with equal fg/bg are the same style regardless of glyph", () => {
        expect(sameStyle(RED, RED2)).toBe(true);
    });

    test("differing fg makes two cells different styles", () => {
        expect(sameStyle(RED, BLUE)).toBe(false);
    });
});

describe("encodeRun — SGR run coalescing", () => {
    test("plain/glyph tiers concatenate glyphs with no SGR at all", () => {
        expect(encodeRun([RED, BLUE], "plain")).toBe("RB");
        expect(encodeRun([RED, BLUE], "glyph")).toBe("RB");
    });

    test("a run of same-style cells emits exactly one SGR prefix for the whole run", () => {
        const encoded = encodeRun([RED, RED2], "truecolor");
        const prefix = sgrPrefix(RED, "truecolor");
        expect(encoded).toBe(`${prefix}Rr`);
        // exactly one escape introducer for the whole run — the coalescing property itself.
        expect(encoded.split("\x1b").length - 1).toBe(1);
    });

    test("a style change mid-run emits a second SGR prefix at the boundary", () => {
        const encoded = encodeRun([RED, BLUE], "truecolor");
        expect(encoded).toBe(`${sgrPrefix(RED, "truecolor")}R${sgrPrefix(BLUE, "truecolor")}B`);
        expect(encoded.split("\x1b").length - 1).toBe(2);
    });
});
