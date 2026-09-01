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
});

describe("sgrPrefix", () => {
    test("plain and glyph tiers never emit an escape, regardless of color", () => {
        expect(sgrPrefix(RED, "plain")).toBe("");
        expect(sgrPrefix(RED, "glyph")).toBe("");
    });

    test("a colorless cell in a color tier emits a bare reset", () => {
        expect(sgrPrefix(PLAIN, "ansi256")).toBe("\x1b[0m");
        expect(sgrPrefix(PLAIN, "truecolor")).toBe("\x1b[0m");
    });

    test("truecolor emits exact 38;2;r;g;b / 48;2;r;g;b params", () => {
        const styled: Cell = { glyph: "x", fg: { r: 10, g: 20, b: 30 }, bg: { r: 1, g: 2, b: 3 } };
        expect(sgrPrefix(styled, "truecolor")).toBe("\x1b[38;2;10;20;30;48;2;1;2;3m");
    });

    test("ansi256 emits 38;5;N / 48;5;N against the cube index", () => {
        expect(sgrPrefix(RED, "ansi256")).toBe(`\x1b[38;5;${ansi256FromRgb(255, 0, 0)}m`);
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
