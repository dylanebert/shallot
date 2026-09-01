import { describe, expect, test } from "bun:test";
import { ansi256FromRgb, cube6, encodeRun, sameStyle, sgrPrefix } from "../src/sgr";
import type { Cell } from "../src/types";

const RED: Cell = { glyph: "R", fg: { r: 255, g: 0, b: 0 }, bg: null };
const RED2: Cell = { glyph: "r", fg: { r: 255, g: 0, b: 0 }, bg: null };
const BLUE: Cell = { glyph: "B", fg: { r: 0, g: 0, b: 255 }, bg: null };
const PLAIN: Cell = { glyph: "x", fg: null, bg: null };

describe("cube6 / ansi256FromRgb", () => {
    test("quantizes the 6-step cube exactly at its own (real, non-evenly-spaced) step values", () => {
        expect(cube6(0)).toBe(0);
        expect(cube6(95)).toBe(1);
        expect(cube6(135)).toBe(2);
        expect(cube6(175)).toBe(3);
        expect(cube6(215)).toBe(4);
        expect(cube6(255)).toBe(5);
    });

    test("maps pure red/green/blue to the expected cube indices (base 16, r*36 + g*6 + b) — every axis-aligned primary and black beat the grayscale ramp on distance", () => {
        expect(ansi256FromRgb(255, 0, 0)).toBe(16 + 36 * 5);
        expect(ansi256FromRgb(0, 255, 0)).toBe(16 + 6 * 5);
        expect(ansi256FromRgb(0, 0, 255)).toBe(16 + 5);
        expect(ansi256FromRgb(0, 0, 0)).toBe(16);
    });

    test("hand-computed (rgb -> index) pairs against the real xterm palette — B2: not the evenly-spaced formula (0,51,102,153,204,255), the real cube levels (0,95,135,175,215,255), nearest cube corner vs. nearest grayscale-ramp value by real distance", () => {
        // (128,64,32): nearest cube corner is (135,95,0) [idx 2,1,0], squared dist 7^2+31^2+32^2 =
        // 2034; nearest grayscale value is avg=75 -> ramp step 7 (value 78), squared dist
        // 50^2+14^2+46^2 = 4812. Cube wins.
        expect(ansi256FromRgb(128, 64, 32)).toBe(16 + 36 * 2 + 6 * 1 + 0); // 94
        // (200,150,100): nearest cube corner is (215,135,95) [idx 4,2,1], squared dist
        // 15^2+15^2+5^2 = 475; nearest grayscale value is avg=150 -> ramp step 14 (value 148),
        // squared dist 52^2+2^2+48^2 = 5012. Cube wins.
        expect(ansi256FromRgb(200, 150, 100)).toBe(16 + 36 * 4 + 6 * 2 + 1); // 173
        // pure white — every channel quantizes to the top cube step, the corner opposite black
        expect(ansi256FromRgb(255, 255, 255)).toBe(16 + 36 * 5 + 6 * 5 + 5); // 231
        // the real rounding boundary is the midpoint between the cube's first two levels, 0 and
        // 95 — 47.5, not 25.5 (the evenly-spaced formula's wrong boundary)
        expect(cube6(47)).toBe(0);
        expect(cube6(48)).toBe(1);
    });

    test("a near-grey color quantizes to the grayscale ramp (232-255), not the color cube, when the ramp is the real nearest neighbour", () => {
        // (120, 118, 122): every channel far from any cube corner (0/95/135/...), but very close
        // to a grayscale ramp value — avg=120, nearest ramp step 11 (value 118), squared dist
        // 4+0+16=20; nearest cube corner (135,95,135) or similar is far worse.
        const index = ansi256FromRgb(120, 118, 122);
        expect(index).toBeGreaterThanOrEqual(232);
        expect(index).toBeLessThanOrEqual(255);
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
