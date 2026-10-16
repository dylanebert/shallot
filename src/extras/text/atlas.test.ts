import { expect, test } from "bun:test";
import type { World } from "../../engine";
import { ensureString, type GlyphAtlas, layoutText } from "./atlas";
import { DEFAULT_FONT, loadFont } from "./font";

const metrics = {
    glyphWidth: 1,
    glyphHeight: 1,
    glyphTop: 1,
    glyphLeft: 0,
    advance: 1,
    u0: 0,
    v0: 0,
    u1: 1,
    v1: 1,
};
const layoutAtlas = {
    font: { unitsPerEm: 10, lineHeight: 15, kerning: () => 0 },
    glyphs: new Map(
        [..."abcdef "].map((char) => [
            char,
            char === " " ? { ...metrics, glyphWidth: 0 } : metrics,
        ]),
    ),
} as unknown as GlyphAtlas;

test("words wrap at spaces into lines within the width", () => {
    const result = layoutText("ab cd ef", layoutAtlas, 1, 5);
    expect(result.width).toBe(5);
    expect(result.glyphs.map((g) => [g.x, g.y])).toEqual([
        [0, 1.5],
        [1, 1.5],
        [3, 1.5],
        [4, 1.5],
        [0, 0],
        [1, 0],
    ]);
    expect(result.height).toBe(2.5);
});

test("an overlong word stands alone without splitting", () => {
    const result = layoutText("ab abcdef cd", layoutAtlas, 1, 3);
    expect(result.glyphs.map((g) => g.y)).toEqual([3, 3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0, 0]);
    expect(result.width).toBe(6);
});

test("newlines break lines, including empty lines", () => {
    const result = layoutText("ab\n\nc", layoutAtlas, 2);
    expect(result.glyphs.map((g) => g.y)).toEqual([6, 6, 0]);
    expect(result.height).toBe(8);
});

test("centred lines share a centre and right lines share a right edge", () => {
    for (const [justify, positions] of [
        [1, [1, 2, 1.5]],
        [2, [2, 3, 3]],
    ] as const) {
        const result = layoutText("ab\nc", layoutAtlas, 1, 4, justify);
        expect(result.glyphs.map((g) => g.x)).toEqual([...positions]);
        expect(result.width).toBe(4);
    }
});

test("unbounded default retains every glyph position, advance and height", () => {
    expect(layoutText("ab cd", layoutAtlas, 2)).toEqual({
        glyphs: [0, 2, 6, 8].map((x) => ({
            x,
            y: 0,
            width: 2,
            height: 2,
            u0: 0,
            v0: 0,
            u1: 1,
            v1: 1,
        })),
        width: 10,
        height: 2,
    });
});

test("blank glyphs advance the layout cursor without emitting quads or taking atlas space", async () => {
    const font = await loadFont(DEFAULT_FONT);
    const atlas = {
        width: 2048,
        height: 2048,
        cursorX: 0,
        cursorY: 0,
        rowHeight: 0,
        glyphs: new Map(),
        font,
        sdfGenerator: { begin() {}, add() {}, flush() {} },
    } as unknown as GlyphAtlas;
    ensureString({} as World, atlas, "ab");
    const cursor = atlas.cursorX;
    const scale = 0.5;
    const joined = layoutText("ab", atlas, scale);
    for (const blank of [" ", "\u00a0"]) {
        expect(font.advance(blank)).toBeGreaterThan(0);
        expect(font.glyphPath(blank)).toBeNull();
        ensureString({} as World, atlas, `a${blank}b`);
        const separated = layoutText(`a${blank}b`, atlas, scale);
        expect(separated.glyphs).toHaveLength(2);
        const kerning =
            (font.kerning("a", blank) + font.kerning(blank, "b") - font.kerning("a", "b")) /
            font.unitsPerEm;
        expect(separated.glyphs[1]!.x - joined.glyphs[1]!.x).toBeCloseTo(
            (font.advance(blank) / font.unitsPerEm + kerning) * scale,
            10,
        );
        expect(atlas.cursorX).toBe(cursor);
    }
});
