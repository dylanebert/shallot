import { expect, test } from "bun:test";
import type { World } from "../../engine";
import { ensureString, type GlyphAtlas, layoutText } from "./atlas";
import { DEFAULT_FONT, loadFont } from "./font";

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
