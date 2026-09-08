import { describe, expect, test } from "bun:test";
import { computeGlyphMetrics, disposeAtlases, type GlyphAtlas } from "./atlas";
import type { Font } from "./font";

// computeGlyphMetrics is a shelf packer (place each glyph on the current row, wrap, or refuse once the
// atlas is full) plus UV math (font-unit box → normalized metrics + atlas-space UV rect). `Font` is a
// duck-typed interface, so a hand-built fake reaches every branch with no real font file or GPU device —
// `createGlyphAtlas`/`ensureString`, the rest of this module, stay untested here: both call into a real
// `GPUDevice`/`SDFGenerator`, outside this function's pure shelf-packing + UV surface.

function fakeFont(overrides: Partial<Font> = {}): Font {
    return {
        unitsPerEm: 1000,
        ascender: 800,
        descender: -200,
        lineGap: 0,
        glyphPath: () => "M0,0L1,1Z",
        glyphBounds: () => [10, -20, 500, 700],
        advance: () => 600,
        kerning: () => 0,
        ...overrides,
    };
}

// texture/textureView/sdfGenerator are never read by computeGlyphMetrics (only ensureString/
// createGlyphAtlas touch them) — a duck-typed data stub, never a device, per the Locked decision's "every
// seam this unit needs already exists as a parameter" bar.
function fakeAtlas(font: Font, size: { width: number; height: number }): GlyphAtlas {
    return {
        texture: null as never,
        textureView: null as never,
        width: size.width,
        height: size.height,
        glyphs: new Map(),
        rowHeight: 0,
        cursorX: 0,
        cursorY: 0,
        font,
        sdfGenerator: null as never,
    };
}

describe("computeGlyphMetrics", () => {
    test("returns null and touches no cursor state when the font has no path or bounds for the char", () => {
        const atlas = fakeAtlas(fakeFont({ glyphPath: () => null }), { width: 1000, height: 1000 });
        expect(computeGlyphMetrics(atlas, " ")).toBeNull();
        expect(atlas.glyphs.size).toBe(0);
        expect(atlas.cursorX).toBe(0);
        expect(atlas.cursorY).toBe(0);
    });

    test("packs the font-unit box into normalized metrics and an atlas-space UV rect", () => {
        // unitsPerEm 1000, bounds [10, -20, 500, 700] -> padding 100 -> paddedBounds [-90, -120, 600, 800]
        const atlas = fakeAtlas(fakeFont(), { width: 1000, height: 1000 });
        const entry = computeGlyphMetrics(atlas, "A");
        const metrics = atlas.glyphs.get("A");

        expect(entry).toEqual({
            path: "M0,0L1,1Z",
            paddedBounds: [-90, -120, 600, 800],
            atlasX: 0,
            atlasY: 0,
        });
        expect(metrics).toBeDefined();
        expect(metrics!.glyphWidth).toBeCloseTo(690 / 1000, 10); // (600 - -90) / unitsPerEm
        expect(metrics!.glyphHeight).toBeCloseTo(920 / 1000, 10); // (800 - -120) / unitsPerEm
        expect(metrics!.glyphTop).toBeCloseTo(800 / 1000, 10);
        expect(metrics!.glyphLeft).toBeCloseTo(-90 / 1000, 10);
        expect(metrics!.advance).toBeCloseTo(600 / 1000, 10);
        // the atlas slot is square (width === height); the UV rect is the slot placed at (cursorX, cursorY)
        // before this call, both zero for the first glyph on a fresh atlas
        expect(metrics!.width).toBe(metrics!.height);
        expect(metrics!.u0).toBe(0);
        expect(metrics!.v0).toBe(0);
        expect(metrics!.u1).toBeCloseTo(metrics!.width / 1000, 10);
        expect(metrics!.v1).toBeCloseTo(metrics!.height / 1000, 10);
    });

    test("advances the cursor by one slot on the same row for a second glyph", () => {
        const atlas = fakeAtlas(fakeFont(), { width: 1000, height: 1000 });
        computeGlyphMetrics(atlas, "A");
        const slot = atlas.glyphs.get("A")!.width;
        const second = computeGlyphMetrics(atlas, "B");
        expect(second!.atlasX).toBe(slot);
        expect(second!.atlasY).toBe(0);
        expect(atlas.cursorX).toBe(slot * 2);
    });

    test("wraps to a new row when the next slot would overflow the atlas width", () => {
        const atlas = fakeAtlas(fakeFont(), { width: 1000, height: 1000 });
        const first = computeGlyphMetrics(atlas, "A")!;
        const slot = atlas.glyphs.get("A")!.width;
        // force the next slot past the width without depending on how many glyphs that actually takes
        atlas.cursorX = atlas.width - slot + 1;
        const wrapped = computeGlyphMetrics(atlas, "B")!;
        expect(wrapped.atlasX).toBe(0);
        expect(wrapped.atlasY).toBe(first.atlasY + slot); // rowHeight carried from the first glyph's row
        expect(atlas.cursorX).toBe(slot);
        expect(atlas.rowHeight).toBe(slot);
    });

    test("throws once the atlas has no room even at column 0", () => {
        const atlas = fakeAtlas(fakeFont(), { width: 1000, height: 50 });
        expect(() => computeGlyphMetrics(atlas, "A")).toThrow("Glyph atlas full");
    });
});

// TextPlugin.dispose() destroys atlas.texture but, before this fix, never called
// atlas.sdfGenerator.destroy() — leaking the generator's own `_intermediateTexture` on every rebuild.
// disposeAtlases is the pure iteration TextPlugin.dispose drives, so this test reaches the real teardown
// path with duck-typed destroy spies instead of a live GPUDevice.
describe("disposeAtlases", () => {
    function fakeDisposable(font: Font): GlyphAtlas & {
        textureDestroyed: boolean;
        generatorDestroyed: boolean;
    } {
        const state = { textureDestroyed: false, generatorDestroyed: false };
        return {
            texture: { destroy: () => (state.textureDestroyed = true) } as unknown as GPUTexture,
            textureView: null as never,
            width: 0,
            height: 0,
            glyphs: new Map(),
            rowHeight: 0,
            cursorX: 0,
            cursorY: 0,
            font,
            sdfGenerator: {
                destroy: () => (state.generatorDestroyed = true),
            } as unknown as GlyphAtlas["sdfGenerator"],
            get textureDestroyed() {
                return state.textureDestroyed;
            },
            get generatorDestroyed() {
                return state.generatorDestroyed;
            },
        };
    }

    test("destroys both the texture and the SDF generator for every atlas", () => {
        const a = fakeDisposable(fakeFont());
        const b = fakeDisposable(fakeFont());
        disposeAtlases([a, b]);
        expect(a.textureDestroyed).toBe(true);
        expect(a.generatorDestroyed).toBe(true);
        expect(b.textureDestroyed).toBe(true);
        expect(b.generatorDestroyed).toBe(true);
    });

    test("skips a hole left by a font that failed to load", () => {
        expect(() => disposeAtlases([undefined])).not.toThrow();
    });
});
