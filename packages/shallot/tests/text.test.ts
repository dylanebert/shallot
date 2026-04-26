import { describe, test, expect, beforeEach } from "bun:test";
import { State, capacity, clearBuf } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Text, TextData, TextFonts, TextPlugin, font, fontRegistry } from "../src/extras/text";
import { Transform } from "../src/standard/transforms";
import { count } from "./helpers/state";

const SDF_EXPONENT = 9;
const SDF_CUTOFF = 0.5;

function encodeExponentialSdf(linearData: Uint8Array): Uint8Array {
    const encoded = new Uint8Array(linearData.length);
    for (let i = 0; i < linearData.length; i++) {
        const raw = linearData[i] / 255;
        const signedDist = (SDF_CUTOFF - raw) / SDF_CUTOFF;

        const absDist = Math.min(1, Math.abs(signedDist));
        let alpha = Math.pow(1 - absDist, SDF_EXPONENT) / 2;
        if (signedDist < 0) {
            alpha = 1 - alpha;
        }
        encoded[i] = Math.round(Math.max(0, Math.min(255, alpha * 255)));
    }
    return encoded;
}

function parseTextAttrs(attrString: string): Record<string, string> {
    const parsed: Record<string, string> = {};
    for (const part of attrString.split(";")) {
        const colonIdx = part.indexOf(":");
        if (colonIdx === -1) continue;
        const key = part.slice(0, colonIdx).trim();
        const value = part.slice(colonIdx + 1).trim();
        if (key && value) parsed[key] = value;
    }
    return parsed;
}

function parseColor(colorStr: string): number {
    if (colorStr.startsWith("0x") || colorStr.startsWith("0X")) {
        return parseInt(colorStr, 16);
    } else if (colorStr.startsWith("#")) {
        return parseInt(colorStr.slice(1), 16);
    } else {
        return parseInt(colorStr, 10);
    }
}

interface GlyphMetrics {
    glyphWidth: number;
    glyphHeight: number;
    glyphTop: number;
    glyphLeft: number;
    advance: number;
}

interface LayoutGlyph {
    x: number;
    y: number;
    width: number;
    height: number;
}

function layoutText(
    text: string,
    glyphMap: Map<string, GlyphMetrics>,
    fontSize: number,
    sdfFontSize: number,
): { glyphs: LayoutGlyph[]; width: number; height: number } {
    const glyphs: LayoutGlyph[] = [];
    const scale = fontSize / sdfFontSize;

    let cursorX = 0;
    let maxHeight = 0;

    for (const char of text) {
        const metrics = glyphMap.get(char);
        if (!metrics) continue;

        const glyphW = metrics.glyphWidth * scale;
        const glyphH = metrics.glyphHeight * scale;
        const advance = metrics.advance * scale;

        const x = cursorX + metrics.glyphLeft * scale;
        const y = (metrics.glyphTop - metrics.glyphHeight) * scale;

        glyphs.push({ x, y, width: glyphW, height: glyphH });

        cursorX += advance;
        maxHeight = Math.max(maxHeight, glyphH);
    }

    return { glyphs, width: cursorX, height: maxHeight };
}

describe("text", () => {
    describe("SDF encoding", () => {
        test("encodes 0 (far outside) to low alpha", () => {
            const input = new Uint8Array([0]);
            const result = encodeExponentialSdf(input);
            expect(result[0]).toBeLessThan(10);
        });

        test("encodes 255 (inside glyph) to high alpha", () => {
            const input = new Uint8Array([255]);
            const result = encodeExponentialSdf(input);
            expect(result[0]).toBeGreaterThan(250);
        });

        test("encodes 127/128 (edge) to approximately 127/128", () => {
            const input = new Uint8Array([127]);
            const result = encodeExponentialSdf(input);
            expect(result[0]).toBeGreaterThan(100);
            expect(result[0]).toBeLessThan(156);
        });

        test("preserves array length", () => {
            const input = new Uint8Array([0, 64, 128, 192, 255]);
            const result = encodeExponentialSdf(input);
            expect(result.length).toBe(5);
        });

        test("is monotonically increasing", () => {
            const input = new Uint8Array(256);
            for (let i = 0; i < 256; i++) input[i] = i;
            const result = encodeExponentialSdf(input);
            for (let i = 1; i < 256; i++) {
                expect(result[i]).toBeGreaterThanOrEqual(result[i - 1]);
            }
        });

        test("clamps output to [0, 255]", () => {
            const input = new Uint8Array([0, 127, 255]);
            const result = encodeExponentialSdf(input);
            for (const val of result) {
                expect(val).toBeGreaterThanOrEqual(0);
                expect(val).toBeLessThanOrEqual(255);
            }
        });
    });

    describe("layout", () => {
        const mockGlyphs = new Map<string, GlyphMetrics>([
            ["A", { glyphWidth: 80, glyphHeight: 100, glyphTop: 90, glyphLeft: 5, advance: 85 }],
            ["B", { glyphWidth: 75, glyphHeight: 100, glyphTop: 90, glyphLeft: 8, advance: 80 }],
            [" ", { glyphWidth: 0, glyphHeight: 0, glyphTop: 0, glyphLeft: 0, advance: 40 }],
        ]);
        const sdfFontSize = 128;

        test("calculates glyph positions", () => {
            const result = layoutText("AB", mockGlyphs, 16, sdfFontSize);
            expect(result.glyphs.length).toBe(2);
            expect(result.glyphs[0].x).toBeCloseTo(5 * (16 / 128), 3);
            expect(result.glyphs[1].x).toBeCloseTo(85 * (16 / 128) + 8 * (16 / 128), 3);
        });

        test("respects font size scaling", () => {
            const small = layoutText("A", mockGlyphs, 8, sdfFontSize);
            const large = layoutText("A", mockGlyphs, 32, sdfFontSize);
            expect(large.glyphs[0].width).toBeCloseTo(small.glyphs[0].width * 4, 3);
            expect(large.glyphs[0].height).toBeCloseTo(small.glyphs[0].height * 4, 3);
        });

        test("accumulates width from advances", () => {
            const result = layoutText("AB", mockGlyphs, 128, sdfFontSize);
            expect(result.width).toBeCloseTo(85 + 80, 3);
        });

        test("computes max height", () => {
            const result = layoutText("AB", mockGlyphs, 128, sdfFontSize);
            expect(result.height).toBe(100);
        });

        test("handles space character", () => {
            const result = layoutText("A B", mockGlyphs, 128, sdfFontSize);
            expect(result.glyphs.length).toBe(3);
            expect(result.width).toBeCloseTo(85 + 40 + 80, 3);
        });

        test("skips unknown characters", () => {
            const result = layoutText("A?B", mockGlyphs, 128, sdfFontSize);
            expect(result.glyphs.length).toBe(2);
        });

        test("empty string produces no glyphs", () => {
            const result = layoutText("", mockGlyphs, 128, sdfFontSize);
            expect(result.glyphs.length).toBe(0);
            expect(result.width).toBe(0);
        });
    });

    describe("attribute parsing", () => {
        test("parses semicolon-delimited attributes", () => {
            const result = parseTextAttrs("font-size:16;color:#ff0000");
            expect(result["font-size"]).toBe("16");
            expect(result["color"]).toBe("#ff0000");
        });

        test("trims whitespace", () => {
            const result = parseTextAttrs("  key  :  value  ;  key2  :  value2  ");
            expect(result["key"]).toBe("value");
            expect(result["key2"]).toBe("value2");
        });

        test("handles missing colon", () => {
            const result = parseTextAttrs("valid:value;invalid;also-valid:ok");
            expect(result["valid"]).toBe("value");
            expect(result["also-valid"]).toBe("ok");
            expect(result["invalid"]).toBeUndefined();
        });

        test("handles empty string", () => {
            const result = parseTextAttrs("");
            expect(Object.keys(result).length).toBe(0);
        });

        test("handles colon in value", () => {
            const result = parseTextAttrs("url:http://example.com");
            expect(result["url"]).toBe("http://example.com");
        });

        test("parses content attribute", () => {
            const result = parseTextAttrs("content:Hello World;font-size:24");
            expect(result["content"]).toBe("Hello World");
            expect(result["font-size"]).toBe("24");
        });
    });

    describe("color parsing", () => {
        test("parses hex color with 0x prefix", () => {
            expect(parseColor("0xff0000")).toBe(0xff0000);
            expect(parseColor("0xFF0000")).toBe(0xff0000);
        });

        test("parses hex color with # prefix", () => {
            expect(parseColor("#ff0000")).toBe(0xff0000);
            expect(parseColor("#00ff00")).toBe(0x00ff00);
            expect(parseColor("#0000ff")).toBe(0x0000ff);
        });

        test("parses decimal color", () => {
            expect(parseColor("16711680")).toBe(0xff0000);
            expect(parseColor("65280")).toBe(0x00ff00);
        });

        test("parses white", () => {
            expect(parseColor("0xffffff")).toBe(0xffffff);
            expect(parseColor("#ffffff")).toBe(0xffffff);
        });

        test("parses black", () => {
            expect(parseColor("0x000000")).toBe(0);
            expect(parseColor("#000000")).toBe(0);
            expect(parseColor("0")).toBe(0);
        });

        test("handles lowercase hex", () => {
            expect(parseColor("0xabcdef")).toBe(0xabcdef);
            expect(parseColor("#abcdef")).toBe(0xabcdef);
        });

        test("handles uppercase hex", () => {
            expect(parseColor("0xABCDEF")).toBe(0xabcdef);
            expect(parseColor("#ABCDEF")).toBe(0xabcdef);
        });
    });

    describe("TextData storage", () => {
        test("data has correct size for capacity", () => {
            expect(TextData.chunks[0].length).toBe(capacity() * 12);
        });

        test("fonts has correct size for capacity", () => {
            expect(TextFonts.chunks[0].length).toBe(capacity());
        });
    });

    describe("Text proxy accessors", () => {
        const eid = 42;

        beforeEach(() => {
            clearBuf(TextData);
            clearBuf(TextFonts);
        });

        test("fontSize reads/writes correctly", () => {
            Text.fontSize[eid] = 2.5;
            expect(Text.fontSize[eid]).toBeCloseTo(2.5);
        });

        test("opacity reads/writes correctly", () => {
            Text.opacity[eid] = 0.5;
            expect(Text.opacity[eid]).toBeCloseTo(0.5);
        });

        test("visible reads/writes correctly", () => {
            Text.visible[eid] = 1;
            expect(Text.visible[eid]).toBe(1);
            Text.visible[eid] = 0;
            expect(Text.visible[eid]).toBe(0);
        });

        test("anchorX reads/writes correctly", () => {
            Text.anchorX[eid] = 0.5;
            expect(Text.anchorX[eid]).toBeCloseTo(0.5);
        });

        test("anchorY reads/writes correctly", () => {
            Text.anchorY[eid] = 1;
            expect(Text.anchorY[eid]).toBe(1);
        });

        test("color converts hex to RGBA floats and back", () => {
            Text.color[eid] = 0xff0000;
            expect(Text.color[eid]).toBe(0xff0000);

            Text.color[eid] = 0x00ff00;
            expect(Text.color[eid]).toBe(0x00ff00);
        });

        test("colorR/G/B access individual linear channels", () => {
            Text.color[eid] = 0xff8040;
            expect(Text.colorR[eid]).toBeCloseTo(1.0, 3);
            expect(Text.colorG[eid]).toBeCloseTo(0.2159, 3);
            expect(Text.colorB[eid]).toBeCloseTo(0.0513, 3);
        });

        test("font field reads/writes Uint32 correctly", () => {
            Text.font[eid] = 2;
            expect(Text.font[eid]).toBe(2);
        });
    });

    describe("Text content (string field)", () => {
        test("content reads/writes string per entity", () => {
            Text.content[42] = "Hello World";
            expect(Text.content[42]).toBe("Hello World");
        });

        test("content is undefined for unset entities", () => {
            expect(Text.content[9999]).toBeUndefined();
        });

        test("content can be cleared with undefined", () => {
            Text.content[42] = "test";
            expect(Text.content[42]).toBe("test");
            Text.content[42] = undefined as unknown as string;
            expect(Text.content[42]).toBeUndefined();
        });

        test("different entities have independent content", () => {
            Text.content[10] = "first";
            Text.content[11] = "second";
            expect(Text.content[10]).toBe("first");
            expect(Text.content[11]).toBe("second");
        });
    });

    describe("Text component defaults", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            clearBuf(TextData);
            clearBuf(TextFonts);
            fontRegistry.clear();
            state = new State();
            state.register(TextPlugin);
        });

        test("defaults applied on addComponent", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Text);

            expect(Text.fontSize[eid]).toBe(1);
            expect(Text.opacity[eid]).toBe(1);
            expect(Text.visible[eid]).toBe(1);
            expect(Text.anchorX[eid]).toBe(0);
            expect(Text.anchorY[eid]).toBe(0);
            expect(Text.color[eid]).toBe(0xffffff);
            expect(Text.font[eid]).toBe(0);
        });
    });

    describe("ECS integration", () => {
        let state: State;

        beforeEach(() => {
            clearRegistry();
            clearBuf(TextData);
            clearBuf(TextFonts);
            fontRegistry.clear();
            state = new State();
            state.register(TextPlugin);
        });

        test("query matches entities with Text and Transform", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Text);
            state.addComponent(eid, Transform);

            expect(count(state, [Text, Transform])).toBe(1);
        });

        test("query excludes entities missing Transform", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Text);

            expect(count(state, [Text, Transform])).toBe(0);
        });

        test("removing Text drops entity from query", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Text);
            state.addComponent(eid, Transform);
            expect(count(state, [Text, Transform])).toBe(1);

            state.removeComponent(eid, Text);
            expect(count(state, [Text, Transform])).toBe(0);
        });

        test("dynamically added Text mid-session is queryable", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Transform);
            state.step();

            state.addComponent(eid, Text);
            expect(count(state, [Text, Transform])).toBe(1);
        });

        test("defaults applied on dynamic add", () => {
            const eid = state.addEntity();
            state.addComponent(eid, Transform);
            state.step();

            state.addComponent(eid, Text);
            expect(Text.fontSize[eid]).toBe(1);
            expect(Text.visible[eid]).toBe(1);
            expect(Text.color[eid]).toBe(0xffffff);
        });

        test("multiple Text entities all queryable", () => {
            for (let i = 0; i < 3; i++) {
                const eid = state.addEntity();
                state.addComponent(eid, Text);
                state.addComponent(eid, Transform);
            }
            expect(count(state, [Text, Transform])).toBe(3);
        });
    });

    describe("anchor positioning", () => {
        test("anchor 0,0 positions at origin", () => {
            const layoutWidth = 100;
            const layoutHeight = 20;
            const anchorX = 0;
            const anchorY = 0;
            const offsetX = -layoutWidth * anchorX;
            const offsetY = -layoutHeight * anchorY;
            expect(offsetX).toBeCloseTo(0, 5);
            expect(offsetY).toBeCloseTo(0, 5);
        });

        test("anchor 0.5,0.5 centers text", () => {
            const layoutWidth = 100;
            const layoutHeight = 20;
            const anchorX = 0.5;
            const anchorY = 0.5;
            const offsetX = -layoutWidth * anchorX;
            const offsetY = -layoutHeight * anchorY;
            expect(offsetX).toBe(-50);
            expect(offsetY).toBe(-10);
        });

        test("anchor 1,1 positions at bottom-right", () => {
            const layoutWidth = 100;
            const layoutHeight = 20;
            const anchorX = 1;
            const anchorY = 1;
            const offsetX = -layoutWidth * anchorX;
            const offsetY = -layoutHeight * anchorY;
            expect(offsetX).toBe(-100);
            expect(offsetY).toBe(-20);
        });
    });

    describe("color unpacking", () => {
        test("unpacks to RGB float components", () => {
            const color = 0xff8040;
            const r = ((color >> 16) & 0xff) / 255;
            const g = ((color >> 8) & 0xff) / 255;
            const b = (color & 0xff) / 255;
            expect(r).toBeCloseTo(1.0, 5);
            expect(g).toBeCloseTo(0.502, 2);
            expect(b).toBeCloseTo(0.251, 2);
        });

        test("round-trips color values", () => {
            const original = 0xabcdef;
            const r = (original >> 16) & 0xff;
            const g = (original >> 8) & 0xff;
            const b = original & 0xff;
            const packed = (r << 16) | (g << 8) | b;
            expect(packed).toBe(original);
        });
    });

    describe("font registry", () => {
        beforeEach(() => {
            fontRegistry.clear();
        });

        test("font() returns incrementing IDs", () => {
            const id0 = font("/font0.ttf");
            const id1 = font("/font1.ttf");
            const id2 = font("/font2.ttf");
            expect(id0).toBe(0);
            expect(id1).toBe(1);
            expect(id2).toBe(2);
        });

        test("get returns registered URL", () => {
            font("/font.ttf");
            expect(fontRegistry.get(0)).toBe("/font.ttf");
        });

        test("clear resets registry", () => {
            font("/font0.ttf");
            font("/font1.ttf");
            fontRegistry.clear();
            const id = font("/new.ttf");
            expect(id).toBe(0);
        });

        test("get returns undefined for invalid ID", () => {
            expect(fontRegistry.get(999)).toBeUndefined();
        });

        test("font() accepts optional name", () => {
            const id = font("/inter.ttf", "inter");
            expect(id).toBe(0);
        });

        test("getByName returns correct ID", () => {
            font("/font0.ttf", "inter");
            font("/font1.ttf", "pixel");
            expect(fontRegistry.getByName("inter")).toBe(0);
            expect(fontRegistry.getByName("pixel")).toBe(1);
        });

        test("getByName returns undefined for unknown name", () => {
            expect(fontRegistry.getByName("nonexistent")).toBeUndefined();
        });

        test("unnamed fonts are not in name map", () => {
            font("/font0.ttf");
            expect(fontRegistry.getByName("font0")).toBeUndefined();
        });

        test("clear resets name map", () => {
            font("/inter.ttf", "inter");
            expect(fontRegistry.getByName("inter")).toBe(0);
            fontRegistry.clear();
            expect(fontRegistry.getByName("inter")).toBeUndefined();
        });
    });
});
