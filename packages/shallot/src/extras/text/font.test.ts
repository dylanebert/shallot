import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { parseFont } from "./font";

// The TTF binary decoder. parseFont walks the sfnt table directory and the glyf
// outlines into the Font query surface. The brand font (Outfit) is the real
// fixture; the rejection paths run on hand-built headers.
const bytes = readFileSync(new URL("../../../../../assets/font.ttf", import.meta.url));
const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
) as ArrayBuffer;
const font = parseFont(buffer);

describe("parseFont — metrics", () => {
    test("reads head/hhea vertical metrics with the expected signs", () => {
        expect(font.unitsPerEm).toBeGreaterThan(0);
        expect(font.ascender).toBeGreaterThan(0);
        expect(font.descender).toBeLessThan(0);
        expect(font.lineGap).toBeGreaterThanOrEqual(0);
    });
});

describe("parseFont — glyphs", () => {
    test("a drawn glyph yields a closed SVG path and a positive-area bound", () => {
        const path = font.glyphPath("A");
        expect(path).not.toBeNull();
        expect(path!.startsWith("M")).toBe(true);
        expect(path!.endsWith("Z")).toBe(true);

        const bounds = font.glyphBounds("A");
        expect(bounds).not.toBeNull();
        const [xMin, yMin, xMax, yMax] = bounds!;
        expect(xMax).toBeGreaterThan(xMin);
        expect(yMax).toBeGreaterThan(yMin);

        expect(font.advance("A")).toBeGreaterThan(0);
    });

    test("the glyph cache returns the identical path on a repeat lookup", () => {
        expect(font.glyphPath("A")).toBe(font.glyphPath("A"));
    });

    test("a whitespace glyph has no outline but still advances the pen", () => {
        // space is an empty glyph (loca start === end → no contours), yet carries
        // an advance width — the null-path / positive-advance split is the edge.
        expect(font.glyphPath(" ")).toBeNull();
        expect(font.advance(" ")).toBeGreaterThan(0);
    });

    test("accented glyphs resolve to a path", () => {
        // diacritics are often composite glyphs (base + accent), which take the
        // numContours < 0 branch into the component assembler.
        for (const char of ["é", "ñ", "ü", "Â"]) {
            expect(font.glyphPath(char)).not.toBeNull();
        }
    });

    test("kerning returns a number for any pair", () => {
        // the contract is a number — 0 when the font carries no legacy kern table.
        expect(typeof font.kerning("A", "V")).toBe("number");
    });
});

describe("parseFont — rejects malformed input", () => {
    test("throws on a buffer that isn't a TTF/OTF", () => {
        expect(() => parseFont(new ArrayBuffer(12))).toThrow(/valid TTF/);
    });

    test("throws when required tables are missing", () => {
        const buf = new ArrayBuffer(12);
        new DataView(buf).setUint32(0, 0x00010000); // valid sfnt magic, numTables stays 0
        expect(() => parseFont(buf)).toThrow(/Missing required font tables/);
    });
});
