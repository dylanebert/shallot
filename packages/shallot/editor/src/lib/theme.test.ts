import { describe, expect, test } from "bun:test";
import { adaptLight, packed, rgb, THEMES, theme } from "./theme";

const channel = (hex: string) => [
    (packed(hex) >> 16) & 0xff,
    (packed(hex) >> 8) & 0xff,
    packed(hex) & 0xff,
];
const luma = (hex: string) => channel(hex).reduce((s, c, i) => s + c * [0.299, 0.587, 0.114][i], 0);

describe("packed — hex string to packed color int", () => {
    test("decodes a 6-digit hex to its 0xRRGGBB value", () => {
        expect(packed("#221e1a")).toBe(0x221e1a);
        expect(packed("#ffffff")).toBe(0xffffff);
        expect(packed("#000000")).toBe(0x000000);
    });
});

describe("rgb — hex string to normalized channels", () => {
    test("splits channels in R,G,B order, normalized to 0..1", () => {
        expect(rgb("#000000")).toEqual([0, 0, 0]);
        expect(rgb("#ffffff")).toEqual([1, 1, 1]);
        // the editor's Z-axis blue decodes to clean fractions — a channel swap would surface here
        const [r, g, b] = rgb("#3366cc");
        expect(r).toBeCloseTo(0.2, 5);
        expect(g).toBeCloseTo(0.4, 5);
        expect(b).toBeCloseTo(0.8, 5);
    });
});

describe("adaptLight — re-light a chromatic token for a light background", () => {
    test("preserves hue family and darkens a bright source", () => {
        const blue = adaptLight("#4a90e2");
        const [r, g, b] = channel(blue);
        expect(b).toBeGreaterThan(r); // still blue-dominant — hue held
        expect(b).toBeGreaterThan(g);
        expect(luma(blue)).toBeLessThan(luma("#4a90e2")); // darker, for contrast on white
    });
});

describe("theme — resolve by id with default fallback", () => {
    test("returns the matching theme", () => {
        expect(theme("light").id).toBe("light");
    });

    test("falls back to the default for an unknown or absent id", () => {
        expect(theme("nope")).toBe(THEMES[0]);
        expect(theme(undefined)).toBe(THEMES[0]);
    });
});
