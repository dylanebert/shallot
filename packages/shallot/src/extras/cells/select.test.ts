import { describe, expect, test } from "bun:test";
import { vec3f } from "typegpu/data";
import { CELL_DIRECTIONAL_GLYPHS, CELL_FILL_GLYPHS } from "./ramp";
import {
    directionalGlyphIndex,
    EDGE_MAGNITUDE_THRESHOLD,
    luma,
    reinhard,
    selectWgsl,
} from "./select";

// directionalGlyphIndex is a typegpu dual CPU/GPU function (`packCell`'s own shape), callable directly
// with no device — the logic-truth surface for the tangent-bucket rotation (`testing.md`'s "CPU execution
// of pure TGSL kernels against deterministic references").
describe("directionalGlyphIndex", () => {
    // each case's *raw gradient bucket* is named in a comment beside it — the naive (unrotated) index a
    // regression that dropped the 90° rotation would produce instead, so this table is two-sided: it
    // proves the rotation ran, not just that some directional index came back.
    const cases: { name: string; gx: number; gy: number; wantChar: string; naiveBucket: number }[] =
        [
            // gradient 0° (points +x, a vertical edge) -> tangent 90° -> "|"
            { name: "gradient +x (vertical edge)", gx: 1, gy: 0, wantChar: "|", naiveBucket: 0 },
            // gradient 90° (points +y, a horizontal edge) -> tangent 0° -> "-"
            { name: "gradient +y (horizontal edge)", gx: 0, gy: 1, wantChar: "-", naiveBucket: 2 },
            // gradient 45° -> tangent 135° -> "\"
            { name: "gradient +45°", gx: 1, gy: 1, wantChar: "\\", naiveBucket: 1 },
            // gradient -45° (folds to 135°) -> tangent 45° -> "/"
            { name: "gradient -45°", gx: 1, gy: -1, wantChar: "/", naiveBucket: 3 },
        ];

    for (const { name, gx, gy, wantChar, naiveBucket } of cases) {
        test(`${name} selects the perpendicular tangent glyph, not the raw gradient bucket`, () => {
            const index = directionalGlyphIndex(gx, gy);
            const char = CELL_DIRECTIONAL_GLYPHS[index - CELL_FILL_GLYPHS.length];
            expect(char).toBe(wantChar);
            // the naive (unrotated) index this would read as if the 90° rotation were dropped — assert
            // the two disagree, or this case can't discriminate the bug it exists to catch
            const naiveChar = CELL_DIRECTIONAL_GLYPHS[naiveBucket];
            expect(char).not.toBe(naiveChar);
        });
    }

    test("indexes past the fill ramp — every directional index is >= CELL_FILL_GLYPHS.length", () => {
        for (const { gx, gy } of cases) {
            expect(directionalGlyphIndex(gx, gy)).toBeGreaterThanOrEqual(CELL_FILL_GLYPHS.length);
        }
    });
});

describe("reinhard", () => {
    test("compresses toward 1 without ever reaching it", () => {
        const black = reinhard(vec3f(0, 0, 0));
        expect(black.x).toBe(0);
        expect(black.y).toBe(0);
        expect(black.z).toBe(0);
        const bright = reinhard(vec3f(9, 9, 9));
        expect(bright.x).toBeCloseTo(0.9, 5);
    });
});

describe("luma", () => {
    test("rec709 weights, white maps to 1", () => {
        expect(luma(vec3f(1, 1, 1))).toBeCloseTo(1, 5);
        expect(luma(vec3f(0, 0, 0))).toBe(0);
    });
});

describe("EDGE_MAGNITUDE_THRESHOLD", () => {
    test("sits strictly between zero and the Sobel kernel's maximum diagonal response", () => {
        // weights sum to 4 per axis; a maximal diagonal step (gx = gy = 4) reads sqrt(32) ≈ 5.657 —
        // the threshold must gate somewhere inside that range or it can never fire, or always fires
        expect(EDGE_MAGNITUDE_THRESHOLD).toBeGreaterThan(0);
        expect(EDGE_MAGNITUDE_THRESHOLD).toBeLessThan(Math.sqrt(2) * 4);
    });
});

describe("selectWgsl", () => {
    test("resolves both kernels with no device", () => {
        const wgsl = selectWgsl();
        expect(wgsl).toContain("fn");
        expect(wgsl.length).toBeGreaterThan(0);
    });
});
