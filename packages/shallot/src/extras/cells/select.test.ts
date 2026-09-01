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
//
// gx/gy fixtures below are derived from concrete 3×3 luma neighborhoods run through selectKernel's own
// Sobel formula (`select.ts`), not picked as abstract numbers — the class of bug this arm exists to catch
// is a y-frame mix-up (grid y-down vs. the glyph labels' y-up visual convention), which an abstract
// `(gx, gy)` pair can't discriminate: the old version of this table asserted `gx=1,gy=1 -> "\\"`, which is
// what you get if you *don't* convert frames, and it passed for a full review round (`specs/shallot-tui.md`
// s3r item 1). Naming the luma neighborhood a case derives from ties the expectation to a real edge
// instead of to whatever the code under test currently computes.
describe("directionalGlyphIndex", () => {
    // l(dx, dy) = 0.5 + A*dx + B*dy over the 8 neighbors selectKernel reads (dx/dy in {-1,0,1}, own
    // excluded) — a smooth luma ramp, not a hard step, so no tap saturates and the tie-break at exactly
    // 45°/135° stays clean. Sobel of a linear field is exact: gx = 8*A, gy = 8*B (grid frame, y-down).
    function sobelOfLinearField(a: number, b: number): { gx: number; gy: number } {
        const l = (dx: number, dy: number) => 0.5 + a * dx + b * dy;
        const l00 = l(-1, -1);
        const l10 = l(0, -1);
        const l20 = l(1, -1);
        const l01 = l(-1, 0);
        const l21 = l(1, 0);
        const l02 = l(-1, 1);
        const l12 = l(0, 1);
        const l22 = l(1, 1);
        const gx = l20 + 2 * l21 + l22 - (l00 + 2 * l01 + l02);
        const gy = l02 + 2 * l12 + l22 - (l00 + 2 * l10 + l20);
        return { gx, gy };
    }

    const cases: { name: string; a: number; b: number; wantChar: string }[] = [
        // brightness rises rightward only (a horizontal ramp, top/bottom rows equal) -> the edge (the
        // ramp's own contour lines) runs vertically -> "|"
        { name: "brighter to the right (vertical edge)", a: 0.25, b: 0, wantChar: "|" },
        // brightness rises downward only (row-down) -> the edge runs horizontally -> "-"
        { name: "brighter toward the bottom (horizontal edge)", a: 0, b: 0.25, wantChar: "-" },
        // top-left darkest, bottom-right brightest: the contour tangent runs bottom-left-to-top-right
        { name: "darkest top-left, brightest bottom-right", a: 0.25, b: 0.25, wantChar: "/" },
        // top-right darkest, bottom-left brightest: the contour tangent runs top-left-to-bottom-right
        { name: "darkest bottom-left, brightest top-right", a: 0.25, b: -0.25, wantChar: "\\" },
    ];

    for (const { name, a, b, wantChar } of cases) {
        test(`${name} selects the perpendicular tangent glyph`, () => {
            const { gx, gy } = sobelOfLinearField(a, b);
            const index = directionalGlyphIndex(gx, gy);
            const char = CELL_DIRECTIONAL_GLYPHS[index - CELL_FILL_GLYPHS.length];
            expect(char).toBe(wantChar);
        });
    }

    // the two diagonal cases above are each other's y-frame-mix-up bug: dropping the grid-y-down ->
    // visual-y-up conversion swaps exactly these two and leaves the two axis-aligned cases untouched
    // (`select.ts`'s own docblock) — assert the swap explicitly so a regression that re-introduces it
    // reds here even if a future edit reorders or renames the table above.
    test("the two diagonal cases are not swapped", () => {
        const bl = sobelOfLinearField(0.25, 0.25);
        const br = sobelOfLinearField(0.25, -0.25);
        const blChar =
            CELL_DIRECTIONAL_GLYPHS[directionalGlyphIndex(bl.gx, bl.gy) - CELL_FILL_GLYPHS.length];
        const brChar =
            CELL_DIRECTIONAL_GLYPHS[directionalGlyphIndex(br.gx, br.gy) - CELL_FILL_GLYPHS.length];
        expect(blChar).toBe("/");
        expect(brChar).toBe("\\");
        expect(blChar).not.toBe(brChar);
    });

    test("indexes past the fill ramp — every directional index is >= CELL_FILL_GLYPHS.length", () => {
        for (const { a, b } of cases) {
            const { gx, gy } = sobelOfLinearField(a, b);
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
    // the kernel's true reachable maximum, derived rather than quoted: gx/gy are each linear in the 8
    // free luma taps (own excluded) selectKernel reads, each tap clamped to [0, 1], so a linear
    // objective (the Sobel magnitude, maximized over a direction) is maximized at a vertex of that
    // 8-cube — brute-forceable over all 256. `gx = gy = 4` (sqrt(32) ≈ 5.657) is NOT reachable: gx and
    // gy share taps with opposite sign (e.g. l02/l20), so driving gx to its own axis maximum caps what
    // gy can reach at the same time (`select.ts`'s own derivation).
    function sobelMagnitude(bits: number): number {
        const l = Array.from({ length: 8 }, (_, i) => (bits >> i) & 1);
        const [l00, l10, l20, l01, l21, l02, l12, l22] = l;
        const gx = l20 + 2 * l21 + l22 - (l00 + 2 * l01 + l02);
        const gy = l02 + 2 * l12 + l22 - (l00 + 2 * l10 + l20);
        return Math.hypot(gx, gy);
    }

    test("the kernel's reachable maximum is sqrt(20), not sqrt(32)", () => {
        let max = 0;
        for (let bits = 0; bits < 256; bits++) max = Math.max(max, sobelMagnitude(bits));
        expect(max).toBeCloseTo(Math.sqrt(20), 10);
    });

    test("sits strictly between zero and the Sobel kernel's reachable maximum", () => {
        let max = 0;
        for (let bits = 0; bits < 256; bits++) max = Math.max(max, sobelMagnitude(bits));
        // the threshold must gate somewhere inside that range or it can never fire, or always fires
        expect(EDGE_MAGNITUDE_THRESHOLD).toBeGreaterThan(0);
        expect(EDGE_MAGNITUDE_THRESHOLD).toBeLessThan(max);
    });
});

describe("selectWgsl", () => {
    test("resolves both kernels with no device", () => {
        const wgsl = selectWgsl();
        expect(wgsl).toContain("fn");
        expect(wgsl.length).toBeGreaterThan(0);
    });
});
