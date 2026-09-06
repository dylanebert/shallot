import { describe, expect, test } from "bun:test";
import { vec2f } from "typegpu/data";
import { cellFootprintPx, drawWgsl, glyphFootprintT } from "./draw";

// The draw pipeline's device-free structural seam: `tgpu.resolve` runs no device, so this proves the vs/fs
// pair type-checks and resolves to valid WGSL text with no adapter — the same shape `grid.ts`'s `gridWgsl`
// and `extras/outline`'s `wgslArtifacts` use. Real-device correctness (the instanced draw's actual pixels)
// is `bun bench`/Playwright tier truth (`testing.md`), not this file's.
describe("drawWgsl", () => {
    test("resolves the vertex + fragment pair with no device", () => {
        const wgsl = drawWgsl();
        expect(wgsl).toContain("@vertex");
        expect(wgsl).toContain("@fragment");
    });

    test("the vertex stage reads @builtin(vertex_index) and @builtin(instance_index) — no vertex buffer", () => {
        const wgsl = drawWgsl();
        expect(wgsl).toContain("vertex_index");
        expect(wgsl).toContain("instance_index");
    });

    test("the vertex stage indexes the glyph-size table and computes the isotropic footprint", () => {
        const wgsl = drawWgsl();
        expect(wgsl).toContain("glyphSize");
        expect(wgsl).toContain(
            "fn cellFootprintPx(size: vec2f, cellW: f32, cellH: f32) -> vec2f {",
        );
        expect(wgsl).toContain("fn glyphFootprintT(corner: vec2f, sizeNorm: vec2f) -> vec2f {");
    });
});

// cellFootprintPx is the s3r item 9 fix's own arithmetic, extracted to a plain TS-callable TGSL function
// so its numbers are asserted directly rather than only through resolved shader text. One TGSL source, dual-mode like
// `cell.ts`'s `packCell` — `cellVertex` resolves the identical body real-device-side.
describe("cellFootprintPx", () => {
    test("a non-square cell scales both axes by the SAME factor — no anisotropic stretch", () => {
        // cellW=16, cellH=30 (a real 80x24-shaped cell): the s3r item 9 defect used cellW on x and cellH
        // on y independently, which would give (0.5*16, 0.5*30) = (8, 15) here — a 1:1 em-space glyph
        // rendered as a 1:1.875 footprint. The fix uses one shared scale, min(16, 30) = 16, so a square
        // glyph (size.x === size.y) stays square on screen.
        const result = cellFootprintPx(vec2f(0.5, 0.5), 16, 30);
        expect(result.x).toBeCloseTo(8, 5);
        expect(result.y).toBeCloseTo(8, 5);
    });

    test("collapses to the old per-axis formula on a square cell — the gym bench fixtures' own shape", () => {
        // every examples/gym/src/scenarios/cells.ts fixture uses cellW === cellH, where
        // min(cellW, cellH) === cellW === cellH and the isotropic scale is indistinguishable from
        // multiplying each axis by its own cell dimension — this is why the s3r item 9 defect was
        // invisible to that gate.
        const result = cellFootprintPx(vec2f(0.3, 0.7), 24, 24);
        expect(result.x).toBeCloseTo(0.3 * 24, 5);
        expect(result.y).toBeCloseTo(0.7 * 24, 5);
    });

    test("a wide-and-short footprint keeps its own aspect, scaled by the narrower cell axis", () => {
        const result = cellFootprintPx(vec2f(0.8, 0.2), 16, 30);
        // scale = min(16, 30) = 16
        expect(result.x).toBeCloseTo(0.8 * 16, 5);
        expect(result.y).toBeCloseTo(0.2 * 16, 5);
        // the two axes preserve the input's own 4:1 ratio — this is what "no stretch" means numerically
        expect(result.x / result.y).toBeCloseTo(0.8 / 0.2, 5);
    });

    test("every footprint stays within the cell on both axes, at the identity (1, 1) em size", () => {
        const result = cellFootprintPx(vec2f(1, 1), 16, 30);
        expect(result.x).toBeLessThanOrEqual(16);
        expect(result.y).toBeLessThanOrEqual(30);
    });

    test("a degenerate zero-em glyph is floored, not NaN or zero-divided downstream", () => {
        const result = cellFootprintPx(vec2f(0, 0), 16, 30);
        expect(Number.isFinite(result.x)).toBe(true);
        expect(Number.isFinite(result.y)).toBe(true);
        expect(result.x).toBeGreaterThan(0);
        expect(result.y).toBeGreaterThan(0);
    });
});

// glyphFootprintT is the full-cell-geometry fix's inverse map: given a corner spanning the WHOLE cell
// (0..1), where does it fall relative to the glyph's own centered footprint? Values outside [0, 1] mean
// "outside the footprint, in the cell's own background margin" — `cellFragment` clamps before sampling
// and gates ink on the unclamped range.
describe("glyphFootprintT", () => {
    test("the cell center is always the footprint center, at any footprint size", () => {
        const wide = glyphFootprintT(vec2f(0.5, 0.5), vec2f(0.9, 0.9));
        const narrow = glyphFootprintT(vec2f(0.5, 0.5), vec2f(0.1, 0.1));
        expect(wide.x).toBeCloseTo(0.5, 5);
        expect(wide.y).toBeCloseTo(0.5, 5);
        expect(narrow.x).toBeCloseTo(0.5, 5);
        expect(narrow.y).toBeCloseTo(0.5, 5);
    });

    test("size (1, 1) is the identity map — a full-cell footprint has no margin", () => {
        const lo = glyphFootprintT(vec2f(0, 0), vec2f(1, 1));
        const hi = glyphFootprintT(vec2f(1, 1), vec2f(1, 1));
        expect(lo.x).toBeCloseTo(0, 5);
        expect(lo.y).toBeCloseTo(0, 5);
        expect(hi.x).toBeCloseTo(1, 5);
        expect(hi.y).toBeCloseTo(1, 5);
    });

    test("a corner in the cell's margin (outside the footprint) extrapolates past [0, 1]", () => {
        // size (0.5, 0.5): a 0.25 margin on every side, so the full-cell corner (0, 0) sits a quarter of
        // the *footprint's own width* outside it on the low side.
        const result = glyphFootprintT(vec2f(0, 0), vec2f(0.5, 0.5));
        expect(result.x).toBeLessThan(0);
        expect(result.y).toBeLessThan(0);
        expect(result.x).toBeCloseTo(-0.5, 5);
        expect(result.y).toBeCloseTo(-0.5, 5);
    });

    test("is the exact inverse of the centered-shrink map it replaced — round-trips through it", () => {
        // the s3r item 2 shrink was `shrunk = margin + corner * size`; glyphFootprintT inverts it:
        // `t = (shrunk - margin) / size` must recover the original `corner`.
        const size = vec2f(0.4, 0.7);
        const corner = vec2f(0.3, 0.9);
        const margin = vec2f((1 - size.x) / 2, (1 - size.y) / 2);
        const shrunk = vec2f(margin.x + corner.x * size.x, margin.y + corner.y * size.y);
        const t = glyphFootprintT(shrunk, size);
        expect(t.x).toBeCloseTo(corner.x, 5);
        expect(t.y).toBeCloseTo(corner.y, 5);
    });

    test("independent width/height footprints are each inverted on their own axis", () => {
        // the mutation this red-proofs: a fix that swaps x/y or shares one axis' margin across both
        // would fail this asymmetric case even though the square case above stays green
        const result = glyphFootprintT(vec2f(1, 0), vec2f(0.2, 0.8));
        // margin.x = 0.4, margin.y = 0.1; t = (corner - margin) / size
        expect(result.x).toBeCloseTo((1 - 0.4) / 0.2, 5);
        expect(result.y).toBeCloseTo((0 - 0.1) / 0.8, 5);
    });
});
