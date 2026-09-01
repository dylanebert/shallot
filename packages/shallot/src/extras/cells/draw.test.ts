import { describe, expect, test } from "bun:test";
import { vec2f } from "typegpu/data";
import { drawWgsl, glyphLocalCorner } from "./draw";

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

    test("the vertex stage indexes the glyph-size table and reads glyphLocalCorner", () => {
        const wgsl = drawWgsl();
        expect(wgsl).toContain("glyphSize");
        expect(wgsl).toContain("fn glyphLocalCorner(corner: vec2f, size: vec2f) -> vec2f {");
    });
});

// glyphLocalCorner is the size-proportional-placement fix's own arithmetic, extracted to a plain TS-callable
// TGSL function so its numbers are asserted directly rather than only through resolved shader text
// (`checks.md`: "an arm over a shader's resolved text is not an arm over its behaviour"). One TGSL source,
// dual-mode like `cell.ts`'s `packCell` — `cellVertex` resolves the identical body real-device-side
// (`bun bench --scenario cells`'s "ramp monotonicity" check drives that path).
describe("glyphLocalCorner", () => {
    test("size (1, 1) leaves the corner unchanged — the pre-fix full-cell footprint", () => {
        const result = glyphLocalCorner(vec2f(0, 0), vec2f(1, 1));
        expect(result.x).toBeCloseTo(0, 5);
        expect(result.y).toBeCloseTo(0, 5);
        const other = glyphLocalCorner(vec2f(1, 1), vec2f(1, 1));
        expect(other.x).toBeCloseTo(1, 5);
        expect(other.y).toBeCloseTo(1, 5);
    });

    test("shrinks and centers the quad to a fractional footprint", () => {
        // size (0.5, 0.5): a 0.25 margin on every side, corner 0..1 mapped into [0.25, 0.75]
        const lo = glyphLocalCorner(vec2f(0, 0), vec2f(0.5, 0.5));
        expect(lo.x).toBeCloseTo(0.25, 5);
        expect(lo.y).toBeCloseTo(0.25, 5);
        const hi = glyphLocalCorner(vec2f(1, 1), vec2f(0.5, 0.5));
        expect(hi.x).toBeCloseTo(0.75, 5);
        expect(hi.y).toBeCloseTo(0.75, 5);
        const center = glyphLocalCorner(vec2f(0.5, 0.5), vec2f(0.5, 0.5));
        expect(center.x).toBeCloseTo(0.5, 5);
        expect(center.y).toBeCloseTo(0.5, 5);
    });

    test("independent width/height scaling — a narrow, tall footprint shrinks each axis its own amount", () => {
        const result = glyphLocalCorner(vec2f(1, 0), vec2f(0.2, 0.8));
        // margin.x = 0.4, margin.y = 0.1
        expect(result.x).toBeCloseTo(0.6, 5);
        expect(result.y).toBeCloseTo(0.1, 5);
    });

    test("a smaller footprint always sits strictly inside a larger one at every non-degenerate corner", () => {
        // the mutation this red-proofs: a fix that only shrinks without centering would leave (0,0) fixed
        // instead of moving it inward — this reds if the margin term is dropped
        const small = glyphLocalCorner(vec2f(0, 0), vec2f(0.3, 0.3));
        const large = glyphLocalCorner(vec2f(0, 0), vec2f(0.9, 0.9));
        expect(small.x).toBeGreaterThan(large.x);
        expect(small.y).toBeGreaterThan(large.y);
    });
});
