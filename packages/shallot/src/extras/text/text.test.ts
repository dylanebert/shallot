import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { font, TextPlugin, text } from "./";
import { Glyph, sdfToSignedDistance, textSrgbToLinear } from "./glyph";
import { SDF_EXPONENT, SdfUniforms, sdfWgsl } from "./sdf";

// The registries are the CPU codec boundary the scene parse/format trait crosses — the glyph layout + GPU
// draw are validated in `bun bench`. These assert the interning is stable and round-trips through the
// public trait, no State or GPU needed. Beside them: the glyph instance buffer's layout (the CPU↔GPU
// boundary the staging writer indexes), the two SDF-decode kernels now callable on the CPU, and the
// structure of the two generator shaders (`sdfWgsl` resolves device-free).
const traits = TextPlugin.traits!.Text;
const parseContent = traits.parse!.content as (raw: string) => number;
const formatContent = traits.format!.content as (id: number) => string;
const parseFont = traits.parse!.font as (name: string) => number;

describe("kitchen text registry surface", () => {
    test("text() interns a string and the trait formats the id back", () => {
        const id = text("Hello");
        expect(formatContent(id)).toBe("Hello");
        expect(parseContent("Hello")).toBe(id); // scene parse interns through the same table
    });

    test("identical strings dedupe to one id, distinct strings don't", () => {
        expect(text("shared")).toBe(text("shared"));
        expect(text("first")).not.toBe(text("second"));
    });

    test("content id 0 is the empty string — the Text.content default", () => {
        expect(text("")).toBe(0);
        expect(formatContent(0)).toBe("");
    });

    test("font() dedupes by name, distinct names get distinct ids", () => {
        const inter = font("/inter.ttf", "inter");
        expect(font("/inter.ttf", "inter")).toBe(inter); // same name, same id
        expect(font("/pixel.ttf", "pixel")).not.toBe(inter);
    });

    test("the font trait parse resolves a name to its id, unknown names to the default 0", () => {
        const id = font("/mono.ttf", "mono");
        expect(parseFont("mono")).toBe(id);
        expect(parseFont("nonexistent")).toBe(0);
    });
});

// The glyph instance buffer's layout: the staging writer derives its indices from the schema and the
// surface's vs reads the struct, so the offsets are pinned against the numbers the hand-written `Glyph`
// WGSL struct had — literals, never re-derived from the schema itself.
describe("Glyph layout", () => {
    const offset = (field: (g: d.Infer<typeof Glyph>) => unknown) =>
        d.memoryLayoutOf(Glyph, field).offset;

    test("three vec4 reads, 48 bytes, eid sharing pos's 16-byte slot", () => {
        expect(d.sizeOf(Glyph)).toBe(48);
        expect(offset((g) => g.pos)).toBe(0);
        // eid rides pos's trailing word, which is what keeps the instance at three vec4 reads
        expect(offset((g) => g.eid)).toBe(12);
        expect(offset((g) => g.uvRect)).toBe(16);
        expect(offset((g) => g.size)).toBe(32);
        expect(offset((g) => g.color)).toBe(40);
        expect(offset((g) => g.pad)).toBe(44);
    });

    test("resolving the struct + both decode helpers emits them under their authored names", () => {
        // no `glyphWgsl` preamble chunk anymore (the surface is a typed `typedTextSurface`, not a raw
        // WGSL splice site) — resolve the same three resources directly, same names, same substance
        const wgsl = tgpu.resolve([Glyph, sdfToSignedDistance, textSrgbToLinear], {
            names: "strict",
        });
        expect(wgsl).toContain("struct Glyph {");
        expect(wgsl).toContain("fn sdfToSignedDistance(sdf: f32, maxDimension: f32) -> f32 {");
        expect(wgsl).toContain("fn textSrgbToLinear(c: vec3f) -> vec3f {");
    });
});

// Both helpers are plain TGSL now, so the decode the fs depends on is checkable without a GPU. The
// exponent-9 curve is exercised at the points where the arithmetic is exact by hand: `2·a` a power of two
// makes its 1/9 root exact, so no value here restates the implementation.
describe("SDF decode on the CPU", () => {
    test("sdf 0.5 is exactly the glyph edge", () => {
        expect(sdfToSignedDistance(0.5, 7)).toBe(0);
    });

    test("sdf > 0.5 is inside (negative), sdf < 0.5 outside (positive), and the two mirror", () => {
        expect(sdfToSignedDistance(0.9, 1)).toBeLessThan(0);
        expect(sdfToSignedDistance(0.1, 1)).toBeGreaterThan(0);
        expect(sdfToSignedDistance(0.9, 1)).toBeCloseTo(-sdfToSignedDistance(0.1, 1), 6);
    });

    test("the exponent-9 root at hand-exact points: 2·a = 2⁻⁹ ⇒ half the max dimension", () => {
        expect(SDF_EXPONENT).toBe(9);
        // a = 2⁻¹⁰, so pow(2·a, 1/9) = pow(2⁻⁹, 1/9) = 2⁻¹ ⇒ absDist = (1 − 0.5)·maxDim
        expect(sdfToSignedDistance(2 ** -10, 4)).toBeCloseTo(2, 5);
        expect(sdfToSignedDistance(1 - 2 ** -10, 4)).toBeCloseTo(-2, 5);
        // a = 0 ⇒ pow(0, 1/9) = 0 ⇒ absDist = maxDim, the far rail on both sides
        expect(sdfToSignedDistance(0, 3)).toBeCloseTo(3, 6);
        expect(sdfToSignedDistance(1, 3)).toBeCloseTo(-3, 6);
    });

    test("textSrgbToLinear: 0 and 1 are fixed points, the breakpoint takes the linear segment", () => {
        expect(textSrgbToLinear(d.vec3f(0)).x).toBe(0);
        // (1 + 0.055) / 1.055 = 1, and 1^2.4 = 1
        expect(textSrgbToLinear(d.vec3f(1)).x).toBeCloseTo(1, 6);
        // 0.04045 is the piecewise boundary and the comparison is `<=`, so it divides by 12.92
        expect(textSrgbToLinear(d.vec3f(0.04045)).x).toBeCloseTo(0.04045 / 12.92, 7);
        // just past it the power segment takes over, and it is the larger of the two branches there
        expect(textSrgbToLinear(d.vec3f(0.05)).x).toBeGreaterThan(0.05 / 12.92);
    });
});

// The two generator passes resolve device-free (the `blitWgsl` seam), so their entry points, bindings and
// fragment output shapes are structurally checkable in `bun test`; the pixels are a `bun bench` concern.
describe("SDF generator shaders", () => {
    test("the distance pass declares its uniform + segment bindings and one vec4 target", () => {
        const wgsl = sdfWgsl().distance;
        expect(wgsl).toContain("@vertex fn distanceVs(");
        expect(wgsl).toContain("@fragment fn distanceFs(");
        expect(wgsl).toContain("@group(0) @binding(0) var<uniform> uniforms: SdfUniforms;");
        expect(wgsl).toContain("@group(0) @binding(1) var<storage, read> segments: array<vec4f>;");
        expect(wgsl).toContain("-> @location(0) vec4f");
        // one instance per segment, so the segment index rides the vertex stage as a flat u32
        expect(wgsl).toContain("@builtin(instance_index) segmentIdx: u32");
        expect(wgsl).toContain("@location(1) @interpolate(flat) segmentIdx: u32,");
        // the r/g crossing counters are one 1/255 step each, additively blended
        expect(wgsl).toContain("0.00392156862745098f");
    });

    test("the finalize pass declares its texture + sampler and flips the falloff when inside", () => {
        const wgsl = sdfWgsl().finalize;
        expect(wgsl).toContain("@vertex fn finalizeVs(");
        expect(wgsl).toContain("@fragment fn finalizeFs(");
        expect(wgsl).toContain("@group(0) @binding(0) var intermediate: texture_2d<f32>;");
        expect(wgsl).toContain("@group(0) @binding(1) var samp: sampler;");
        expect(wgsl).toContain("-> @location(0) vec4f");
        // inside ⇔ the two crossing counters disagree; the target is r8unorm, so all four lanes carry it
        expect(wgsl).toContain("let inside = (color.x != color.y);");
        expect(wgsl).toContain("return vec4f(val, val, val, val);");
    });

    test("the SdfUniforms layout the per-glyph write fills", () => {
        const wgsl = sdfWgsl().distance;
        expect(wgsl).toContain("struct SdfUniforms {");
        expect(d.sizeOf(SdfUniforms)).toBe(32);
        expect(d.memoryLayoutOf(SdfUniforms, (u) => u.glyphBounds).offset).toBe(0);
        expect(d.memoryLayoutOf(SdfUniforms, (u) => u.maxDistance).offset).toBe(16);
        expect(d.memoryLayoutOf(SdfUniforms, (u) => u.exponent).offset).toBe(20);
    });
});
