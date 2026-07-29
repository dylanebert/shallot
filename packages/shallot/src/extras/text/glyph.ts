// The glyph instance buffer's CPU↔GPU boundary: the `Glyph` layout the staging writer fills and the
// surface's vs reads, plus the two pure decode helpers its fs calls. A sibling of the producer (index.ts)
// rather than one of its exports — a WGSL chunk and a codegen schema are not author surface (exports.md
// "Barrel rules"), and the test imports this file directly.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { chunk, spliceNs } from "../../engine/utils/core";
import { SDF_EXPONENT } from "./sdf";

/**
 * one glyph instance: a glyph-local quad origin + the owning entity id, the atlas uv rect, the world-space
 * quad size, and a packed sRGBA color. Three vec4 reads — `pos` shares its 16-byte slot with `eid`
 * (matching the line segment's pos+width packing), so the trailing `pad` is what rounds the struct to
 * that stride. The CPU↔GPU layout of the glyph instance buffer, so the staging writer's indices and its
 * byte size both derive from here.
 */
export const Glyph = d.struct({
    pos: d.vec3f,
    eid: d.u32,
    uvRect: d.vec4f,
    size: d.vec2f,
    color: d.u32,
    pad: d.u32,
});

/** one glyph instance's byte size, from {@link Glyph} — the staging + GPU buffer stride. @internal */
export const GLYPH_BYTES = d.sizeOf(Glyph);
/** the same stride in 4-byte lanes, the staging typed-array views' step. @internal */
export const GLYPH_FLOATS = GLYPH_BYTES / 4;

// the f32/u32 index of one Glyph field, from the schema — the staging writer fills flat typed-array
// views, so this is what keeps them and the WGSL struct one layout
const at = <T extends d.BaseData>(schema: T, field: (p: d.Infer<T>) => unknown) =>
    d.memoryLayoutOf(schema, field).offset / 4;

/** each {@link Glyph} field's lane index within one instance, the staging writer's offsets.
 *  @internal */
export const GLYPH_AT = {
    pos: at(Glyph, (g) => g.pos),
    eid: at(Glyph, (g) => g.eid),
    uvRect: at(Glyph, (g) => g.uvRect),
    size: at(Glyph, (g) => g.size),
    color: at(Glyph, (g) => g.color),
} as const;

/**
 * decode a single-channel SDF texel to a world-space signed distance — negative inside the glyph,
 * positive outside, 0 exactly on the edge. The atlas stores the falloff exponent-encoded and folded
 * about 0.5 (`sdf > 0.5` is inside), so the decode unfolds it, takes the {@link SDF_EXPONENT}-th root,
 * and rescales by the glyph's largest world dimension. Valve "Improved Alpha-Tested Magnification".
 *
 * @example let signedDist = sdfToSignedDistance(sdf, max(size.x, size.y));
 */
export const sdfToSignedDistance = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((sdf, maxDimension) => {
    "use gpu";
    const a = std.select(sdf, 1 - sdf, sdf > 0.5);
    const absDist = (1 - std.pow(2 * a, 1 / SDF_EXPONENT)) * maxDimension;
    // f32-typed rails: a bare `1` / `-1` transpiles i32 and lands an implicit conversion in the emitted
    // WGSL (the integer-literal law, spec Approach 2a)
    return absDist * std.select(d.f32(1), d.f32(-1), sdf > 0.5);
});

/** the sRGB→linear transfer function (the exact piecewise form, not the γ-2.2 approximation): the packed
 *  glyph color is sRGB-encoded and the color target is linear, so it decodes before the blend.
 *  @example col = vec4f(textSrgbToLinear(unp.rgb), unp.a * alpha); */
export const textSrgbToLinear = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    const lo = std.div(c, 12.92);
    const hi = std.pow(std.div(std.add(c, d.vec3f(0.055)), 1.055), d.vec3f(2.4));
    return std.select(hi, lo, std.le(c, d.vec3f(0.04045)));
});

/** the glyph surface's module-scope preamble: the {@link Glyph} instance struct plus the SDF decode +
 *  sRGB transfer its fs calls. @internal */
export const glyphWgsl = chunk(
    "glyphWgsl",
    [Glyph, sdfToSignedDistance, textSrgbToLinear],
    spliceNs,
);
