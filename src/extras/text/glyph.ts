import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { SDF_EXPONENT } from "./sdf";

/** Shared CPU/GPU instance layout; staging stride and field offsets derive from this schema.
 * Positions and size are local, before the entity transform; color is packed sRGBA. */
export const Glyph = d.struct({
    pos: d.vec3f,
    eid: d.u32,
    uvRect: d.vec4f,
    size: d.vec2f,
    color: d.u32,
    billboard: d.u32,
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
    billboard: at(Glyph, (g) => g.billboard),
} as const;

/**
 * decode a single-channel SDF texel to a world-space signed distance — negative inside the glyph,
 * positive outside, 0 exactly on the edge. The atlas stores the falloff exponent-encoded and folded
 * about 0.5 (`sdf > 0.5` is inside), so the decode unfolds it, takes the {@link SDF_EXPONENT}-th root,
 * and rescales by the glyph's largest world dimension. Valve "Improved Alpha-Tested Magnification".
 */
export const sdfToSignedDistance = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((sdf, maxDimension) => {
    "use gpu";
    const a = std.select(sdf, 1 - sdf, sdf > 0.5);
    const absDist = (1 - std.pow(2 * a, 1 / SDF_EXPONENT)) * maxDimension;
    // Explicit f32 values avoid implicit integer conversions in emitted WGSL.
    return absDist * std.select(d.f32(1), d.f32(-1), sdf > 0.5);
});

/** Decodes packed sRGB glyph colors to linear RGB before blending. */
export const textSrgbToLinear = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    const lo = std.div(c, 12.92);
    const hi = std.pow(std.div(std.add(c, d.vec3f(0.055)), 1.055), d.vec3f(2.4));
    return std.select(hi, lo, std.le(c, d.vec3f(0.04045)));
});
