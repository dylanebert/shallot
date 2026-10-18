import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

const address = tgpu.fn(
    [d.i32, d.i32, d.u32],
    d.i32,
)((index, size, mode) => {
    "use gpu";
    if (mode === 0) return std.clamp(index, 0, size - 1);
    const period = std.select(size, size * 2, mode === 2);
    const wrapped = ((index % period) + period) % period;
    return std.select(wrapped, period - 1 - wrapped, wrapped >= size);
});

// A level's size from the base size, not textureDimensions(images, level): llvmpipe (Mesa 25.2.8,
// Dawn Vulkan) answers every lane of a quad with one lane's level, so lanes on a different level
// addressed past their own and loaded zero.
const levelSize = tgpu.fn(
    [d.texture2dArray(), d.i32],
    d.vec2i,
)((images, level) => {
    "use gpu";
    const base = std.textureDimensions(images);
    const shift = d.u32(level);
    return d.vec2i(d.i32(std.max(base.x >>> shift, 1)), d.i32(std.max(base.y >>> shift, 1)));
});

const texel = tgpu.fn(
    [d.texture2dArray(), d.vec2i, d.i32, d.i32, d.vec2u],
    d.vec4f,
)((images, position, layer, level, modes) => {
    "use gpu";
    const size = levelSize(images, level);
    return std.textureLoad(
        images,
        d.vec2i(address(position.x, size.x, modes.x), address(position.y, size.y, modes.y)),
        layer,
        level,
    );
});

const filtered = tgpu.fn(
    [d.texture2dArray(), d.vec2f, d.i32, d.i32, d.u32, d.vec2u],
    d.vec4f,
)((images, uv, layer, level, filter, modes) => {
    "use gpu";
    const position = std.mul(uv, d.vec2f(levelSize(images, level)));
    if (filter === 0) {
        return texel(images, d.vec2i(std.floor(position)), layer, level, modes);
    }
    const centered = std.sub(position, d.vec2f(0.5));
    const base = d.vec2i(std.floor(centered));
    const weight = std.fract(centered);
    const a = std.mix(
        texel(images, base, layer, level, modes),
        texel(images, std.add(base, d.vec2i(1, 0)), layer, level, modes),
        weight.x,
    );
    const b = std.mix(
        texel(images, std.add(base, d.vec2i(0, 1)), layer, level, modes),
        texel(images, std.add(base, d.vec2i(1, 1)), layer, level, modes),
        weight.x,
    );
    return std.mix(a, b, weight.y);
});

// Address texel taps, not just UVs: repeat's linear footprint straddles the opposite edges.
// Derivatives precede addressing so wrap boundaries cannot spuriously select coarser mips.
/**
 * @internal Shared fragment sampling for the built-in image consumers; dx/dy are unwrapped UV
 * derivatives. Level of detail is isotropic, from the larger footprint axis, as with anisotropy 1.
 */
export const sampleImage = tgpu.fn(
    [d.texture2dArray(), d.sampler(), d.texture2d(), d.vec2f, d.i32, d.vec2f, d.vec2f],
    d.vec4f,
)((images, sampler, settings, uv, layer, dx, dy) => {
    "use gpu";
    const settingsHeight = d.i32(std.textureDimensions(settings).y);
    const s = d.vec4u(
        std.round(
            std.mul(
                std.textureLoad(settings, d.vec2i(0, std.min(layer, settingsHeight - 1)), 0),
                255,
            ),
        ),
    );
    // s.w packs the min filter (bit 0) with the mipmap filter (bit 1); the shared sampler is all linear.
    if (s.x === 0 && s.y === 0 && s.z === 1 && s.w === 3) {
        return std.textureSampleGrad(images, sampler, uv, layer, dx, dy);
    }
    const dimensions = d.vec2f(std.textureDimensions(images));
    const x = std.mul(dx, dimensions);
    const y = std.mul(dy, dimensions);
    const lod = 0.5 * std.log2(std.max(std.dot(x, x), std.dot(y, y)));
    const last = d.f32(std.textureNumLevels(images) - 1);
    const filter = std.select(s.w & 1, s.z, lod <= 0);
    if (s.w >>> 1 === 0) {
        const level = d.i32(std.clamp(std.floor(lod + 0.5), 0, last));
        return filtered(images, uv, layer, level, filter, s.xy);
    }
    const clamped = std.clamp(lod, 0, last);
    const level = d.i32(std.floor(clamped));
    const finer = filtered(images, uv, layer, level, filter, s.xy);
    const coarser = filtered(
        images,
        uv,
        layer,
        d.i32(std.min(d.f32(level + 1), last)),
        filter,
        s.xy,
    );
    return std.mix(finer, coarser, std.fract(clamped));
});
