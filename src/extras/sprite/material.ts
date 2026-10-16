import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Billboard, billboardCorner, xformMat } from "../../core/rendering";
import type { World } from "../../engine";
import { engineLayout, materialLayout, materialType } from "../../standard/rendering";
import { createSpriteSource } from "./table";

/** One type-local material row: image, tint and fill are data, never shader registrations. */
export const SpriteMaterialInput = d
    .struct({
        size: d.vec2f,
        anchor: d.vec2f,
        alphaMode: d.vec2f,
        image: d.u32,
        color: d.f32,
        opacity: d.f32,
        fillAmount: d.f32,
        fillMethod: d.u32,
        billboard: d.u32,
        flags: d.u32,
    })
    .$name("SpriteMaterialInput");

/** @internal Local lower-left corner shared by vertex placement and conservative bounds. */
export const spriteOffset = tgpu.fn(
    [SpriteMaterialInput],
    d.vec2f,
)((sprite) => {
    "use gpu";
    let anchor = d.vec2f(0.5);
    if (sprite.flags !== 0) anchor = std.add(sprite.anchor, d.vec2f(0.5));
    return std.mul(sprite.size, std.neg(anchor));
});

/** @internal Sprite owns this World's parameter rows for exactly each entity's Sprite membership. */
export const SpriteSource = {
    create: (world: World) => createSpriteSource(world, SpriteMaterialInput, spriteOffset),
};

const layout = materialLayout(SpriteMaterialInput, {
    images: { type: "texture-2d-array" },
    imageSampler: { type: "sampler" },
});

const spriteSrgbToLinear = tgpu
    .fn(
        [d.vec3f],
        d.vec3f,
    )((c) => {
        "use gpu";
        const lo = std.div(c, 12.92);
        const hi = std.pow(std.div(std.add(c, d.vec3f(0.055)), 1.055), d.vec3f(2.4));
        return std.select(hi, lo, std.le(c, d.vec3f(0.04045)));
    })
    .$name("spriteSrgbToLinear");

/** Fill mask: radial clockwise from 12 o'clock, vertical bottom-up, horizontal left-to-right. */
const spriteFillMask = tgpu
    .fn(
        [d.f32, d.u32, d.vec2f],
        d.f32,
    )((fillAmount, fillMethod, uv) => {
        "use gpu";
        const mode = fillMethod & 0xffff;
        if (mode === 0) return d.f32(1);
        const amount = std.floor(std.clamp(fillAmount, 0, 1) * 65535 + 0.5) / 65535;
        let t = uv.x;
        if (mode === 1) {
            const delta = std.sub(uv, d.vec2f(0.5, 0.5));
            t = std.fract(std.atan2(delta.x, -delta.y) / 6.283185307179586);
        } else if (mode === 2) {
            t = 1 - uv.y;
        }
        return std.select(d.f32(0), d.f32(1), t <= amount);
    })
    .$name("spriteFillMask");

/** Sprite parameters occupy one table; each row supplies its own AlphaMode. */
export const SpriteMaterialType = materialType({
    name: "SpriteMaterial",
    layout,
    source: SpriteSource,
    fragmentInputs: { localPos: true },
    defaults: {
        size: d.vec2f(1),
        anchor: d.vec2f(0),
        alphaMode: d.vec2f(1, 0.5),
        image: 0,
        color: 0xffffff,
        opacity: 1,
        fillAmount: 1,
        fillMethod: 0,
        billboard: Billboard.Screen,
        flags: 0,
    },
    depthPass: { prepass: true, shadows: true },
    vertex: (input) => {
        "use gpu";
        const sprite = input.parameters;
        const local = std.add(spriteOffset(sprite), std.mul(input.localPos.xy, sprite.size));
        const transform = xformMat(input.xform);
        const corner = billboardCorner(
            std.min(sprite.billboard, 2),
            transform,
            engineLayout.$.view.right.xyz,
            engineLayout.$.view.up.xyz,
            local.x,
            local.y,
        );
        return {
            world: d.vec4f(corner, 1),
            worldNormal: input.worldNormal,
        };
    },
    fragment: (ctx) => {
        "use gpu";
        const sprite = ctx.parameters;
        const uv = d.vec2f(ctx.localPos.x, 1 - ctx.localPos.y);
        const texel = std.textureSample(
            layout.$.images,
            layout.$.imageSampler,
            uv,
            d.i32(sprite.image),
        );
        const hex = d.u32(sprite.color);
        const tint = d.vec4f(
            d.f32((hex >>> 16) & 255) / 255,
            d.f32((hex >>> 8) & 255) / 255,
            d.f32(hex & 255) / 255,
            std.floor(std.clamp(sprite.opacity, 0, 1) * 255 + 0.5) / 255,
        );
        const mask = spriteFillMask(sprite.fillAmount, sprite.fillMethod, uv);
        const rgb = std.mul(texel.xyz, spriteSrgbToLinear(tint.xyz));
        const alpha = texel.w * tint.w * mask;
        return d.vec4f(rgb, alpha);
    },
});
