import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { xformMat } from "../../engine/utils";
import {
    engineLayout,
    MaterialVertexInput,
    materialFragmentContext,
    materialLayout,
    materialType,
    materialVertexOutput,
} from "../../standard/rendering";
import { screenCorner, worldCorner, yLockedCorner } from "./billboard";
import { SpriteBillboard } from "./pack";

/** One type-local material row: image, tint and fill are data, never shader registrations. */
export const SpriteMaterialInput = d
    .struct({
        offset: d.vec2f,
        size: d.vec2f,
        layer: d.u32,
        color: d.u32,
        fill: d.u32,
        billboard: d.u32,
    })
    .$name("SpriteMaterialInput");

const layout = materialLayout(SpriteMaterialInput, {
    spriteAtlas: { type: "texture-2d-array" },
    spriteSamp: { type: "sampler" },
});
const SpriteVertexOutput = materialVertexOutput();
const SpriteFragmentContext = materialFragmentContext();

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
        [d.u32, d.vec2f],
        d.f32,
    )((fill, uv) => {
        "use gpu";
        const mode = fill >>> 16;
        if (mode === 0) return d.f32(1);
        const amount = d.f32(fill & 0xffff) / 65535;
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

const spriteVertex = tgpu
    .fn(
        [MaterialVertexInput],
        SpriteVertexOutput,
    )((input) => {
        "use gpu";
        const sprite = SpriteMaterialInput(layout.$.materialParameters[input.eid]);
        const local = std.add(sprite.offset, std.mul(input.localPos.xy, sprite.size));
        const transform = xformMat(input.xform);
        let corner = worldCorner(transform, local.x, local.y);
        if (sprite.billboard === SpriteBillboard.Screen) {
            corner = screenCorner(
                transform,
                engineLayout.$.view.right.xyz,
                engineLayout.$.view.up.xyz,
                local.x,
                local.y,
            );
        } else if (sprite.billboard === SpriteBillboard.YLocked) {
            corner = yLockedCorner(
                transform,
                engineLayout.$.view.right.xyz,
                engineLayout.$.view.up.xyz,
                local.x,
                local.y,
            );
        }
        return SpriteVertexOutput({
            world: d.vec4f(corner, 1),
            worldNormal: input.worldNormal,
        } as never);
    })
    .$name("spriteVertex");

function spriteFragment(clip: boolean) {
    return tgpu
        .fn(
            [SpriteFragmentContext],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            const sprite = SpriteMaterialInput(layout.$.materialParameters[ctx.eid]);
            const uv = d.vec2f(ctx.localPos.x, 1 - ctx.localPos.y);
            const texel = std.textureSample(
                layout.$.spriteAtlas,
                layout.$.spriteSamp,
                uv,
                d.i32(sprite.layer),
            );
            const tint = std.unpack4x8unorm(sprite.color);
            const mask = spriteFillMask(sprite.fill, uv);
            const rgb = std.mul(texel.xyz, spriteSrgbToLinear(tint.xyz));
            const alpha = texel.w * tint.w * mask;
            if (clip) {
                if (alpha < 0.5) std.discard();
                return d.vec4f(rgb, 1);
            }
            return d.vec4f(rgb, alpha);
        })
        .$name(clip ? "spriteClipFragment" : "spriteAlphaFragment");
}

const defaults = {
    offset: d.vec2f(0),
    size: d.vec2f(1),
    layer: 0,
    color: 0xffffffff,
    fill: 0,
    billboard: SpriteBillboard.Screen,
};

/** Opaque/cutout sprites retain standard depth and shadow-atlas participation. */
export const SpriteMaterialType = materialType({
    name: "SpriteMaterial",
    parameters: SpriteMaterialInput,
    layout,
    fragmentInputs: { localPos: true },
    vertex: spriteVertex,
    fragment: spriteFragment(true),
    defaults,
    blend: "clip",
    depthPass: { prepass: true, shadows: true },
});

/** Alpha sprites share the same schema/shader and use the alpha route's fixed-function blend state. */
export const SpriteAlphaMaterialType = materialType({
    name: "SpriteAlphaMaterial",
    parameters: SpriteMaterialInput,
    layout,
    fragmentInputs: { localPos: true },
    vertex: spriteVertex,
    fragment: spriteFragment(false),
    defaults,
    blend: "alpha",
    depthPass: { prepass: false, shadows: false },
});
