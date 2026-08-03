// The typed sprite surface: six per-bucket registrations — (screen | y | world) billboard × (clip |
// alpha) blend — sharing one `surfaceLayout` (the gltf-trio / skin shape). Sprite adopts the
// `eids`+`transforms` instancing convention (its own layout declares both), so `VsIn.eid`/`VsIn.xform`
// replace the hand-rolled `transforms[spriteData[iid].eid]` lookup and the engine's instanced
// `tag = eid` default applies for free — the authored tag line the string contract needed is gone.
// `spriteData` is itself eid-indexed (packed at `eid * SPRITE_FLOATS`, see pack.ts), not slot-indexed:
// the point/cascade shadow atlas re-gathers casters mesh-major across combos and preserves only
// `eid` in its own `eids` lane, so `iid` is a re-gather index in the shadow pass, not sprite's slot.
// Both the vs and the fs read `spriteData[eid]` off the engine's own `VsIn.eid`/`ctx.eid` — no custom
// varying crosses the vs→fs seam, since `eid` is already a built-in interstage for an instanced
// surface. The billboard corner math lives in billboard.ts (the executable spec the vs calls
// directly, so a test exercising it pins the real production code, not a hand-derived mirror).

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Xform, xformMat } from "../../engine/utils/core";
import {
    engineLayout,
    fsCtxSchema,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "../../standard/sear/core";
import { screenCorner, worldCorner, yLockedCorner } from "./billboard";

const VARIANTS = ["screen", "y", "world"] as const;

export const surfaceName = (bucket: number): string =>
    `sprite-${VARIANTS[bucket >> 1]}${bucket & 1 ? "-alpha" : ""}`;

/** one sprite instance: quad-local offset + size, the owning eid, the array layer, a packed sRGBA tint,
 *  and the packed fill (unorm16 amount | mode << 16) — 32 bytes / two vec4 reads. `pack.ts`'s
 *  `SPRITE_BYTES`/`SPRITE_FLOATS` derive from this schema's own size, which is what its offset
 *  arithmetic (`eid * SPRITE_FLOATS`) rests on. The `eid` field itself is dead — neither the vs nor
 *  the fs reads it, since both already have the real eid from `VsIn.eid`/`ctx.eid` (the surface's own
 *  `eids` binding, the instancing convention) — kept in place rather than relayouting the packed
 *  buffer for one word. */
export const SpriteData = d
    .struct({
        offset: d.vec2f,
        size: d.vec2f,
        eid: d.u32,
        layer: d.u32,
        color: d.u32,
        fill: d.u32,
    })
    .$name("SpriteData");

const layout = surfaceLayout({
    spriteData: { type: "storage", element: SpriteData },
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    spriteAtlas: { type: "texture-2d-array" },
    spriteSamp: { type: "sampler" },
});

/** the sRGB→linear transfer function: the packed tint is sRGB-encoded and the color target is linear
 *  (texture rgb is already linear — the array is rgba8unorm-srgb, hardware-decoded on sample). */
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

/** fill mask from the packed (unorm16 amount | mode << 16) word: 1 inside the leading fraction of the
 *  image, 0 past it. Radial sweeps clockwise from 12 o'clock; vertical fills bottom-up (uv.y is
 *  image-down); horizontal left-to-right. Mode 0 is unfilled (always 1). */
const spriteFillMask = tgpu
    .fn(
        [d.u32, d.vec2f],
        d.f32,
    )((fill, uv) => {
        "use gpu";
        const mode = fill >> 16;
        if (mode === 0) return d.f32(1);
        const amount = d.f32(fill & 0xffff) / 65535;
        let t = uv.x;
        if (mode === 1) {
            const dd = std.sub(uv, d.vec2f(0.5, 0.5));
            t = std.fract(std.atan2(dd.x, -dd.y) / 6.283185307179586);
        } else if (mode === 2) {
            t = 1 - uv.y;
        }
        return std.select(d.f32(0), d.f32(1), t <= amount);
    })
    .$name("spriteFillMask");

const SpritePatch = vsPatchSchema();
const SpriteCtx = fsCtxSchema();

// per-billboard vertex chunk. `t` is the reconstructed instance matrix — `xformMat`, not `xformPoint`,
// because the billboard corner math needs the columns (camera-basis substitution + per-axis scale),
// not a single transformed point. `variant` is a captured JS string (compile-time), so the branch below
// folds away at trace time — one generated fn per bucket, never a runtime dispatch.
function spriteVs(variant: (typeof VARIANTS)[number]) {
    return tgpu
        .fn(
            [VsIn],
            SpritePatch,
        )((vsIn) => {
            "use gpu";
            const s = SpriteData(layout.$.spriteData[vsIn.eid]);
            const lp = std.add(s.offset, std.mul(vsIn.localPos.xy, s.size));
            const t = xformMat(vsIn.xform);
            let corner = d.vec3f(0);
            if (variant === "screen") {
                corner = screenCorner(
                    t,
                    engineLayout.$.view.right.xyz,
                    engineLayout.$.view.up.xyz,
                    lp.x,
                    lp.y,
                );
            } else if (variant === "y") {
                corner = yLockedCorner(
                    t,
                    engineLayout.$.view.right.xyz,
                    engineLayout.$.view.up.xyz,
                    lp.x,
                    lp.y,
                );
            } else {
                corner = worldCorner(t, lp.x, lp.y);
            }
            return SpritePatch({
                world: d.vec4f(corner, 1),
                worldNormal: vsIn.worldNormal,
                clip: d.vec4f(0),
            } as never);
        })
        .$name(`sprite${variant}Vs`);
}

// unlit icon shading: texture × sRGB-decoded tint, masked by the per-instance fill. Clip discards below
// the 0.5 cutout (opacity shrinks the cutout); alpha has no discard. The uv override the string contract
// authored in the vs (`uv = (localPos.x, 1 - localPos.y)`) needs no varying — it's a pure function of
// `ctx.localPos`, which the typed ctx already carries.
export function spriteFs(alpha: boolean) {
    return tgpu
        .fn(
            [SpriteCtx],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            const s = SpriteData(layout.$.spriteData[ctx.eid]);
            const uv = d.vec2f(ctx.localPos.x, 1 - ctx.localPos.y);
            const tex = std.textureSample(
                layout.$.spriteAtlas,
                layout.$.spriteSamp,
                uv,
                d.i32(s.layer),
            );
            const unp = std.unpack4x8unorm(s.color);
            const mask = spriteFillMask(s.fill, uv);
            const rgb = std.mul(tex.xyz, spriteSrgbToLinear(unp.xyz));
            // `alpha` is a captured JS boolean (the factory's own argument, not a GPU value) — the
            // if/else folds to whichever arm at build time (the outline `maskFragment` precedent), so
            // the clip variant never carries the alpha blend and the alpha variant never carries the
            // discard
            if (alpha) {
                return d.vec4f(rgb, tex.w * unp.w * mask);
            } else {
                if (tex.w * unp.w * mask < 0.5) std.discard();
                return d.vec4f(rgb, 1);
            }
        })
        .$name(`sprite${alpha ? "Alpha" : "Clip"}Fs`);
}

export function spriteSurface(bucket: number) {
    const variant = VARIANTS[bucket >> 1];
    const alpha = (bucket & 1) === 1;
    return {
        name: surfaceName(bucket),
        layout,
        fragmentInputs: { localPos: true } as const,
        blend: alpha ? ("alpha" as const) : ("clip" as const),
        vs: spriteVs(variant),
        fs: spriteFs(alpha),
    };
}
