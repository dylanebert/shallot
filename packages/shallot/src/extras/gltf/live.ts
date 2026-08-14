import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { State } from "../../engine";
import { unpackLdrColor, Xform, xformNormal, xformPoint } from "../../engine/utils/core";
import {
    fsCtxSchema,
    registerSurface,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "../../standard/sear/core";
import { LIVE_SKIN_VS, SkinParams } from "../skin/core";
import { MaterialData } from "./palette";
import { materialFns } from "./shade";

// The `skin-live` surface trio: the glTF PBR material path over the engine's live joint-palette substrate
// (`extras/skin`). The runtime-posed twin of the VAT skin surface (skin.ts) — same material `shadePbr` path,
// same alpha-mode split (opaque / MASK clip / BLEND); its vs blends the live joint palette each frame where
// the VAT vs samples two baked textures. Registered beside the VAT surfaces by `GltfPlugin.initialize`;
// because skinning is a VS warp, sear's prepass + shadow passes deform for free, exactly like the VAT.
//
// The substrate half — the palette buffer, its block layout, the pose-write API, and the WGSL spliced below —
// is `extras/skin`; only the material composition lives here. A producer that poses a rig by hand needs
// `SkinPlugin` alone; it needs this file only for glTF-material shading.

// the bindings every live-skin surface declares: the instancing convention (eids + transforms) + the
// substrate's three (`skin` / `skinData` / `skinParams`, skin/core) + the shared material palette + the
// material's texture arrays/sampler (shared with the textured + VAT paths). Storage count is 5
// (eids/transforms/skin/materialData/skinData) + sear's shared 5 = 10, the ceiling (gpu.md), zero headroom:
// folding `color` into the palette header (`liveTintWgsl`) is what buys the room for `skinData` versus the
// VAT skin surface's `color` binding. The texture arrays + the `skinParams` uniform are separate limits, not
// storage. Declaration order is binding order, so the substrate's three are placed individually rather than
// spread.
const liveLayout = surfaceLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    skin: { type: "storage", element: d.vec4f },
    materialData: { type: "storage", element: MaterialData },
    skinData: { type: "storage", element: d.vec4u },
    albedo0: { type: "texture-2d-array" },
    albedo1: { type: "texture-2d-array" },
    albedo2: { type: "texture-2d-array" },
    albedo3: { type: "texture-2d-array" },
    mr: { type: "texture-2d-array" },
    normalTex: { type: "texture-2d-array" },
    occlusion: { type: "texture-2d-array" },
    emissive: { type: "texture-2d-array" },
    albedoSamp: { type: "sampler" },
    skinParams: { type: "uniform", struct: SkinParams },
});
const LivePatch = vsPatchSchema();
const LiveCtx = fsCtxSchema();
// 0.12 removed `layout.bound`; the dereferenced `layout.$.x` throws outside an actual TGSL body ("Direct
// access to buffer values..."), so a raw-WGSL-string `$uses` external can't bind a per-field value —
// the whole `$` proxy rides as one external and the WGSL text's own `bound.x` dot chain defers the field
// read to resolution time (see standard/sear/pipelines.ts's `typedVaryingVs` for the same fix in full).
const liveBound = liveLayout.$;

// The dynamic vec4 lane reads are the one WGSL-bodied leaf the 4a lock names. The surface itself is still
// a typed fn: schemas own every argument/resource and `$uses` binds the raw body's free names.
const liveVs = tgpu
    .fn(
        [VsIn],
        LivePatch,
    )(/* wgsl */ `(vsIn: VsIn) -> LivePatch {
    let vidx = vsIn.vidx;
    let eid = vsIn.eid;
    var localPos = vsIn.localPos;
    let localNormal = vsIn.localNormal;
    var world = vsIn.world;
    var worldNormal = vsIn.worldNormal;
${LIVE_SKIN_VS.replace(/\bskinData\b/g, "bound.skinData")
    .replace(/\bskinParams\b/g, "bound.skinParams")
    .replace(/\bskin\b/g, "bound.skin")
    .replace("let xf = transforms[eid];", "let xf = vsIn.xform;")}
    return LivePatch(world, worldNormal, vec4f(0.0));
}`)
    .$uses({
        VsIn,
        LivePatch,
        bound: liveBound,
        Xform,
        xformPoint,
        xformNormal,
    })
    .$name("skinLiveVs");

const liveTint = tgpu
    .fn(
        [d.u32],
        d.vec4f,
    )((eid) => {
        "use gpu";
        const base = d.u32(liveLayout.$.skin[eid].x);
        return unpackLdrColor(liveLayout.$.skinData[base].x);
    })
    .$name("liveTint");

function liveFs(variant: number, mode: "opaque" | "clip" | "blend") {
    const { sampleAlbedo, shadePbr } = materialFns(liveLayout, variant);
    const clip = mode === "clip";
    const blend = mode === "blend";
    return tgpu
        .fn(
            [LiveCtx],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            const mid = d.u32(liveLayout.$.skin[ctx.eid].y);
            const tex = sampleAlbedo(mid, ctx.uv);
            const tint = liveTint(ctx.eid);
            const rgb = shadePbr(
                mid,
                ctx.uv,
                std.mul(tex.xyz, tint.xyz),
                std.normalize(ctx.worldNormal),
                ctx.world,
            );
            if (clip && tex.w * tint.w < liveLayout.$.materialData[mid].cutoff) std.discard();
            return d.vec4f(rgb, blend ? tex.w * tint.w : 1);
        })
        .$name("skinLiveFs");
}

// the three alpha-mode surfaces share LIVE_SKIN_VS (the substrate's palette blend) + the `shadePbr` material
// path; only the blend mode + cutout discard differ, exactly like the VAT `skin*` trio. `mid` is the folded
// `skin[eid].y` palette index; the per-instance tint comes from the header (`liveTint`), not a `color`
// binding.
/**
 * register the three alpha-mode live-skin surfaces — opaque `skin-live` / MASK `skin-live-clip` (cutout →
 * holed shadows) / BLEND `skin-live-blend`. The runtime-posed twin of the VAT `registerSkinSurfaces`: the
 * same material `shadePbr` path + alpha split, over the live joint palette (`LiveSkin`, `extras/skin`) where
 * the VAT samples baked textures. Called by `GltfPlugin.initialize` beside the VAT surfaces.
 */
export function registerLiveSkinSurfaces(state: State): void {
    for (const [name, blend, mode] of [
        ["skin-live", undefined, "opaque"],
        ["skin-live-clip", "clip", "clip"],
        ["skin-live-blend", "alpha", "blend"],
    ] as const) {
        registerSurface(state, {
            name,
            layout: liveLayout,
            fragmentInputs: { uv: true },
            blend,
            vs: liveVs,
            fs: liveFs(0, mode),
            specialize: (variant) => ({ vs: liveVs, fs: liveFs(variant, mode) }),
        });
    }
}

/** the live-skin surface name per glTF alphaMode — the importer routes each live instance by its
 *  material's mode, the `skinSurface` twin. */
export function liveSkinSurface(alphaMode: "OPAQUE" | "MASK" | "BLEND"): string {
    return alphaMode === "MASK"
        ? "skin-live-clip"
        : alphaMode === "BLEND"
          ? "skin-live-blend"
          : "skin-live";
}
