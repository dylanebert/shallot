import type { Binding, Surface } from "../../standard/render/core";
import { Surfaces } from "../../standard/render/core";
import { LIVE_SKIN_VS, liveTintWgsl, skinBindings, skinParamsWgsl } from "../skin/core";
import { ALBEDO_NAMES } from "./image";
import { materialPreamble } from "./shade";

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
const liveSkinBindings: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    skin: skinBindings.skin,
    materialData: { type: "storage", element: "MaterialData" },
    skinData: skinBindings.skinData,
    ...Object.fromEntries(ALBEDO_NAMES.map((n) => [n, { type: "texture-2d-array" } as Binding])),
    mr: { type: "texture-2d-array" },
    normalTex: { type: "texture-2d-array" },
    occlusion: { type: "texture-2d-array" },
    emissive: { type: "texture-2d-array" },
    albedoSamp: { type: "sampler" },
    skinParams: skinBindings.skinParams,
};

// SkinParams + the tint helper (both the substrate's) + the material map-set helpers (shadePbr /
// sampleAlbedo, specialized per material map-set — the same specialization the VAT + textured surfaces use)
const liveSkinPreamble = (variant: number) =>
    `${skinParamsWgsl()}\n${liveTintWgsl()}\n${materialPreamble(variant)}`;

// the three alpha-mode surfaces share LIVE_SKIN_VS (the substrate's palette blend) + the `shadePbr` material
// path; only the blend mode + cutout discard differ, exactly like the VAT `skin*` trio. `mid` is the folded
// `skin[eid].y` palette index; the per-instance tint comes from the header (`liveTint`), not a `color`
// binding.
const liveSkinSurfaces: Surface[] = [
    {
        name: "skin-live",
        bindings: liveSkinBindings,
        specialize: (variant) => ({ preamble: liveSkinPreamble(variant) }),
        vs: LIVE_SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let base = sampleAlbedo(mid, uv).rgb * liveTint(eid).rgb;
        col = vec4<f32>(shadePbr(mid, uv, base, normalize(worldNormal), world), 1.0);`,
    },
    {
        name: "skin-live-clip",
        blend: "clip",
        bindings: liveSkinBindings,
        specialize: (variant) => ({ preamble: liveSkinPreamble(variant) }),
        vs: LIVE_SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let tex = sampleAlbedo(mid, uv);
        let c = liveTint(eid);
        let rgb = shadePbr(mid, uv, tex.rgb * c.rgb, normalize(worldNormal), world);
        if (tex.a * c.a < materialData[mid].cutoff) { discard; }
        col = vec4<f32>(rgb, 1.0);`,
    },
    {
        name: "skin-live-blend",
        blend: "alpha",
        bindings: liveSkinBindings,
        specialize: (variant) => ({ preamble: liveSkinPreamble(variant) }),
        vs: LIVE_SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let tex = sampleAlbedo(mid, uv) * liveTint(eid);
        col = vec4<f32>(shadePbr(mid, uv, tex.rgb, normalize(worldNormal), world), tex.a);`,
    },
];

/**
 * register the three alpha-mode live-skin surfaces — opaque `skin-live` / MASK `skin-live-clip` (cutout →
 * holed shadows) / BLEND `skin-live-blend`. The runtime-posed twin of the VAT `registerSkinSurfaces`: the
 * same material `shadePbr` path + alpha split, over the live joint palette (`LiveSkin`, `extras/skin`) where
 * the VAT samples baked textures. Called by `GltfPlugin.initialize` beside the VAT surfaces.
 */
export function registerLiveSkinSurfaces(): void {
    for (const s of liveSkinSurfaces) Surfaces.register(s);
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
