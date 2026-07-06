import { Compute } from "../../engine";
import type { Ktx2Image } from "./basis";
import { ALBEDO_NAMES, albedoSampler, fallback1x1 } from "./image";
import { MATERIAL_STRIDE } from "./palette";

// The glTF texture boundary — the decoded (deviceless) + assembled (GPU) texture types and the publish/
// fallback primitives, in one place so the decode (index.ts), the union assembly (union.ts), and the plugin
// lifecycle share them without a circular import. Decoded types are the union's input; assembled types are
// what gets published under the global binding names a textured/skin surface reads.

// the data-map texture-array names the textured + skin surfaces bind. The slot keys double as the
// `Compute.textures` names + the binding names the shade helpers (shade.ts) sample. baseColor rides the
// per-size-bucket set ALBEDO_NAMES; these are the four single-array data-map slots.
export const DATA_NAMES = ["mr", "normalTex", "occlusion", "emissive"] as const;
export type DataSlots = Record<(typeof DATA_NAMES)[number], GPUTexture>;

// the per-data-slot format: color slots decode sRGB→linear in hardware, data slots stay linear
export const DATA_FORMAT: Record<(typeof DATA_NAMES)[number], GPUTextureFormat> = {
    mr: "rgba8unorm",
    normalTex: "rgba8unorm",
    occlusion: "rgba8unorm",
    emissive: "rgba8unorm-srgb",
};

/** a decoded texture image, deviceless: shared by baseColor (union.ts buckets these by size) and the data
 *  maps. PNG/JPEG carry an `ImageBitmap` (resized into a shared bitmap bucket/array on upload, so any size
 *  collapses there); KTX2 carry the transcoded compressed block + the source bytes (the union re-transcodes
 *  to RGBA only when sizes don't match, the spill / warned last resort). */
export type DecodedImage =
    | { kind: "bitmap"; bitmap: ImageBitmap }
    | { kind: "compressed"; image: Ktx2Image; bytes: Uint8Array };

/** one data-map slot's decoded layers + the per-material layer index (`-1` = the material has no image for
 *  the slot). A KTX2 source transcodes to the role's block format (normal → BC5, AO → BC4, mr/emissive → BC7
 *  on a BC device); the union concatenates every asset's layers into one array per slot, taking the compressed
 *  path when every layer is a uniform-size block and falling back to one RGBA array otherwise. */
export interface DecodedMap {
    images: DecodedImage[];
    layer: Int32Array;
}

/** the decoded texture payload: every image decoded to its deviceless GPU-ready form + the per-material
 *  references the union rebases into the shared arrays + palette. Nothing uploaded, nothing bucketed (the
 *  bucketing is a union-level decision, made at assembly across the active set). */
export interface DecodedTextures {
    // the used baseColor images (deduped), in `used`-order; `albedoRef[material]` indexes into this
    albedo: DecodedImage[];
    // per material → albedo image index (-1 = factor-only / no baseColor texture)
    albedoRef: Int32Array;
    // each data-map slot's used images + per-material layer, keyed by DATA_NAMES (one source of truth)
    maps: Record<(typeof DATA_NAMES)[number], DecodedMap>;
    // whether any material carries a real texture (false → the scene stays on sear's solid default)
    textured: boolean;
}

/** the assembled (uploaded) texture GPU resources: the size-bucketed baseColor arrays (padded to
 *  ALBEDO_NAMES so every `albedo{b}` binding resolves), the four data-map arrays, one shared sampler, and the
 *  per-material palette. Built by {@link beginUnion} (union.ts) per active-set + held by the asset cache;
 *  {@link publishTextures} points the surfaces at it. */
export interface AssembledTextures {
    albedo: GPUTexture[];
    data: DataSlots;
    sampler: GPUSampler;
    palette: GPUBuffer;
}

/** the GPU resources behind an assembled texture set, for the deferred free (the union's albedo arrays + data
 *  arrays + palette buffer): invalidate / clearGltfCache destroy these behind the submit fence. */
export function textureResources(set: AssembledTextures): (GPUTexture | GPUBuffer)[] {
    const res: (GPUTexture | GPUBuffer)[] = [...set.albedo, set.palette];
    for (const name of DATA_NAMES) res.push(set.data[name]);
    return res;
}

/**
 * point the textured + skin surfaces' bindings at an assembled texture set: a pointer-publish, no free
 * (the asset cache / union memo owns the real set; the build-scoped fallback is freed in dispose). `albedo`
 * is padded to ALBEDO_NAMES, so an unused bucket's binding still resolves.
 */
export function publishTextures(set: AssembledTextures): void {
    set.albedo.forEach((tex, b) => {
        Compute.textures.set(ALBEDO_NAMES[b], tex);
    });
    for (const name of DATA_NAMES) Compute.textures.set(name, set.data[name]);
    Compute.samplers.set("albedoSamp", set.sampler);
    Compute.buffers.set("materialData", set.palette);
}

// the build-scoped 1×1 fallback texture set, published at warm so the textured surfaces' draws resolve (and
// no-op, since no entity uses them yet) before a load resolves — no startup "binding not published" warning,
// the same fallback shape sear's shadow map uses. A real load re-publishes the cache-owned union over it,
// and dispose frees it.
let _fallback: AssembledTextures | null = null;

/**
 * build + publish the build-scoped 1×1 fallback set (every albedo bucket + data slot a 1×1, an empty
 * palette) so the textured draws bind cleanly before a load resolves. Tracked so dispose frees it; a real
 * load re-publishes the cache-owned union over it.
 */
export function fallbackTextures(device: GPUDevice): void {
    const albedo = ALBEDO_NAMES.map(() => fallback1x1(device, "rgba8unorm-srgb"));
    const data = Object.fromEntries(
        DATA_NAMES.map((name) => [name, fallback1x1(device, DATA_FORMAT[name])]),
    ) as DataSlots;
    const palette = device.createBuffer({
        label: "gltf-material-data-fallback",
        size: MATERIAL_STRIDE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _fallback = { albedo, data, sampler: albedoSampler(device), palette };
    publishTextures(_fallback);
}

/** free the build-scoped fallback set (GltfPlugin.dispose). The cache-owned union arrays + palette survive,
 *  freed only by invalidate / clearGltfCache. */
export function disposeTextureFallbacks(): void {
    if (!_fallback) return;
    for (const tex of _fallback.albedo) tex.destroy();
    for (const name of DATA_NAMES) _fallback.data[name].destroy();
    _fallback.palette.destroy();
    _fallback = null;
}
