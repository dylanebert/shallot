import * as d from "typegpu/data";
import { chunk, spliceNs } from "../../engine/utils/core";
import type { GltfMaterial } from "./gltf";

// the per-material PBR palette boundary: the `MaterialData` schema + its CPU packer in one file, so the
// CPU↔GPU byte layout has a single source of truth (boundaries are where layout bugs hide). One MaterialData
// (64 B = 4 vec4) per material, indexed by the per-instance `Textured` id; the textured surfaces read it.

/**
 * one material's PBR palette entry. The `*Layer` fields are the per-slot `texture_2d_array` layer indices,
 * `-1` when the material carries no image for that slot (so they are signed); `albedoBucket` is the
 * size-bucket its baseColor lives in, which `sampleAlbedo` switches on. Four vec4-aligned rows, mixed
 * i32/u32/f32 — {@link packMaterials} derives every write offset from here.
 */
export const MaterialData = d.struct({
    layer: d.i32,
    cutoff: d.f32,
    metallic: d.f32,
    roughness: d.f32,
    mrLayer: d.i32,
    normalLayer: d.i32,
    occLayer: d.i32,
    emisLayer: d.i32,
    emissive: d.vec3f,
    normalScale: d.f32,
    occStrength: d.f32,
    albedoBucket: d.u32,
    pad1: d.f32,
    pad2: d.f32,
});

/** the per-material stride in bytes, from {@link MaterialData} — the one source of truth for the layout. */
export const MATERIAL_STRIDE = d.sizeOf(MaterialData);

/** the WGSL {@link MaterialData} struct a textured / skinned surface declares its `materialData` storage
 *  binding over. Splice at module scope. */
export const materialDataWgsl = chunk("materialDataWgsl", [MaterialData], spliceNs);

// the 4-byte-lane index of one field — the packer writes through flat i32/u32/f32 views of one buffer, so
// the schema stays the one source for where each value lands (a reordered struct moves the writes with it)
const at = (field: (m: d.Infer<typeof MaterialData>) => unknown) =>
    d.memoryLayoutOf(MaterialData, field).offset / 4;

const AT = {
    layer: at((m) => m.layer),
    cutoff: at((m) => m.cutoff),
    metallic: at((m) => m.metallic),
    roughness: at((m) => m.roughness),
    mrLayer: at((m) => m.mrLayer),
    normalLayer: at((m) => m.normalLayer),
    occLayer: at((m) => m.occLayer),
    emisLayer: at((m) => m.emisLayer),
    emissive: at((m) => m.emissive),
    normalScale: at((m) => m.normalScale),
    occStrength: at((m) => m.occStrength),
    albedoBucket: at((m) => m.albedoBucket),
} as const;

/** the per-slot per-material array layer (`-1` = the material has no image for that slot). */
export interface SlotLayers {
    albedo: Int32Array;
    mr: Int32Array;
    normal: Int32Array;
    occ: Int32Array;
    emis: Int32Array;
    /** the size-bucket each material's baseColor lives in (`sampleAlbedo` switches on it); 0 when absent. */
    albedoBucket: Uint32Array;
}

/**
 * pack the per-material PBR palette to the {@link MaterialData} byte layout, each field written through the
 * typed view matching its declared type. Pure (no device), so the struct offsets are unit-pinned. A
 * factor-only material keeps its `-1` albedo layer (like the data slots): a textured instance rides sear's
 * solid `default` and never samples it, but a skinned one always lands on the skin surface, where
 * `sampleAlbedo` reads the `-1` and returns the glTF default white (shade.ts).
 */
export function packMaterials(materials: GltfMaterial[], layers: SlotLayers): ArrayBuffer {
    const palette = new ArrayBuffer(Math.max(MATERIAL_STRIDE, materials.length * MATERIAL_STRIDE));
    const u = new Uint32Array(palette);
    const i = new Int32Array(palette);
    const f = new Float32Array(palette);
    const lanes = MATERIAL_STRIDE / 4;
    materials.forEach((m, k) => {
        const o = k * lanes;
        i[o + AT.layer] = layers.albedo[k];
        f[o + AT.cutoff] = m.cutoff;
        f[o + AT.metallic] = m.metallic;
        f[o + AT.roughness] = m.roughness;
        i[o + AT.mrLayer] = layers.mr[k];
        i[o + AT.normalLayer] = layers.normal[k];
        i[o + AT.occLayer] = layers.occ[k];
        i[o + AT.emisLayer] = layers.emis[k];
        f[o + AT.emissive] = m.emissive[0];
        f[o + AT.emissive + 1] = m.emissive[1];
        f[o + AT.emissive + 2] = m.emissive[2];
        f[o + AT.normalScale] = m.normalScale;
        f[o + AT.occStrength] = m.occStrength;
        u[o + AT.albedoBucket] = layers.albedoBucket[k];
    });
    return palette;
}
