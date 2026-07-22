import type { GltfMaterial } from "./gltf";

// the per-material PBR palette boundary: the WGSL `MaterialData` struct + its CPU packer in one file, so the
// CPU↔GPU byte layout has a single source of truth (boundaries are where layout bugs hide). One MaterialData
// (64 B = 4 vec4) per material, indexed by the per-instance `Textured` id; the textured surfaces read it.

export const MATERIAL_STRIDE = 64;

export const MATERIAL_DATA_WGSL = /* wgsl */ `
struct MaterialData {
    layer: i32, cutoff: f32, metallic: f32, roughness: f32,
    mrLayer: i32, normalLayer: i32, occLayer: i32, emisLayer: i32,
    emissive: vec3<f32>, normalScale: f32,
    occStrength: f32, albedoBucket: u32, pad1: f32, pad2: f32,
}`;

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
 * pack the per-material PBR palette to the {@link MATERIAL_DATA_WGSL} byte layout. Pure (no device), so the
 * struct offsets are unit-pinned. A factor-only material keeps its `-1` albedo layer (like the data slots):
 * a textured instance rides sear's solid `default` and never samples it, but a skinned one always lands on the
 * skin surface, where `sampleAlbedo` reads the `-1` and returns the glTF default white (shade.ts).
 */
export function packMaterials(materials: GltfMaterial[], layers: SlotLayers): ArrayBuffer {
    const palette = new ArrayBuffer(Math.max(MATERIAL_STRIDE, materials.length * MATERIAL_STRIDE));
    const u = new Uint32Array(palette);
    const i = new Int32Array(palette);
    const f = new Float32Array(palette);
    materials.forEach((m, k) => {
        const o = k * 16;
        i[o] = layers.albedo[k];
        f[o + 1] = m.cutoff;
        f[o + 2] = m.metallic;
        f[o + 3] = m.roughness;
        i[o + 4] = layers.mr[k];
        i[o + 5] = layers.normal[k];
        i[o + 6] = layers.occ[k];
        i[o + 7] = layers.emis[k];
        f[o + 8] = m.emissive[0];
        f[o + 9] = m.emissive[1];
        f[o + 10] = m.emissive[2];
        f[o + 11] = m.normalScale;
        f[o + 12] = m.occStrength;
        u[o + 13] = layers.albedoBucket[k];
    });
    return palette;
}
