import type { GltfMaterial } from "./gltf";
import { ALBEDO_BUCKETS, ALBEDO_NAMES } from "./image";
import { MATERIAL_DATA_WGSL } from "./palette";

// baseColor lives in one of ALBEDO_BUCKETS size-bucketed arrays (image.ts); `sampleAlbedo` switches on the
// per-material `albedoBucket` to sample the right one. The switch is dynamically uniform per draw — each glTF
// mesh has one material, and the pack groups a draw by (surface, mesh), so every instance in a draw shares the
// bucket — so it's a single non-divergent branch. `textureSampleGrad` (explicit derivatives computed before
// the switch) is what lets the sample sit inside the branch: unlike `textureSample` it needs no uniform
// control flow, while still mip-sampling correctly.
const SAMPLE_ALBEDO = /* wgsl */ `
fn sampleAlbedo(mid: u32, uv: vec2<f32>) -> vec4<f32> {
    let layer = i32(materialData[mid].layer);
    let ddx = dpdx(uv);
    let ddy = dpdy(uv);
    switch materialData[mid].albedoBucket {
${ALBEDO_NAMES.map(
    (name, b) =>
        `        ${b === ALBEDO_BUCKETS - 1 ? "default" : `case ${b}u`}: { return textureSampleGrad(${name}, albedoSamp, uv, layer, ddx, ddy); }`,
).join("\n")}
    }
}`;

// the four optional data maps a textured material's map-set is composed of — the compile-time
// specialization key (Bevy's `StandardMaterialKey` bitflags / the `#ifdef USE_*MAP` idiom). baseColor is
// not a bit: `load()` only routes a material to a textured surface when it carries one, so albedo is always
// present. These four gate the throwaway samples a sparse-map material would otherwise pay. The bitmask is
// constant per registered mesh (a mesh is one glTF primitive = one material), so it specializes the
// `(surface, mesh)` draw's pipeline — no per-instance branch, no new pack axis.
export const MAP_NORMAL = 1;
export const MAP_MR = 2;
export const MAP_OCC = 4;
export const MAP_EMIS = 8;
export const MAP_ALL = MAP_NORMAL | MAP_MR | MAP_OCC | MAP_EMIS;

/**
 * the material map-set bitmask (a mesh's {@link Mesh.variant} / the {@link materialPreamble} specialization
 * key) for a decoded material — one bit per present data map, `0` for a factor-only or absent material
 * (albedo-only). The bits read the same `*Image` fields the per-material palette layers derive from, so a
 * set bit and a `>= 0` palette layer always agree — the guarantee that lets the specialized preamble sample
 * a present map with no `*Layer >= 0` gate.
 */
export function mapSet(m: GltfMaterial | undefined): number {
    if (!m) return 0;
    return (
        (m.normalImage !== undefined ? MAP_NORMAL : 0) |
        (m.mrImage !== undefined ? MAP_MR : 0) |
        (m.occImage !== undefined ? MAP_OCC : 0) |
        (m.emissiveImage !== undefined ? MAP_EMIS : 0)
    );
}

/**
 * the metallic-roughness shade helpers specialized to a material map-set `mapset` (a bitmask of the
 * `MAP_*` bits) — spliced into a `(surface, mesh)` draw's module scope. Each helper takes the palette
 * index `mid`; a present map samples its `texture_2d_array` directly (the bitmask guarantees its
 * `*Layer >= 0`, so no `select`/`max` gate), an absent one uses the factor alone and emits **no sample**
 * — that's the win over an unconditional ubershader off-L2 (the absent sample is a real DRAM fetch on the
 * integrated floor). `perturbNormal` only reconstructs the tangent frame (Schüler's screen-space
 * cotangent — no TANGENT attribute) when the normal map is present; otherwise it returns the geometric
 * normal with no derivatives. The all-maps form (`MAP_ALL`) is the unconditional shader minus the now-dead
 * gates. Sear compiles one variant per distinct map-set a scene loads (Bevy's on-demand specialization).
 */
export function materialPreamble(mapset: number): string {
    const has = (bit: number) => (mapset & bit) !== 0;
    return /* wgsl */ `
${MATERIAL_DATA_WGSL}
${SAMPLE_ALBEDO}

fn pbrOf(mid: u32, uv: vec2<f32>, baseRgb: vec3<f32>) -> Pbr {
    let md = materialData[mid];
${
    has(MAP_MR)
        ? `    let mrTex = textureSample(mr, albedoSamp, uv, md.mrLayer);
    let metallic = md.metallic * mrTex.b;
    let roughness = md.roughness * mrTex.g;`
        : `    let metallic = md.metallic;
    let roughness = md.roughness;`
}
${
    has(MAP_OCC)
        ? `    let occ = 1.0 + md.occStrength * (textureSample(occlusion, albedoSamp, uv, md.occLayer).r - 1.0);`
        : `    let occ = 1.0;`
}
    return Pbr(baseRgb, metallic, roughness, occ, 0.04);
}

fn emissiveOf(mid: u32, uv: vec2<f32>) -> vec3<f32> {
    let md = materialData[mid];
${
    has(MAP_EMIS)
        ? `    return md.emissive * textureSample(emissive, albedoSamp, uv, md.emisLayer).rgb;`
        : `    return md.emissive;`
}
}

fn perturbNormal(mid: u32, N: vec3<f32>, world: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
${
    has(MAP_NORMAL)
        ? `    let md = materialData[mid];
    let tex = textureSample(normalTex, albedoSamp, uv, md.normalLayer);
    // reconstruct Z from XY (tangent-space normals have z > 0): a BC5 two-channel map stores only RG, and a
    // unit normal makes this identical to a stored Z for the RGBA path too — one path for both encodings
    let nxy = (tex.xy * 2.0 - 1.0) * md.normalScale;
    let n = vec3<f32>(nxy, sqrt(max(0.0, 1.0 - dot(nxy, nxy))));
    let dp1 = dpdx(world); let dp2 = dpdy(world);
    let duv1 = dpdx(uv); let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, N);
    let dp1perp = cross(N, dp1);
    let T = dp2perp * duv1.x + dp1perp * duv2.x;
    let B = dp2perp * duv1.y + dp1perp * duv2.y;
    let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
    return normalize(mat3x3<f32>(T * invmax, B * invmax, N) * n);`
        : `    return N;`
}
}

fn shadePbr(mid: u32, uv: vec2<f32>, baseRgb: vec3<f32>, geoN: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let n = perturbNormal(mid, geoN, world, uv);
    return litPbr(pbrOf(mid, uv, baseRgb), n, world) + emissiveOf(mid, uv);
}
`;
}
