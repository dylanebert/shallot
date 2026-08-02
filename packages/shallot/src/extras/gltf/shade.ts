import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { litPbr, Pbr } from "../../standard/sear/core";
import type { GltfMaterial } from "./gltf";
import { ALBEDO_BUCKETS, ALBEDO_NAMES } from "./image";
import { MaterialData, materialDataWgsl } from "./palette";

// baseColor lives in one of ALBEDO_BUCKETS size-bucketed arrays (image.ts); `sampleAlbedo` switches on the
// per-material `albedoBucket` to sample the right one. The switch is dynamically uniform per draw — each glTF
// mesh has one material, and the pack groups a draw by (surface, mesh), so every instance in a draw shares the
// bucket — so it's a single non-divergent branch. `textureSampleGrad` (explicit derivatives computed before
// the switch) is what lets the sample sit inside the branch: unlike `textureSample` it needs no uniform
// control flow, while still mip-sampling correctly.
//
// A material with no baseColor image carries `layer < 0` (union.ts fills -1). The textured path never routes
// such a material here — `load()` sends a factor-only material to the solid `default` surface — but a skinned
// one always lands on the skin surface, image or not, so it reaches this function. Return the glTF default
// baseColor (white) so `albedo × baseColorFactor` yields the factor, not a sample of the black absent-slot
// fallback. The guard is per-draw uniform (`layer` is), so it costs no divergence.
const SAMPLE_ALBEDO = /* wgsl */ `
fn sampleAlbedo(mid: u32, uv: vec2<f32>) -> vec4<f32> {
    let layer = materialData[mid].layer;
    // derivatives before the guard: dpdx/dpdy need uniform control flow, so they can't sit after a branch
    let ddx = dpdx(uv);
    let ddy = dpdy(uv);
    if (layer < 0) { return vec4<f32>(1.0); }
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
 * key) for a decoded material: one bit per present data map, `0` for a factor-only or absent material
 * (albedo-only). The bits read the same `*Image` fields the per-material palette layers derive from, so a
 * set bit and a `>= 0` palette layer always agree, the guarantee that lets the specialized preamble sample
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

type MaterialBound = {
    materialData: d.Infer<typeof MaterialData>[];
    albedo0: d.texture2dArray;
    albedo1: d.texture2dArray;
    albedo2: d.texture2dArray;
    albedo3: d.texture2dArray;
    mr: d.texture2dArray;
    normalTex: d.texture2dArray;
    occlusion: d.texture2dArray;
    emissive: d.texture2dArray;
    albedoSamp: d.sampler;
};

/** typed material helpers specialized to one mesh's map-set. The captured booleans fold absent texture
 * samples out of the emitted shader, preserving the legacy specialization without string splicing. */
export function materialFns(layout: { $: unknown }, mapset: number) {
    const b = layout.$ as MaterialBound;
    const hasNormal = (mapset & MAP_NORMAL) !== 0;
    const hasMr = (mapset & MAP_MR) !== 0;
    const hasOcc = (mapset & MAP_OCC) !== 0;
    const hasEmis = (mapset & MAP_EMIS) !== 0;

    const sampleAlbedo = tgpu
        .fn(
            [d.u32, d.vec2f],
            d.vec4f,
        )((mid, uv) => {
            "use gpu";
            const md = MaterialData(b.materialData[mid]);
            const ddx = std.dpdx(uv);
            const ddy = std.dpdy(uv);
            if (md.layer < 0) return d.vec4f(1);
            if (md.albedoBucket === d.u32(0))
                return std.textureSampleGrad(b.albedo0, b.albedoSamp, uv, md.layer, ddx, ddy);
            if (md.albedoBucket === d.u32(1))
                return std.textureSampleGrad(b.albedo1, b.albedoSamp, uv, md.layer, ddx, ddy);
            if (md.albedoBucket === d.u32(2))
                return std.textureSampleGrad(b.albedo2, b.albedoSamp, uv, md.layer, ddx, ddy);
            return std.textureSampleGrad(b.albedo3, b.albedoSamp, uv, md.layer, ddx, ddy);
        })
        .$name("sampleAlbedo");

    const shadePbr = tgpu
        .fn(
            [d.u32, d.vec2f, d.vec3f, d.vec3f, d.vec3f],
            d.vec3f,
        )((mid, uv, baseRgb, geoN, world) => {
            "use gpu";
            const md = MaterialData(b.materialData[mid]);
            let metallic = d.f32(md.metallic);
            let roughness = d.f32(md.roughness);
            if (hasMr) {
                const mrTex = std.textureSample(b.mr, b.albedoSamp, uv, md.mrLayer);
                metallic *= mrTex.z;
                roughness *= mrTex.y;
            }
            let occ = d.f32(1);
            if (hasOcc) {
                occ =
                    1 +
                    md.occStrength *
                        (std.textureSample(b.occlusion, b.albedoSamp, uv, md.occLayer).x - 1);
            }
            let emissive = d.vec3f(md.emissive);
            if (hasEmis) {
                emissive = d.vec3f(
                    std.mul(
                        emissive,
                        std.textureSample(b.emissive, b.albedoSamp, uv, md.emisLayer).xyz,
                    ),
                );
            }
            let normal = d.vec3f(geoN);
            if (hasNormal) {
                const tex = std.textureSample(b.normalTex, b.albedoSamp, uv, md.normalLayer);
                const nxy = std.mul(std.sub(std.mul(tex.xy, 2), 1), md.normalScale);
                const n = d.vec3f(nxy, std.sqrt(std.max(0, 1 - std.dot(nxy, nxy))));
                const dp1 = std.dpdx(world);
                const dp2 = std.dpdy(world);
                const duv1 = std.dpdx(uv);
                const duv2 = std.dpdy(uv);
                const dp2perp = std.cross(dp2, geoN);
                const dp1perp = std.cross(geoN, dp1);
                const tangent = std.add(std.mul(dp2perp, duv1.x), std.mul(dp1perp, duv2.x));
                const bitangent = std.add(std.mul(dp2perp, duv1.y), std.mul(dp1perp, duv2.y));
                const invmax = std.inverseSqrt(
                    std.max(std.dot(tangent, tangent), std.dot(bitangent, bitangent)),
                );
                normal = d.vec3f(
                    std.normalize(
                        std.mul(
                            d.mat3x3f(std.mul(tangent, invmax), std.mul(bitangent, invmax), geoN),
                            n,
                        ),
                    ),
                );
            }
            const pbr = Pbr({
                albedo: baseRgb,
                metallic,
                roughness,
                occlusion: occ,
                dielectric: 0.04,
            });
            return d.vec3f(std.add(litPbr(pbr, normal, world), emissive));
        })
        .$name("shadePbr");

    return { sampleAlbedo, shadePbr };
}

/**
 * the metallic-roughness shade helpers specialized to a material map-set `mapset` (a bitmask of the
 * `MAP_*` bits), spliced into a `(surface, mesh)` draw's module scope. Each helper takes the palette
 * index `mid`; a present map samples its `texture_2d_array` directly (the bitmask guarantees its
 * `*Layer >= 0`, so no `select`/`max` gate), an absent one uses the factor alone and emits **no sample**.
 * That's the win over an unconditional ubershader off-L2 (the absent sample is a real DRAM fetch on the
 * integrated floor). `perturbNormal` only reconstructs the tangent frame (Schüler's screen-space
 * cotangent, no TANGENT attribute) when the normal map is present; otherwise it returns the geometric
 * normal with no derivatives. The all-maps form (`MAP_ALL`) is the unconditional shader minus the now-dead
 * gates. Sear compiles one variant per distinct map-set a scene loads (Bevy's on-demand specialization).
 */
export function materialPreamble(mapset: number): string {
    const has = (bit: number) => (mapset & bit) !== 0;
    return /* wgsl */ `
${materialDataWgsl()}
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
