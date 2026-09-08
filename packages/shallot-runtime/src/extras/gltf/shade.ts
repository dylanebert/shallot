import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { litPbr, Pbr } from "../../standard/sear/core";
import type { GltfMaterial } from "./gltf";
import { MaterialData } from "./palette";

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
 * the material map-set bitmask (a mesh's {@link Mesh.variant} / the {@link materialFns} specialization
 * key) for a decoded material: one bit per present data map, `0` for a factor-only or absent material
 * (albedo-only). The bits read the same `*Image` fields the per-material palette layers derive from, so a
 * set bit and a `>= 0` palette layer always agree, the guarantee that lets the specialized helpers sample
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
 * samples out of the emitted shader without string splicing. */
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
