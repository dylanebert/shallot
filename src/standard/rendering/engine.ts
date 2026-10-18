// The canonical engine substrate: the pass-invariant group-0 layout every standard pipeline binds,
// plus `lit` / `litPbr` / `clusterOf` and their four private seams, authored
// once against that layout.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { MeshQuant } from "../../core/mesh";
import { FrameGpu, ViewUniforms } from "../../core/rendering";
import { clusterCell, LightClusters } from "./cluster";
import { IndirectLightInput, indirectLight } from "./indirect";
import { distanceAttenuation, LightingGpu, PointLightGpu, spotFactor } from "./lighting";
import { brdf, brdfSphere, Pbr, pointShadowRef } from "./shade";

/**
 * the canonical engine group-0 layout: every pass-invariant binding a standard pipeline reads — frame / view /
 * lighting uniforms and two storage tables: clustered lights and mesh dequantization. `vertices` is deliberately absent: it's pass-variant (color binds the 16 B main stream,
 * prepass/shadow the 8 B position stream) and moves into the material-type group (2). Pinned at group 0.
 */
export const engineLayout = tgpu
    .bindGroupLayout({
        frame: { uniform: FrameGpu, visibility: ["vertex", "fragment"] },
        view: { uniform: ViewUniforms, visibility: ["vertex", "fragment"] },
        lighting: { uniform: LightingGpu, visibility: ["vertex", "fragment"] },
        pointLights: {
            storage: LightClusters,
            access: "readonly",
            visibility: ["vertex", "fragment"],
        },
        meshQuant: {
            storage: d.arrayOf(MeshQuant),
            access: "readonly",
            visibility: ["vertex", "fragment"],
        },
    })
    .$idx(0);

/** the sun-shadow seam: the sun's visibility at this fragment (1.0 = unshadowed), multiplied into the
 *  sun term only. Filled by a real-reference color-pass caller via `sampleSunShadow`; the default
 *  (vertex stage, shadowless frames) is fully lit. */
export const sunVisibility = tgpu.privateVar(d.f32, 1);
/** the point-light seam: the fragment's world position, read by {@link litPbr}'s
 *  clustered loop. Zero in the vs / prepass entries, where the point term is never enabled. */
export const fragWorld = tgpu.privateVar(d.vec3f, d.vec3f(0));
/** the point-light seam: `@builtin(position)`, read by {@link clusterOf} to map the fragment to its
 *  froxel cluster. */
export const fragCoord = tgpu.privateVar(d.vec4f, d.vec4f(0));
/** the point-light seam's enable flag: 0 in the vs / prepass entries (no point contribution), 1 in the
 *  color fs. */
export const pointScale = tgpu.privateVar(d.f32, 0);

const pointShadowOf = pointShadowRef();

// Vertex shading statically reaches the point receiver even when pointScale is zero.
// Depth pipelines need a fully-lit stub to avoid binding or sampling the atlas they write.
// Keep the real receiver's pointShadowOf name and signature so the emitted call site is unchanged.
export const pointShadowStub = tgpu
    .fn(
        [PointLightGpu, d.vec3f, d.vec3f, d.u32],
        d.f32,
    )((_light, _normal, _fragWorld, _method) => {
        "use gpu";
        return 1;
    })
    .$name("pointShadowOf");

/** the receiver seam the scaffold calls through: defaults to the real {@link pointShadowRef}, which
 *  color pipelines sample; a depth pipeline binds {@link pointShadowStub} instead
 *  (`root.with(pointShadowSlot, pointShadowStub)`). */
export const pointShadowSlot = tgpu.slot(pointShadowOf).$name("pointShadowSlot");

/** the fragment's slot-major cluster index ({@link clusterCell}). ViewUniforms depth recovers from the position
 *  builtin: perspective clip.w is the view depth (`fragCoord.w = 1/clip.w`); orthographic depth is
 *  linear in `fragCoord.z`.
 */
export const clusterOf = tgpu.fn(
    [],
    d.u32,
)(() => {
    "use gpu";
    const near = engineLayout.$.view.projection.x;
    const far = engineLayout.$.view.projection.y;
    let viewZ = 1 / fragCoord.$.w;
    if (engineLayout.$.view.projection.z < 0.5) {
        viewZ = near + fragCoord.$.z * (far - near);
    }
    return clusterCell(
        fragCoord.$.x / engineLayout.$.view.resolution.x,
        fragCoord.$.y / engineLayout.$.view.resolution.y,
        viewZ,
        near,
        far,
        d.u32(engineLayout.$.view.projection.w),
    );
});

/** per-pixel (or per-vertex) metallic-roughness shading.
 *  Reads the sun-shadow / point-cluster seam: {@link sunVisibility} and {@link pointScale} /
 *  {@link fragWorld} are fs-scaffold privates, so a vs-side call gets neither point lights nor sun
 *  shadows. At metallic 0 / dielectric 0 specular vanishes and the response is Lambert at any roughness.
 */
export const litPbr = tgpu.fn(
    [Pbr, d.vec3f, d.vec3f, d.vec3f],
    d.vec3f,
)((s, normal, world, V) => {
    "use gpu";
    const exposure = engineLayout.$.view.exposure;
    const indirect = indirectLight(
        IndirectLightInput({
            worldPosition: world,
            normal,
            view: V,
            materialOcclusion: s.occlusion,
            ambientRadiance: engineLayout.$.view.ambientColor.rgb,
        }),
    );
    let radiance = d.vec3f(std.mul(std.mul(indirect, exposure), s.albedo));
    for (let i = d.u32(0); i < engineLayout.$.lighting.directionalCount; i++) {
        const light = engineLayout.$.lighting.directionalLights[i];
        let visibility = d.f32(1);
        if (light.params.y > 0) visibility = sunVisibility.$;
        radiance = d.vec3f(
            std.add(
                radiance,
                std.mul(
                    std.mul(light.color.rgb, light.params.x * exposure * visibility),
                    brdf(s, normal, V, std.neg(light.direction.xyz)),
                ),
            ),
        );
    }
    if (pointScale.$ !== 0) {
        const entry = engineLayout.$.pointLights.grid[clusterOf()];
        for (let i = d.u32(0); i < entry.y; i++) {
            const light = PointLightGpu(
                engineLayout.$.pointLights.lights.lights[
                    engineLayout.$.pointLights.indices[entry.x + i]
                ],
            );
            const toLight = std.sub(light.posRange.xyz, fragWorld.$);
            const distSq = std.dot(toLight, toLight);
            const radiusSq = light.params.x * light.params.x;
            const dist = std.sqrt(std.max(distSq, 1e-8));
            const L = std.div(toLight, dist);
            const f =
                distanceAttenuation(distSq, light.posRange.w, radiusSq) *
                spotFactor(light, L) *
                pointShadowSlot.$(
                    light,
                    normal,
                    fragWorld.$,
                    d.u32(engineLayout.$.view.shadowFilteringMethod),
                );
            // Preserve product order to retain f32 rounding.
            radiance = d.vec3f(
                std.add(
                    radiance,
                    std.mul(
                        std.mul(light.color.rgb, f * exposure),
                        brdfSphere(s, normal, V, L, dist, light.params.x),
                    ),
                ),
            );
        }
    }
    return radiance;
});

/** Camera-exposed linear colour of a diffuse surface under ambient, sun and clustered point/spot lights,
 *  before fog: StandardMaterial's response at metallic 0, which has no specular. Use it in a custom
 *  material's fragment to light its own base colour; for metallic, roughness, occlusion or emissive,
 *  use `applyPbrLighting` over `pbrInputFromMaterial`. `normal` is world-space and unit length. */
export const lit = tgpu.fn(
    [d.vec3f, d.vec3f],
    d.vec3f,
)((baseColor, normal) => {
    "use gpu";
    const world = fragWorld.$;
    const V = std.normalize(std.sub(engineLayout.$.view.eye.xyz, world));
    return litPbr(
        Pbr({ albedo: baseColor, metallic: 0, roughness: 1, occlusion: 1, dielectric: 0 }),
        normal,
        world,
        V,
    );
});
