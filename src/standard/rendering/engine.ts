// The canonical engine substrate: the pass-invariant group-0 layout every standard pipeline binds,
// plus `lit` / `litPbr` / `lightFactor` / `pointFactor` / `clusterOf` and their four private seams, authored
// once against that layout.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { FrameGpu, ViewUniforms } from "../../core/rendering";
import { MeshQuant } from "../../engine/utils";
import { clusterCell, LightClusters } from "./cluster";
import { IndirectLightInput, indirectLight } from "./indirect";
import { distanceAttenuation, LightingGpu, PointLightGpu, spotFactor } from "./lighting";
import { brdf, brdfSphere, halfLambert, Pbr, pointShadowRef } from "./shade";

const PI = Math.PI;

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
/** the point-light seam: the fragment's world position, read by {@link pointFactor} / {@link litPbr}'s
 *  clustered loop. Zero in the vs / prepass entries, where the point term is never enabled. */
export const fragWorld = tgpu.privateVar(d.vec3f, d.vec3f(0));
/** the point-light seam: `@builtin(position)`, read by {@link clusterOf} to map the fragment to its
 *  froxel cluster. */
export const fragCoord = tgpu.privateVar(d.vec4f, d.vec4f(0));
/** the point-light seam's enable flag: 0 in the vs / prepass entries (no point contribution), 1 in the
 *  color fs. */
export const pointScale = tgpu.privateVar(d.f32, 0);

// `pointShadowRef()`/the shade.ts module memoize the real callable reference across every consumer
// (the fog kernel's own `pointShadowOf` local is the precedent) — resolved once here, in ordinary
// module-scope JS, so the scaffold bodies below close over the real `tgpu.fn` value directly.
const pointShadowOf = pointShadowRef();

/**
 * the depth-pipeline stub for the point/spot shadow receiver — bound via {@link pointShadowSlot}
 * in depth/prepass pipelines: same signature, fully lit. A surface's own `vs` chunk can reach {@link litPbr} /
 * {@link lightFactor} (per-vertex shading), which statically pull the real receiver — in a prepass /
 * shadow-atlas pipeline that would sample the very atlas being written (a usage hazard) and read
 * bindings those passes never bind. Named `pointShadowOf` so the emitted call site is identical either
 * way; `pointScale`'s 0 default already makes the real one dynamically dead in those stages.
 */
export const pointShadowStub = tgpu
    .fn(
        [PointLightGpu, d.vec3f, d.vec3f],
        d.f32,
    )((_light, _normal, _fragWorld) => {
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

/** the clustered point/spot diffuse sum for one fragment normal. Zero when {@link pointScale} is 0
 *  (the vs / prepass entries).
 */
export const pointFactor = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((normal) => {
    "use gpu";
    let sum = d.vec3f(0);
    if (pointScale.$ === 0) return sum;
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
        const L = std.mul(toLight, std.inverseSqrt(std.max(distSq, 0.0001)));
        const diff = halfLambert(std.dot(normal, L));
        sum = d.vec3f(
            std.add(
                sum,
                std.mul(
                    light.color.rgb,
                    (distanceAttenuation(distSq, light.posRange.w, radiusSq) *
                        diff *
                        spotFactor(light, L) *
                        pointShadowSlot.$(light, normal, fragWorld.$) *
                        engineLayout.$.view.exposure) /
                        PI,
                ),
            ),
        );
    }
    return sum;
});

/** Photometric ambient, directional and clustered point illumination. Exposure is camera-local. */
export const lightFactor = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((normal) => {
    "use gpu";
    const exposure = engineLayout.$.view.exposure;
    let direct = d.vec3f(0);
    for (let i = d.u32(0); i < engineLayout.$.lighting.directionalCount; i++) {
        const light = engineLayout.$.lighting.directionalLights[i];
        let visibility = d.f32(1);
        if (light.params.y > 0) visibility = sunVisibility.$;
        const cosine = halfLambert(std.dot(normal, std.neg(light.direction.xyz)));
        direct = d.vec3f(
            std.add(
                direct,
                std.mul(light.color.rgb, (light.params.x * exposure * cosine * visibility) / PI),
            ),
        );
    }
    const world = fragWorld.$;
    const view = std.normalize(std.sub(engineLayout.$.view.eye.xyz, world));
    const indirect = indirectLight(
        IndirectLightInput({
            worldPosition: world,
            normal,
            view,
            materialOcclusion: 1,
            ambientRadiance: engineLayout.$.view.ambientColor.rgb,
        }),
    );
    return std.add(std.add(std.mul(indirect, exposure), direct), pointFactor(normal));
});

/** `baseColor * lightFactor(normal)`.
 */
export const lit = tgpu.fn(
    [d.vec3f, d.vec3f],
    d.vec3f,
)((baseColor, normal) => {
    "use gpu";
    return std.mul(baseColor, lightFactor(normal));
});

/** per-pixel (or per-vertex) metallic-roughness shading.
 *  Shares the sun-shadow / point-cluster seam with {@link lightFactor}: {@link sunVisibility} and
 *  {@link pointScale} / {@link fragWorld} are the same fs-scaffold privates, so a vs-side call gets
 *  neither point lights nor sun shadows, same as the diffuse path. At metallic 0 / roughness 1 /
 *  dielectric 0 this reduces to `lit(s.albedo, normal)` exactly.
 */
export const litPbr = tgpu.fn(
    [Pbr, d.vec3f, d.vec3f],
    d.vec3f,
)((s, normal, world) => {
    "use gpu";
    const V = std.normalize(std.sub(engineLayout.$.view.eye.xyz, world));
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
                pointShadowSlot.$(light, normal, fragWorld.$);
            // (light.color.rgb * f) * brdfSphere(...), left-to-right like the shipped shader
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
