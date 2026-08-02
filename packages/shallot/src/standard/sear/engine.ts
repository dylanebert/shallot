// The canonical typed engine substrate: the pass-invariant group-0 layout every typed sear pipeline binds,
// plus `lit` / `litPbr` / `lightFactor` / `pointFactor` / `clusterOf` and their four private seams, authored
// once against that layout. `engineScaffoldWgsl()` is the device-free structural seam for those same refs.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { MeshQuant } from "../../engine/utils/core";
import {
    clusterCell,
    distanceAttenuation,
    FrameGpu,
    LightingGpu,
    PointLightGpu,
    PointLights,
    spotFactor,
    View,
} from "../render/core";
import { brdf, brdfSphere, halfLambert, Pbr, pointShadowRef } from "./shade";

/**
 * the canonical engine group-0 layout (4a-ii design lock): every pass-invariant binding a sear pipeline
 * reads — frame / view / lighting uniforms, the compacted point-light list, the froxel light grid +
 * index pool, and the per-mesh dequant table. `vertices` is deliberately absent: it's pass-variant
 * (color binds the 16 B main stream, prepass/shadow the 8 B position stream) and moves into the surface
 * group (2) at 4a-ii-c, per the design lock. Pinned at group 0.
 */
export const engineLayout = tgpu
    .bindGroupLayout({
        frame: { uniform: FrameGpu, visibility: ["vertex", "fragment"] },
        view: { uniform: View, visibility: ["vertex", "fragment"] },
        lighting: { uniform: LightingGpu, visibility: ["vertex", "fragment"] },
        pointLights: {
            storage: PointLights,
            access: "readonly",
            visibility: ["vertex", "fragment"],
        },
        lightGrid: {
            storage: d.arrayOf(d.vec2u),
            access: "readonly",
            visibility: ["vertex", "fragment"],
        },
        lightIndices: {
            storage: d.arrayOf(d.u32),
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
 *  sun term only. Filled by a real-reference color-pass caller via `sampleSunShadow`, matching the raw
 *  scaffold's `fragmentBody` fill; the default (vertex stage, shadowless frames) is fully lit. */
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
 * the depth-pipeline stub for the point/spot shadow receiver — `SHADOW_STUB_WGSL`'s typed twin
 * (codegen.ts): same signature, fully lit. A surface's own `vs` chunk can reach {@link litPbr} /
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

/** the receiver seam the scaffold calls through: defaults to the real {@link pointShadowRef} (color
 *  pipelines emit exactly the raw path's sampling receiver); a depth pipeline binds
 *  {@link pointShadowStub} instead (`root.with(pointShadowSlot, pointShadowStub)` — the typed twin of
 *  the raw prepass module's shadow stubs). */
export const pointShadowSlot = tgpu.slot(pointShadowOf).$name("pointShadowSlot");

/** the fragment's slot-major cluster index ({@link clusterCell}). View depth recovers from the position
 *  builtin: perspective clip.w is the view depth (`fragCoord.w = 1/clip.w`); orthographic depth is
 *  linear in `fragCoord.z`.
 *
 * @example let cluster = clusterOf();
 */
export const clusterOf = tgpu.fn(
    [],
    d.u32,
)(() => {
    "use gpu";
    const near = engineLayout.$.view.cluster.x;
    const far = engineLayout.$.view.cluster.y;
    let viewZ = 1 / fragCoord.$.w;
    if (engineLayout.$.view.cluster.z < 0.5) {
        viewZ = near + fragCoord.$.z * (far - near);
    }
    return clusterCell(
        fragCoord.$.x / engineLayout.$.view.resolution.x,
        fragCoord.$.y / engineLayout.$.view.resolution.y,
        viewZ,
        near,
        far,
        d.u32(engineLayout.$.view.cluster.w),
    );
});

/** the clustered point/spot diffuse sum for one fragment normal. Zero when {@link pointScale} is 0
 *  (the vs / prepass entries).
 *
 * @example let sum = pointFactor(worldNormal);
 */
export const pointFactor = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((normal) => {
    "use gpu";
    let sum = d.vec3f(0);
    if (pointScale.$ === 0) return sum;
    const entry = engineLayout.$.lightGrid[clusterOf()];
    for (let i = d.u32(0); i < entry.y; i++) {
        const light = PointLightGpu(
            engineLayout.$.pointLights.lights[engineLayout.$.lightIndices[entry.x + i]],
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
                    distanceAttenuation(distSq, light.posRange.w, radiusSq) *
                        diff *
                        spotFactor(light, L) *
                        pointShadowSlot.$(light, normal, fragWorld.$),
                ),
            ),
        );
    }
    return sum;
});

/** ambient + sun·halfLambert·shadow + the clustered point sum, callable in a vs for per-vertex shading
 *  (where {@link sunVisibility} /
 *  {@link pointScale} sit at their defaults: fully-lit sun, zero point contribution).
 *
 * @example let factor = lightFactor(worldNormal);
 */
export const lightFactor = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((normal) => {
    "use gpu";
    const L = std.neg(engineLayout.$.lighting.sunDirection.xyz);
    const sun = halfLambert(std.dot(normal, L));
    return std.add(
        std.add(
            std.mul(
                engineLayout.$.lighting.ambientColor.rgb,
                engineLayout.$.lighting.ambientColor.a,
            ),
            // (sunColor.rgb * sun) * sunVisibility, left-to-right like the shipped shader — grouping
            // sun*sunVisibility first (a natural JS-side simplification) shifts the f32 rounding
            std.mul(std.mul(engineLayout.$.lighting.sunColor.rgb, sun), sunVisibility.$),
        ),
        pointFactor(normal),
    );
});

/** `baseColor * lightFactor(normal)`.
 *
 * @example let col = lit(albedo, worldNormal);
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
 *
 * @example let radiance = litPbr(surface, worldNormal, world);
 */
export const litPbr = tgpu.fn(
    [Pbr, d.vec3f, d.vec3f],
    d.vec3f,
)((s, normal, world) => {
    "use gpu";
    const V = std.normalize(std.sub(engineLayout.$.view.eye.xyz, world));
    // left-to-right, like the shipped shader's `a * b * c * d`: ((ambient.rgb * ambient.a) * albedo) * occlusion
    let radiance = d.vec3f(
        std.mul(
            std.mul(
                std.mul(
                    engineLayout.$.lighting.ambientColor.rgb,
                    engineLayout.$.lighting.ambientColor.a,
                ),
                s.albedo,
            ),
            s.occlusion,
        ),
    );
    radiance = d.vec3f(
        std.add(
            radiance,
            std.mul(
                std.mul(engineLayout.$.lighting.sunColor.rgb, sunVisibility.$),
                brdf(s, normal, V, std.neg(engineLayout.$.lighting.sunDirection.xyz)),
            ),
        ),
    );
    if (pointScale.$ !== 0) {
        const entry = engineLayout.$.lightGrid[clusterOf()];
        for (let i = d.u32(0); i < entry.y; i++) {
            const light = PointLightGpu(
                engineLayout.$.pointLights.lights[engineLayout.$.lightIndices[entry.x + i]],
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
                        std.mul(light.color.rgb, f),
                        brdfSphere(s, normal, V, L, dist, light.params.x),
                    ),
                ),
            );
        }
    }
    return radiance;
});

/** the typed engine scaffold's emitted WGSL: `clusterOf` / `pointFactor` / `lightFactor` / `lit` /
 *  `litPbr` + the four `var<private>` seams, self-contained (own resolve, not the shared `spliceNs`) —
 *  device-free structural seam for `engine.test.ts`. */
export function engineScaffoldWgsl(): string {
    return tgpu.resolve([clusterOf, pointFactor, lightFactor, lit, litPbr], { names: "strict" });
}
