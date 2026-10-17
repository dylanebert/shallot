import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { ShadowFilteringMethodMode, ViewUniforms } from "../../core/rendering";
import {
    clusterCell,
    IndirectLightInput,
    indirectLight,
    LightClusters,
    LightingGpu,
    MAX_POINT_CASTERS,
    PointLightGpu,
    pointCastersSchema,
    pointShadowRef,
    SunShadow,
    sampleSunShadow,
    tileRectsSchema,
} from "../../standard/rendering";
import {
    directionalInScatter,
    FOG_MAX_STEPS,
    FogGpu,
    fogComposite,
    fogDensity,
    ign,
    inScatterContribution,
    reconstructWorld,
    WORKGROUP,
} from "./march";

/** group 0: the per-camera resources — the resolved scene + depth to march, the storage-texture write
 *  target, and the `ViewUniforms` / `Fog` uniforms. Rebuilt per camera (`index.ts`'s `_views` cache). @internal */
export const fogLayout0 = tgpu
    .bindGroupLayout({
        // "sceneTex", not "scene" — fogComposite's own `scene` parameter would collide with a binding
        // named identically once both land in the same resolve, forcing a confusing rename on one of them
        sceneTex: { texture: d.texture2d(d.f32), visibility: ["compute"] },
        depthTex: { texture: d.textureDepth2d(), visibility: ["compute"] },
        output: {
            storageTexture: d.textureStorage2d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        view: { uniform: ViewUniforms, visibility: ["compute"] },
        fog: { uniform: FogGpu, visibility: ["compute"] },
    })
    .$idx(0);

/** MSAA variant: compute loads sample zero from the main pass's multisampled depth lane. */
export const fogLayout0Multisampled = tgpu
    .bindGroupLayout({
        sceneTex: { texture: d.texture2d(d.f32), visibility: ["compute"] },
        depthTex: { texture: d.textureDepthMultisampled2d(), visibility: ["compute"] },
        output: {
            storageTexture: d.textureStorage2d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        view: { uniform: ViewUniforms, visibility: ["compute"] },
        fog: { uniform: FogGpu, visibility: ["compute"] },
    })
    .$idx(0);

// Schema instances are scoped to this resolve, independent of standard's declarations.
const _pointCastersGpu = pointCastersSchema();
const _tileRectsGpu = tileRectsSchema(MAX_POINT_CASTERS * 6);

/** group 1: the camera-independent light + shadow service — render's compacted lights + light grid, standard's
 *  point atlas + caster uniform + comparison sampler + sun shadow map + params, and the `Lighting` UBO.
 *  Cached across cameras on resource identity (`index.ts`'s `_lights`). @internal */
export const fogLayout1 = tgpu
    .bindGroupLayout({
        pointLights: { storage: LightClusters, access: "readonly", visibility: ["compute"] },
        pointAtlas: { texture: d.textureDepth2d(), visibility: ["compute"] },
        pointShadows: { uniform: _pointCastersGpu, visibility: ["compute"] },
        shadowSamp: { sampler: "comparison", visibility: ["compute"] },
        shadowMap: { texture: d.textureDepth2d(), visibility: ["compute"] },
        sunShadow: { uniform: SunShadow, visibility: ["compute"] },
        lighting: { uniform: LightingGpu, visibility: ["compute"] },
        tileRects: { uniform: _tileRectsGpu, visibility: ["compute"] },
    })
    .$idx(1);

// Resolve the host-side getter outside TGSL so the kernel captures a transpileable tgpu.fn.
const pointShadowOf = pointShadowRef();

const loadDepth = tgpu.fn(
    [d.vec2i],
    d.f32,
)((px) => {
    "use gpu";
    return std.textureLoad(fogLayout0.$.depthTex, px, 0);
});
const loadDepthMultisampled = tgpu.fn(
    [d.vec2i],
    d.f32,
)((px) => {
    "use gpu";
    return std.textureLoad(fogLayout0Multisampled.$.depthTex, px, 0);
});

/** the fog march compute kernel: per pixel, reconstruct the camera→fragment segment, then fuse the
 *  extinction march ({@link fogDensity} / `fogComposite`) with clustered point/spot + directional in-scatter
 *  ({@link inScatterContribution} / {@link directionalInScatter}), shadowed by standard's point atlas
 *  ({@link pointShadowOf}) and the selected directional cascade map ({@link sampleSunShadow}). @internal */
function createFogKernel(
    layout: typeof fogLayout0 | typeof fogLayout0Multisampled,
    readDepth: typeof loadDepth | typeof loadDepthMultisampled,
) {
    return (
        tgpu
            .computeFn({
                workgroupSize: [WORKGROUP, WORKGROUP],
                in: { gid: d.builtin.globalInvocationId },
            })((input) => {
                "use gpu";
                const dim = std.textureDimensions(layout.$.output);
                if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
                const px = d.vec2i(input.gid.xy);
                const scn = std.textureLoad(layout.$.sceneTex, px, 0).xyz;
                // Sample zero avoids resolving multisampled depth, approximating coverage at an MSAA edge.
                const depth = readDepth(px);
                if (depth === 0 && layout.$.view.projection.z > 0.5) {
                    // Infinite perspective depth marks no opaque surface; avoid unprojecting it as a finite point.
                    std.textureStore(layout.$.output, px, d.vec4f(scn, 1));
                    return;
                }

                // reconstruct the camera→fragment segment from the near plane to the fragment depth (reverse-Z:
                // near = ndc depth 1.0)
                const uv = std.div(std.add(d.vec2f(input.gid.xy), 0.5), d.vec2f(dim));
                const nearWorld = reconstructWorld(layout.$.view.invViewProj, uv, 1);
                const fragWorld = reconstructWorld(layout.$.view.invViewProj, uv, depth);
                const seg = std.sub(fragWorld, nearWorld);
                const dist = std.length(seg);
                const dir = std.div(seg, std.max(dist, 1e-6));

                // `cfg`/`camView` — not `fog`/`view` — to avoid colliding with the group-0 binding keys of the same
                // name (a local named identically to its own binding forces a confusing double-pointer alias)
                const cfg = layout.$.fog;
                const density = cfg.march.x;
                const base = cfg.march.y;
                const falloff = cfg.march.z;
                const jitter = cfg.march.w;
                const steps = std.max(d.u32(cfg.extra.x), d.u32(1));
                const g = cfg.extra.y;
                const albedo = cfg.extra.z;
                const gain = cfg.extra.w;
                const offset = std.mix(0.5, ign(d.vec2f(input.gid.xy)), jitter);

                // the froxel lookup along this pixel's ray: tile-xy is the pixel, the z-slice is the step's view
                // depth (matches standard's clusterOf for the same world point, perspective + ortho alike)
                const camView = layout.$.view;
                const near = camView.projection.x;
                const far = camView.projection.y;
                const forward = std.neg(std.cross(camView.right.xyz, camView.up.xyz));
                const slot = d.u32(camView.projection.w);

                // the extinction + in-scatter march, fused on one front-to-back midpoint sweep: `trans` is the
                // transmittance so far, `inScatter` the radiance scattered toward the camera
                const ds = dist / d.f32(steps);
                let trans = d.f32(1);
                let inScatter = d.vec3f(0);
                let i = d.u32(0);
                while (i < FOG_MAX_STEPS) {
                    if (i >= steps) break;
                    const p = std.add(nearWorld, std.mul(dir, (d.f32(i) + offset) * ds));
                    const dens = fogDensity(p, density, base, falloff);
                    const sampleTrans = std.exp(-dens * ds);
                    const viewZ = std.max(std.dot(std.sub(p, camView.eye.xyz), forward), near);
                    const entry =
                        fogLayout1.$.pointLights.grid[
                            clusterCell(uv.x, uv.y, viewZ, near, far, slot)
                        ];
                    const ambient = indirectLight(
                        IndirectLightInput({
                            worldPosition: p,
                            normal: std.neg(dir),
                            view: std.neg(dir),
                            materialOcclusion: 1,
                            ambientRadiance: camView.ambientColor.rgb,
                        }),
                    );
                    let lstep = std.mul(ambient, camView.exposure / (4 * Math.PI));
                    let j = d.u32(0);
                    while (j < entry.y) {
                        // the array-element read is a pointer, not a copy — wrap in the element schema to
                        // pass it by value into inScatterContribution / pointShadowRef
                        const light = PointLightGpu(
                            fogLayout1.$.pointLights.lights.lights[
                                fogLayout1.$.pointLights.indices[entry.x + j]
                            ],
                        );
                        // Negative params.x marks VolumetricLight.
                        if (light.params.x >= 0) {
                            j = j + 1;
                            continue;
                        }
                        const shadow = pointShadowOf(
                            light,
                            d.vec3f(0),
                            p,
                            d.u32(layout.$.view.shadowFilteringMethod),
                        );
                        lstep = d.vec3f(
                            std.add(
                                lstep,
                                std.mul(
                                    inScatterContribution(light, p, dir, g),
                                    shadow * camView.exposure,
                                ),
                            ),
                        );
                        j = j + 1;
                    }
                    // Every directional light carrying VolumetricLight scatters; only the selected shadow caster
                    // samples the directional atlas, while the other enabled lights remain unshadowed. Each step
                    // takes one hardware tap, as Bevy's march does: the per-pixel jitter already softens the edge.
                    let k = d.u32(0);
                    while (k < fogLayout1.$.lighting.directionalCount) {
                        const light = fogLayout1.$.lighting.directionalLights[k];
                        if (light.params.z > 0) {
                            let visibility = d.f32(1);
                            if (light.params.y > 0) {
                                visibility = sampleSunShadow(
                                    p,
                                    d.vec3f(0),
                                    d.u32(ShadowFilteringMethodMode.Hardware2x2),
                                );
                            }
                            lstep = d.vec3f(
                                std.add(
                                    lstep,
                                    std.mul(
                                        directionalInScatter(
                                            std.mul(
                                                light.color.rgb,
                                                light.params.x * camView.exposure,
                                            ),
                                            light.direction.xyz,
                                            dir,
                                            g,
                                        ),
                                        visibility,
                                    ),
                                ),
                            );
                        }
                        k = k + 1;
                    }
                    // Keep the separate vector scales: reassociation changes f32 rounding.
                    inScatter = d.vec3f(
                        std.add(
                            inScatter,
                            std.mul(std.mul(lstep, trans * albedo * gain), 1 - sampleTrans),
                        ),
                    );
                    trans = trans * sampleTrans;
                    i = i + 1;
                }

                // Shadow receivers use free binding names invisible to the call-graph walk.
                // Feed these reads into the output so transpilation retains their declarations.
                const forcedZero =
                    (fogLayout1.$.pointShadows.casters[0].pos.x +
                        fogLayout1.$.tileRects.rects[0].x +
                        fogLayout1.$.sunShadow.enabled +
                        std.textureSampleCompareLevel(
                            fogLayout1.$.pointAtlas,
                            fogLayout1.$.shadowSamp,
                            d.vec2f(0, 0),
                            0,
                        ) +
                        std.textureSampleCompareLevel(
                            fogLayout1.$.shadowMap,
                            fogLayout1.$.shadowSamp,
                            d.vec2f(0, 0),
                            0,
                        )) *
                    0;

                const t = trans + forcedZero;
                const outc = std.add(fogComposite(scn, cfg.color.xyz, t), inScatter);
                std.textureStore(layout.$.output, px, d.vec4f(outc, 1));
            })
            // "fogMarch", not "fog" — the kernel's own name shares one WGSL namespace with its bindings, and a
            // group-0 binding is keyed "fog" (the config uniform); naming both the same forces a confusing
            // collision-avoidance rename on the binding instead
            .$name("fogMarch")
    );
}

export const fogKernel = createFogKernel(fogLayout0, loadDepth);
export const fogKernelMultisampled = createFogKernel(fogLayout0Multisampled, loadDepthMultisampled);
