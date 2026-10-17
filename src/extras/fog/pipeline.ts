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
    FOG_SLICES,
    FogGpu,
    fogComposite,
    fogDensity,
    inScatterContribution,
    reconstructWorld,
    WORKGROUP,
} from "./medium";

export const fogGridLayout = tgpu
    .bindGroupLayout({
        output: {
            storageTexture: d.textureStorage3d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        view: { uniform: ViewUniforms, visibility: ["compute"] },
        fog: { uniform: FogGpu, visibility: ["compute"] },
    })
    .$idx(0);

/** Per-camera scene application resources. */
export const fogLayout0 = tgpu
    .bindGroupLayout({
        grid: { texture: d.texture3d(d.f32), visibility: ["compute"] },
        gridSampler: { sampler: "filtering", visibility: ["compute"] },
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
        grid: { texture: d.texture3d(d.f32), visibility: ["compute"] },
        gridSampler: { sampler: "filtering", visibility: ["compute"] },
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

/** Light and integrate one frustum column per invocation. */
function createGridKernel() {
    const layout = fogGridLayout;
    return tgpu
        .computeFn({
            workgroupSize: [WORKGROUP, WORKGROUP],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const dim = std.textureDimensions(layout.$.output);
            if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
            const px = d.vec2i(input.gid.xy);
            const uv = std.div(std.add(d.vec2f(input.gid.xy), 0.5), d.vec2f(dim.xy));
            const nearWorld = reconstructWorld(layout.$.view.invViewProj, uv, 1);
            const farWorld = reconstructWorld(layout.$.view.invViewProj, uv, 0.0001);
            const dir = std.normalize(std.sub(farWorld, nearWorld));

            // `cfg`/`camView` — not `fog`/`view` — to avoid colliding with the group-0 binding keys of the same
            // name (a local named identically to its own binding forces a confusing double-pointer alias)
            const cfg = layout.$.fog;
            const density = cfg.medium.x;
            const base = cfg.medium.y;
            const falloff = cfg.medium.z;
            const extent = std.max(cfg.medium.w - layout.$.view.projection.x, 0.001);
            const g = cfg.extra.y;
            const albedo = cfg.extra.z;
            const gain = cfg.extra.w;

            const camView = layout.$.view;
            const near = camView.projection.x;
            const far = camView.projection.y;
            const forward = std.neg(std.cross(camView.right.xyz, camView.up.xyz));
            const slot = d.u32(camView.projection.w);

            const rayScale = 1 / std.max(std.dot(dir, forward), 1e-6);
            std.textureStore(layout.$.output, d.vec3i(px, 0), d.vec4f(0, 0, 0, 1));
            let trans = d.f32(1);
            let inScatter = d.vec3f(0);
            let i = d.u32(0);
            while (i < FOG_SLICES) {
                const a = d.f32(i) / FOG_SLICES;
                const b = d.f32(i + 1) / FOG_SLICES;
                // Quadratic view-depth boundaries crowd slices near the camera.
                const start = extent * a * a;
                const end = extent * b * b;
                const ds = (end - start) * rayScale;
                const p = std.add(nearWorld, std.mul(dir, (start + end) * 0.5 * rayScale));
                const dens = fogDensity(p, density, base, falloff);
                const sampleTrans = std.exp(-dens * ds);
                const viewZ = std.max(std.dot(std.sub(p, camView.eye.xyz), forward), near);
                const entry =
                    fogLayout1.$.pointLights.grid[clusterCell(uv.x, uv.y, viewZ, near, far, slot)];
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
                        d.u32(ShadowFilteringMethodMode.Hardware2x2),
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
                // Only the selected directional caster samples the atlas; every volumetric sun scatters.
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
                                        std.mul(light.color.rgb, light.params.x * camView.exposure),
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
                // Hillaire's closed-form cell integral S*(1-exp(-sigma*ds))/sigma:
                // S = sigma*albedo*lstep, so sigma cancels, including the clear-medium limit.
                inScatter = d.vec3f(
                    std.add(
                        inScatter,
                        std.mul(std.mul(lstep, trans * albedo * gain), 1 - sampleTrans),
                    ),
                );
                trans = trans * sampleTrans;
                std.textureStore(
                    layout.$.output,
                    d.vec3i(px, d.i32(i + 1)),
                    d.vec4f(inScatter, trans),
                );
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

            std.textureStore(
                layout.$.output,
                d.vec3i(px, FOG_SLICES),
                d.vec4f(inScatter, trans + forcedZero),
            );
        })
        .$name("fogGrid");
}

export const fogGridKernel = createGridKernel();

function createApplyKernel(
    layout: typeof fogLayout0 | typeof fogLayout0Multisampled,
    readDepth: typeof loadDepth | typeof loadDepthMultisampled,
) {
    return tgpu
        .computeFn({
            workgroupSize: [WORKGROUP, WORKGROUP],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const dim = std.textureDimensions(layout.$.output);
            if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
            const px = d.vec2i(input.gid.xy);
            const scn = std.textureLoad(layout.$.sceneTex, px, 0).xyz;
            const depth = readDepth(px);
            if (depth === 0 && layout.$.view.projection.z > 0.5) {
                std.textureStore(layout.$.output, px, d.vec4f(scn, 1));
                return;
            }
            const uv = std.div(std.add(d.vec2f(input.gid.xy), 0.5), d.vec2f(dim));
            const p = reconstructWorld(layout.$.view.invViewProj, uv, depth);
            const camView = layout.$.view;
            const forward = std.neg(std.cross(camView.right.xyz, camView.up.xyz));
            const viewZ = std.dot(std.sub(p, camView.eye.xyz), forward);
            const extent = std.max(layout.$.fog.medium.w - camView.projection.x, 0.001);
            const slice = std.sqrt(std.clamp((viewZ - camView.projection.x) / extent, 0, 1));
            // Texels hold slice boundaries, including the identity at the near plane.
            const z = (slice * FOG_SLICES + 0.5) / (FOG_SLICES + 1);
            const integrated = std.textureSampleLevel(
                layout.$.grid,
                layout.$.gridSampler,
                d.vec3f(uv, z),
                0,
            );
            const outc = std.add(
                fogComposite(scn, layout.$.fog.color.xyz, integrated.w),
                integrated.xyz,
            );
            std.textureStore(layout.$.output, px, d.vec4f(outc, 1));
        })
        .$name("fogApply");
}

export const fogKernel = createApplyKernel(fogLayout0, loadDepth);
export const fogKernelMultisampled = createApplyKernel(
    fogLayout0Multisampled,
    loadDepthMultisampled,
);
