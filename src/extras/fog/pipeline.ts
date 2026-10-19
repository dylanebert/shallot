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
    equiangularInScatter,
    FOG_SLICES,
    FogGpu,
    fogDensity,
    henyeyGreenstein,
    reconstructWorld,
    WORKGROUP,
} from "./medium";
import { HALTON, HISTORY_WEIGHT, TEMPORAL_FRAMES } from "./temporal";

export const FogHistory = d.struct({
    viewProj: d.mat4x4f,
    eye: d.vec4f,
    forward: d.vec4f,
    range: d.vec4f,
    /** x indexes the Halton sequence; w enables history. */
    jitter: d.vec4f,
});

export const fogGridLayout = tgpu
    .bindGroupLayout({
        feedback: {
            storageTexture: d.textureStorage3d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        surface: {
            storageTexture: d.textureStorage2d("r32float", "write-only"),
            visibility: ["compute"],
        },
        history: { texture: d.texture3d(d.f32), visibility: ["compute"] },
        historySampler: { sampler: "filtering", visibility: ["compute"] },
        temporal: { uniform: FogHistory, visibility: ["compute"] },
        view: { uniform: ViewUniforms, visibility: ["compute"] },
        fog: { uniform: FogGpu, visibility: ["compute"] },
    })
    .$idx(0);

/** Godot's separable fog filter, with Gaussian weights: one 7-tap pass along x into the spent history texture. */
export const fogFilterLayout = tgpu
    .bindGroupLayout({
        output: {
            storageTexture: d.textureStorage3d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        lighting: { texture: d.texture3d(d.f32), visibility: ["compute"] },
        surface: {
            texture: d.texture2d(d.f32),
            sampleType: "unfilterable-float",
            visibility: ["compute"],
        },
        view: { uniform: ViewUniforms, visibility: ["compute"] },
        fog: { uniform: FogGpu, visibility: ["compute"] },
    })
    .$idx(0);

export const fogIntegrateLayout = tgpu
    .bindGroupLayout({
        filtered: {
            storageTexture: d.textureStorage3d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        output: {
            storageTexture: d.textureStorage3d("rgba16float", "write-only"),
            visibility: ["compute"],
        },
        lighting: { texture: d.texture3d(d.f32), visibility: ["compute"] },
        surface: {
            texture: d.texture2d(d.f32),
            sampleType: "unfilterable-float",
            visibility: ["compute"],
        },
        view: { uniform: ViewUniforms, visibility: ["compute"] },
        fog: { uniform: FogGpu, visibility: ["compute"] },
    })
    .$idx(0);

export const fogReadLayout = tgpu
    .bindGroupLayout({
        grid: { texture: d.texture3d(d.f32), visibility: ["fragment"] },
        feedback: { texture: d.texture3d(d.f32), visibility: ["fragment"] },
        filtered: { texture: d.texture3d(d.f32), visibility: ["fragment"] },
        gridSampler: { sampler: "filtering", visibility: ["fragment"] },
        view: { uniform: ViewUniforms, visibility: ["fragment"] },
        fog: { uniform: FogGpu, visibility: ["fragment"] },
        lighting: { uniform: LightingGpu, visibility: ["fragment"] },
    })
    .$idx(3);

export const fogLayout0 = tgpu
    .bindGroupLayout({
        depthTex: { texture: d.textureDepth2d(), visibility: ["fragment"] },
    })
    .$idx(0);

/** Shared multisampled prepass depth: equal-depth pixels apply once; edges apply per sample. */
export const fogLayout0Multisampled = tgpu
    .bindGroupLayout({
        depthTex: { texture: d.textureDepthMultisampled2d(), visibility: ["fragment"] },
    })
    .$idx(0);

/** The camera's prepass depth, which bounds where each grid column samples the medium. */
export const fogGridDepthLayout = tgpu
    .bindGroupLayout({
        depthTex: { texture: d.textureDepth2d(), visibility: ["compute"] },
    })
    .$idx(2);

export const fogGridDepthLayoutMultisampled = tgpu
    .bindGroupLayout({
        depthTex: { texture: d.textureDepthMultisampled2d(), visibility: ["compute"] },
    })
    .$idx(2);

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
    [d.vec2i, d.u32],
    d.f32,
)((px, sample) => {
    "use gpu";
    return std.textureLoad(fogLayout0Multisampled.$.depthTex, px, sample);
});

const loadGridDepth = tgpu.fn(
    [d.vec2f, d.vec2f],
    d.f32,
)((uv, resolution) => {
    "use gpu";
    const size = d.vec2u(resolution);
    const px = std.min(d.vec2u(std.mul(uv, d.vec2f(size))), std.sub(size, d.vec2u(1)));
    return std.textureLoad(fogGridDepthLayout.$.depthTex, d.vec2i(px), 0);
});
const loadGridDepthMultisampled = tgpu.fn(
    [d.vec2f, d.vec2f],
    d.f32,
)((uv, resolution) => {
    "use gpu";
    const size = d.vec2u(resolution);
    const px = std.min(d.vec2u(std.mul(uv, d.vec2f(size))), std.sub(size, d.vec2u(1)));
    return std.textureLoad(fogGridDepthLayoutMultisampled.$.depthTex, d.vec2i(px), 0);
});

/** Stratified equiangular samples per volumetric point or spot light and cell. */
const LIGHT_STRATA = 2;

/** Light and reproject one frustum column per invocation. */
function createGridKernel(multisampled: boolean) {
    const layout = fogGridLayout;
    const loadDepth = multisampled ? loadGridDepthMultisampled : loadGridDepth;
    return tgpu
        .computeFn({
            workgroupSize: [WORKGROUP, WORKGROUP],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const dim = std.textureDimensions(layout.$.feedback);
            if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
            const px = d.vec2i(input.gid.xy);
            // Decorrelate neighbouring columns without changing their shared ray offset. Godot's volumetric fog
            // gives every cell one Halton phase per frame; HDRP hashes only the depth variate per pixel. Each column
            // here starts the sequence at its own phase, so the spatial filter below averages neighbours that sample
            // different offsets within one frame. Under a shared phase every column moves together each frame, and
            // the filter would leave the frame-to-frame swing that stage 6 measured it removing.
            const seed = (input.gid.x + input.gid.y * dim.x) * 747796405 + 2891336453;
            const word = ((seed >>> ((seed >>> 28) + 4)) ^ seed) * 277803737;
            const phaseOffset = ((word >>> 22) ^ word) % TEMPORAL_FRAMES;
            const jitter =
                HALTON.$[(d.u32(layout.$.temporal.jitter.x) + phaseOffset) % TEMPORAL_FRAMES];
            const uv = std.div(std.add(d.vec2f(input.gid.xy), 0.5), d.vec2f(dim.xy));
            const sampleUv = std.div(std.add(d.vec2f(input.gid.xy), jitter.xy), d.vec2f(dim.xy));
            const nearWorld = reconstructWorld(layout.$.view.invViewProj, uv, 1);
            const farWorld = reconstructWorld(layout.$.view.invViewProj, uv, 0.0001);
            // Phase uses the unjittered ray, so moving the sample cannot jitter anisotropy.
            const dir = std.normalize(std.sub(farWorld, nearWorld));
            // The angle one froxel subtends: twice HDRP's _HalfVoxelArcLength.
            const edgeUv = std.div(
                std.add(d.vec2f(input.gid.xy), d.vec2f(0.5, 1)),
                d.vec2f(dim.xy),
            );
            const edgeNear = reconstructWorld(layout.$.view.invViewProj, edgeUv, 1);
            const froxelArc =
                2 *
                std.length(
                    std.sub(
                        std.normalize(
                            std.sub(
                                reconstructWorld(layout.$.view.invViewProj, edgeUv, 0.0001),
                                edgeNear,
                            ),
                        ),
                        dir,
                    ),
                );
            const sampleNear = reconstructWorld(layout.$.view.invViewProj, sampleUv, 1);
            const sampleDir = std.normalize(
                std.sub(reconstructWorld(layout.$.view.invViewProj, sampleUv, 0.0001), sampleNear),
            );

            // `cfg`/`camView` — not `fog`/`view` — to avoid colliding with the group-0 binding keys of the same
            // name (a local named identically to its own binding forces a confusing double-pointer alias)
            const cfg = layout.$.fog;
            const density = cfg.medium.x;
            const base = cfg.medium.y;
            const falloff = cfg.medium.z;
            const extent = std.max(cfg.medium.w - layout.$.view.projection.x, 0.001);
            const g = cfg.extra.y;

            const camView = layout.$.view;
            const near = camView.projection.x;
            const far = camView.projection.y;
            const forward = std.neg(std.cross(camView.right.xyz, camView.up.xyz));
            const slot = d.u32(camView.projection.w);

            const rayScale = 1 / std.max(std.dot(dir, forward), 1e-6);
            // Centre depth keeps the spatial-filter mask independent of temporal jitter.
            const centerDepth = loadDepth(uv, camView.resolution);
            let centerSurface = d.f32(extent);
            if (centerDepth > 0) {
                const hit = reconstructWorld(camView.invViewProj, uv, centerDepth);
                centerSurface = std.dot(std.sub(hit, camView.eye.xyz), forward) - near;
            }
            std.textureStore(layout.$.surface, px, d.vec4f(centerSurface));
            const sceneDepth = loadDepth(sampleUv, camView.resolution);
            let surface = d.f32(extent);
            if (sceneDepth > 0) {
                const hit = reconstructWorld(layout.$.view.invViewProj, sampleUv, sceneDepth);
                surface = std.dot(std.sub(hit, camView.eye.xyz), forward) - near;
            }
            // Shadow receivers use free binding names invisible to the call-graph walk.
            // Feed these reads into the feedback so transpilation retains their declarations.
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
            let i = d.u32(0);
            while (i < FOG_SLICES) {
                const a = d.f32(i) / FOG_SLICES;
                const b = d.f32(i + 1) / FOG_SLICES;
                // Quadratic view-depth boundaries crowd slices near the camera.
                const start = extent * a * a;
                const end = extent * b * b;
                const center = std.add(nearWorld, std.mul(dir, (start + end) * 0.5 * rayScale));
                const sampleScale = 1 / std.max(std.dot(sampleDir, forward), 1e-6);
                // Unlike HDRP's skipped far cells, near-side light fills them for trilinear and filtered reads.
                const limit = surface * 0.999;
                const p = std.add(
                    sampleNear,
                    std.mul(
                        sampleDir,
                        std.mix(std.min(start, limit), std.min(end, limit), jitter.z) * sampleScale,
                    ),
                );
                let dens = fogDensity(p, density, base, falloff);
                // Local lights sample the centre ray, bounded where the jittered sample is. The centre's own depth
                // repeated near-side light past a silhouette: a band above a pillar rose 14% (stage 10).
                const lightStart = std.min(start, limit) * rayScale;
                const lightEnd = std.min(end, limit) * rayScale;

                const viewZ = std.max(std.dot(std.sub(p, camView.eye.xyz), forward), near);
                const entry =
                    fogLayout1.$.pointLights.grid[
                        clusterCell(sampleUv.x, sampleUv.y, viewZ, near, far, slot)
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
                // Uniform ambient radiance against a normalized phase integrates to itself (Hillaire 2015).
                let lstep = std.mul(ambient, camView.exposure);
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
                    // HDRP raises a light's radius to the froxel's size at the segment's entry, so a ray through
                    // the lamp cannot sample its singular core; it compares a squared radius with a length, so
                    // this takes the intent, the froxel's width, and keeps the glow's brightness (stage 10).
                    // Local lights take the column's centre ray, as HDRP takes the voxel centre for anisotropy to
                    // keep it out of the jitter: a ray jittered above or below a spot's apex inside one column
                    // flips the lamp's cell between dark and lit. Two depth strata settle the cone's penumbra
                    // along the segment. A lamp swings 0.0160 with neither, 0.0098 or 0.0088 with one, 0.0011 with both.
                    // As in HDRP, the light's shadow and cone are read at its own samples while density, ambient
                    // and the sun keep the jittered one; the light list is the jittered sample's cluster, and
                    // the falloff is HDRP's 1 / (d² + r²) rather than standard's 1 / max(d², r²).
                    const lightPhase =
                        camView.exposure *
                        henyeyGreenstein(
                            g,
                            std.dot(dir, std.normalize(std.sub(light.posRange.xyz, center))),
                        );
                    const froxelWidth = froxelArc * lightStart;
                    let stratum = d.u32(0);
                    while (stratum < LIGHT_STRATA) {
                        const sample = equiangularInScatter(
                            light,
                            nearWorld,
                            dir,
                            lightStart,
                            lightEnd,
                            (d.f32(stratum) + jitter.z) / LIGHT_STRATA,
                            froxelWidth * froxelWidth,
                        );
                        const shadow = pointShadowOf(
                            light,
                            d.vec3f(0),
                            sample.position,
                            d.u32(ShadowFilteringMethodMode.Hardware2x2),
                        );
                        lstep = d.vec3f(
                            std.add(
                                lstep,
                                std.mul(sample.radiance, (shadow * lightPhase) / LIGHT_STRATA),
                            ),
                        );
                        stratum = stratum + 1;
                    }
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
                // Reproject complete lighting as Godot does: a current phase ratio flickers
                // when jitter changes the balance of ambient and shadowed direct light.
                // Angular scattering in history remains approximate when the camera turns.
                const prev = layout.$.temporal;
                const clip = std.mul(prev.viewProj, d.vec4f(center, 1));
                const prevUv = std.add(std.mul(std.div(clip.xy, clip.w), d.vec2f(0.5, -0.5)), 0.5);
                const depth = std.dot(std.sub(center, prev.eye.xyz), prev.forward.xyz);
                const sliceDepth = (depth - prev.range.x) / prev.range.y;
                // Centres lie at (i²+i+0.5)/N²; invert that, not the slice boundaries.
                const z = fogCellCoordinate(depth, prev.range.x, prev.range.y);
                if (
                    prev.jitter.w > 0 &&
                    clip.w > 0 &&
                    prevUv.x > 0 &&
                    prevUv.x < 1 &&
                    prevUv.y > 0 &&
                    prevUv.y < 1 &&
                    depth > prev.range.x &&
                    sliceDepth < 1
                ) {
                    const old = std.textureSampleLevel(
                        layout.$.history,
                        layout.$.historySampler,
                        d.vec3f(prevUv, z),
                        0,
                    );
                    lstep = std.mix(
                        lstep,
                        std.mul(old.rgb, camView.exposure / std.max(prev.range.z, 1e-8)),
                        HISTORY_WEIGHT,
                    );
                    dens = std.mix(dens, old.a, HISTORY_WEIGHT);
                }
                std.textureStore(
                    layout.$.feedback,
                    d.vec3i(px, d.i32(i)),
                    d.vec4f(lstep, dens + forcedZero),
                );
                i = i + 1;
            }
        })
        .$name("fogGrid");
}

export const fogGridKernel = createGridKernel(false);
export const fogGridKernelMultisampled = createGridKernel(true);

// A sampled Gaussian of σ = 1.6 cells, its ±3 tails at 17% of the centre. Godot's weights
// (0.071303 … 0.214607) fit σ ≈ 2.0 cut at ±3, about 1.5σ: nearly a box, whose separable product draws a hot
// cell as a square (stage 10). σ = 1.6 is the narrowest that keeps stage 6's shaft-edge claims with margin
// natively and on lavapipe, where 1.5 reached 0.00499 of their 0.005.
const GAUSSIAN = tgpu.const(
    d.arrayOf(d.f32, 7),
    [0.044147, 0.117223, 0.210611, 0.256038, 0.210611, 0.117223, 0.044147],
);

/**
 * Blur extinction-weighted light (σt·L, σt) along x, as Godot blurs σs·L with σt; the integration kernel blurs
 * along y and divides by σt. Blurring L and σt apart would light dense shadowed air with its thin lit neighbour.
 */
export const fogFilterKernel = tgpu
    .computeFn({
        workgroupSize: [WORKGROUP, WORKGROUP],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const dim = std.textureDimensions(fogFilterLayout.$.output);
        if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
        const cell = d.vec3i(input.gid);
        const last = d.i32(dim.x) - 1;
        const extent = std.max(
            fogFilterLayout.$.fog.medium.w - fogFilterLayout.$.view.projection.x,
            0.001,
        );
        const a = d.f32(cell.z) / FOG_SLICES;
        const b = (d.f32(cell.z) + 1) / FOG_SLICES;
        const surface = std.textureLoad(fogFilterLayout.$.surface, cell.xy, 0).x;
        const weight = std.clamp((surface - extent * a * a) / (extent * (b * b - a * a)), 0, 1);
        const own = std.textureLoad(fogFilterLayout.$.lighting, cell, 0);
        // Neither HDRP's 3×3 nor Godot's 7-tap fog filter is depth-aware; blend continuously at the surface.
        if (weight <= 0) {
            std.textureStore(
                fogFilterLayout.$.output,
                cell,
                d.vec4f(std.mul(own.xyz, own.w), own.w),
            );
            return;
        }
        let sum = d.vec4f(0);
        let k = d.i32(-3);
        while (k <= 3) {
            const tap = std.textureLoad(
                fogFilterLayout.$.lighting,
                d.vec3i(std.clamp(cell.x + k, 0, last), cell.y, cell.z),
                0,
            );
            sum = std.add(sum, std.mul(d.vec4f(std.mul(tap.xyz, tap.w), tap.w), GAUSSIAN.$[k + 3]));
            k = k + 1;
        }
        std.textureStore(
            fogFilterLayout.$.output,
            cell,
            std.mix(d.vec4f(std.mul(own.xyz, own.w), own.w), sum, weight),
        );
    })
    .$name("fogFilter");

/**
 * Filter along y and integrate one frustum column per invocation. As in Godot and HDRP, reprojected lighting is
 * blurred across columns before integration while history keeps it unfiltered: neighbours sample different
 * jitter phases, so a shadow edge that one column's depth sample flips across averages within the frame.
 */
export const fogIntegrateKernel = tgpu
    .computeFn({
        workgroupSize: [WORKGROUP, WORKGROUP],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const dim = std.textureDimensions(fogIntegrateLayout.$.output);
        if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
        const px = d.vec2i(input.gid.xy);
        const last = d.i32(dim.y) - 1;
        const uv = std.div(std.add(d.vec2f(input.gid.xy), 0.5), d.vec2f(dim.xy));
        const camView = fogIntegrateLayout.$.view;
        const nearWorld = reconstructWorld(camView.invViewProj, uv, 1);
        const dir = std.normalize(
            std.sub(reconstructWorld(camView.invViewProj, uv, 0.0001), nearWorld),
        );
        const forward = std.neg(std.cross(camView.right.xyz, camView.up.xyz));
        const rayScale = 1 / std.max(std.dot(dir, forward), 1e-6);
        const cfg = fogIntegrateLayout.$.fog;
        const extent = std.max(cfg.medium.w - camView.projection.x, 0.001);
        const albedo = cfg.extra.z;
        const gain = cfg.extra.w;
        const surface = std.textureLoad(fogIntegrateLayout.$.surface, px, 0).x;
        std.textureStore(fogIntegrateLayout.$.output, d.vec3i(px, 0), d.vec4f(0, 0, 0, 1));
        let trans = d.f32(1);
        let inScatter = d.vec3f(0);
        let i = d.u32(0);
        while (i < FOG_SLICES) {
            const a = d.f32(i) / FOG_SLICES;
            const b = d.f32(i + 1) / FOG_SLICES;
            const ds = extent * (b * b - a * a) * rayScale;
            const slice = d.i32(i);
            let cell = d.vec4f(0);
            const weight = std.clamp((surface - extent * a * a) / (extent * (b * b - a * a)), 0, 1);
            const wall = weight <= 0;
            let k = std.select(d.i32(-3), d.i32(0), wall);
            const radius = std.select(d.i32(3), d.i32(0), wall);
            while (k <= radius) {
                cell = std.add(
                    cell,
                    std.mul(
                        std.textureLoad(
                            fogIntegrateLayout.$.lighting,
                            d.vec3i(px.x, std.clamp(px.y + k, 0, last), slice),
                            0,
                        ),
                        std.select(GAUSSIAN.$[k + 3], 1, wall),
                    ),
                );
                k = k + 1;
            }
            const own = std.textureLoad(fogIntegrateLayout.$.lighting, d.vec3i(px, slice), 0);
            cell = std.mix(own, cell, weight);
            std.textureStore(fogIntegrateLayout.$.filtered, d.vec3i(px, slice), cell);
            const opticalDepth = cell.w * ds;
            const sampleTrans = std.exp(-opticalDepth);
            // Hillaire's closed-form cell integral S*(1-exp(-sigma*ds))/sigma, with S = albedo*(sigma*lstep)
            // filtered; the series keeps the clear-medium limit S*ds.
            let span = ds * (1 - opticalDepth * 0.5);
            if (opticalDepth > 1e-3) span = (1 - sampleTrans) / cell.w;
            inScatter = d.vec3f(
                std.add(inScatter, std.mul(cell.xyz, trans * albedo * gain * span)),
            );
            trans = trans * sampleTrans;
            std.textureStore(
                fogIntegrateLayout.$.output,
                d.vec3i(px, slice + 1),
                d.vec4f(inScatter, trans),
            );
            i = i + 1;
        }
    })
    .$name("fogIntegrate");

const fogCellCoordinate = tgpu.fn(
    [d.f32, d.f32, d.f32],
    d.f32,
)((depth, near, extent) => {
    "use gpu";
    return std.sqrt(std.max((depth - near) / extent - 0.25 / (FOG_SLICES * FOG_SLICES), 0));
});

const fogTail = tgpu
    .fn(
        [FogGpu, ViewUniforms, d.vec3f, d.vec3f, d.f32, d.vec4f],
        d.vec4f,
    )((cfg, view, start, dir, distance, integrated) => {
        "use gpu";
        const x = cfg.medium.z * dir.y * distance;
        // The series avoids cancellation for horizontal rays and zero height falloff.
        let integral = d.f32(1 - x * 0.5 + (x * x) / 6);
        if (std.abs(x) > 0.001) integral = (1 - std.exp(-x)) / x;
        const opticalDepth =
            fogDensity(start, cfg.medium.x, cfg.medium.y, cfg.medium.z) * distance * integral;
        const trans = std.exp(-opticalDepth);
        const light = fogTailLight(cfg, view, dir);
        return d.vec4f(
            std.add(
                integrated.xyz,
                std.mul(light, integrated.w * cfg.extra.z * cfg.extra.w * (1 - trans)),
            ),
            integrated.w * trans,
        );
    })
    .$name("fogTail");

// The distant medium has no resolved occluders or local lights.
const fogTailLight = tgpu
    .fn(
        [FogGpu, ViewUniforms, d.vec3f],
        d.vec3f,
    )((cfg, view, dir) => {
        "use gpu";
        let light = std.mul(view.ambientColor.rgb, view.exposure);
        let k = d.u32(0);
        while (k < fogReadLayout.$.lighting.directionalCount) {
            const sun = fogReadLayout.$.lighting.directionalLights[k];
            if (sun.params.z > 0) {
                light = d.vec3f(
                    std.add(
                        light,
                        directionalInScatter(
                            std.mul(sun.color.rgb, sun.params.x * view.exposure),
                            sun.direction.xyz,
                            dir,
                            cfg.extra.y,
                        ),
                    ),
                );
            }
            k = k + 1;
        }
        return light;
    })
    .$name("fogTailLight");

/**
 * Read camera-exposed local in-scatter radiance per unit scattering, before the medium's albedo: Fog's own
 * light plus the scene's lights times lightIntensity.
 * A particle adds albedo × coverage × readFogInScatter(p) × readFog(p).a; apply no exposure.
 * Returns zero without FogPlugin/Fog, with NoFog, at/before near, or beyond maxDistance. Bind
 * fogReadBindGroup at group 3 with fogReadLayout after this frame's fog draw system. Light is trilinear
 * at cell resolution, filtered across columns like the haze, with phase toward the camera along the cell ray;
 * history lags moving lights and cameras.
 */
export const readFogInScatter = tgpu
    .fn(
        [d.vec3f],
        d.vec3f,
    )((point) => {
        "use gpu";
        const view = fogReadLayout.$.view;
        const cfg = fogReadLayout.$.fog;
        if (cfg.medium.x === 0) return d.vec3f(0);
        const forward = std.neg(std.cross(view.right.xyz, view.up.xyz));
        const viewZ = std.dot(std.sub(point, view.eye.xyz), forward);
        if (viewZ <= view.projection.x) return d.vec3f(0);
        const clip = std.mul(view.viewProj, d.vec4f(point, 1));
        const uv = std.add(std.mul(std.div(clip.xy, clip.w), d.vec2f(0.5, -0.5)), 0.5);
        const nearWorld = reconstructWorld(view.invViewProj, uv, 1);
        const dir = std.normalize(std.sub(point, nearWorld));
        const cosine = std.max(std.dot(dir, forward), 1e-6);
        if (viewZ > cfg.extra.x * cosine || cfg.extra.x <= 0) return d.vec3f(0);
        const own = std.mul(cfg.color.xyz, view.exposure);
        if (viewZ > cfg.medium.w)
            return std.add(own, std.mul(fogTailLight(cfg, view, dir), cfg.extra.w));
        const z = fogCellCoordinate(
            viewZ,
            view.projection.x,
            std.max(cfg.medium.w - view.projection.x, 0.001),
        );
        const cell = std.textureSampleLevel(
            fogReadLayout.$.filtered,
            fogReadLayout.$.gridSampler,
            d.vec3f(uv, z),
            0,
        );
        let light = std.textureSampleLevel(
            fogReadLayout.$.feedback,
            fogReadLayout.$.gridSampler,
            d.vec3f(uv, z),
            0,
        ).rgb;
        // Clear air retains its local light even though it scatters none into the haze.
        if (cell.w > 0) light = std.div(cell.rgb, cell.w);
        return std.add(own, std.mul(light, cfg.extra.w));
    })
    .$name("readFogInScatter");

/**
 * Read added RGB radiance (camera-exposed) and alpha transmittance to a world-space point, bounded by
 * Fog.maxDistance. A custom fragment shader resolves this TGSL function through TypeGPU and binds
 * fogReadBindGroup(world, camera) at group 3 with fogReadLayout. Raw WebGPU pipeline layouts must fill
 * unused earlier indices with explicit empty layouts: TypeGPU leaves sparse group indices unfilled.
 */
export const readFog = tgpu
    .fn(
        [d.vec3f],
        d.vec4f,
    )((point) => {
        "use gpu";
        const camView = fogReadLayout.$.view;
        const cfg = fogReadLayout.$.fog;
        if (cfg.medium.x === 0) return d.vec4f(0, 0, 0, 1);
        const clip = std.mul(camView.viewProj, d.vec4f(point, 1));
        const uv = std.add(std.mul(std.div(clip.xy, clip.w), d.vec2f(0.5, -0.5)), 0.5);
        const forward = std.neg(std.cross(camView.right.xyz, camView.up.xyz));
        const nearWorld = reconstructWorld(camView.invViewProj, uv, 1);
        const dir = std.normalize(std.sub(point, nearWorld));
        const surfaceZ = std.dot(std.sub(point, camView.eye.xyz), forward);
        const cosine = std.max(std.dot(dir, forward), 1e-6);
        const viewZ = std.min(surfaceZ, std.max(camView.projection.x, cfg.extra.x * cosine));
        const extent = std.max(cfg.medium.w - camView.projection.x, 0.001);
        const slice = std.sqrt(std.clamp((viewZ - camView.projection.x) / extent, 0, 1));
        // Texels hold slice boundaries, including the identity at the near plane.
        const z = (slice * FOG_SLICES + 0.5) / (FOG_SLICES + 1);
        const integrated = std.textureSampleLevel(
            fogReadLayout.$.grid,
            fogReadLayout.$.gridSampler,
            d.vec3f(uv, z),
            0,
        );
        let volume = d.vec4f(integrated);
        if (viewZ > cfg.medium.w) {
            const rayScale = 1 / cosine;
            volume = fogTail(
                cfg,
                camView,
                std.add(nearWorld, std.mul(dir, extent * rayScale)),
                dir,
                (viewZ - cfg.medium.w) * rayScale,
                integrated,
            );
        }
        // The medium's own light is uniform, so its in-scatter along any density is albedo · L · (1 − T).
        const own = std.mul(cfg.color.xyz, camView.exposure * cfg.extra.z * (1 - volume.w));
        return d.vec4f(std.add(own, volume.xyz), volume.w);
    })
    .$name("readFog");

const applyFog = tgpu.fn(
    [d.vec4f, d.f32],
    d.vec4f,
)((position, depth) => {
    "use gpu";
    const uv = std.div(position.xy, fogReadLayout.$.view.resolution);
    const camView = fogReadLayout.$.view;
    let point = reconstructWorld(camView.invViewProj, uv, depth);
    if (depth === 0 && camView.projection.z > 0.5) {
        const nearWorld = reconstructWorld(camView.invViewProj, uv, 1);
        const dir = std.normalize(std.sub(nearWorld, camView.eye.xyz));
        point = std.add(
            camView.eye.xyz,
            std.mul(dir, std.max(fogReadLayout.$.fog.extra.x, camView.projection.x + 0.001)),
        );
    }
    return readFog(point);
});

export const fogKernel = tgpu
    .fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })((input) => {
        "use gpu";
        return applyFog(input.position, loadDepth(d.vec2i(input.position.xy)));
    })
    .$name("fogApply");

// Exact depth equality keeps every discontinuity on the sample-rate path. Sloped surfaces may
// also take that path; a tolerance could instead merge distinct surfaces and lose their fog edge.
const isEdge = tgpu.fn(
    [d.vec2i],
    d.bool,
)((pixel) => {
    "use gpu";
    const z = loadDepthMultisampled(pixel, 0);
    // Unrolled for core rendering's SAMPLE_COUNT = 4 to keep sample loads and short-circuiting static.
    return (
        z !== loadDepthMultisampled(pixel, 1) ||
        z !== loadDepthMultisampled(pixel, 2) ||
        z !== loadDepthMultisampled(pixel, 3)
    );
});

export const fogKernelInterior = tgpu
    .fragmentFn({ in: { position: d.builtin.position }, out: d.vec4f })((input) => {
        "use gpu";
        const pixel = d.vec2i(input.position.xy);
        if (isEdge(pixel)) std.discard();
        return applyFog(input.position, loadDepthMultisampled(pixel, 0));
    })
    .$name("fogApply");

export const fogKernelMultisampled = tgpu
    .fragmentFn({
        in: { position: d.builtin.position, sample: d.builtin.sampleIndex },
        out: d.vec4f,
    })((input) => {
        "use gpu";
        if (!isEdge(d.vec2i(input.position.xy))) std.discard();
        return applyFog(
            input.position,
            loadDepthMultisampled(d.vec2i(input.position.xy), input.sample),
        );
    })
    .$name("fogApply");
