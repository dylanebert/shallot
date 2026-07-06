// Fog — opt-in volumetric atmosphere. A compute pass marches each pixel camera→scene-depth, fusing
// **extinction** (uniform haze + exponential height fog, fading the scene toward the haze color) with
// **in-scatter** — the light shafts a `Volumetric` light opts into: the clustered point/spot cones
// shadowed by sear's point atlas, plus the directional sun shaft shadowed by sear's sun map (the same
// froxel grid + shadow service sear's lit path uses, bound through `render/core` + `sear/core`), so
// occluders cast dark shafts. It runs through the `sceneTransform` seam (after sear's color pass, before
// glaze's tonemap), so the result is part of the HDR scene the tonemap rolls off — the same pre-glaze slot
// orrstead's fog uses. A scene opts in with one `Fog` singleton; a camera opts in with sear's `Depth` lane
// (the march needs scene depth). Both absent → the pass no-ops, no auto-add. The march primitives + the Fog
// uniform layout live in `./march`, shared verbatim with the fog probe and pinned by a TS oracle (extinction
// + clustered + sun in-scatter).
import type { Plugin, System } from "../../engine";
import { Compute, f32, formatHex, sparse, u32 } from "../../engine";
import { OCT_ENCODE_WGSL } from "../../engine/utils/core";
import { GlazeSystem } from "../glaze";
import { Camera, RenderPlugin } from "../render";
import {
    LIGHTING_STRUCT_WGSL,
    LIGHTING_UNIFORM_SIZE,
    LightCull,
    Lighting,
    OverlaySystem,
    POINT_LIGHTS_STRUCT_WGSL,
    Render,
    sceneTransform,
    VIEW_BYTES,
    VIEW_STRIDE,
    VIEW_STRUCT_WGSL,
    Views,
} from "../render/core";
import { Sear, SearPlugin } from "../sear";
import {
    ColorSystem,
    casterWgsl,
    LIGHT_EVAL_WGSL,
    pointAtlasView,
    pointShadowWgsl,
    SAMPLE_SUN_SHADOW_WGSL,
    SHADOW_PARAMS_BYTES,
    SUN_SHADOW_STRUCT_WGSL,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
} from "../sear/core";
import {
    FOG_BYTES,
    FOG_FLOATS,
    FOG_INSCATTER_WGSL,
    FOG_MARCH_WGSL,
    FOG_MAX_STEPS,
    FOG_STRUCT_WGSL,
    WORKGROUP,
} from "./march";
import { packFog } from "./pack";

/**
 * the scene's volumetric atmosphere: one per scene (a singleton). The fog pass marches each pixel from the
 * camera to the scene depth, accumulating extinction and fading the scene toward `color`. `density` is the
 * base haze thickness; `heightFalloff` makes it an exponential height fog (denser low, thinning with
 * altitude above `heightBase`); `steps` / `jitter` trade march cost for banding. The scattering knobs
 * (`absorption` / `scattering` / `anisotropy` / `scatterIntensity`) shape volumetric light shafts.
 *
 * @example
 * ```
 * <a fog="density: 0.04; color: 0xb5c4d8; height-base: 0; height-falloff: 0.15" />
 * ```
 */
export const Fog = {
    /** base extinction coefficient: how fast the scene fades into haze with distance (0 = clear) */
    density: sparse(f32),
    /** hex sRGB haze color the scene fades toward (e.g. 0xb5c4d8) */
    color: sparse(f32),
    /** absorbed fraction of extinction [0,1]; the rest scatters (the scattering albedo for light shafts) */
    absorption: sparse(f32),
    /** in-scatter strength: how brightly light shafts glow in the haze */
    scattering: sparse(f32),
    /** Henyey-Greenstein anisotropy [-1,1]: 0 even glow, →1 forward (bright halo toward a light) */
    anisotropy: sparse(f32),
    /** world height where density equals `density`: the base of the height falloff */
    heightBase: sparse(f32),
    /** exponential density falloff per world unit above `heightBase` (0 = uniform haze, no height fog) */
    heightFalloff: sparse(f32),
    /** raymarch step count along each pixel's ray (clamped to 256); more = smoother, costlier */
    steps: sparse(u32),
    /** per-pixel step jitter [0,1] that breaks march banding into noise (0 = fixed midpoint sampling) */
    jitter: sparse(f32),
    /** overall multiplier on in-scatter brightness */
    scatterIntensity: sparse(f32),
};

function code(): string {
    return /* wgsl */ `
${VIEW_STRUCT_WGSL}
${FOG_STRUCT_WGSL}
${POINT_LIGHTS_STRUCT_WGSL}
${LIGHTING_STRUCT_WGSL}
${SUN_SHADOW_STRUCT_WGSL}

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var depthTex: texture_depth_2d;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var<uniform> fog: Fog;

// group 1: the clustered-light + shadow service. The compacted lights + light grid are render's (the same
// buffers sear's color FS binds, the cull already ran this frame); the point atlas + caster uniform + the
// sun shadow map + params + the comparison sampler are sear's, handed over by sear/core; the Lighting UBO
// is render's (the sun dir/color + the volumetric opt-in flag). The fog march evaluates the same lit,
// shadowed cones + sun shaft sear does — one source of truth, the relocatable chunks below
@group(1) @binding(0) var<storage, read> pointLights: PointLights;
@group(1) @binding(1) var<storage, read> lightGrid: array<vec2<u32>>;
@group(1) @binding(2) var<storage, read> lightIndices: array<u32>;
${casterWgsl()}
@group(1) @binding(3) var pointAtlas: texture_depth_2d;
@group(1) @binding(4) var<uniform> pointShadows: PointCasters;
@group(1) @binding(5) var shadowSamp: sampler_comparison;
@group(1) @binding(6) var shadowMap: texture_depth_2d;
@group(1) @binding(7) var<uniform> sunShadow: SunShadow;
@group(1) @binding(8) var<uniform> lighting: Lighting;
@group(1) @binding(9) var<uniform> tileRects: TileRects;

${OCT_ENCODE_WGSL}
${LIGHT_EVAL_WGSL}
${pointShadowWgsl()}
${SAMPLE_SUN_SHADOW_WGSL}
${FOG_MARCH_WGSL}
${FOG_INSCATTER_WGSL}

fn reconstructWorld(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let h = view.invViewProj * vec4<f32>(ndc, 1.0);
    return h.xyz / h.w;
}

// interleaved gradient noise (Jimenez 2014) — a cheap per-pixel offset that turns march banding into noise
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(output);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let px = vec2<i32>(gid.xy);
    let scn = textureLoad(scene, px, 0).rgb;
    let depth = textureLoad(depthTex, px, 0);

    // reconstruct the camera→fragment segment from the near plane to the fragment depth. Going through both
    // planes (not eye→fragment) is correct for an orthographic camera too — its rays don't share one eye.
    // Reverse-Z: the near plane is NDC z=1 (near→1, far→0), so reconstruct it at depth 1.0 — depth 0.0 is the
    // far plane, which would invert the march (close fragments over-extincted, far ones not fogged at all)
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dim);
    let nearWorld = reconstructWorld(uv, 1.0);
    let fragWorld = reconstructWorld(uv, depth);
    let seg = fragWorld - nearWorld;
    let dist = length(seg);
    let dir = seg / max(dist, 1e-6);

    let density = fog.march.x;
    let base = fog.march.y;
    let falloff = fog.march.z;
    let jitter = fog.march.w;
    let steps = max(u32(fog.extra.x), 1u);
    let g = fog.extra.y;
    let absorption = fog.extra.z;
    let gain = fog.extra.w;
    let offset = mix(0.5, ign(vec2<f32>(gid.xy)), jitter);

    // the froxel lookup along this pixel's ray: tile-xy is the pixel (fixed along the ray), the z-slice is
    // the step's view depth = its distance along the camera forward (matches sear's clusterOf for the same
    // world point, perspective + ortho alike)
    let near = view.cluster.x;
    let far = view.cluster.y;
    let forward = -cross(view.right.xyz, view.up.xyz);
    let slot = u32(view.cluster.w);

    // the extinction + in-scatter march, fused on one front-to-back midpoint sweep. trans is the
    // transmittance to the current step's start (the product of per-step exp(-density·ds) = exp(-sum), the
    // same extinction integral fogTransmittance closes). Per step: gather this froxel's volumetric lights'
    // source, then integrate the in-scatter over the step analytically
    let ds = dist / f32(steps);
    let albedo = 1.0 - absorption;
    var trans = 1.0;
    var inScatter = vec3<f32>(0.0);
    for (var i = 0u; i < ${FOG_MAX_STEPS}u; i = i + 1u) {
        if (i >= steps) { break; }
        let p = nearWorld + dir * ((f32(i) + offset) * ds);
        let dens = fogDensity(p, density, base, falloff);
        let sampleTrans = exp(-dens * ds);
        let viewZ = max(dot(p - view.eye.xyz, forward), near);
        let entry = lightGrid[clusterCell(uv.x, uv.y, viewZ, near, far, slot)];
        var lstep = vec3<f32>(0.0);
        for (var j = 0u; j < entry.y; j = j + 1u) {
            let light = pointLights.lights[lightIndices[entry.x + j]];
            // params.x < 0 is the Volumetric flag; a plain light has no shaft (skip it). A volumetric point
            // light with no surface normal passes vec3(0) to the shadow lookup — zero normal-bias offset
            if (light.params.x >= 0.0) { continue; }
            let shadow = pointShadowOf(light, vec3<f32>(0.0), p);
            lstep += inScatterContribution(light, p, dir, g) * shadow;
        }
        // the sun (directional) shaft, additive with the clustered cones on the same accumulator (so it gets
        // the same transmittance · scattering weighting). sunDirection.w is the Volumetric opt-in flag; the
        // sun shadow map darkens it behind occluders (vec3(0) normal = zero bias, in-volume), reading the
        // enabled:0 fallback as fully lit when the sun has no Shadow
        if (lighting.sunDirection.w > 0.0) {
            lstep += sunInScatter(lighting.sunColor.rgb, lighting.sunDirection.xyz, dir, g)
                * sampleSunShadow(p, vec3<f32>(0.0));
        }
        // energy-conserving in-scatter (Hillaire/Frostbite): the source integrated over the step with its
        // own extinction = albedo·(1−e^{−σ_t·ds})·gain, the analytic twin of the haze's (1−T), weighted by
        // the transmittance to the step start. Brightness is step-count-stable (a coarse march converges to
        // the same value as a fine one) and consistent with the fogComposite extinction half
        inScatter += trans * albedo * gain * lstep * (1.0 - sampleTrans);
        trans = trans * sampleTrans;
    }

    let t = trans;
    let outc = fogComposite(scn, fog.color.rgb, t) + inScatter;
    textureStore(output, px, vec4<f32>(outc, 1.0));
}
`;
}

const _fog = {
    pipeline: null as GPUComputePipeline | null,
    layout: null as GPUBindGroupLayout | null,
    light: null as GPUBindGroupLayout | null,
    buffer: null as GPUBuffer | null,
};

const _staging = new Float32Array(FOG_FLOATS);

// the camera-independent light + shadow service group (group 1), cached on the identities of the resources
// it binds. The cull already binned every shading view this frame, so one group serves all cameras; sear's
// shadow resources can flip identity (the atlas allocates lazily, the sun map toggles with a casting frame),
// so rebuild only when one changes — sear's `shadowGroup` idiom, not a per-frame allocation
let _lights: { keys: (GPUBuffer | GPUTextureView | GPUSampler)[]; group: GPUBindGroup } | null =
    null;

// per-camera group 0 (scene / depth / output / view / fog), cached per eid on the `sceneTransform` read +
// write + the depth view — all three reallocate only on a resize, so the group rebuilds then, not every frame
const _views = new Map<
    number,
    { read: GPUTextureView; write: GPUTextureView; depth: GPUTextureView; group: GPUBindGroup }
>();

function fogLights(device: GPUDevice): GPUBindGroup {
    const atlas = pointAtlasView()!;
    const casters = Compute.buffers.get("pointShadows")!;
    const tileRects = Compute.buffers.get("pointTileRects")!;
    const sampler = shadowSampler()!;
    const sunMap = sunShadowView()!;
    const sunParams = sunShadowParams()!;
    const keys = [
        LightCull.lights!,
        LightCull.grid!,
        LightCull.indices!,
        atlas,
        casters,
        sampler,
        sunMap,
        sunParams,
        Lighting.buffer,
        tileRects,
    ];
    const cached = _lights;
    if (cached && keys.every((k, i) => cached.keys[i] === k)) return cached.group;
    const group = device.createBindGroup({
        label: "fog-lights",
        layout: _fog.light!,
        entries: [
            { binding: 0, resource: { buffer: LightCull.lights! } },
            { binding: 1, resource: { buffer: LightCull.grid! } },
            { binding: 2, resource: { buffer: LightCull.indices! } },
            { binding: 3, resource: atlas },
            { binding: 4, resource: { buffer: casters } },
            { binding: 5, resource: sampler },
            { binding: 6, resource: sunMap },
            { binding: 7, resource: { buffer: sunParams } },
            { binding: 8, resource: { buffer: Lighting.buffer } },
            { binding: 9, resource: { buffer: tileRects } },
        ],
    });
    _lights = { keys, group };
    return group;
}

/**
 * the fog march, per camera: reads the resolved scene (`view.framebuffer`) + the camera's depth lane,
 * marches each pixel through the atmosphere, and writes the haze-composited scene back through the
 * `sceneTransform` scratch so glaze tonemaps it. No-op unless the scene has a {@link Fog} singleton and the
 * camera carries sear's `Depth` lane (the march needs scene depth, no auto-add). Ordered after sear's
 * color pass and before glaze.
 */
export const FogSystem: System = {
    name: "fog",
    group: "draw",
    annotations: { mode: "always" },
    after: [ColorSystem],
    // a scene-transform effect runs before the overlay anchor, so a screen-space overlay (outline)
    // composites on top of the haze rather than getting marched over by it (render.md "the post-color seam")
    before: [GlazeSystem, OverlaySystem],
    update(state) {
        const encoder = Render.encoder;
        const device = Compute.device;
        if (!encoder || !device || !_fog.pipeline || !_fog.layout || !_fog.light || !_fog.buffer)
            return;
        const fogEid = state.only([Fog]);
        if (fogEid < 0) return;
        packFog(fogEid, _staging);
        device.queue.writeBuffer(_fog.buffer, 0, _staging as Float32Array<ArrayBuffer>);
        // a null resource is a wiring bug, not a frame to skip (gpu firehose rule) — fogLights asserts them
        const lights = fogLights(device);
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer || !view.depth) continue;
            const { read, write } = sceneTransform(view, eid);
            let cam = _views.get(eid);
            if (!cam || cam.read !== read || cam.write !== write || cam.depth !== view.depth) {
                cam = {
                    read,
                    write,
                    depth: view.depth,
                    group: device.createBindGroup({
                        label: `fog/${eid}`,
                        layout: _fog.layout,
                        entries: [
                            { binding: 0, resource: read },
                            { binding: 1, resource: view.depth },
                            { binding: 2, resource: write },
                            {
                                binding: 3,
                                resource: { buffer: Render.viewBuffer, size: VIEW_BYTES },
                            },
                            { binding: 4, resource: { buffer: _fog.buffer, size: FOG_BYTES } },
                        ],
                    }),
                };
                _views.set(eid, cam);
            }
            const pass = encoder.beginComputePass({
                label: `fog/${eid}`,
                timestampWrites: Compute.span?.("fog:march"),
            });
            pass.setPipeline(_fog.pipeline);
            pass.setBindGroup(0, cam.group, [view.slot * VIEW_STRIDE]);
            pass.setBindGroup(1, lights);
            pass.dispatchWorkgroups(
                Math.ceil(view.width / WORKGROUP),
                Math.ceil(view.height / WORKGROUP),
            );
            pass.end();
        }
    },
};

/**
 * volumetric atmosphere (fog + height fog). Opt-in: add `FogPlugin` to the plugin set, give the scene one
 * {@link Fog} singleton, and give the rendering camera sear's `Depth` lane. The march composites pre-glaze
 * via the `sceneTransform` seam.
 */
export const FogPlugin: Plugin = {
    name: "Fog",
    components: { Fog },
    traits: {
        Fog: {
            singleton: true,
            defaults: () => ({
                density: 0.02,
                color: 0xb5c4d8,
                absorption: 0,
                scattering: 1,
                anisotropy: 0,
                heightBase: 0,
                heightFalloff: 0,
                steps: 32,
                jitter: 1,
                scatterIntensity: 1,
            }),
            format: { color: formatHex },
        },
    },
    systems: [FogSystem],
    dependencies: [RenderPlugin, SearPlugin],

    async warm() {
        const { device } = Compute;
        if (!device) return;
        _fog.buffer?.destroy();
        _fog.buffer = device.createBuffer({
            label: "fog-config",
            size: FOG_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        _fog.layout = device.createBindGroupLayout({
            label: "fog",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "float" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "depth" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba16float" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: VIEW_BYTES },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform", minBindingSize: FOG_BYTES },
                },
            ],
        });
        // group 1: the clustered-light + point/spot shadow service (render's compacted lights + light grid,
        // sear's atlas + caster uniform + comparison sampler) — the live resources bound via `fogLights`
        _fog.light = device.createBindGroupLayout({
            label: "fog-lights",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "depth" },
                },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: { type: "comparison" } },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "depth" },
                },
                {
                    binding: 7,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform", minBindingSize: SHADOW_PARAMS_BYTES },
                },
                {
                    binding: 8,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform", minBindingSize: LIGHTING_UNIFORM_SIZE },
                },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        const module = device.createShaderModule({ label: "fog", code: code() });
        _fog.pipeline = await device.createComputePipelineAsync({
            label: "fog",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_fog.layout, _fog.light] }),
            compute: { module, entryPoint: "main" },
        });
        // the layouts just changed identity — drop any group cached against the prior build
        _lights = null;
        _views.clear();
    },

    dispose() {
        _fog.buffer?.destroy();
        _fog.buffer = null;
        _fog.pipeline = null;
        _fog.layout = null;
        _fog.light = null;
        _lights = null;
        _views.clear();
    },
};
