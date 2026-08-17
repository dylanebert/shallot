// Fog — opt-in volumetric atmosphere. A compute pass marches each pixel camera→scene-depth, fusing
// **extinction** (uniform haze + exponential height fog, fading the scene toward the haze color) with
// **in-scatter** — the light shafts a `Volumetric` light opts into: the clustered point/spot cones
// shadowed by sear's point atlas, plus the directional sun shaft shadowed by sear's sun map (the same
// froxel grid + shadow service sear's lit path uses, bound through `render/core` + `sear/core`), so
// occluders cast dark shafts. It runs through the `sceneTransform` seam (after sear's color pass, before
// glaze's tonemap), so the result is part of the HDR scene the tonemap rolls off. A scene opts in with one `Fog` singleton; a camera opts in with sear's `Depth` lane
// (the march needs scene depth). Both absent → the pass no-ops, no auto-add. The march primitives + the Fog
// uniform schema live in `./march`; the typed pipeline (the two bind-group layouts + the compute kernel
// calling them) lives in `./pipeline`. Both the kernel and the CPU-side oracle the gym fog probe diffs
// against are the same TGSL source (extinction + clustered + sun in-scatter) — this file is the ECS/system/
// plugin half: the component, the per-frame uniform pack, and the per-camera dispatch.
import type { TgpuBindGroup, TgpuBuffer, TgpuComputePipeline, UniformFlag } from "typegpu";
import type { Plugin, System } from "../../engine";
import { Compute, f32, formatHex, sparse, u32 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { GlazeSystem } from "../glaze";
import { Camera, RenderPlugin } from "../render";
import { LightCull, Lighting, OverlaySystem, Render, sceneTransform, Views } from "../render/core";
import { Sear, SearPlugin } from "../sear";
import {
    ColorSystem,
    DEPTH_FORMAT,
    pointAtlasView,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
} from "../sear/core";
import { FOG_FLOATS, FogGpu, WORKGROUP } from "./march";
import { packFog } from "./pack";
import { fogKernel, fogLayout0, fogLayout1 } from "./pipeline";

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

const _fog = {
    pipeline: null as TgpuComputePipeline | null,
    buffer: null as (TgpuBuffer<typeof FogGpu> & UniformFlag) | null,
};

const _staging = new Float32Array(FOG_FLOATS);

type LightsGroup = TgpuBindGroup<(typeof fogLayout1)["entries"]>;
type ViewGroup = TgpuBindGroup<(typeof fogLayout0)["entries"]>;

// the camera-independent light + shadow service group (group 1), cached on the identities of the resources
// it binds. The cull already binned every shading view this frame, so one group serves all cameras; sear's
// shadow resources can flip identity (the atlas allocates lazily, the sun map toggles with a casting frame),
// so rebuild only when one changes — sear's `shadowGroup` idiom, not a per-frame allocation
let _lights: { keys: (GPUBuffer | GPUTextureView | GPUSampler)[]; group: LightsGroup } | null =
    null;

// per-camera group 0 (scene / depth / output / view / fog), cached per eid on the `sceneTransform` read +
// write + the depth view (all three reallocate only on a resize, so the group rebuilds then, not every
// frame) and the view slot (the per-slot View buffer it binds — a per-slot-buffer design)
const _views = new Map<
    number,
    {
        read: GPUTextureView;
        write: GPUTextureView;
        depth: GPUTextureView;
        slot: number;
        group: ViewGroup;
    }
>();

function fogLights(): LightsGroup {
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
    const group = Compute.root.createBindGroup(fogLayout1, {
        pointLights: LightCull.lights!,
        lightGrid: LightCull.grid!,
        lightIndices: LightCull.indices!,
        pointAtlas: atlas,
        pointShadows: casters,
        shadowSamp: sampler,
        shadowMap: sunMap,
        sunShadow: sunParams,
        lighting: Lighting.buffer,
        tileRects,
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
        if (!encoder || !Compute.device || !_fog.pipeline || !_fog.buffer) return;
        const fogEid = state.only([Fog]);
        if (fogEid < 0) return;
        packFog(fogEid, _staging);
        _fog.buffer.write(_staging.buffer as ArrayBuffer);
        // a null resource is a wiring bug, not a frame to skip (gpu firehose rule) — fogLights asserts them
        const lights = fogLights();
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer || !view.depth) continue;
            const { read, write } = sceneTransform(view, eid);
            let cam = _views.get(eid);
            if (
                !cam ||
                cam.read !== read ||
                cam.write !== write ||
                cam.depth !== view.depth ||
                cam.slot !== view.slot
            ) {
                cam = {
                    read,
                    write,
                    depth: view.depth,
                    slot: view.slot,
                    group: Compute.root.createBindGroup(fogLayout0, {
                        sceneTex: read,
                        depthTex: view.depth,
                        output: write,
                        view: Render.viewBuffers[view.slot],
                        fog: _fog.buffer,
                    }),
                };
                _views.set(eid, cam);
            }
            const pass = encoder.beginComputePass({
                label: `fog/${eid}`,
                timestampWrites: Compute.span?.("fog:march"),
            });
            _fog.pipeline
                .with(cam.group)
                .with(lights)
                .with(pass)
                .dispatchWorkgroups(
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
        const device = Compute.device;
        if (!device) return;
        _fog.buffer?.destroy();
        _fog.buffer = Compute.root.createBuffer(FogGpu).$usage("uniform").$name("fog-config");
        _fog.pipeline = Compute.root.createComputePipeline({ compute: fogKernel }).$name("fog");
        // the pipeline just changed identity — drop any group cached against the prior build
        _lights = null;
        _views.clear();

        // typegpu has no async pipeline creation, so Dawn defers the real compile to first dispatch — and
        // the march runs every frame `Fog` + `Depth` are both present, so an unfired compile would land the
        // stall on whichever frame that is. Group 1's real resources (the light/shadow service) exist by the
        // time this runs (deferred past every plugin's `warm`); group 0 is genuinely per-camera,
        // so the forcer stands in 1×1 throwaways, like glaze's / outline's
        precompile("fog", () => {
            const src = device.createTexture({
                label: "fog-precompile-scene",
                size: { width: 1, height: 1 },
                format: "rgba16float",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            const depth = device.createTexture({
                label: "fog-precompile-depth",
                size: { width: 1, height: 1 },
                format: DEPTH_FORMAT,
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            const dst = device.createTexture({
                label: "fog-precompile-out",
                size: { width: 1, height: 1 },
                format: "rgba16float",
                usage: GPUTextureUsage.STORAGE_BINDING,
            });
            const group0 = Compute.root.createBindGroup(fogLayout0, {
                sceneTex: src.createView(),
                depthTex: depth.createView(),
                output: dst.createView(),
                view: Render.viewBuffers[0],
                fog: _fog.buffer!,
            });
            const bound = _fog.pipeline!.with(group0).with(fogLights());
            bound.dispatchWorkgroups(0);
            // the dispatch is already submitted and ran no invocation, so the throwaways are dead here
            src.destroy();
            depth.destroy();
            dst.destroy();
            return bound;
        });
    },

    dispose() {
        _fog.buffer?.destroy();
        _fog.buffer = null;
        _fog.pipeline = null;
        _lights = null;
        _views.clear();
    },
};
