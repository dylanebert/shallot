// Fog — opt-in volumetric atmosphere. A compute pass marches each pixel camera→scene-depth, fusing
// **extinction** (uniform haze + exponential height fog, fading the scene toward the haze color) with
// **in-scatter** — the light shafts a `Volumetric` light opts into: the clustered point/spot cones
// shadowed by sear's point atlas, plus the directional sun shaft shadowed by sear's sun map (the same
// froxel grid + shadow service sear's lit path uses, bound through `render` + `sear`), so
// occluders cast dark shafts. It runs through the `sceneTransform` seam (after sear's color pass, before
// glaze's tonemap), so the result is part of the HDR scene the tonemap rolls off. A scene opts in with one `Fog` singleton; a camera opts in with sear's `Depth` lane
// (the march needs scene depth). Both absent → the pass no-ops, no auto-add. The march primitives + the Fog
// uniform schema live in `./march`; the typed pipeline (the two bind-group layouts + the compute kernel
// calling them) lives in `./pipeline`. Both the kernel and the CPU-side oracle
// are the same TGSL source (extinction + clustered + sun in-scatter) — this file is the ECS/system/
// plugin half: the component, the per-frame uniform pack, and the per-camera dispatch.
import type { TgpuBindGroup, TgpuBuffer, TgpuComputePipeline, UniformFlag } from "typegpu";
import {
    Camera,
    LightCull,
    Lighting,
    OverlaySystem,
    Render,
    RenderPlugin,
    sceneTransform,
    Views,
} from "../../core/rendering";
import type { Plugin, State, System } from "../../engine";
import { Compute, f32, formatHex, u32 } from "../../engine";
import { precompile, worldResource } from "../../engine/runtime";
import {
    ColorSystem,
    DEPTH_FORMAT,
    pointAtlasView,
    Sear,
    SearPlugin,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
} from "../../standard/rendering";
import { GlazeSystem } from "../../transitional/glaze";
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
    density: f32,
    /** hex sRGB haze color the scene fades toward (e.g. 0xb5c4d8) */
    color: f32,
    /** absorbed fraction of extinction [0,1]; the rest scatters (the scattering albedo for light shafts) */
    absorption: f32,
    /** in-scatter strength: how brightly light shafts glow in the haze */
    scattering: f32,
    /** Henyey-Greenstein anisotropy [-1,1]: 0 even glow, →1 forward (bright halo toward a light) */
    anisotropy: f32,
    /** world height where density equals `density`: the base of the height falloff */
    heightBase: f32,
    /** exponential density falloff per world unit above `heightBase` (0 = uniform haze, no height fog) */
    heightFalloff: f32,
    /** raymarch step count along each pixel's ray (clamped to 256); more = smoother, costlier */
    steps: u32,
    /** per-pixel step jitter [0,1] that breaks march banding into noise (0 = fixed midpoint sampling) */
    jitter: f32,
    /** overall multiplier on in-scatter brightness */
    scatterIntensity: f32,
};

interface FogState {
    fog: {
        pipeline: TgpuComputePipeline | null;
        buffer: (TgpuBuffer<typeof FogGpu> & UniformFlag) | null;
    };
    staging: Float32Array;
    lights: { keys: (GPUBuffer | GPUTextureView | GPUSampler)[]; group: LightsGroup } | null;
    views: Map<
        number,
        {
            read: GPUTextureView;
            write: GPUTextureView;
            depth: GPUTextureView;
            slot: number;
            group: ViewGroup;
        }
    >;
}

const fogStateKey = { create: () => createFogState() };
const createFogState = (): FogState => ({
    fog: { pipeline: null, buffer: null },
    staging: new Float32Array(FOG_FLOATS),
    lights: null,
    views: new Map(),
});
const fogState = () => worldResource(fogStateKey);

function initializeFogState(state: State): void {
    state.resource(fogStateKey);
}

const _fog = new Proxy({} as FogState["fog"], {
    get(_target, key) {
        return fogState().fog[key as keyof FogState["fog"]];
    },
    set(_target, key, value) {
        (fogState().fog as unknown as Record<PropertyKey, unknown>)[key] = value;
        return true;
    },
});

type LightsGroup = TgpuBindGroup<(typeof fogLayout1)["entries"]>;
type ViewGroup = TgpuBindGroup<(typeof fogLayout0)["entries"]>;

// the camera-independent light + shadow service group (group 1), cached on the identities of the resources
// it binds. The cull already binned every shading view this frame, so one group serves all cameras; sear's
// shadow resources can flip identity (the atlas allocates lazily, the sun map toggles with a casting frame),
// so rebuild only when one changes — sear's `shadowGroup` idiom, not a per-frame allocation

// per-camera group 0 (scene / depth / output / view / fog), cached per eid on the `sceneTransform` read +
// write + the depth view (all three reallocate only on a resize, so the group rebuilds then, not every
// frame) and the view slot (the per-slot View buffer it binds — a per-slot-buffer design)

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
    const cached = fogState().lights;
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
    fogState().lights = { keys, group };
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
    after: [ColorSystem],
    // a scene-transform effect runs before the overlay anchor, so a screen-space overlay (outline)
    // composites on top of the haze rather than getting marched over by it
    before: [GlazeSystem, OverlaySystem],
    update(state) {
        const encoder = Render.encoder;
        if (!encoder || !Compute.device || !_fog.pipeline || !_fog.buffer) return;
        const fogEid = state.only([Fog]);
        if (fogEid < 0) return;
        packFog(fogEid, fogState().staging);
        _fog.buffer.write(fogState().staging.buffer as ArrayBuffer);
        // a null resource is a wiring bug, not a frame to skip (gpu firehose rule) — fogLights asserts them
        const lights = fogLights();
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer || !view.depth) continue;
            const { read, write } = sceneTransform(view, eid);
            let cam = fogState().views.get(eid);
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
                fogState().views.set(eid, cam);
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

    initialize(state) {
        initializeFogState(state);
    },

    async warm() {
        const device = Compute.device;
        if (!device) return;
        _fog.buffer?.destroy();
        _fog.buffer = Compute.root.createBuffer(FogGpu).$usage("uniform").$name("fog-config");
        _fog.pipeline = Compute.root.createComputePipeline({ compute: fogKernel }).$name("fog");
        // the pipeline just changed identity — drop any group cached against the prior build
        fogState().lights = null;
        fogState().views.clear();

        // typegpu creates pipelines synchronously, so Dawn defers the real compile — and
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
        fogState().lights = null;
        fogState().views.clear();
    },
};

// fog's extension + diagnostics surface. The march WGSL chunks (so a custom pass — or the fog probe —
// splices the same integration the production `FogSystem` runs), the `Fog` uniform layout + `packFog`, and
// the CPU-side march oracles the GPU readback is diffed against. Every march primitive is one TGSL function
// that resolves to the spliced WGSL and runs on the CPU, so the chunks and the oracle are the same source.
// The extinction half (`fogMarchWgsl` / `fogTransmittance`), the clustered in-scatter half (light
// shafts — `fogInScatterWgsl` / `henyeyGreenstein` / `fogInScatter`), and the sun half (the directional
// shaft — `sunInScatter` / `fogSunInScatter`). The happy path (`Fog`, `FogPlugin`) is on the index barrel.
export type { FogScatter, FogSun } from "./march";
export {
    FOG_BYTES,
    FOG_FLOATS,
    FOG_MAX_STEPS,
    FogGpu,
    fogComposite,
    fogDensity,
    fogInScatter,
    fogInScatterWgsl,
    fogMarchWgsl,
    fogStructWgsl,
    fogSunInScatter,
    fogTransmittance,
    heightOpticalDepth,
    henyeyGreenstein,
    inScatterContribution,
    reconstructWorld,
    sunInScatter,
    WORKGROUP,
} from "./march";
export { packFog } from "./pack";
