// Fog — opt-in volumetric atmosphere. A compute pass marches each pixel camera→scene-depth, fusing
// **extinction** (uniform haze + exponential height fog, fading the scene toward the haze color) with
// **in-scatter** — every point, spot or directional light carrying `VolumetricLight`: point/spot sources use
// standard's point atlas, and only the selected directional shadow caster uses the directional atlas; other
// enabled directionals scatter unshadowed. Occluders still cast dark shafts for shadowed sources.
// It runs through the `sceneTransform` seam after the main pass and before tonemapping.
// A scene opts in with one `Fog` singleton; the plugin requests core's shared depth lane for each standard camera
// unless it carries `NoFog`. The march primitives + the Fog
// uniform schema live in `./march`; the pipeline (the two bind-group layouts + the compute kernel
// calling them) lives in `./pipeline`. Both the kernel and the CPU-side oracle
// are the same TGSL source (extinction + clustered + directional in-scatter) — this file is the ECS/system/
// plugin half: the component, the per-frame uniform pack, and the per-camera dispatch.
import type { TgpuBindGroup, TgpuBuffer, TgpuComputePipeline, UniformFlag } from "typegpu";
import {
    Camera,
    DEPTH_FORMAT,
    DepthPrepassRequests,
    MainPassSystem,
    OverlaySystem,
    RenderContext,
    RenderingPlugin,
    SAMPLE_COUNT,
    sceneTransform,
    TonemappingSystem,
    Views,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { component, f32, u32 } from "../../engine";
import { precompile } from "../../engine/runtime";
import {
    LightCull,
    Lighting,
    pointAtlasView,
    StandardRenderer,
    StandardRenderingPlugin,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
} from "../../standard/rendering";
import { FOG_FLOATS, FogGpu, WORKGROUP } from "./march";
import { packFog } from "./pack";
import {
    fogKernel,
    fogKernelMultisampled,
    fogLayout0,
    fogLayout0Multisampled,
    fogLayout1,
} from "./pipeline";

/**
 * the scene's volumetric atmosphere: one per scene (a singleton). The fog pass marches each pixel from the
 * camera to the scene depth, accumulating extinction and fading the scene toward `color`. `density` is the
 * base haze thickness; `heightFalloff` makes it an exponential height fog (denser low, thinning with
 * altitude above `heightBase`); `steps` / `jitter` trade march cost for banding. The scattering knobs
 * (`absorption` / `scattering` / `anisotropy` / `scatterIntensity`) shape volumetric light shafts.
 */
export const Fog = component(
    "Fog",
    {
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
    },
    {
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
    },
);

/** Camera marker that opts out of the world's {@link Fog} atmosphere. */
export const NoFog = component("NoFog", {});

const FOG_SINGLETON_QUERY = [Fog];
const FOG_CAMERA_QUERY = [Camera, StandardRenderer];

interface FogState {
    fog: {
        pipeline: TgpuComputePipeline | null;
        multisampledPipeline: TgpuComputePipeline | null;
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
            multisampled: boolean;
            group: ViewGroup;
        }
    >;
}

const fogStateKey = { create: () => createFogState() };
const createFogState = (): FogState => ({
    fog: { pipeline: null, multisampledPipeline: null, buffer: null },
    staging: new Float32Array(FOG_FLOATS),
    lights: null,
    views: new Map(),
});
const fogState = (world: World) => world.resource(fogStateKey);

function initializeFogState(world: World): void {
    world.resource(fogStateKey);
}

type LightsGroup = TgpuBindGroup<(typeof fogLayout1)["entries"]>;
type ViewGroup =
    | TgpuBindGroup<(typeof fogLayout0)["entries"]>
    | TgpuBindGroup<(typeof fogLayout0Multisampled)["entries"]>;

// the camera-independent light + shadow service group (group 1), cached on the identities of the resources
// it binds. The cull already binned every shading view this frame, so one group serves all cameras; standard's
// shadow resources can flip identity (the atlas allocates lazily, the sun map toggles with a casting frame),
// so rebuild only when one changes — standard's `shadowGroup` idiom, not a per-frame allocation

// per-camera group 0 (scene / depth / output / view / fog), cached per eid on the `sceneTransform` read +
// write + the depth view (all three reallocate only on a resize, so the group rebuilds then, not every
// frame) and the view slot (the per-slot ViewUniforms buffer it binds — a per-slot-buffer design)

function fogLights(world: World): LightsGroup {
    const _lightCull = world.resource(LightCull);
    const _lighting = world.resource(Lighting);

    const atlas = pointAtlasView(world)!;
    const casters = world.gpu.buffers.get("pointShadows")!;
    const tileRects = world.gpu.buffers.get("pointTileRects")!;
    const sampler = shadowSampler(world)!;
    const sunMap = sunShadowView(world)!;
    const sunParams = sunShadowParams(world)!;
    const keys = [
        _lightCull.lights!,
        atlas,
        casters,
        sampler,
        sunMap,
        sunParams,
        _lighting.buffer,
        tileRects,
    ];
    const cached = fogState(world).lights;
    if (cached && keys.every((k, i) => cached.keys[i] === k)) return cached.group;
    const group = world.gpu.root.createBindGroup(fogLayout1, {
        pointLights: _lightCull.lights!,
        pointAtlas: atlas,
        pointShadows: casters,
        shadowSamp: sampler,
        shadowMap: sunMap,
        sunShadow: sunParams,
        lighting: _lighting.buffer,
        tileRects,
    });
    fogState(world).lights = { keys, group };
    return group;
}

/**
 * the fog march, per camera: reads the resolved scene (`view.framebuffer`) + the camera's depth lane,
 * marches each pixel through the atmosphere, and writes the haze-composited scene back through the
 * `sceneTransform` scratch so tonemapping reads it. No-op unless the scene has a {@link Fog} singleton and the
 * camera is a standard view without {@link NoFog}; the plugin requests the shared depth lane. Ordered after the
 * main color pass and before tonemapping.
 */
const FogSystem: System = {
    name: "fog",
    group: "draw",
    after: [MainPassSystem],
    // a scene-transform effect runs before the overlay anchor, so a screen-space overlay (outline)
    // composites on top of the haze rather than getting marched over by it
    before: [TonemappingSystem, OverlaySystem],
    update(world) {
        const _render = world.resource(RenderContext);
        const _fogState = world.resource(fogStateKey);

        if (
            !world.gpu.device ||
            !_fogState.fog.pipeline ||
            !_fogState.fog.multisampledPipeline ||
            !_fogState.fog.buffer
        )
            return;
        const fogEid = world.only(FOG_SINGLETON_QUERY);
        if (fogEid < 0) return;
        packFog(world, fogEid, fogState(world).staging);
        _fogState.fog.buffer.write(fogState(world).staging.buffer as ArrayBuffer);
        // a null resource is a wiring bug, not a frame to skip (gpu firehose rule) — fogLights asserts them
        const lights = fogLights(world);
        for (const eid of world.query(FOG_CAMERA_QUERY)) {
            if (world.has(eid, NoFog)) continue;
            const view = world.resource(Views).get(eid);
            if (!view?.framebuffer || !view.depth) continue;
            const multisampled = world.storage(Camera).antialias.get(eid) !== 0;
            const pipeline = multisampled
                ? _fogState.fog.multisampledPipeline
                : _fogState.fog.pipeline;
            const { read, write } = sceneTransform(world, view, eid);
            let cam = fogState(world).views.get(eid);
            if (
                !cam ||
                cam.read !== read ||
                cam.write !== write ||
                cam.depth !== view.depth ||
                cam.slot !== view.slot ||
                cam.multisampled !== multisampled
            ) {
                cam = {
                    read,
                    write,
                    depth: view.depth,
                    slot: view.slot,
                    multisampled,
                    group: multisampled
                        ? world.gpu.root.createBindGroup(fogLayout0Multisampled, {
                              sceneTex: read,
                              depthTex: view.depth,
                              output: write,
                              view: _render.viewBuffers[view.slot],
                              fog: _fogState.fog.buffer,
                          })
                        : world.gpu.root.createBindGroup(fogLayout0, {
                              sceneTex: read,
                              depthTex: view.depth,
                              output: write,
                              view: _render.viewBuffers[view.slot],
                              fog: _fogState.fog.buffer,
                          }),
                };
                fogState(world).views.set(eid, cam);
            }
            const pass = world.frameEncoder()!.beginComputePass({
                label: `fog/${eid}`,
                timestampWrites: world.gpu.span?.("fog:march"),
            });
            pipeline
                .with(cam.group as never)
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
 * {@link Fog} singleton; fog requests the shared camera depth lane, and {@link NoFog} opts a camera out. The march
 * composites before tonemapping via the `sceneTransform` seam.
 */
export const FogPlugin: Plugin = {
    gpu: {},
    name: "Fog",
    components: [Fog, NoFog],

    systems: [FogSystem],
    dependencies: [RenderingPlugin, StandardRenderingPlugin],

    initialize(world) {
        initializeFogState(world);
        world
            .resource(DepthPrepassRequests)
            .push(
                (world, eid) =>
                    world.only(FOG_SINGLETON_QUERY) >= 0 &&
                    world.has(eid, StandardRenderer) &&
                    !world.has(eid, NoFog),
            );
    },

    async warm(world: World) {
        const _fogState = world.resource(fogStateKey);

        const device = world.gpu.device;
        if (!device) return;
        _fogState.fog.buffer?.destroy();
        _fogState.fog.buffer = world.gpu.root
            .createBuffer(FogGpu)
            .$usage("uniform")
            .$name("fog-config");
        _fogState.fog.pipeline = world.gpu.root
            .createComputePipeline({ compute: fogKernel })
            .$name("fog");
        _fogState.fog.multisampledPipeline = world.gpu.root
            .createComputePipeline({ compute: fogKernelMultisampled })
            .$name("fog-multisampled");
        // the pipeline just changed identity — drop any group cached against the prior build
        fogState(world).lights = null;
        fogState(world).views.clear();

        // typegpu creates pipelines synchronously, so Dawn defers the real compile — and
        // the march runs every frame `Fog` + a camera without `NoFog` are both present, so an unfired compile would land the
        // stall on whichever frame that is. Group 1's real resources (the light/shadow service) exist by the
        // time this runs (deferred past every plugin's `warm`); group 0 is genuinely per-camera,
        // so the forcer stands in 1×1 throwaways, like tonemapping's / outline's
        precompile(world, "fog", () => {
            const _fogState = world.resource(fogStateKey);

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
            const group0 = world.gpu.root.createBindGroup(fogLayout0, {
                sceneTex: src.createView(),
                depthTex: depth.createView(),
                output: dst.createView(),
                view: world.resource(RenderContext).viewBuffers[0],
                fog: _fogState.fog.buffer!,
            });
            const bound = _fogState.fog.pipeline!.with(group0).with(fogLights(world));
            src.destroy();
            depth.destroy();
            dst.destroy();
            return bound;
        });
        precompile(world, "fog-msaa", () => {
            const _fogState = world.resource(fogStateKey);

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
                sampleCount: SAMPLE_COUNT,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
            const dst = device.createTexture({
                label: "fog-precompile-out",
                size: { width: 1, height: 1 },
                format: "rgba16float",
                usage: GPUTextureUsage.STORAGE_BINDING,
            });
            const group0 = world.gpu.root.createBindGroup(fogLayout0Multisampled, {
                sceneTex: src.createView(),
                depthTex: depth.createView(),
                output: dst.createView(),
                view: world.resource(RenderContext).viewBuffers[0],
                fog: _fogState.fog.buffer!,
            });
            const bound = _fogState.fog.multisampledPipeline!.with(group0).with(fogLights(world));
            src.destroy();
            depth.destroy();
            dst.destroy();
            return bound;
        });
    },

    dispose(world: World) {
        const _fogState = world.resource(fogStateKey);

        _fogState.fog.buffer?.destroy();
        _fogState.fog.buffer = null;
        _fogState.fog.pipeline = null;
        _fogState.fog.multisampledPipeline = null;
        fogState(world).lights = null;
        fogState(world).views.clear();
    },
};
