import type { TgpuBuffer, TgpuComputePipeline, UniformFlag } from "typegpu";
import {
    Camera,
    DepthPrepassRequests,
    MainPassSystem,
    OverlaySystem,
    RenderContext,
    RenderingPlugin,
    sceneTransform,
    TonemappingSystem,
    Views,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { component, f32 } from "../../engine";
import {
    beginComputePass,
    dispatchWorkgroups2,
    endComputePass,
    precompile,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";
import {
    indirectLightSlot,
    indirectLightSources,
    LightCull,
    Lighting,
    pointAtlasView,
    pointShadowCasters,
    pointShadowRects,
    StandardRenderer,
    StandardRenderingPlugin,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
} from "../../standard/rendering";
import { FOG_FLOATS, FOG_HEIGHT, FOG_SLICES, FOG_WIDTH, FogGpu, WORKGROUP } from "./medium";
import { packFog } from "./pack";
import {
    fogGridKernel,
    fogGridLayout,
    fogKernel,
    fogKernelMultisampled,
    fogLayout0,
    fogLayout0Multisampled,
    fogLayout1,
} from "./pipeline";
import { TEMPORAL_FRAMES } from "./temporal";

/**
 * The scene's volumetric atmosphere: one per scene (a singleton). A frustum grid accumulates extinction
 * and in-scattered light up to `viewDistance`; beyond it, analytic height fog continues with unshadowed
 * ambient and directional in-scatter. `maxDistance` bounds fog along each ray, including a perspective
 * camera's clear background. `density` is the
 * base haze thickness; `heightFalloff` makes it an exponential height fog (denser low, thinning with
 * altitude above `heightBase`). Extinction is
 * density × (absorption + scattering); the scattered fraction is scattering / (absorption + scattering).
 */
export const Fog = component(
    "Fog",
    {
        /** Base medium density in inverse world units (0 = clear), multiplied by the extinction coefficients. */
        density: f32,
        /** hex sRGB haze color the scene fades toward (e.g. 0xb5c4d8) */
        color: f32,
        /** Nonnegative absorption coefficient; density × absorption removes light without scattering. */
        absorption: f32,
        /** Nonnegative scattering coefficient; density × scattering scatters light toward and away from the viewer. */
        scattering: f32,
        /** Henyey-Greenstein asymmetry (-1,1): 0 isotropic, positive values scatter forward toward a light. */
        scatteringAsymmetry: f32,
        /** world height where density equals `density`: the base of the height falloff */
        heightBase: f32,
        /** exponential density falloff per world unit above `heightBase` (0 = uniform haze, no height fog) */
        heightFalloff: f32,
        /** Far view-space depth of the shadowed volumetric grid, in world units; unshadowed height fog continues beyond it. */
        viewDistance: f32,
        /** Maximum fog distance from the camera along each ray, in world units (default 5000).
         * The perspective clear background lies here; farther surfaces take the same fog. Values below
         * `viewDistance` end within the grid, without an analytic tail; nonpositive values disable fog.
         * Set beyond the camera's far plane to blend distant geometry into the sky. */
        maxDistance: f32,
        /** Nonphysical multiplier on direct and ambient in-scattered light; 1 leaves the light unadjusted.
         * Defaults to 1/π as a tuned look value. */
        lightIntensity: f32,
    },
    {
        defaults: () => ({
            density: 0.02,
            color: 0xb5c4d8,
            absorption: 0,
            scattering: 1,
            scatteringAsymmetry: 0,
            heightBase: 0,
            heightFalloff: 0,
            viewDistance: 64,
            maxDistance: 5000,
            lightIntensity: 1 / Math.PI,
        }),
    },
);

/** Camera marker that opts out of the world's {@link Fog} atmosphere. */
export const NoFog = component("NoFog", {});

const FOG_SINGLETON_QUERY = [Fog];
const FOG_CAMERA_QUERY = [Camera, StandardRenderer];

interface FogState {
    fog: {
        gridPipeline: TgpuComputePipeline | null;
        pipeline: TgpuComputePipeline | null;
        multisampledPipeline: TgpuComputePipeline | null;
        buffer: (TgpuBuffer<typeof FogGpu> & UniformFlag) | null;
    };
    staging: Float32Array;
    lights: { keys: (GPUBuffer | GPUTextureView | GPUSampler)[]; group: GPUBindGroup } | null;
    views: Map<
        number,
        {
            grid: GPUTexture;
            width: number;
            height: number;
            feedback: GPUTexture[];
            temporal: GPUBuffer;
            staging: Float32Array<ArrayBuffer>;
            valid: boolean;
            config: Float32Array;
            parity: number;
            camera: number;
            gridGroups: GPUBindGroup[];
            gridPass: GPUComputePassDescriptor;
            read: GPUTextureView;
            write: GPUTextureView;
            depth: GPUTextureView;
            slot: number;
            multisampled: boolean;
            bindGroup: GPUBindGroup;
            pass: GPUComputePassDescriptor;
        }
    >;
}

const fogStateKey = { create: () => createFogState() };
const createFogState = (): FogState => ({
    fog: { gridPipeline: null, pipeline: null, multisampledPipeline: null, buffer: null },
    staging: new Float32Array(FOG_FLOATS),
    lights: null,
    views: new Map(),
});
const fogState = (world: World) => world.resource(fogStateKey);

function clearViews(state: FogState): void {
    for (const cam of state.views.values()) destroyView(cam);
    state.views.clear();
}

function destroyView(cam: FogState["views"] extends Map<number, infer V> ? V : never): void {
    cam.grid.destroy();
    for (const texture of cam.feedback) texture.destroy();
    cam.temporal.destroy();
}

// One light group serves all cameras. Shadow resources can change identity, invalidating the cache.
function fogLights(world: World): GPUBindGroup {
    const _lightCull = world.resource(LightCull);
    const _lighting = world.resource(Lighting);

    const atlas = pointAtlasView(world)!;
    const casters = pointShadowCasters(world)!;
    const tileRects = pointShadowRects(world)!;
    const sampler = shadowSampler(world)!;
    const sunMap = sunShadowView(world)!;
    const sunParams = sunShadowParams(world)!;
    const cached = fogState(world).lights;
    if (
        cached &&
        cached.keys[0] === _lightCull.lights &&
        cached.keys[1] === atlas &&
        cached.keys[2] === casters &&
        cached.keys[3] === sampler &&
        cached.keys[4] === sunMap &&
        cached.keys[5] === sunParams &&
        cached.keys[6] === _lighting.buffer &&
        cached.keys[7] === tileRects
    )
        return cached.group;
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
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(fogLayout1, {
            pointLights: _lightCull.lights!,
            pointAtlas: atlas,
            pointShadows: casters,
            shadowSamp: sampler,
            shadowMap: sunMap,
            sunShadow: sunParams,
            lighting: _lighting.buffer,
            tileRects,
        }),
    );
    fogState(world).lights = { keys, group };
    return group;
}

/**
 * Per camera, light and integrate the fog grid, then apply it at the scene's opaque depth through the
 * `sceneTransform` scratch so tonemapping reads it. No-op unless the scene has a {@link Fog} singleton and the
 * camera is a standard view without {@link NoFog}; the plugin requests the shared depth lane. Ordered after the
 * main color pass and before tonemapping.
 */
const FogSystem: System = {
    name: "fog",
    group: "draw",
    after: [MainPassSystem],
    // a scene-transform effect runs before the overlay anchor, so a screen-space overlay (outline)
    // composites on top of the haze rather than receiving fog
    before: [TonemappingSystem, OverlaySystem],
    update(world) {
        const _render = world.resource(RenderContext);
        const _fogState = world.resource(fogStateKey);

        if (
            !_fogState.fog.gridPipeline ||
            !_fogState.fog.pipeline ||
            !_fogState.fog.multisampledPipeline ||
            !_fogState.fog.buffer
        )
            return;
        const fogEid = world.only(FOG_SINGLETON_QUERY);
        if (fogEid < 0) {
            clearViews(_fogState);
            return;
        }
        for (const [eid, cam] of _fogState.views) {
            if (
                !world.has(eid, Camera) ||
                !world.has(eid, StandardRenderer) ||
                world.has(eid, NoFog) ||
                !world.resource(Views).get(eid)?.framebuffer
            ) {
                destroyView(cam);
                _fogState.views.delete(eid);
            }
        }
        packFog(world, fogEid, fogState(world).staging);
        _fogState.fog.buffer.write(fogState(world).staging.buffer as ArrayBuffer);
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
                cam.camera !== world.ref(eid) ||
                cam.width !== view.width ||
                cam.height !== view.height ||
                cam.read !== read ||
                cam.write !== write ||
                cam.depth !== view.depth ||
                cam.slot !== view.slot ||
                cam.multisampled !== multisampled
            ) {
                if (cam) destroyView(cam);
                const grid = world.gpu.device.createTexture({
                    label: `fog-grid/${eid}`,
                    dimension: "3d",
                    size: [
                        Math.min(FOG_WIDTH, view.width),
                        Math.min(FOG_HEIGHT, view.height),
                        FOG_SLICES + 1,
                    ],
                    format: "rgba16float",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                const gridView = grid.createView();
                const feedback = [0, 1].map(() =>
                    world.gpu.device.createTexture({
                        label: `fog-lighting/${eid}`,
                        dimension: "3d",
                        size: [grid.width, grid.height, FOG_SLICES],
                        format: "rgba16float",
                        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                    }),
                );
                const temporal = world.gpu.device.createBuffer({
                    label: `fog-history/${eid}`,
                    size: 128,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                const historySampler = world.gpu.device.createSampler({
                    minFilter: "linear",
                    magFilter: "linear",
                });
                const feedbackViews = feedback.map((texture) => texture.createView());
                const gridGroups = [0, 1].map((parity) =>
                    world.gpu.root.unwrap(
                        world.gpu.root.createBindGroup(fogGridLayout, {
                            output: gridView,
                            feedback: feedbackViews[parity],
                            history: feedbackViews[1 - parity],
                            historySampler,
                            temporal,
                            view: _render.viewBuffers[view.slot],
                            fog: _fogState.fog.buffer!,
                        }),
                    ),
                );
                const gridSampler = world.gpu.device.createSampler({
                    minFilter: "linear",
                    magFilter: "linear",
                });
                const group = multisampled
                    ? world.gpu.root.createBindGroup(fogLayout0Multisampled, {
                          grid: gridView,
                          gridSampler,
                          sceneTex: read,
                          depthTex: view.depth,
                          output: write,
                          view: _render.viewBuffers[view.slot],
                          fog: _fogState.fog.buffer,
                      })
                    : world.gpu.root.createBindGroup(fogLayout0, {
                          grid: gridView,
                          gridSampler,
                          sceneTex: read,
                          depthTex: view.depth,
                          output: write,
                          view: _render.viewBuffers[view.slot],
                          fog: _fogState.fog.buffer,
                      });
                cam = {
                    grid,
                    width: view.width,
                    height: view.height,
                    feedback,
                    temporal,
                    staging: new Float32Array(32),
                    valid: false,
                    config: new Float32Array(FOG_FLOATS),
                    parity: 0,
                    camera: world.ref(eid),
                    gridGroups,
                    gridPass: { label: `fog-grid/${eid}` },
                    read,
                    write,
                    depth: view.depth,
                    slot: view.slot,
                    multisampled,
                    bindGroup: world.gpu.root.unwrap(group),
                    pass: { label: `fog/${eid}` },
                };
                fogState(world).views.set(eid, cam);
            }
            for (let i = 0; i < FOG_FLOATS; i++) {
                if (cam.config[i] !== _fogState.staging[i]) cam.valid = false;
                cam.config[i] = _fogState.staging[i];
            }
            cam.staging[28] = world.gpu.frame % TEMPORAL_FRAMES;
            cam.staging[31] = cam.valid ? 1 : 0;
            world.gpu.device.queue.writeBuffer(cam.temporal, 0, cam.staging);
            const current = _render.viewStaging;
            const start = view.slot * 64;
            for (let i = 0; i < 16; i++) cam.staging[i] = current[start + i];
            for (let i = 0; i < 3; i++) cam.staging[16 + i] = current[start + 32 + i];
            const rx = current[start + 20],
                ry = current[start + 21],
                rz = current[start + 22];
            const ux = current[start + 24],
                uy = current[start + 25],
                uz = current[start + 26];
            cam.staging[20] = -(ry * uz - rz * uy);
            cam.staging[21] = -(rz * ux - rx * uz);
            cam.staging[22] = -(rx * uy - ry * ux);
            cam.staging[24] = current[start + 28];
            cam.staging[25] = Math.max(_fogState.staging[7] - cam.staging[24], 0.001);
            cam.staging[26] = current[start + 56];
            cam.gridPass.timestampWrites = world.gpu.span?.("fog:grid");
            const gridPass = beginComputePass(world.frameEncoder()!, cam.gridPass);
            setComputePipeline(gridPass, world.gpu.root.unwrap(_fogState.fog.gridPipeline));
            setComputeBindGroup(gridPass, 0, cam.gridGroups[cam.parity]);
            setComputeBindGroup(gridPass, 1, lights);
            dispatchWorkgroups2(
                gridPass,
                Math.ceil(cam.grid.width / WORKGROUP),
                Math.ceil(cam.grid.height / WORKGROUP),
            );
            endComputePass(gridPass);
            cam.valid = true;
            cam.parity = 1 - cam.parity;
            cam.pass.timestampWrites = world.gpu.span?.("fog:apply");
            const pass = beginComputePass(world.frameEncoder()!, cam.pass);
            setComputePipeline(pass, world.gpu.root.unwrap(pipeline));
            setComputeBindGroup(pass, 0, cam.bindGroup);
            setComputeBindGroup(pass, 1, lights);
            dispatchWorkgroups2(
                pass,
                Math.ceil(view.width / WORKGROUP),
                Math.ceil(view.height / WORKGROUP),
            );
            endComputePass(pass);
        }
    },
};

/**
 * volumetric atmosphere (fog + height fog). Opt-in: add `FogPlugin` to the plugin set, give the scene one
 * {@link Fog} singleton; fog requests the shared camera depth lane, and {@link NoFog} opts a camera out. The grid
 * composites before tonemapping via the `sceneTransform` seam.
 */
export const FogPlugin: Plugin = {
    gpu: {},
    name: "Fog",
    components: [Fog, NoFog],

    systems: [FogSystem],
    dependencies: [RenderingPlugin, StandardRenderingPlugin],

    initialize(world) {
        const state = world.resource(fogStateKey);
        world.own({ destroy: () => clearViews(state) });
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

        _fogState.fog.buffer?.destroy();
        _fogState.fog.buffer = world.gpu.root
            .createBuffer(FogGpu)
            .$usage("uniform")
            .$name("fog-config");
        const root = world.gpu.root.with(indirectLightSlot, indirectLightSources(world));
        _fogState.fog.gridPipeline = root
            .createComputePipeline({ compute: fogGridKernel })
            .$name("fog-grid");
        _fogState.fog.pipeline = root.createComputePipeline({ compute: fogKernel }).$name("fog");
        _fogState.fog.multisampledPipeline = root
            .createComputePipeline({ compute: fogKernelMultisampled })
            .$name("fog-multisampled");
        fogState(world).lights = null;
        clearViews(_fogState);

        precompile(world, "fog-grid", () => _fogState.fog.gridPipeline!);
        precompile(world, "fog", () => _fogState.fog.pipeline!);
        precompile(world, "fog-msaa", () => _fogState.fog.multisampledPipeline!);
    },

    dispose(world: World) {
        const _fogState = world.resource(fogStateKey);

        _fogState.fog.buffer?.destroy();
        _fogState.fog.buffer = null;
        _fogState.fog.gridPipeline = null;
        _fogState.fog.pipeline = null;
        _fogState.fog.multisampledPipeline = null;
        fogState(world).lights = null;
        clearViews(_fogState);
    },
};
