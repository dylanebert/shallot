import tgpu, { type TgpuBuffer, type TgpuComputePipeline, type UniformFlag } from "typegpu";
import * as d from "typegpu/data";
import {
    Camera,
    DepthPrepassRequests,
    EXPOSURE_EV100_BLENDER,
    fullscreenVertex,
    MainPassSystem,
    PrepassSystem,
    RenderContext,
    RenderingPlugin,
    RenderPhases,
    SAMPLE_COUNT,
    Views,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { component, f32 } from "../../engine";
import {
    beginComputePass,
    dispatchWorkgroups2,
    dispatchWorkgroups3,
    endComputePass,
    precompile,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";
import {
    FogInput,
    indirectLightSlot,
    indirectLightSources,
    LightCull,
    Lighting,
    LightingGpu,
    pointAtlasView,
    pointShadowCasters,
    pointShadowRects,
    registerFogSource,
    ShadowMapSystem,
    StandardRenderer,
    StandardRenderingPlugin,
    shadowSampler,
    sunShadowParams,
    sunShadowView,
} from "../../standard/rendering";
import { FOG_FLOATS, FOG_HEIGHT, FOG_SLICES, FOG_WIDTH, FogGpu, WORKGROUP } from "./medium";
import { packFog } from "./pack";
import {
    fogFilterKernel,
    fogFilterLayout,
    fogGridDepthLayout,
    fogGridDepthLayoutMultisampled,
    fogGridKernel,
    fogGridKernelMultisampled,
    fogGridLayout,
    fogIntegrateKernel,
    fogIntegrateLayout,
    fogKernel,
    fogKernelInterior,
    fogKernelMultisampled,
    fogLayout0,
    fogLayout0Multisampled,
    fogLayout1,
    fogReadLayout,
    readFog,
} from "./pipeline";

export { fogReadLayout, readFog, readFogInScatter } from "./pipeline";

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
        /** Hex sRGB color of the medium's own ambient light, which it scatters without any scene light,
         * so the scene fades toward it as fog thickens. */
        color: f32,
        /** Radiance of `color` in cd/m², scaled by camera exposure like `AmbientLight.brightness`.
         * The default reads as `color` at the default exposure, EV100 9.7. */
        brightness: f32,
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
        /** Fog endpoint distance from the camera along each ray, in world units (default 5000).
         * Fog begins at the near plane, like the grid; an endpoint at or before it gives no fog.
         * The perspective clear background lies at this endpoint; farther surfaces take the same fog. Values below
         * `viewDistance` end within the grid, without an analytic tail; nonpositive values disable fog.
         * Set beyond the camera's far plane to blend distant geometry into the sky. */
        maxDistance: f32,
        /** Nonphysical multiplier on the scene's direct and ambient light in the medium; 1 leaves it unadjusted.
         * Defaults to 1/π as a tuned look value. */
        lightIntensity: f32,
    },
    {
        defaults: () => ({
            density: 0.02,
            color: 0xb5c4d8,
            brightness: 1.2 * 2 ** EXPOSURE_EV100_BLENDER,
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

const emptyReadKey = {
    create: () => ({
        texture: null as GPUTexture | null,
        buffer: null as (TgpuBuffer<typeof FogGpu> & UniformFlag) | null,
        lighting: null as (TgpuBuffer<typeof LightingGpu> & UniformFlag) | null,
        groups: new Map<number, { buffer: object; group: GPUBindGroup }>(),
    }),
};

/**
 * Per-camera group for readFog at group 3, valid after the fog draw system and for the current frame.
 * Returns an identity read when FogPlugin or the Fog singleton is absent, or the camera has NoFog.
 * Identity reads require only CorePipelinePlugin, not standard rendering or a lighting resource.
 * The world owns the resources. Custom raw pipeline layouts must fill unused groups below 3 with
 * explicit empty layouts; TypeGPU leaves sparse group indices unfilled.
 */
export function fogReadBindGroup(world: World, eid: number): GPUBindGroup {
    const active = fogState(world).views.get(eid);
    if (active && world.only(FOG_SINGLETON_QUERY) >= 0 && !world.has(eid, NoFog))
        return active.readGroups[1 - active.parity];
    const view = world.resource(Views).get(eid);
    if (!view) throw new Error(`fog read requires a rendering camera: ${eid}`);
    const state = world.resource(emptyReadKey);
    const root = world.gpu.root;
    if (!state.texture) {
        state.texture = world.gpu.device.createTexture({
            dimension: "3d",
            size: [1, 1, 1],
            format: "rgba16float",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        world.own(state.texture);
        state.buffer = root.createBuffer(FogGpu).$usage("uniform");
        state.lighting = root.createBuffer(LightingGpu).$usage("uniform");
    }
    const buffer = world.resource(RenderContext).viewBuffers[view.slot];
    let cached = state.groups.get(eid);
    if (!cached || cached.buffer !== buffer) {
        cached = {
            buffer,
            group: root.unwrap(
                root.createBindGroup(fogReadLayout, {
                    grid: state.texture.createView(),
                    feedback: state.texture.createView(),
                    filtered: state.texture.createView(),
                    gridSampler: world.gpu.device.createSampler({
                        minFilter: "linear",
                        magFilter: "linear",
                    }),
                    view: buffer,
                    fog: state.buffer!,
                    lighting: state.lighting!,
                }),
            ),
        };
        state.groups.set(eid, cached);
    }
    return cached.group;
}

const fogSource = tgpu.fn(
    [FogInput],
    d.vec4f,
)((input) => {
    "use gpu";
    return readFog(input.worldPosition);
});

interface FogState {
    fog: {
        gridPipeline: TgpuComputePipeline | null;
        gridPipelineMultisampled: TgpuComputePipeline | null;
        filterPipeline: TgpuComputePipeline | null;
        integratePipeline: TgpuComputePipeline | null;
        pipelines: Map<string, GPURenderPipeline>;
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
            filtered: GPUTexture;
            surface: GPUTexture;
            temporal: GPUBuffer;
            staging: Float32Array<ArrayBuffer>;
            valid: boolean;
            config: Float32Array;
            parity: number;
            camera: number;
            gridGroups: GPUBindGroup[];
            filterGroups: GPUBindGroup[];
            integrateGroups: GPUBindGroup[];
            gridPass: GPUComputePassDescriptor;
            lighting: GPUBuffer;
            format: GPUTextureFormat;
            pipeline: GPURenderPipeline;
            interiorPipeline: GPURenderPipeline | null;
            readGroups: GPUBindGroup[];
            depth: GPUTextureView;
            slot: number;
            multisampled: boolean;
            bindGroup: GPUBindGroup;
            gridDepthGroup: GPUBindGroup;
        }
    >;
}

const fogStateKey = { create: () => createFogState() };
const createFogState = (): FogState => ({
    fog: {
        gridPipeline: null,
        gridPipelineMultisampled: null,
        filterPipeline: null,
        integratePipeline: null,
        pipelines: new Map(),
        buffer: null,
    },
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
    cam.filtered.destroy();
    cam.surface.destroy();
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

function applyPipeline(
    world: World,
    format: GPUTextureFormat,
    multisampled: boolean,
    interior = false,
): GPURenderPipeline {
    const pipelines = fogState(world).fog.pipelines;
    const samples = multisampled ? SAMPLE_COUNT : 1;
    const key = `${format}:${samples}:${interior}`;
    let pipeline = pipelines.get(key);
    if (!pipeline) {
        const device = world.gpu.device;
        const code = tgpu.resolve(
            [
                fullscreenVertex.$name("fullscreenVertex"),
                multisampled ? (interior ? fogKernelInterior : fogKernelMultisampled) : fogKernel,
            ],
            { names: "strict" },
        );
        const module = device.createShaderModule({ label: "fog-apply", code });
        // TypeGPU leaves sparse group indices unfilled; WebGPU requires explicit empty layouts.
        const empty = device.createBindGroupLayout({ entries: [] });
        pipeline = device.createRenderPipeline({
            label: "fog-apply",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [
                    world.gpu.root.unwrap(multisampled ? fogLayout0Multisampled : fogLayout0),
                    empty,
                    empty,
                    world.gpu.root.unwrap(fogReadLayout),
                ],
            }),
            vertex: { module, entryPoint: "fullscreenVertex" },
            fragment: {
                module,
                entryPoint: "fogApply",
                targets: [
                    {
                        format,
                        blend: {
                            color: { srcFactor: "one", dstFactor: "src-alpha", operation: "add" },
                            alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
                        },
                    },
                ],
            },
            depthStencil: {
                format: "depth32float",
                depthCompare: "always",
                depthWriteEnabled: false,
            },
            multisample: { count: samples },
        });
        pipelines.set(key, pipeline);
    }
    return pipeline;
}

/** Integrates each standard camera's grid after shadows, light culling and the depth prepass, before the main pass. */
const FogSystem: System = {
    name: "fog",
    group: "draw",
    after: [ShadowMapSystem, PrepassSystem],
    before: [MainPassSystem],
    update(world) {
        const _render = world.resource(RenderContext);
        const _fogState = world.resource(fogStateKey);

        if (
            !_fogState.fog.gridPipeline ||
            !_fogState.fog.gridPipelineMultisampled ||
            !_fogState.fog.filterPipeline ||
            !_fogState.fog.integratePipeline ||
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
            const format = view.framebufferFormat ?? _render.format;
            const lighting = world.resource(Lighting).buffer;
            let cam = fogState(world).views.get(eid);
            if (
                !cam ||
                cam.camera !== world.ref(eid) ||
                cam.width !== view.width ||
                cam.height !== view.height ||
                cam.lighting !== lighting ||
                cam.depth !== view.depth ||
                cam.slot !== view.slot ||
                cam.multisampled !== multisampled ||
                cam.format !== format
            ) {
                const cached =
                    cam && cam.format === format && cam.multisampled === multisampled
                        ? cam
                        : undefined;
                const pipeline = cached
                    ? cached.pipeline
                    : applyPipeline(world, format, multisampled);
                const interiorPipeline = cached
                    ? cached.interiorPipeline
                    : multisampled
                      ? applyPipeline(world, format, true, true)
                      : null;
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
                const surface = world.gpu.device.createTexture({
                    label: `fog-surface/${eid}`,
                    size: [grid.width, grid.height],
                    format: "r32float",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                const surfaceView = surface.createView();
                const feedback = [0, 1].map(() =>
                    world.gpu.device.createTexture({
                        label: `fog-lighting/${eid}`,
                        dimension: "3d",
                        size: [grid.width, grid.height, FOG_SLICES],
                        format: "rgba16float",
                        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                    }),
                );
                const filtered = world.gpu.device.createTexture({
                    label: `fog-filtered/${eid}`,
                    dimension: "3d",
                    size: [grid.width, grid.height, FOG_SLICES],
                    format: "rgba16float",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                const filteredView = filtered.createView();
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
                            feedback: feedbackViews[parity],
                            surface: surfaceView,
                            history: feedbackViews[1 - parity],
                            historySampler,
                            temporal,
                            view: _render.viewBuffers[view.slot],
                            fog: _fogState.fog.buffer!,
                        }),
                    ),
                );
                // The x pass writes into the spent history; the integration reads it there.
                const filterGroups = [0, 1].map((parity) =>
                    world.gpu.root.unwrap(
                        world.gpu.root.createBindGroup(fogFilterLayout, {
                            output: feedbackViews[1 - parity],
                            lighting: feedbackViews[parity],
                            surface: surfaceView,
                            view: _render.viewBuffers[view.slot],
                            fog: _fogState.fog.buffer!,
                        }),
                    ),
                );
                const integrateGroups = [0, 1].map((parity) =>
                    world.gpu.root.unwrap(
                        world.gpu.root.createBindGroup(fogIntegrateLayout, {
                            output: gridView,
                            filtered: filteredView,
                            lighting: feedbackViews[1 - parity],
                            surface: surfaceView,
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
                          depthTex: view.depth,
                      })
                    : world.gpu.root.createBindGroup(fogLayout0, { depthTex: view.depth });
                const readGroups = feedbackViews.map((feedback) =>
                    world.gpu.root.unwrap(
                        world.gpu.root.createBindGroup(fogReadLayout, {
                            grid: gridView,
                            filtered: filteredView,
                            feedback,
                            gridSampler,
                            view: _render.viewBuffers[view.slot],
                            fog: _fogState.fog.buffer!,
                            lighting,
                        }),
                    ),
                );
                cam = {
                    grid,
                    width: view.width,
                    height: view.height,
                    feedback,
                    filtered,
                    surface,
                    temporal,
                    staging: new Float32Array(32),
                    valid: false,
                    config: new Float32Array(FOG_FLOATS),
                    parity: 0,
                    camera: world.ref(eid),
                    gridGroups,
                    filterGroups,
                    integrateGroups,
                    gridPass: { label: `fog-grid/${eid}` },
                    lighting,
                    format,
                    pipeline,
                    interiorPipeline,
                    readGroups,
                    depth: view.depth,
                    slot: view.slot,
                    multisampled,
                    bindGroup: world.gpu.root.unwrap(group),
                    gridDepthGroup: world.gpu.root.unwrap(
                        multisampled
                            ? world.gpu.root.createBindGroup(fogGridDepthLayoutMultisampled, {
                                  depthTex: view.depth,
                              })
                            : world.gpu.root.createBindGroup(fogGridDepthLayout, {
                                  depthTex: view.depth,
                              }),
                    ),
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
            setComputePipeline(
                gridPass,
                world.gpu.root.unwrap(
                    cam.multisampled
                        ? _fogState.fog.gridPipelineMultisampled
                        : _fogState.fog.gridPipeline,
                ),
            );
            setComputeBindGroup(gridPass, 0, cam.gridGroups[cam.parity]);
            setComputeBindGroup(gridPass, 1, lights);
            setComputeBindGroup(gridPass, 2, cam.gridDepthGroup);
            const groupsX = Math.ceil(cam.grid.width / WORKGROUP);
            const groupsY = Math.ceil(cam.grid.height / WORKGROUP);
            dispatchWorkgroups2(gridPass, groupsX, groupsY);
            setComputePipeline(gridPass, world.gpu.root.unwrap(_fogState.fog.filterPipeline));
            setComputeBindGroup(gridPass, 0, cam.filterGroups[cam.parity]);
            dispatchWorkgroups3(gridPass, groupsX, groupsY, FOG_SLICES);
            setComputePipeline(gridPass, world.gpu.root.unwrap(_fogState.fog.integratePipeline));
            setComputeBindGroup(gridPass, 0, cam.integrateGroups[cam.parity]);
            dispatchWorkgroups2(gridPass, groupsX, groupsY);
            endComputePass(gridPass);
            cam.valid = true;
            cam.parity = 1 - cam.parity;
        }
    },
};

/**
 * volumetric atmosphere (fog + height fog). Opt-in: add `FogPlugin` to the plugin set, give the scene one
 * {@link Fog} singleton; fog requests the shared camera depth lane, and {@link NoFog} opts a camera out. The grid
 * composites opaque fog inside the main pass before transparent draws.
 */
export const FogPlugin: Plugin = {
    gpu: {},
    name: "Fog",
    components: [Fog, NoFog],

    systems: [FogSystem],
    dependencies: [RenderingPlugin, StandardRenderingPlugin],

    initialize(world) {
        const state = world.resource(fogStateKey);
        registerFogSource(world, "fog", {
            source: fogSource,
            layout: fogReadLayout,
            bindGroup: fogReadBindGroup,
        });
        world.own({ destroy: () => clearViews(state) });
        world.resource(RenderPhases).push({
            afterOpaque(world, eid, _view, pass) {
                const cam = fogState(world).views.get(eid);
                if (!cam) return;
                pass.setPipeline(cam.pipeline);
                pass.setBindGroup(0, cam.bindGroup);
                pass.setBindGroup(3, cam.readGroups[1 - cam.parity]);
                if (cam.interiorPipeline) {
                    pass.setPipeline(cam.interiorPipeline);
                    pass.draw(3);
                    pass.setPipeline(cam.pipeline);
                }
                pass.draw(3);
            },
        });
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
        _fogState.fog.gridPipelineMultisampled = root
            .createComputePipeline({ compute: fogGridKernelMultisampled })
            .$name("fog-grid-msaa");
        _fogState.fog.gridPipeline = root
            .createComputePipeline({ compute: fogGridKernel })
            .$name("fog-grid");
        _fogState.fog.filterPipeline = world.gpu.root
            .createComputePipeline({ compute: fogFilterKernel })
            .$name("fog-filter");
        _fogState.fog.integratePipeline = world.gpu.root
            .createComputePipeline({ compute: fogIntegrateKernel })
            .$name("fog-integrate");
        _fogState.fog.pipelines.clear();
        const format = world.resource(RenderContext).format;
        const pipeline = applyPipeline(world, format, false);
        const multisampledPipeline = applyPipeline(world, format, true);
        const interiorPipeline = applyPipeline(world, format, true, true);
        fogState(world).lights = null;
        clearViews(_fogState);

        precompile(world, "fog-grid", () => [
            _fogState.fog.gridPipeline!,
            _fogState.fog.gridPipelineMultisampled!,
        ]);
        precompile(world, "fog-filter", () => _fogState.fog.filterPipeline!);
        precompile(world, "fog-integrate", () => _fogState.fog.integratePipeline!);
        precompile(world, "fog", () => [pipeline]);
        precompile(world, "fog-msaa", () => [multisampledPipeline, interiorPipeline]);
    },

    dispose(world: World) {
        const _fogState = world.resource(fogStateKey);

        _fogState.fog.buffer?.destroy();
        _fogState.fog.buffer = null;
        _fogState.fog.gridPipeline = null;
        _fogState.fog.gridPipelineMultisampled = null;
        _fogState.fog.filterPipeline = null;
        _fogState.fog.integratePipeline = null;
        _fogState.fog.pipelines.clear();
        fogState(world).lights = null;
        clearViews(_fogState);
    },
};
