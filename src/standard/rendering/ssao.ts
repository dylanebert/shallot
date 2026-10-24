import type { TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import {
    Camera,
    CorePipelinePlugin,
    DepthPrepassRequests,
    MainPassSystem,
    NormalPrepassRequests,
    PrepassSystem,
    RenderContext,
    type View,
    Views,
} from "../../core/rendering";
import {
    component,
    f32,
    type Plugin,
    precompile,
    type System,
    u32,
    type World,
} from "../../engine";
import {
    aoKernel,
    aoLayout,
    copyKernel,
    copyLayout,
    copyMsaaLayout,
    denoiseKernel,
    denoiseLayout,
    mipKernel,
    mipLayout,
    SsaoSettings,
} from "./ssao-shaders";

export const ScreenSpaceAmbientOcclusionQualityLevel = {
    Low: 0,
    Medium: 1,
    High: 2,
    Ultra: 3,
    Custom: 4,
} as const;
/** Visibility bitmask AO on a camera. Requires ScreenSpaceAmbientOcclusionPlugin.
 * Occludes indirect PBR fragment lighting; vertex-lit materials are not re-lit.
 * Noise is fixed without temporal antialiasing.
 * radius and constantObjectThickness are in world units; radius must be positive.
 * Custom uses positive sliceCount and samplesPerSliceSide (1–16, fitting the 32-bit visibility mask).
 * Remove the component to disable the effect. */
export const ScreenSpaceAmbientOcclusion = component(
    "ScreenSpaceAmbientOcclusion",
    {
        qualityLevel: u32,
        radius: f32,
        constantObjectThickness: f32,
        sliceCount: u32,
        samplesPerSliceSide: u32,
    },
    {
        defaults: () => ({
            qualityLevel: ScreenSpaceAmbientOcclusionQualityLevel.High,
            radius: 0.7285,
            constantObjectThickness: 0.25,
            sliceCount: 3,
            samplesPerSliceSide: 3,
        }),
    },
);
const counts = [
    [1, 2],
    [2, 2],
    [3, 3],
    [9, 3],
] as const;
interface CameraAo {
    width: number;
    height: number;
    multisampled: boolean;
    depth: GPUTextureView;
    normal: GPUTextureView;
    viewBuffer: GPUBuffer;
    textures: GPUTexture[];
    output: GPUTextureView;
    settings: GPUBuffer;
    staging: Float32Array;
    words: Uint32Array;
    copy: GPUBindGroup;
    mips: GPUBindGroup[];
    ao: GPUBindGroup;
    denoise: GPUBindGroup;
}
export const ssaoState = {
    create: () => ({
        cameras: new Map<number, CameraAo>(),
        slots: [] as (GPUTextureView | null)[],
        white: null as GPUTextureView | null,
        noise: null as GPUTextureView | null,
        copy: null as TgpuComputePipeline | null,
        copyMsaa: null as TgpuComputePipeline | null,
        mip: null as TgpuComputePipeline | null,
        ao: null as TgpuComputePipeline | null,
        denoise: null as TgpuComputePipeline | null,
    }),
};
export function ssaoTexture(world: World, slot: number): GPUTextureView {
    const state = world.resource(ssaoState);
    if (!state.white) {
        const texture = world.gpu.device.createTexture({
            label: "standard:ssao-white",
            size: [1, 1],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.gpu.device.queue.writeTexture({ texture }, new Float32Array([1]), {}, [1, 1]);
        state.white = texture.createView();
    }
    return state.slots[slot] ?? state.white;
}
function retire(world: World, camera: CameraAo) {
    for (const texture of camera.textures) world.retireGpuResource(texture);
    world.retireGpuResource(camera.settings);
}
function allocate(world: World, view: View, multisampled: boolean): CameraAo {
    const { root, device } = world.gpu;
    const state = world.resource(ssaoState);
    const { width, height } = view;
    const textures: GPUTexture[] = [];
    function texture(format: GPUTextureFormat, mipLevelCount = 1) {
        const result = device.createTexture({
            label: "standard:ssao",
            size: [width, height],
            mipLevelCount,
            format,
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });
        textures.push(result);
        return result;
    }
    const mipCount = Math.min(5, Math.floor(Math.log2(Math.min(width, height))) + 1);
    // WebGPU-core compute storage is rgba16float; its red channel supplies Bevy's filtered depth.
    const depth = texture("rgba16float", mipCount);
    const normal = texture("rgba16float");
    const noisy = texture("r32float");
    const edges = texture("r32uint");
    const output = texture("r32float").createView();
    const settings = device.createBuffer({
        label: "standard:ssao-settings",
        size: d.sizeOf(SsaoSettings),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const staging = new Float32Array(d.sizeOf(SsaoSettings) / 4);
    const copyResources = {
        depth: view.depth!,
        normals: view.normal!,
        outputDepth: depth.createView({ baseMipLevel: 0, mipLevelCount: 1 }),
        outputNormals: normal.createView(),
    };
    const copy = root.unwrap(
        multisampled
            ? root.createBindGroup(copyMsaaLayout, copyResources)
            : root.createBindGroup(copyLayout, copyResources),
    );
    const mips: GPUBindGroup[] = [];
    // One output mip per dispatch keeps the port below WebGPU's four-storage-texture floor.
    for (let mip = 1; mip < mipCount; mip++)
        mips.push(
            root.unwrap(
                root.createBindGroup(mipLayout, {
                    depth: depth.createView({ baseMipLevel: mip - 1, mipLevelCount: 1 }),
                    output: depth.createView({ baseMipLevel: mip, mipLevelCount: 1 }),
                }),
            ),
        );
    const ao = root.unwrap(
        root.createBindGroup(aoLayout, {
            view: world.resource(RenderContext).viewBuffers[view.slot],
            settings,
            depth: depth.createView(),
            depthSampler: device.createSampler({
                magFilter: "linear",
                minFilter: "linear",
                mipmapFilter: "linear",
            }),
            normals: normal.createView(),
            noise: state.noise!,
            output: noisy.createView(),
            edges: edges.createView(),
        }),
    );
    const denoise = root.unwrap(
        root.createBindGroup(denoiseLayout, {
            noisy: noisy.createView(),
            edges: edges.createView(),
            output,
        }),
    );
    return {
        width,
        height,
        multisampled,
        depth: view.depth!,
        normal: view.normal!,
        viewBuffer: world.resource(RenderContext).viewBuffers[view.slot]!,
        textures,
        output,
        settings,
        staging,
        words: new Uint32Array(staging.buffer),
        copy,
        mips,
        ao,
        denoise,
    };
}
function dispatch(
    world: World,
    label: string,
    pipeline: TgpuComputePipeline,
    group: GPUBindGroup,
    width: number,
    height: number,
) {
    const pass = world
        .frameEncoder()!
        .beginComputePass({ label, timestampWrites: world.gpu.span?.(label) });
    pass.setPipeline(world.gpu.root.unwrap(pipeline));
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
}
const SsaoSystem: System = {
    name: "ScreenSpaceAmbientOcclusion",
    group: "draw",
    after: [PrepassSystem],
    before: [MainPassSystem],
    update(world) {
        const state = world.resource(ssaoState);
        const views = world.resource(Views);
        state.slots.fill(null);
        for (const [eid, camera] of state.cameras) {
            const view = views.get(eid);
            if (!world.has(eid, ScreenSpaceAmbientOcclusion) || !view?.framebuffer) {
                retire(world, camera);
                state.cameras.delete(eid);
            }
        }
        for (const [eid, view] of views) {
            if (
                !view.framebuffer ||
                !view.depth ||
                !view.normal ||
                !world.has(eid, ScreenSpaceAmbientOcclusion)
            )
                continue;
            const c = world.storage(ScreenSpaceAmbientOcclusion);
            const radius = c.radius.get(eid);
            const thickness = c.constantObjectThickness.get(eid);
            const quality = c.qualityLevel.get(eid);
            const slices = quality === 4 ? c.sliceCount.get(eid) : counts[quality]?.[0];
            const samples = quality === 4 ? c.samplesPerSliceSide.get(eid) : counts[quality]?.[1];
            if (
                !(radius > 0) ||
                !Number.isFinite(radius) ||
                !(thickness >= 0) ||
                !Number.isFinite(thickness) ||
                !slices ||
                !samples ||
                samples > 16
            )
                throw new Error("invalid ScreenSpaceAmbientOcclusion settings");
            const msaa = world.storage(Camera).antialias.get(eid) !== 0;
            let camera = state.cameras.get(eid);
            if (
                !camera ||
                camera.width !== view.width ||
                camera.height !== view.height ||
                camera.multisampled !== msaa ||
                camera.depth !== view.depth ||
                camera.normal !== view.normal ||
                camera.viewBuffer !== world.resource(RenderContext).viewBuffers[view.slot]
            ) {
                if (camera) retire(world, camera);
                camera = allocate(world, view, msaa);
                state.cameras.set(eid, camera);
            }
            camera.staging.set(view.viewFromWorld!, 0);
            camera.staging[16] = radius;
            camera.staging[17] = thickness;
            const staging = world.resource(RenderContext).viewStaging;
            const at = view.slot * 64;
            camera.staging[18] = Math.hypot(staging[at], staging[at + 4], staging[at + 8]);
            camera.words[19] = slices;
            camera.words[20] = samples;
            world.gpu.device.queue.writeBuffer(camera.settings, 0, camera.staging);
            dispatch(
                world,
                "standard:ssao-preprocess",
                msaa ? state.copyMsaa! : state.copy!,
                camera.copy,
                view.width,
                view.height,
            );
            for (let mip = 0; mip < camera.mips.length; mip++)
                dispatch(
                    world,
                    `standard:ssao-mip${mip + 1}`,
                    state.mip!,
                    camera.mips[mip]!,
                    Math.max(1, view.width >> (mip + 1)),
                    Math.max(1, view.height >> (mip + 1)),
                );
            dispatch(
                world,
                "standard:ssao-evaluate",
                state.ao!,
                camera.ao,
                view.width,
                view.height,
            );
            dispatch(
                world,
                "standard:ssao-denoise",
                state.denoise!,
                camera.denoise,
                view.width,
                view.height,
            );
            state.slots[view.slot] = camera.output;
        }
    },
};
/** Bevy's visibility bitmask AO, supplied normals and fixed Hilbert/R2 noise.
 * Multisampled prepasses use sample zero for depth and normal. */
export const ScreenSpaceAmbientOcclusionPlugin: Plugin = {
    name: "ScreenSpaceAmbientOcclusion",
    dependencies: [CorePipelinePlugin],
    components: [ScreenSpaceAmbientOcclusion],
    systems: [SsaoSystem],
    initialize(world) {
        const requested = (world: World, eid: number) =>
            world.has(eid, ScreenSpaceAmbientOcclusion);
        world.resource(DepthPrepassRequests).push(requested);
        world.resource(NormalPrepassRequests).push(requested);
    },
    warm(world) {
        const state = world.resource(ssaoState);
        const { root, device } = world.gpu;
        const noise = new Uint32Array(64 * 64);
        // Bevy uploads an [x][y] LUT; texture coordinates therefore address hilbert(y, x).
        for (let y = 0; y < 64; y++)
            for (let x = 0; x < 64; x++) {
                let hx = y,
                    hy = x,
                    index = 0;
                for (let level = 32; level > 0; level >>= 1) {
                    const rx = (hx & level) > 0 ? 1 : 0;
                    const ry = (hy & level) > 0 ? 1 : 0;
                    index += level * level * ((3 * rx) ^ ry);
                    if (ry === 0) {
                        if (rx === 1) {
                            hx = 63 - hx;
                            hy = 63 - hy;
                        }
                        [hx, hy] = [hy, hx];
                    }
                }
                noise[y * 64 + x] = index;
            }
        const texture = device.createTexture({
            label: "standard:ssao-hilbert",
            size: [64, 64],
            format: "r32uint",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture({ texture }, noise, { bytesPerRow: 256 }, [64, 64]);
        state.noise = texture.createView();
        state.copy = root.createComputePipeline({ compute: copyKernel(false) });
        state.copyMsaa = root.createComputePipeline({ compute: copyKernel(true) });
        state.mip = root.createComputePipeline({ compute: mipKernel });
        state.ao = root.createComputePipeline({ compute: aoKernel() });
        state.denoise = root.createComputePipeline({ compute: denoiseKernel });
        precompile(world, "standard:ssao", () => [
            state.copy,
            state.copyMsaa,
            state.mip,
            state.ao,
            state.denoise,
        ]);
    },
};
