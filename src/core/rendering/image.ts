import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { checkTextureLimits, precompile, Registry, type World } from "../../engine";
import { fullscreenVertex } from "./tonemapping";

/** @internal Native 2D images, indexed by independent color and data ids. */
export const imageTextures = {
    create: () => ({
        textures: [[], []] as GPUTexture[][],
        loads: [[], []] as (Promise<void> | undefined)[][],
        warm: false,
        mipLoads: new Map<GPUTextureFormat, Promise<void>>(),
        samplers: new Map<string, GPUSampler>(),
        imageSamplers: [[], []] as GPUSampler[][],
        fallback: [] as GPUTexture[],
        warned: [new Set<number>(), new Set<number>()],
        // Ids registered before warm on a host with no image decoder, such as Node; they bind silently.
        undecoded: [0, 0],
    }),
};

/**
 * After warm, return the world-owned hardware sampler shared by images with equal effective
 * descriptors. False isSrgb selects the independent data ids. Unknown ids use the default sampler.
 */
export function imageSampler(
    world: World,
    id: number,
    settings: Pick<ImageLoaderSettings, "isSrgb"> = {},
): GPUSampler {
    const isSrgb = settings.isSrgb !== false;
    const state = world.resource(imageTextures);
    const role = isSrgb ? 0 : 1;
    const cached = state.imageSamplers[role][id];
    if (cached) return cached;
    const registry = world.resource(isSrgb ? Images : DataImages);
    const values = registry.get(registry.name(id)!)?.sampler ?? [0, 0, 1, 3];
    const key = values.join(",");
    const cache = world.resource(imageTextures).samplers;
    let sampler = cache.get(key);
    if (!sampler) {
        const address = ["clamp-to-edge", "repeat", "mirror-repeat"] as const;
        sampler = world.gpu.device.createSampler({
            addressModeU: address[values[0]],
            addressModeV: address[values[1]],
            magFilter: values[2] ? "linear" : "nearest",
            minFilter: values[3] & 1 ? "linear" : "nearest",
            mipmapFilter: values[3] & 2 ? "linear" : "nearest",
        });
        cache.set(key, sampler);
    }
    state.imageSamplers[role][id] = sampler;
    return sampler;
}

/**
 * After warm, return a registered image's native-size texture, owned by the world until disposal;
 * callers must not destroy it. False isSrgb selects the independent data ids. An unloaded or unknown
 * id returns a role-local 1×1 fallback and warns once per nonnegative id, except an id registered
 * before warm on a host with no image decoder.
 */
export function imageTexture(
    world: World,
    id: number,
    settings: Pick<ImageLoaderSettings, "isSrgb"> = {},
): GPUTexture {
    const isSrgb = settings.isSrgb !== false;
    const state = world.resource(imageTextures);
    const role = isSrgb ? 0 : 1;
    const texture = state.textures[role][id];
    if (texture) return texture;
    if (id >= state.undecoded[role] && !state.warned[role].has(id)) {
        state.warned[role].add(id);
        console.warn(
            `[Rendering] ${isSrgb ? "color" : "data"} image ${id} has no native texture; binding fallback`,
        );
    }
    return state.fallback[role];
}

/**
 * After warm, return a 2D view of a registered image's native-size texture. The world owns the
 * texture until disposal; callers must not destroy it. False isSrgb selects the independent data
 * ids. An unloaded or unknown id returns a role-local 1×1 fallback view and warns once per
 * nonnegative id, except an id registered before warm on a host with no image decoder.
 */
export function imageView(
    world: World,
    id: number,
    settings: Pick<ImageLoaderSettings, "isSrgb"> = {},
): GPUTextureView {
    return imageTexture(world, id, settings).createView();
}

/** Addressing outside the image, matching Bevy's ImageAddressMode. */
export const ImageAddressMode = { ClampToEdge: 0, Repeat: 1, MirrorRepeat: 2 } as const;
export type ImageAddressMode = (typeof ImageAddressMode)[keyof typeof ImageAddressMode];

/** Texel mixing, matching Bevy's ImageFilterMode. */
export const ImageFilterMode = { Nearest: 0, Linear: 1 } as const;
export type ImageFilterMode = (typeof ImageFilterMode)[keyof typeof ImageFilterMode];

/** The 2D addressing and filters of Bevy's ImageSamplerDescriptor; anisotropy stays 1. */
export interface ImageSamplerDescriptor {
    addressModeU?: ImageAddressMode;
    addressModeV?: ImageAddressMode;
    magFilter?: ImageFilterMode;
    minFilter?: ImageFilterMode;
    /** Filtering between mip levels, matching Bevy's mipmap_filter. */
    mipmapFilter?: ImageFilterMode;
}

/** Color interpretation and sampling, matching Bevy's ImageLoaderSettings. */
export interface ImageLoaderSettings {
    /** False registers a linear data image; its id belongs to the data registry, not the color registry. */
    isSrgb?: boolean;
    sampler?: ImageSamplerDescriptor;
}

/** World-local color image sources, uploaded to native-size textures at warm. */
export const Images = {
    create: () => new Registry<{ name: string; source: string | Blob; sampler: number[] }>(),
};
/** World-local linear image sources; ids are independent of color image ids. */
export const DataImages = { create: Images.create };

/**
 * Register an image; after warm its fetch, decode and upload start immediately. Await prepareImages
 * before drawing to avoid the role fallback while loading. The returned id selects an image in the
 * world's native-size sRGB textures, or its linear data textures when isSrgb is false. Color and
 * data ids are independent; a consumer must select the registration's role.
 * Within one role, URLs with the same effective sampler return the same id; different samplers return distinct ids.
 * Each Blob registration returns a new id, distinct from all URLs. Omitted sampler fields take
 * Bevy's ImagePlugin default, ImageSamplerDescriptor::linear(): clamp-to-edge, linear texel filters
 * and linear filtering between levels of the full mip chain.
 */
export function registerImage(
    world: World,
    source: string | Blob,
    settings: ImageLoaderSettings = {},
): number {
    const images = world.resource(settings.isSrgb === false ? DataImages : Images);
    world.resource(imageTextures).imageSamplers[settings.isSrgb === false ? 1 : 0].length = 0;
    const s = settings.sampler;
    // Minification and mip filters share the last sampler value's low two bits.
    const sampler = [
        s?.addressModeU ?? 0,
        s?.addressModeV ?? 0,
        s?.magFilter ?? 1,
        (s?.minFilter ?? 1) | ((s?.mipmapFilter ?? 1) << 1),
    ];
    let name: string;
    if (typeof source === "string") name = `url:${JSON.stringify([source, sampler])}`;
    else {
        let next = images.size;
        while (images.id(`blob:${next}`) !== undefined) next++;
        name = `blob:${next}`;
    }
    const id = images.register({
        name,
        source,
        sampler,
    });
    if (world.resource(imageTextures).warm)
        startImage(world, id, settings.isSrgb === false ? 1 : 0);
    return id;
}

const PIXEL_PNG =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII=";

function transparentPixel(): Blob {
    const bytes = Uint8Array.from(atob(PIXEL_PNG), (c) => c.charCodeAt(0));
    return new Blob([bytes], { type: "image/png" });
}

const IMAGE_FORMATS = ["rgba8unorm-srgb", "rgba8unorm"] as const;

/** Decode into native-size 2D textures with full mips and role-local 1×1 fallbacks. */
export async function warmImages(world: World): Promise<void> {
    const state = world.resource(imageTextures);
    for (const [role, format] of IMAGE_FORMATS.entries()) {
        const fallback = world.gpu.device.createTexture({
            label: "image-native-fallback",
            size: [1, 1],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        world.own(fallback);
        state.fallback[role] = fallback;
    }
    state.warm = true;
    for (const [role, images] of [world.resource(Images), world.resource(DataImages)].entries()) {
        if (typeof createImageBitmap === "undefined") state.undecoded[role] = images.size;
        for (let id = 0; id < images.size; id++) startImage(world, id, role);
    }
    await prepareImages(world);
}

/**
 * After warm, await the native textures and mip uploads of all color and data images registered
 * before this call. Later registrations are not included. Before warm this is unnecessary and
 * resolves without loading. Failed fetches or decodes warn and resolve with a transparent pixel. Disposal
 * resolves pending preparation without creating or uploading textures for the disposed world.
 */
export async function prepareImages(world: World): Promise<void> {
    if (world.disposed) return;
    const state = world.resource(imageTextures);
    await Promise.all([...state.loads[0], ...state.loads[1]]);
}

function startImage(world: World, id: number, role: number): void {
    const state = world.resource(imageTextures);
    if (state.loads[role][id]) return;
    const signal = world.signal;
    const load = loadImage(world, id, role);
    state.loads[role][id] = new Promise<void>((resolve, reject) => {
        const abort = () => resolve();
        signal.addEventListener("abort", abort, { once: true });
        load.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
    });
}

async function loadImage(world: World, id: number, role: number): Promise<void> {
    if (typeof createImageBitmap === "undefined") return;
    const state = world.resource(imageTextures);
    const images = world.resource(role === 0 ? Images : DataImages);
    const source = images.get(images.name(id)!)!.source;
    const device = world.gpu.device;
    const decode = (blob: Blob) =>
        createImageBitmap(blob, {
            premultiplyAlpha: "none",
            colorSpaceConversion: "none",
        });
    let bitmap: ImageBitmap;
    try {
        let blob: Blob;
        if (typeof source === "string") {
            const response = await fetch(source, { signal: world.signal });
            if (!response.ok) throw new Error(`${response.status}`);
            blob = await response.blob();
        } else blob = source;
        if (world.disposed) return;
        // glTF bytes define the color values; straight alpha preserves cutout and blend edges.
        bitmap = await decode(blob);
    } catch (error) {
        if (world.disposed) return;
        console.warn(`[Rendering] image ${id} (${source}) failed to load:`, error);
        bitmap = await decode(transparentPixel());
    }
    try {
        if (world.disposed) return;
        const format = IMAGE_FORMATS[role];
        if (Math.max(bitmap.width, bitmap.height) > 1) {
            let ready = state.mipLoads.get(format);
            if (!ready) {
                ready = precompile(world, `image-mipmap:${role}`, () => [
                    blitPipeline(world, format),
                ]);
                state.mipLoads.set(format, ready);
            }
            await ready;
            if (world.disposed) return;
        }
        checkTextureLimits(
            "[render] a native image",
            { width: bitmap.width, height: bitmap.height, layers: 1 },
            device.limits,
            "Reduce the image dimensions.",
        );
        const texture = device.createTexture({
            label: `image-native:${role}:${id}`,
            size: [bitmap.width, bitmap.height],
            format,
            mipLevelCount: mipLevels(Math.max(bitmap.width, bitmap.height)),
            usage:
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        world.own(texture);
        state.textures[role][id] = texture;
        imageSampler(world, id, { isSrgb: role === 0 });
        device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [
            bitmap.width,
            bitmap.height,
        ]);
        genMipmapsLayer(world, device, texture, 0, texture.mipLevelCount, format);
    } finally {
        bitmap.close();
    }
}

/** Full mip-chain length down to 1×1, from the texture's largest dimension in pixels. */
export function mipLevels(size: number): number {
    return Math.floor(Math.log2(size)) + 1;
}

// Color mips average in linear space through sRGB sampling/store conversion; data maps use linear formats.
const blitLayout = tgpu.bindGroupLayout({
    src: { texture: d.texture2d(d.f32) },
    samp: { sampler: "filtering" },
});

const blitFs = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
})((input) => {
    "use gpu";
    return std.textureSample(blitLayout.$.src, blitLayout.$.samp, input.uv);
});

// Pipeline caches are world-owned so a rebuilt world cannot reuse a torn-down device's pipelines.
export const blitPipelinesKey = { create: () => new Map<string, TgpuRenderPipeline>() };

function blitPipelines(world: World): Map<string, TgpuRenderPipeline> {
    return world.resource(blitPipelinesKey);
}

export function initializeImageState(world: World): void {
    world.resource(blitPipelinesKey);
}

function blitPipeline(world: World, format: GPUTextureFormat): TgpuRenderPipeline {
    const cached = blitPipelines(world).get(format);
    if (cached) return cached;
    const pipeline = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment: blitFs,
            targets: { format },
            primitive: { topology: "triangle-list" },
        })
        .$name("image-mipmap");
    blitPipelines(world).set(format, pipeline);
    return pipeline;
}

export function genMipmapsLayer(
    world: World,
    device: GPUDevice,
    texture: GPUTexture,
    layer: number,
    levels: number,
    format: GPUTextureFormat,
): void {
    if (levels <= 1) return;
    const pipeline = blitPipeline(world, format);
    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    const encoder = device.createCommandEncoder({ label: "image-mipmap" });
    const sub = (level: number) =>
        texture.createView({
            dimension: "2d",
            baseMipLevel: level,
            mipLevelCount: 1,
            baseArrayLayer: layer,
            arrayLayerCount: 1,
        });
    for (let level = 1; level < levels; level++) {
        const group = world.gpu.root.createBindGroup(blitLayout, {
            src: sub(level - 1),
            samp: sampler,
        });
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: sub(level), loadOp: "clear", storeOp: "store" }],
        });
        pipeline.with(group).with(pass).draw(3);
        pass.end();
    }
    device.queue.submit([encoder.finish()]);
}
