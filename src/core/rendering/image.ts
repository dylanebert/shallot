import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { checkTextureLimits, Registry, type World } from "../../engine";
import { fullscreenVertex } from "./tonemapping";

/** @internal Native 2D images, indexed by independent color and data ids. */
export const imageTextures = {
    create: () => ({
        textures: [[], []] as GPUTexture[][],
        samplers: new Map<string, GPUSampler>(),
        imageSamplers: [[], []] as GPUSampler[][],
        fallback: [] as GPUTexture[],
        warned: [new Set<number>(), new Set<number>()],
    }),
};

/** @internal Hardware sampler shared by images with equal effective descriptors. */
export function imageSampler(world: World, id: number, isSrgb = true): GPUSampler {
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

/** @internal Missing native textures bind a role-local fallback and warn once per id. */
export function imageTexture(world: World, id: number, isSrgb = true): GPUTexture {
    const state = world.resource(imageTextures);
    const role = isSrgb ? 0 : 1;
    const texture = state.textures[role][id];
    if (texture) return texture;
    if (id >= 0 && !state.warned[role].has(id)) {
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
 * nonnegative id.
 */
export function imageView(
    world: World,
    id: number,
    settings: Pick<ImageLoaderSettings, "isSrgb"> = {},
): GPUTextureView {
    return imageTexture(world, id, settings.isSrgb !== false).createView();
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
 * Register before warm, normally in a plugin's initialize. The returned id selects an image in the
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
    if (typeof source === "string") {
        return images.register({
            name: `url:${JSON.stringify([source, sampler])}`,
            source,
            sampler,
        });
    }
    let next = images.size;
    while (images.id(`blob:${next}`) !== undefined) next++;
    return images.register({ name: `blob:${next}`, source, sampler });
}

const PIXEL_PNG =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAADUlEQVR42mNgYGD4DwABBAEAX+XBlwAAAABJRU5ErkJggg==";

function transparentPixel(): Blob {
    const bytes = Uint8Array.from(atob(PIXEL_PNG), (c) => c.charCodeAt(0));
    return new Blob([bytes], { type: "image/png" });
}

/** Decode into native-size 2D textures with full mips and role-local 1×1 fallbacks. */
export async function warmImages(world: World): Promise<void> {
    const images = world.resource(Images);
    const dataImages = world.resource(DataImages);
    const device = world.gpu.device;
    const loadImages = async (
        images: ReturnType<typeof Images.create>,
        format: GPUTextureFormat,
    ) => {
        const role = format === "rgba8unorm-srgb" ? 0 : 1;
        const own = world.resource(imageTextures);
        const fallback = device.createTexture({
            label: "image-native-fallback",
            size: [1, 1],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        world.own(fallback);
        own.fallback[role] = fallback;
        if (images.size > 0 && typeof createImageBitmap !== "undefined") {
            const blobs = await Promise.all(
                Array.from({ length: images.size }, async (_, id) => {
                    const source = images.get(images.name(id)!)!.source;
                    if (typeof source !== "string") return source;
                    try {
                        const response = await fetch(source);
                        if (!response.ok) throw new Error(`${response.status}`);
                        return await response.blob();
                    } catch (error) {
                        console.warn(`[Rendering] image ${id} (${source}) failed to load:`, error);
                        return transparentPixel();
                    }
                }),
            );
            const native = await decodeImages(blobs);
            for (const [id, bitmap] of native.entries()) {
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
                own.textures[role][id] = texture;
                imageSampler(world, id, role === 0);
                device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [
                    bitmap.width,
                    bitmap.height,
                ]);
                genMipmapsLayer(world, device, texture, 0, texture.mipLevelCount, format);
                bitmap.close();
            }
        }
    };
    await loadImages(images, "rgba8unorm-srgb");
    await loadImages(dataImages, "rgba8unorm");
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

function decodeImages(blobs: Blob[]): Promise<ImageBitmap[]> {
    // glTF image bytes define the color values. Straight alpha preserves cutout and blend edges.
    return Promise.all(
        blobs.map((blob) =>
            createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" }),
        ),
    );
}
