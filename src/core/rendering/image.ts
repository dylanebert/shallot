// Array textures support tiling and per-instance layer selection in one draw, but all layers must
// share dimensions. Group sources by size and resize each group to a common capped square; use
// sRGB for color maps and a linear format for data maps so sampling preserves their values.

import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { checkTextureLimits, Registry, type World } from "../../engine";
import { fullscreenVertex } from "./tonemapping";

/** World-local size-class image arrays and sampler, populated at RenderingPlugin warm. */
export const ImageBindings = {
    create: () => ({
        images: null! as GPUTexture,
        images1024: null! as GPUTexture,
        images2048: null! as GPUTexture,
        dataImages: null! as GPUTexture,
        dataImages1024: null! as GPUTexture,
        dataImages2048: null! as GPUTexture,
        imageSampler: null! as GPUSampler,
        imageSettings: null! as GPUTexture,
    }),
};

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
    /** False registers a linear data image; its id belongs to the data array, not the color array. */
    isSrgb?: boolean;
    sampler?: ImageSamplerDescriptor;
}

/** World-local image sources. An empty registry still allocates one fallback layer at warm. */
export const Images = {
    create: () => new Registry<{ name: string; source: string | Blob; sampler: number[] }>(),
};
/** World-local linear images; ids occupy an independent array from color images. */
export const DataImages = { create: Images.create };

/**
 * Register before warm, normally in a plugin's initialize. The returned id selects an image in the
 * world's shared sRGB arrays, or its linear data arrays when isSrgb is false. Color and data ids
 * are independent; a consumer must sample the arrays matching the registration's role.
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

/** Decode registered images into size-class arrays with full mips; empty bands bind 1×1 fallbacks. */
export async function warmImages(world: World): Promise<void> {
    const images = world.resource(Images);
    const dataImages = world.resource(DataImages);
    const device = world.gpu.device;
    const locations: number[][][] = [];
    const loadArray = async (
        images: ReturnType<typeof Images.create>,
        format: GPUTextureFormat,
    ) => {
        const bands: ImageBitmap[][] = [[], [], []];
        const rows: number[][] = [];
        locations.push(rows);
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
            for (const bitmap of native) {
                const side = Math.max(bitmap.width, bitmap.height);
                const band = side > 1024 ? 2 : side > 512 ? 1 : 0;
                rows.push([band + 1, bands[band].length]);
                bands[band].push(bitmap);
            }
        }
        const textures: GPUTexture[] = [];
        for (const band of bands) {
            const texture =
                band.length > 0
                    ? await arrayFromBitmaps(world, device, band, 2048, format)
                    : device.createTexture({
                          label: "images-fallback",
                          size: [1, 1, 1],
                          format,
                          usage: GPUTextureUsage.TEXTURE_BINDING,
                      });
            world.own(texture);
            textures.push(texture);
        }
        return textures;
    };
    const bindings = world.resource(ImageBindings);
    [bindings.images, bindings.images1024, bindings.images2048] = await loadArray(
        images,
        "rgba8unorm-srgb",
    );
    [bindings.dataImages, bindings.dataImages1024, bindings.dataImages2048] = await loadArray(
        dataImages,
        "rgba8unorm",
    );
    const colorCount = Math.max(1, images.size);
    const settingsCount = colorCount + Math.max(1, dataImages.size);
    // A sampled texture keeps sampler metadata out of the storage-buffer binding floor.
    const imageSettings = device.createTexture({
        label: "image-settings",
        size: [1, settingsCount + 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // R packs the seven sampler bits; GB hold the 16-bit layer; A marks packed rows and the band.
    // Unmarked rows retain the original sampler bytes and band zero at layer=id.
    const data = new Uint8Array((settingsCount + 1) * 4);
    for (const [role, registry] of [images, dataImages].entries()) {
        const offset = role === 0 ? 0 : colorCount;
        for (let id = 0; id < Math.max(1, registry.size); id++) {
            const row = (offset + id) * 4;
            const sampler = registry.get(registry.name(id)!)?.sampler ?? [0, 0, 1, 3];
            const location = locations[role][id];
            if (location) {
                const [band, layer] = location;
                data.set(
                    [
                        sampler[0] | (sampler[1] << 2) | (sampler[2] << 4) | (sampler[3] << 5),
                        layer & 255,
                        layer >>> 8,
                        128 | (band - 1),
                    ],
                    row,
                );
            } else {
                data.set(sampler, row);
            }
        }
    }
    // The final row's marker distinguishes the color count from a legacy fallback sampler row.
    data.set([colorCount & 255, colorCount >>> 8, 0, 64], settingsCount * 4);
    device.queue.writeTexture({ texture: imageSettings }, data, { bytesPerRow: 4 }, [
        1,
        settingsCount + 1,
    ]);
    world.own(imageSettings);
    bindings.imageSettings = imageSettings;
    bindings.imageSampler = device.createSampler({
        label: "images",
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
    });
}

/** mip levels for a square texture of `size` px: the full chain down to 1×1. */
export function mipLevels(size: number): number {
    return Math.floor(Math.log2(size)) + 1;
}

/** Square layer size in pixels: largest source dimension, capped at `cap` and floored at 1. */
export function commonSize(dims: { w: number; h: number }[], cap = 2048): number {
    let max = 1;
    for (const dim of dims) max = Math.max(max, dim.w, dim.h);
    return Math.min(cap, Math.max(1, max));
}

const ARRAY_FORMAT: GPUTextureFormat = "rgba8unorm-srgb";

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

// Per-layer submissions let staged builders budget a complete mip chain per frame.
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

/**
 * decode `blobs` to a `texture_2d_array`: one layer per blob, all resized to a common (capped) square
 * size, sRGB-encoded, with a full mip chain. The returned texture binds as a surface's
 * `{ type: "texture", dimension: "2d-array" }` and is sampled `array[layer]`. Async (decode); call
 * from a load path, not a hot frame.
 */
export async function imageArray(
    world: World,
    device: GPUDevice,
    blobs: Blob[],
    cap = 2048,
    format: GPUTextureFormat = ARRAY_FORMAT,
): Promise<GPUTexture> {
    return arrayFromBitmaps(world, device, await decodeImages(blobs), cap, format);
}

function decodeImages(blobs: Blob[]): Promise<ImageBitmap[]> {
    // glTF image bytes define the color values. Straight alpha preserves cutout and blend edges.
    return Promise.all(
        blobs.map((blob) =>
            createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" }),
        ),
    );
}

/**
 * Allocates an unfilled array whose layers are `size` pixels square, with `levels` mip levels.
 * Synchronous allocation is separate from uploadLayer so callers can stage uploads across frames.
 * Refuses layer counts and dimensions beyond device limits before texture creation.
 */
export function allocArray(
    device: GPUDevice,
    size: number,
    layers: number,
    levels: number,
    format: GPUTextureFormat = ARRAY_FORMAT,
): GPUTexture {
    checkTextureLimits(
        "[render] an image array (glTF baseColor / sprite atlas)",
        { width: size, height: size, layers },
        device.limits,
        "Reduce the number of distinct textures (array layers) or lower the size cap.",
    );
    return device.createTexture({
        label: "image-array",
        size: { width: size, height: size, depthOrArrayLayers: Math.max(1, layers) },
        format,
        mipLevelCount: levels,
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
}

/**
 * Resamples an already-decoded bitmap to `size` pixels square without decoding again, then uploads
 * one array layer and its mip chain. A staged caller can budget one complete layer per frame.
 */
export async function uploadLayer(
    world: World,
    device: GPUDevice,
    texture: GPUTexture,
    bitmap: ImageBitmap,
    layer: number,
    size: number,
    levels: number,
    format: GPUTextureFormat = ARRAY_FORMAT,
): Promise<void> {
    const native = device.createTexture({
        label: "image-resample-source",
        size: [bitmap.width, bitmap.height],
        format,
        mipLevelCount: mipLevels(Math.max(bitmap.width, bitmap.height)),
        usage:
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture: native },
        { width: bitmap.width, height: bitmap.height },
    );
    try {
        genMipmapsLayer(world, device, native, 0, native.mipLevelCount, format);
        // Implicit derivatives choose the source footprint's LOD; trilinear filtering blends
        // its neighboring mip averages rather than sampling distant level-zero texels.
        const group = world.gpu.root.createBindGroup(blitLayout, {
            src: native.createView(),
            samp: device.createSampler({
                magFilter: "linear",
                minFilter: "linear",
                mipmapFilter: "linear",
            }),
        });
        const encoder = device.createCommandEncoder({ label: "image-resample" });
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: texture.createView({
                        dimension: "2d",
                        baseArrayLayer: layer,
                        arrayLayerCount: 1,
                        baseMipLevel: 0,
                        mipLevelCount: 1,
                    }),
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.setViewport(0, 0, size, size, 0, 1);
        blitPipeline(world, format).with(group).with(pass).draw(3);
        pass.end();
        device.queue.submit([encoder.finish()]);
    } finally {
        native.destroy();
    }
    genMipmapsLayer(world, device, texture, layer, levels, format);
}

/**
 * Uploads already-decoded bitmaps sequentially as common-size square array layers with a full mip chain;
 * `cap` is in pixels. Color defaults to sRGB; pass a linear format for data maps.
 * Uploads every layer before returning rather than spreading the work across frames. Callers that need
 * staged uploads use allocArray and uploadLayer; decoding remains the caller's choice.
 */
export async function arrayFromBitmaps(
    world: World,
    device: GPUDevice,
    native: ImageBitmap[],
    cap = 2048,
    format: GPUTextureFormat = ARRAY_FORMAT,
): Promise<GPUTexture> {
    const size = commonSize(
        native.map((b) => ({ w: b.width, h: b.height })),
        cap,
    );
    const levels = mipLevels(size);
    const texture = allocArray(device, size, native.length, levels, format);
    for (let layer = 0; layer < native.length; layer++) {
        await uploadLayer(world, device, texture, native[layer], layer, size, levels, format);
    }
    return texture;
}
