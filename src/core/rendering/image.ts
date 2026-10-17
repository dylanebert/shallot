// Array textures support tiling and per-instance layer selection in one draw, but all layers must
// share dimensions. Resize varied sources to a common capped square; use the default sRGB format
// for color maps and pass a linear format for data maps so sampling does not corrupt their values.

import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { checkTextureLimits, Registry, type World } from "../../engine";

/** World-local decoded image array and sampler, populated at RenderingPlugin warm. */
export const ImageBindings = {
    create: () => ({ images: null! as GPUTexture, imageSampler: null! as GPUSampler }),
};

/** World-local image sources. An empty registry still allocates one fallback layer at warm. */
export const Images = { create: () => new Registry<{ name: string; source: string | Blob }>() };

/**
 * Register before warm, normally in a plugin's initialize. The returned id selects a layer of the
 * world's shared sRGB image array, used by Sprite and StandardMaterial.baseColorTexture.
 * Repeated URLs return the same id; each Blob registration returns a new id, distinct from all URLs.
 */
export function registerImage(world: World, source: string | Blob): number {
    const images = world.resource(Images);
    if (typeof source === "string") return images.register({ name: `url:${source}`, source });
    let next = images.size;
    while (images.id(`blob:${next}`) !== undefined) next++;
    return images.register({ name: `blob:${next}`, source });
}

const PIXEL_PNG =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAADUlEQVR42mNgYGD4DwABBAEAX+XBlwAAAABJRU5ErkJggg==";

function transparentPixel(): Blob {
    const bytes = Uint8Array.from(atob(PIXEL_PNG), (c) => c.charCodeAt(0));
    return new Blob([bytes], { type: "image/png" });
}

export async function warmImages(world: World): Promise<void> {
    const images = world.resource(Images);
    const device = world.gpu.device;
    let texture: GPUTexture;
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
        texture = await imageArray(world, device, blobs);
    } else {
        texture = device.createTexture({
            label: "images-fallback",
            size: [1, 1, 1],
            format: "rgba8unorm-srgb",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
    }
    world.own(texture);
    const bindings = world.resource(ImageBindings);
    bindings.images = texture;
    bindings.imageSampler = device.createSampler({
        label: "images",
        magFilter: "linear",
        minFilter: "linear",
    });
    world.gpu.textures.set("images", texture);
    world.gpu.samplers.set("imageSampler", bindings.imageSampler);
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

const blitVs = tgpu.vertexFn({
    in: { vi: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, uv: d.vec2f },
})((input) => {
    "use gpu";
    const uv = d.vec2f(d.f32((input.vi << 1) & 2), d.f32(input.vi & 2));
    return {
        pos: d.vec4f(uv.x * 2 - 1, uv.y * 2 - 1, 0, 1),
        uv: d.vec2f(uv.x, 1 - uv.y),
    };
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
            vertex: blitVs,
            fragment: blitFs,
            targets: { format },
            primitive: { topology: "triangle-list" },
        })
        .$name("image-mipmap");
    blitPipelines(world).set(format, pipeline);
    return pipeline;
}

// Per-layer submissions let staged builders budget a complete mip chain per frame.
function genMipmapsLayer(
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
 * size, sRGB-encoded, with a full mip chain. The returned texture binds as a surface's `texture-2d-array`
 * and is sampled `array[layer]`. Async (decode); call from a load path, not a hot frame.
 */
export async function imageArray(
    world: World,
    device: GPUDevice,
    blobs: Blob[],
    cap = 2048,
    format: GPUTextureFormat = ARRAY_FORMAT,
): Promise<GPUTexture> {
    // straight (non-premultiplied) alpha: the MASK cutout compares `tex.a` and the shader multiplies the
    // straight rgb, so premultiplying (the browser default) would darken cutout edges and the factor blend
    const native = await Promise.all(
        blobs.map((b) => createImageBitmap(b, { premultiplyAlpha: "none" })),
    );
    return arrayFromBitmaps(world, device, native, cap, format);
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
    const sized =
        bitmap.width === size && bitmap.height === size
            ? bitmap
            : await createImageBitmap(bitmap, {
                  resizeWidth: size,
                  resizeHeight: size,
                  resizeQuality: "high",
                  premultiplyAlpha: "none",
              });
    device.queue.copyExternalImageToTexture(
        { source: sized },
        { texture, origin: { x: 0, y: 0, z: layer } },
        { width: size, height: size },
    );
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
