// The image→GPU path the engine otherwise lacks: decode a set of image blobs (browser-native via
// `createImageBitmap`, free on web + webview), resize them onto one shared layer size, upload them as the
// layers of a `texture_2d_array`, and fill the mip chain. The shared substrate two producers sample — glTF
// baseColor (the surface samples `albedo[layer]`, one draw shades many materials) and the sprite atlas (one
// layer per icon) — so it lives in render, inward of both, not in either extra.
//
// `texture_2d_array` (not a bindless `binding_array`, which WebGPU lacks, nor an atlas, which can't tile)
// is the firehose-native choice: the per-instance layer index is just another GPU column, draws stay flat.
// Its one constraint is uniform layer size, so varied sources resize to a common size (capped). Layers are
// `rgba8unorm-srgb` by default, so sampling decodes sRGB→linear in hardware and the mip blit downsamples in
// linear; a data-map caller passes a linear format instead.

import tgpu, { type TgpuRenderPipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute, checkTextureLimits } from "../../engine";

/** mip levels for a square texture of `size` px: the full chain down to 1×1. */
export function mipLevels(size: number): number {
    return Math.floor(Math.log2(size)) + 1;
}

/**
 * the common (square) layer size for an image array: the largest source dimension, capped at `cap`
 * (downscaling larger sources) and floored at 1. The size-uniformity decision in one function. Every
 * layer resizes to this, so the array's one-size constraint is satisfied without size-bucketing.
 */
export function commonSize(dims: { w: number; h: number }[], cap = 2048): number {
    let max = 1;
    for (const dim of dims) max = Math.max(max, dim.w, dim.h);
    return Math.min(cap, Math.max(1, max));
}

const ARRAY_FORMAT: GPUTextureFormat = "rgba8unorm-srgb";

// the mipmap-blit kernel: a fullscreen triangle sampling the previous mip with a linear filter and
// writing the next. `blitVs`/`blitFs` are format-independent (the target format only affects the
// pipeline's color target, not the shading), so one kernel pair serves every format; `blitPipeline`
// below compiles one pipeline per format — srgb for color slots (albedo, emissive), linear `rgba8unorm`
// for data slots (normal, metallic-roughness, occlusion) where an sRGB decode would corrupt the values.
const blitLayout = tgpu.bindGroupLayout({
    src: { texture: d.texture2d(d.f32) },
    samp: { sampler: "filtering" },
});

const blitVs = tgpu.vertexFn({
    in: { vi: d.builtin.vertexIndex },
    out: { pos: d.builtin.position, uv: d.vec2f },
})((input) => {
    "use gpu";
    // a single triangle covering the viewport; uv 0..1 across the covered quad
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

// pipelines bind to the root that created them (device-scoped, memoized — `engine/runtime/gpu.ts`), so a
// stale entry from a torn-down device must not be reused; keyed like the pre-port cache, by format plus
// the device identity it was built against.
const _blit = new Map<string, { device: GPUDevice; pipeline: TgpuRenderPipeline }>();

function blitPipeline(device: GPUDevice, format: GPUTextureFormat): TgpuRenderPipeline {
    // the pipeline comes from `Compute.root`, which is scoped to `Compute.device`, while the caller's
    // encoder / sampler / views come from the device it passed — so a mismatch is a cross-device pass,
    // a validation error at draw time with nothing pointing here. One root per device lands at 4a
    // (the typed extension surface); until then the contract is explicit rather than silent
    if (device !== Compute.device)
        throw new Error(
            "[render] the image array path builds its blit pipeline on Compute.device — pass that device, " +
                "or call requestGPU with yours first",
        );
    const cached = _blit.get(format);
    if (cached && cached.device === device) return cached.pipeline;
    const pipeline = Compute.root
        .createRenderPipeline({
            vertex: blitVs,
            fragment: blitFs,
            targets: { format },
            primitive: { topology: "triangle-list" },
        })
        .$name("image-mipmap");
    _blit.set(format, { device, pipeline });
    return pipeline;
}

/** the emitted mipmap-blit WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function blitWgsl(): string {
    return tgpu.resolve([blitVs, blitFs], { names: "strict" });
}

// fill mips 1..levels of ONE layer by blitting from the level above, one render pass per level, one submit.
// Sampling decodes sRGB→linear and the store re-encodes, so the downsample averages in linear (gamma-correct).
// Per-layer so a staged builder can budget a layer's blit chain as one frame's unit (the union upload spread).
function genMipmapsLayer(
    device: GPUDevice,
    texture: GPUTexture,
    layer: number,
    levels: number,
    format: GPUTextureFormat,
): void {
    if (levels <= 1) return;
    const pipeline = blitPipeline(device, format);
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
        const group = Compute.root.createBindGroup(blitLayout, {
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
 *
 * @example
 * const atlas = await imageArray(device, blobs);
 * Compute.textures.set("spriteAtlas", atlas);
 */
export async function imageArray(
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
    return arrayFromBitmaps(device, native, cap, format);
}

/**
 * allocate (but don't fill) a `texture_2d_array` for `layers` square `size`-px layers with a full mip chain:
 * the cheap, synchronous half of {@link arrayFromBitmaps}, split out so a staged builder uploads layers across
 * frames ({@link uploadLayer}). Fails loud + clear here (the array-layer / 2D-size limit) rather than at an
 * opaque createTexture validation error.
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
 * fill one layer of an array texture: resize the bitmap to the common `size` (re-samples, no second decode),
 * copy it into `layer`, and blit its mip chain. The per-layer upload unit a staged builder budgets one of per
 * frame; the synchronous-uploading {@link arrayFromBitmaps} loops it.
 */
export async function uploadLayer(
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
    genMipmapsLayer(device, texture, layer, levels, format);
}

/**
 * upload decoded `ImageBitmap`s as a `texture_2d_array`: resized to a common (capped) square, sRGB-encoded,
 * mip-filled. The shared core of {@link imageArray} (which decodes blobs first) and any caller with
 * already-decoded bitmaps (the glTF KTX2 RGBA-decode fallback). Takes already-decoded bitmaps so each caller
 * decodes its own way. Synchronous-uploading; the glTF union stages the same primitives across frames instead.
 */
export async function arrayFromBitmaps(
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
        await uploadLayer(device, texture, native[layer], layer, size, levels, format);
    }
    return texture;
}
