// The glTF-specific half of baseColor binding: the size-bucket policy (a `texture_2d_array` is one size, so
// varied-size compressed baseColors split across a small set of arrays), the compressed (KTX2 block) upload,
// and the per-material fallback texture + sampler. The generic image→`texture_2d_array` path it builds on —
// decode/resize/mip a bitmap set — lives in `standard/render` (`imageArray` / `arrayFromBitmaps`), shared
// with the sprite atlas.

import type { Ktx2Image } from "./basis";

// A `texture_2d_array` forces one size + format across its layers, and a block-compressed texture can't be
// resized, so baseColors of differing sizes can't share one compressed array. WebGPU has no bindless, so the
// firehose answer is a fixed, small set of arrays — one per size bucket — selected per-instance in the FS
// (`sampleAlbedo` switches on the per-material `albedoBucket`). The cap is a **binding budget**, not a scene
// guess: each bucket is one sampled `texture_2d_array`, and the tightest surface (`skin`) also binds 4 data
// maps + 2 VAT textures + 2 shadow maps, so `ALBEDO_BUCKETS + 8 <= maxSampledTexturesPerShaderStage` (spec
// min 16) caps it at 8. 4 leaves headroom (room for a future data map) and covers the size spread real glTFs
// mix; more distinct sizes than buckets spill the rarest to one decoded-RGBA array (the warned last resort).
/** the number of baseColor size-bucket `texture_2d_array`s: varied-size compressed textures split across
 *  this many arrays, the rarest sizes past it spilling to one decoded-RGBA array. */
export const ALBEDO_BUCKETS = 4;

/** the per-bucket binding names the textured + skin surfaces declare, `albedo0..albedo{N-1}`. */
export const ALBEDO_NAMES = Array.from({ length: ALBEDO_BUCKETS }, (_, b) => `albedo${b}`);

/**
 * upload pre-transcoded KTX2 images as a compressed `texture_2d_array`, one layer per image, each carrying
 * its own transcoded mip chain straight from the Basis transcoder (no GPU blit; compressed mips are already
 * downsampled). All images must share dimensions + format + block size, the array's one-size constraint,
 * which the caller guarantees. Binds + samples identically to the bitmap arrays: `albedo[layer]`.
 *
 * @example
 * const albedo = compressedAlbedoArray(device, images);
 * Compute.textures.set("albedo", albedo);
 */
export function compressedAlbedoArray(device: GPUDevice, images: Ktx2Image[]): GPUTexture {
    const texture = allocCompressed(device, images);
    images.forEach((image, layer) => {
        writeCompressedLayer(device, texture, image, layer);
    });
    return texture;
}

/** allocate (but don't fill) the compressed `texture_2d_array` for `images`: dims/format/mip count from the
 *  first image (all share them). The staged half of {@link compressedAlbedoArray}; fill via
 *  {@link writeCompressedLayer}. `label` names the array (the data maps pass their slot name). */
export function allocCompressed(
    device: GPUDevice,
    images: Ktx2Image[],
    label = "gltf-albedo",
): GPUTexture {
    const ref = images[0];
    return device.createTexture({
        label,
        size: { width: ref.width, height: ref.height, depthOrArrayLayers: images.length },
        format: ref.format,
        mipLevelCount: ref.mips.length,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
}

/** write one image's transcoded mip chain into `layer` of a compressed array (cheap `writeTexture`, no blit).
 *  The per-layer upload unit the staged union budgets one of per frame. */
export function writeCompressedLayer(
    device: GPUDevice,
    texture: GPUTexture,
    image: Ktx2Image,
    layer: number,
): void {
    for (const mip of image.mips) {
        // compressed copies address data in blocks: one row is ceil(w / blockDim) blocks, and the byte
        // length divides evenly by the block-row count, so bytesPerRow = data / rows
        const rows = Math.ceil(mip.height / image.blockDim);
        device.queue.writeTexture(
            { texture, mipLevel: mip.level, origin: { x: 0, y: 0, z: layer } },
            mip.data as Uint8Array<ArrayBuffer>,
            { offset: 0, bytesPerRow: mip.data.length / rows, rowsPerImage: rows },
            { width: mip.width, height: mip.height, depthOrArrayLayers: 1 },
        );
    }
}

/**
 * a 1×1 single-layer array, the absent-slot fallback so a surface binding always resolves (a missing texture
 * would skip the whole draw). Color slots are sRGB, data slots (normal / metallic-roughness / occlusion)
 * linear, but the palette layer is `-1` for every material on a fallback, so the sample is always discarded.
 */
export function fallback1x1(device: GPUDevice, format: GPUTextureFormat): GPUTexture {
    return device.createTexture({
        label: "gltf-fallback",
        size: { width: 1, height: 1, depthOrArrayLayers: 1 },
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING,
    });
}

/** the default albedo sampler: trilinear + `repeat` wrap (glTF's default; each layer tiles within itself)
 *  + anisotropy for the grazing tiled surfaces. One sampler serves every textured glTF material. */
export function albedoSampler(device: GPUDevice): GPUSampler {
    return device.createSampler({
        label: "gltf-albedo",
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat",
        maxAnisotropy: 16,
    });
}
