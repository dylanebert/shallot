import { createApp } from "../../../../engine/app";
import { uploadLayer } from "../../image";

const target = 32;
const sizes = [8, 32, 512];
const sources = sizes.map((size) => {
    const bytes = new Uint8ClampedArray(size * size * 4);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const offset = (y * size + x) * 4;
            // The downscale's subpixel stripes alias at level zero, but average to 127.5.
            bytes.set(
                size > target
                    ? [x % 4 === 0 ? 255 : 85, y % 4 === 0 ? 255 : 85, 64, 255]
                    : [x * 4, y * 4, 64, 255],
                offset,
            );
        }
    }
    return bytes;
});
const bitmaps = await Promise.all(
    sizes.map(async (size, layer) => {
        const canvas = new OffscreenCanvas(size, size);
        canvas.getContext("2d")!.putImageData(new ImageData(sources[layer]!, size, size), 0, 0);
        return createImageBitmap(await canvas.convertToBlob(), {
            premultiplyAlpha: "none",
            colorSpaceConversion: "none",
        });
    }),
);
const app = await createApp({
    defaults: false,
    plugins: [
        {
            name: "ImageResampleProof",
            gpu: {},
            async warm(world) {
                const device = world.gpu.device;
                const texture = device.createTexture({
                    size: [target, target, sizes.length],
                    format: "rgba8unorm",
                    usage:
                        GPUTextureUsage.COPY_SRC |
                        GPUTextureUsage.RENDER_ATTACHMENT |
                        GPUTextureUsage.TEXTURE_BINDING,
                });
                const buffer = device.createBuffer({
                    size: 256 * target * sizes.length,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                });
                try {
                    for (let layer = 0; layer < sizes.length; layer++)
                        await uploadLayer(
                            world,
                            device,
                            texture,
                            bitmaps[layer]!,
                            layer,
                            target,
                            1,
                            "rgba8unorm",
                        );
                    const encoder = device.createCommandEncoder();
                    encoder.copyTextureToBuffer(
                        { texture },
                        { buffer, bytesPerRow: 256, rowsPerImage: target },
                        [target, target, sizes.length],
                    );
                    device.queue.submit([encoder.finish()]);
                    await buffer.mapAsync(GPUMapMode.READ);
                    const actual = new Uint8Array(buffer.getMappedRange());
                    for (let layer = 0; layer < sizes.length; layer++) {
                        const size = sizes[layer]!;
                        const scale = size / target;
                        for (let y = 0; y < target; y++)
                            for (let x = 0; x < target; x++)
                                for (let c = 0; c < 4; c++) {
                                    let sum = 0;
                                    for (
                                        let sy = Math.floor(y * scale);
                                        sy < Math.ceil((y + 1) * scale);
                                        sy++
                                    ) {
                                        for (
                                            let sx = Math.floor(x * scale);
                                            sx < Math.ceil((x + 1) * scale);
                                            sx++
                                        ) {
                                            const weight =
                                                (Math.min((x + 1) * scale, sx + 1) -
                                                    Math.max(x * scale, sx)) *
                                                (Math.min((y + 1) * scale, sy + 1) -
                                                    Math.max(y * scale, sy));
                                            sum +=
                                                sources[layer]![(sy * size + sx) * 4 + c]! * weight;
                                        }
                                    }
                                    const expected = sum / (scale * scale);
                                    const value = actual[(layer * target + y) * 256 + x * 4 + c]!;
                                    // Upscale bilinear differs from a box's nearest texel by at most 2 here;
                                    // mip-chain unorm rounding admits one byte in exact and downscale cases.
                                    const bound = size < target ? 2 : 1;
                                    if (Math.abs(value - expected) > bound)
                                        throw new Error(
                                            JSON.stringify({
                                                layer,
                                                x,
                                                y,
                                                c,
                                                value,
                                                expected,
                                                bound,
                                            }),
                                        );
                                }
                    }
                    buffer.unmap();
                } finally {
                    buffer.destroy();
                    texture.destroy();
                }
            },
        },
    ],
});
await app.dispose();
for (const bitmap of bitmaps) bitmap.close();
document.body.textContent = "upscale, exact and downscale match CPU box reference";
