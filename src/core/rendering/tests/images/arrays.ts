import { createApp } from "../../../../engine/app";
import { uploadLayer } from "../../image";

const format = new URLSearchParams(location.search).get("format") as GPUTextureFormat;
const colors = [
    [32, 32, 32, 255],
    [96, 32, 32, 255],
];
const bitmaps = await Promise.all(
    colors.map(async (color) => {
        const canvas = new OffscreenCanvas(64, 64);
        const context = canvas.getContext("2d")!;
        context.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        context.fillRect(0, 0, 64, 64);
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
            name: "ImageArrayUpload",
            gpu: {},
            async warm(world) {
                const device = world.gpu.device;
                for (let array = 0; array < 32; array++) {
                    const texture = device.createTexture({
                        size: [64, 64, 2],
                        format,
                        usage:
                            GPUTextureUsage.COPY_DST |
                            GPUTextureUsage.COPY_SRC |
                            GPUTextureUsage.TEXTURE_BINDING |
                            GPUTextureUsage.RENDER_ATTACHMENT,
                    });
                    const buffer = device.createBuffer({
                        size: 256 * 64 * 2,
                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                    });
                    try {
                        for (let layer = 0; layer < 2; layer++) {
                            await uploadLayer(
                                world,
                                device,
                                texture,
                                bitmaps[layer]!,
                                layer,
                                64,
                                1,
                                format,
                            );
                        }
                        const encoder = device.createCommandEncoder();
                        encoder.copyTextureToBuffer(
                            { texture },
                            { buffer, bytesPerRow: 256, rowsPerImage: 64 },
                            [64, 64, 2],
                        );
                        device.queue.submit([encoder.finish()]);
                        await buffer.mapAsync(GPUMapMode.READ);
                        const bytes = new Uint8Array(buffer.getMappedRange());
                        for (let layer = 0; layer < 2; layer++) {
                            for (let y = 0; y < 64; y++) {
                                for (let x = 0; x < 64; x++) {
                                    const offset = (layer * 64 + y) * 256 + x * 4;
                                    const actual = Array.from(bytes.subarray(offset, offset + 4));
                                    if (
                                        actual.some(
                                            (value, channel) =>
                                                Math.abs(value - colors[layer]![channel]!) > 1,
                                        )
                                    ) {
                                        throw new Error(
                                            JSON.stringify({
                                                array,
                                                layer,
                                                x,
                                                y,
                                                format,
                                                expected: colors[layer],
                                                actual,
                                            }),
                                        );
                                    }
                                }
                            }
                        }
                        buffer.unmap();
                    } finally {
                        buffer.destroy();
                        texture.destroy();
                    }
                }
            },
        },
    ],
});
await app.dispose();
for (const bitmap of bitmaps) bitmap.close();
document.body.textContent = "32 arrays preserve both layers";
