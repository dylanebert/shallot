import { createApp } from "../../../../engine/app";
import { probeTexture, rawDevice } from "../../../../engine/runtime";
import { imageTextures, registerImage } from "../../image";
import { RenderingPlugin } from "../../substrate";

const size = 512;
const level = 4;
const targetSize = size >> level;
const pixels = new Uint8ClampedArray(size * size * 4);
for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
        // Subpixel stripes must average rather than alias when reduced from 512 to 32.
        pixels.set(
            [x % 4 === 0 ? 255 : 85, y % 4 === 0 ? 255 : 85, x >> 1, 255],
            (y * size + x) * 4,
        );
const canvas = new OffscreenCanvas(size, size);
canvas.getContext("2d")!.putImageData(new ImageData(pixels, size, size), 0, 0);
const source = await canvas.convertToBlob();
let id = -1;
const app = await createApp({
    defaults: false,
    plugins: [
        RenderingPlugin,
        {
            name: "NativeImageMips",
            initialize(world) {
                id = registerImage(world, source, { isSrgb: false });
            },
        },
    ],
});
try {
    const world = app.world;
    const texture = world.resource(imageTextures).textures[1][id]!;
    // This diagnostic owns its readback pipeline outside game pipeline preparation.
    const device = rawDevice(world.gpu.device);
    const module = device.createShaderModule({
        code: `
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return textureLoad(image, vec2i(p.xy), 0);
}`,
    });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    });
    const target = world.gpu.device.createTexture({
        size: [targetSize, targetSize],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(target);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(
        0,
        device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: texture.createView({ baseMipLevel: level, mipLevelCount: 1 }),
                },
            ],
        }),
    );
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const actual = new Uint8Array((await probeTexture(world, target)).bytes);
    const scale = 1 << level;
    let maxError = 0;
    for (let y = 0; y < targetSize; y++)
        for (let x = 0; x < targetSize; x++)
            for (let c = 0; c < 4; c++) {
                // Independent reference: average the original texels, not previous GPU mip levels.
                let sum = 0;
                for (let sy = y * scale; sy < (y + 1) * scale; sy++)
                    for (let sx = x * scale; sx < (x + 1) * scale; sx++)
                        sum += pixels[(sy * size + sx) * 4 + c]!;
                const expected = sum / (scale * scale);
                maxError = Math.max(
                    maxError,
                    Math.abs(actual[(y * targetSize + x) * 4 + c]! - expected),
                );
            }
    document.body.textContent = JSON.stringify({
        width: texture.width,
        height: texture.height,
        mips: texture.mipLevelCount,
        maxError,
    });
} finally {
    await app.dispose();
}
