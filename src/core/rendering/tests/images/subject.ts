import { createApp } from "../../../../engine/app";
import { rawDevice } from "../../../../engine/runtime";
import { ImageBindings, registerImage } from "../../image";
import { RenderingPlugin } from "../../substrate";

// The authored PNG has sRGB primaries, a linear (gamma 1) ICC transfer curve and
// one opaque RGBA texel [64, 128, 192, 255]. The profile was made with Little CMS
// cmsCreateRGBProfile(D65, sRGB primaries, cmsBuildGamma(1)); no external artwork.
const app = await createApp({
    defaults: false,
    plugins: [
        RenderingPlugin,
        {
            name: "ImageProfile",
            initialize(world) {
                registerImage(world, new URL("./linear-rgb.png", import.meta.url).href);
                // A second, larger source also exercises resizing the registered profile image.
                if (location.search)
                    registerImage(
                        world,
                        new Blob(
                            [
                                Uint8Array.from(
                                    atob(
                                        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAADklEQVR4nGP4DwUMMAYAj4IP8TylVlEAAAAASUVORK5CYII=",
                                    ),
                                    (c) => c.charCodeAt(0),
                                ),
                            ],
                            { type: "image/png" },
                        ),
                    );
            },
        },
    ],
});
// This diagnostic owns its raw readback pipeline, outside the game's pipeline warm-up.
const device = rawDevice(app.world.gpu.device);
const texture = app.world.resource(ImageBindings).images;
const module = device.createShaderModule({
    code: `
@group(0) @binding(0) var image: texture_2d_array<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(p[i], 0, 1);
}
@fragment fn fs() -> @location(0) vec4f {
    return textureLoad(image, vec2i(0), 0, 0);
}`,
});
const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm-srgb" }] },
});
const target = device.createTexture({
    size: [1, 1],
    format: "rgba8unorm-srgb",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const buffer = device.createBuffer({
    size: 256,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store" }],
});
pass.setPipeline(pipeline);
pass.setBindGroup(
    0,
    device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: texture.createView({ dimension: "2d-array" }) }],
    }),
);
pass.draw(3);
pass.end();
encoder.copyTextureToBuffer({ texture: target }, { buffer, bytesPerRow: 256 }, [1, 1]);
device.queue.submit([encoder.finish()]);
await buffer.mapAsync(GPUMapMode.READ);
const texel = Array.from(new Uint8Array(buffer.getMappedRange()).slice(0, 4));
buffer.unmap();
buffer.destroy();
target.destroy();
await app.dispose();
document.body.textContent = JSON.stringify(texel);
