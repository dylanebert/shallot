import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { probeTexture } from "../../engine";
import { type ImageAddressMode, ImageBindings, type ImageFilterMode, registerImage } from "./image";
import { sampleImage } from "./image-sampling";
import { RenderingPlugin } from "./substrate";
import { fullscreenVertex } from "./tonemapping";

setDefaultTimeout(TIMEOUT.gpu);
const modes = ["clamp-to-edge", "repeat", "mirror-repeat"] as const;
const filters = ["nearest", "linear"] as const;
const cases = modes.flatMap((u, addressModeU) =>
    modes.flatMap((v, addressModeV) =>
        filters.flatMap((magFilter, mag) =>
            filters.map((minFilter, min) => ({
                u,
                v,
                magFilter,
                minFilter,
                sampler: {
                    addressModeU: addressModeU as ImageAddressMode,
                    addressModeV: addressModeV as ImageAddressMode,
                    magFilter: mag as ImageFilterMode,
                    minFilter: min as ImageFilterMode,
                },
            })),
        ),
    ),
);
const layout = tgpu.bindGroupLayout({
    images: { texture: d.texture2dArray() },
    settings: { texture: d.texture2d() },
    sampler: { sampler: "filtering" },
    reference: { sampler: "filtering" },
    config: { uniform: d.vec4f },
});
const fragment = tgpu.fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
    "use gpu";
    // Both axes cross negative coordinates and multiple seams, including fractional footprints.
    const uv = d.vec2f(input.pos.x / 16 - 2, input.pos.y / 16 - 2);
    const layer = d.i32(layout.$.config.x);
    const dx = d.vec2f(layout.$.config.y, 0);
    const dy = d.vec2f(0, layout.$.config.y);
    if (layout.$.config.z !== 0) {
        return std.textureSampleGrad(layout.$.images, layout.$.reference, uv, layer, dx, dy);
    }
    return sampleImage(layout.$.images, layout.$.sampler, layout.$.settings, uv, layer, dx, dy);
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "ImageSamplerComparison",
                initialize(world) {
                    for (const c of cases) {
                        const isDefault =
                            c.u === "clamp-to-edge" &&
                            c.v === "clamp-to-edge" &&
                            c.magFilter === "linear" &&
                            c.minFilter === "linear";
                        registerImage(
                            world,
                            new Blob(),
                            isDefault ? undefined : { sampler: c.sampler },
                        );
                    }
                },
            },
        ],
    },
]);

function prepare() {
    const { world } = subjects()[0];
    const { device, root } = world.gpu;
    const bindings = world.resource(ImageBindings);
    const image = device.createTexture({
        size: [8, 8, cases.length],
        mipLevelCount: 4,
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(image);
    for (let level = 0; level < 4; level++) {
        const size = 8 >> level;
        const pixels = new Uint8Array(size * size * cases.length * 4);
        for (let layer = 0; layer < cases.length; layer++)
            for (let y = 0; y < size; y++)
                for (let x = 0; x < size; x++) {
                    pixels.set(
                        [x % 2 ? 255 : 0, y % 2 ? 255 : 0, level * 70, 255],
                        ((layer * size + y) * size + x) * 4,
                    );
                }
        device.queue.writeTexture(
            { texture: image, mipLevel: level },
            pixels,
            { bytesPerRow: size * 4, rowsPerImage: size },
            [size, size, cases.length],
        );
    }
    const pipeline = root.createRenderPipeline({
        vertex: fullscreenVertex,
        fragment,
        targets: { format: "rgba8unorm" },
    });
    const output = device.createTexture({
        size: [64, 64],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(output);
    const config = root.createBuffer(d.vec4f).$usage("uniform");
    return { world, device, root, bindings, image, pipeline, output, config };
}
let prepared: ReturnType<typeof prepare>;
beforeAll(() => {
    prepared = prepare();
});

for (const mode of [0, 1, 2]) {
    test(`${modes[mode]} images match hardware addressing and filters across seams and mip levels (including the default)`, async () => {
        const { world, device, root, bindings, image, pipeline, output, config } = prepared;
        for (let layer = 0; layer < cases.length; layer++) {
            const c = cases[layer];
            if (c.sampler.addressModeU !== mode) continue;
            const group = root.createBindGroup(layout, {
                images: image.createView(),
                settings: bindings.imageSettings.createView(),
                sampler: bindings.imageSampler,
                reference: device.createSampler({
                    addressModeU: c.u,
                    addressModeV: c.v,
                    magFilter: c.magFilter,
                    minFilter: c.minFilter,
                }),
                config,
            });
            for (const gradient of [0, 0.03125, 0.14, 0.18, 0.25, 0.5, 2]) {
                const results: Uint8Array[] = [];
                for (const reference of [0, 1]) {
                    config.write(d.vec4f(layer, gradient, reference, 0));
                    const encoder = device.createCommandEncoder();
                    const pass = encoder.beginRenderPass({
                        colorAttachments: [
                            { view: output.createView(), loadOp: "clear", storeOp: "store" },
                        ],
                    });
                    pipeline.with(group).with(pass).draw(3);
                    pass.end();
                    device.queue.submit([encoder.finish()]);
                    results.push(new Uint8Array((await probeTexture(world, output)).bytes));
                }
                let maximum = 0;
                for (let i = 0; i < results[0].length; i++)
                    maximum = Math.max(maximum, Math.abs(results[0][i] - results[1][i]));
                expect(maximum, JSON.stringify({ c, gradient })).toBeLessThanOrEqual(1);
            }
        }
    });
}
