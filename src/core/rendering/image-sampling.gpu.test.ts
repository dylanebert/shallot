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
// sampleImage reads each setting at one place: U and V only in their axis's addressing, mag and min
// only in the texel filter chosen by the sign of the level of detail, mipmap only in the level
// branch; texel positions and levels come from the pixel and gradient, which every case sweeps. No
// term depends on two settings jointly, except the hardware-path gate on the exact default. So the
// cases are the default, each single-setting departure from it, and rows completing every pair of
// settings (U x V, U or V x each filter, filter x filter), rather than the 72-case cross product.
const L = 1;
const N = 0;
const rows: [number, number, number, number, number][] = [
    [0, 0, L, L, L],
    [0, 0, N, L, L],
    [0, 0, L, N, L],
    [0, 0, L, L, N],
    [1, 0, L, L, L],
    [2, 0, L, L, L],
    [0, 1, L, L, L],
    [0, 2, L, L, L],
    [1, 1, N, N, N],
    [1, 2, N, N, L],
    [2, 1, N, L, N],
    [2, 2, L, N, N],
];
const cases = rows.map(([addressModeU, addressModeV, mag, min, mip]) => ({
    u: modes[addressModeU],
    v: modes[addressModeV],
    magFilter: filters[mag],
    minFilter: filters[min],
    mipmapFilter: filters[mip],
    sampler: {
        addressModeU: addressModeU as ImageAddressMode,
        addressModeV: addressModeV as ImageAddressMode,
        magFilter: mag as ImageFilterMode,
        minFilter: min as ImageFilterMode,
        mipmapFilter: mip as ImageFilterMode,
    },
}));
// One 64x64 tile per gradient; the top row of tiles samples Shallot's path, the bottom the reference.
const GRADIENTS = [0, 0.03125, 0.14, 0.18, 0.25, 0.35, 0.5, 0.7, 2];
const TILE = 64;
const layout = tgpu.bindGroupLayout({
    images: { texture: d.texture2dArray() },
    settings: { texture: d.texture2d() },
    sampler: { sampler: "filtering" },
    reference: { sampler: "filtering" },
    config: { uniform: d.vec4f },
    gradients: { uniform: d.arrayOf(d.vec4f, GRADIENTS.length) },
});
const fragment = tgpu.fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
    "use gpu";
    const tile = std.floor(std.div(input.pos.xy, TILE));
    const local = std.sub(input.pos.xy, std.mul(tile, TILE));
    // Both axes cross negative coordinates and multiple seams, including fractional footprints.
    const uv = d.vec2f(local.x / 16 - 2, local.y / 16 - 2);
    const layer = d.i32(layout.$.config.x);
    const gradient = layout.$.gradients[d.i32(tile.x)].x;
    const dx = d.vec2f(gradient, 0);
    const dy = d.vec2f(0, gradient);
    if (tile.y !== 0) {
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
                            c.minFilter === "linear" &&
                            c.mipmapFilter === "linear";
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
        size: [TILE * GRADIENTS.length, TILE * 2],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(output);
    const config = root.createBuffer(d.vec4f).$usage("uniform");
    const gradients = root
        .createBuffer(
            d.arrayOf(d.vec4f, GRADIENTS.length),
            GRADIENTS.map((gradient) => d.vec4f(gradient, 0, 0, 0)),
        )
        .$usage("uniform");
    return { world, device, root, bindings, image, pipeline, output, config, gradients };
}
let prepared: ReturnType<typeof prepare>;

/** Renders one case's tiles and reads them back: Shallot's path on top, the reference below. */
async function render(layer: number): Promise<Uint8Array> {
    const { world, device, root, bindings, image, pipeline, output, config, gradients } = prepared;
    const c = cases[layer];
    const group = root.createBindGroup(layout, {
        images: image.createView(),
        settings: bindings.imageSettings.createView(),
        sampler: bindings.imageSampler,
        reference: device.createSampler({
            addressModeU: c.u,
            addressModeV: c.v,
            magFilter: c.magFilter,
            minFilter: c.minFilter,
            mipmapFilter: c.mipmapFilter,
        }),
        config,
        gradients,
    });
    config.write(d.vec4f(layer, 0, 0, 0));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: output.createView(), loadOp: "clear", storeOp: "store" }],
    });
    pipeline.with(group).with(pass).draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return new Uint8Array((await probeTexture(world, output)).bytes);
}

// The first draw compiles the pipeline: setup, outside the first test's budget.
beforeAll(async () => {
    prepared = prepare();
    await render(0);
});

for (const mode of [0, 1, 2]) {
    test(`${modes[mode]} images match hardware addressing, texel and mipmap filters across seams and between mip levels (including the default)`, async () => {
        const width = TILE * GRADIENTS.length;
        for (let layer = 0; layer < cases.length; layer++) {
            const c = cases[layer];
            if (c.sampler.addressModeU !== mode) continue;
            const bytes = await render(layer);
            const maxima = GRADIENTS.map(() => 0);
            for (let y = 0; y < TILE; y++)
                for (let x = 0; x < width; x++)
                    for (let channel = 0; channel < 4; channel++) {
                        const i = (y * width + x) * 4 + channel;
                        const difference = Math.abs(bytes[i] - bytes[i + TILE * width * 4]);
                        const tile = Math.floor(x / TILE);
                        maxima[tile] = Math.max(maxima[tile], difference);
                    }
            // Hardware's level-of-detail fraction is coarser than f32: blending two levels
            // differed by 2 at a 0.16 fraction on Apple M4 Max (Metal).
            const tolerance = c.mipmapFilter === "linear" ? 2 : 1;
            for (const [index, gradient] of GRADIENTS.entries())
                expect(maxima[index], JSON.stringify({ c, gradient })).toBeLessThanOrEqual(
                    tolerance,
                );
        }
    });
}
