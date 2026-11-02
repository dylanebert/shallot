import { createApp } from "../../../../engine/app";
import { probeTexture, rawDevice } from "../../../../engine/runtime";
import { imageTexture, imageTextures, prepareImages, registerImage } from "../../image";
import { RenderingPlugin } from "../../substrate";

const canvas = new OffscreenCanvas(8, 8);
const context = canvas.getContext("2d")!;
const pixels = new Uint8ClampedArray(8 * 8 * 4);
for (let i = 0; i < pixels.length; i++) pixels[i] = (i * 37) % 256;
context.putImageData(new ImageData(pixels, 8, 8), 0, 0);
const blob = await canvas.convertToBlob();
const url = URL.createObjectURL(blob);
const warm: number[] = [];
const app = await createApp({
    defaults: false,
    plugins: [
        RenderingPlugin,
        {
            name: "LateImages",
            initialize(world) {
                for (const isSrgb of [true, false])
                    for (const source of [url, blob])
                        warm.push(registerImage(world, source, { isSrgb }));
            },
        },
    ],
});
const world = app.world;
const device = rawDevice(world.gpu.device);
// The wrapper counts engine allocations, not this diagnostic's raw readback allocations.
let textures = 0;
let groups = 0;
const counted = world.gpu.device;
const createTexture = counted.createTexture.bind(counted);
const createBindGroup = counted.createBindGroup.bind(counted);
counted.createTexture = (...args) => {
    textures++;
    return createTexture(...args);
};
counted.createBindGroup = (...args) => {
    groups++;
    return createBindGroup(...args);
};

async function bytes(texture: GPUTexture, level: number): Promise<number[]> {
    const width = Math.max(1, texture.width >> level);
    const height = Math.max(1, texture.height >> level);
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
        fragment: { module, entryPoint: "fs", targets: [{ format: texture.format }] },
    });
    const target = device.createTexture({
        size: [width, height],
        format: texture.format,
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
    const result = Array.from(new Uint8Array((await probeTexture(world, target)).bytes));
    target.destroy();
    return result;
}
try {
    let equal = true;
    const originals = world.resource(imageTextures).textures.map((role) => role.slice());
    let index = 0;
    for (const isSrgb of [true, false]) {
        for (const source of [URL.createObjectURL(blob), blob]) {
            const id = registerImage(world, source, { isSrgb });
            await prepareImages(world);
            const texture = imageTexture(world, id, isSrgb);
            const original = originals[isSrgb ? 0 : 1][warm[index++]!];
            for (let level = 0; level < original.mipLevelCount; level++)
                equal &&=
                    JSON.stringify(await bytes(texture, level)) ===
                    JSON.stringify(await bytes(original, level));
            await world.step(0);
        }
    }
    const loaded = { textures, groups };
    for (let i = 0; i < 3; i++) await world.step(0);
    const steady = { textures: textures - loaded.textures, groups: groups - loaded.groups };
    const warn = console.warn;
    const warnings: string[] = [];
    console.warn = (...args) => {
        warnings.push(String(args[0]));
        warn(...args);
    };
    let corrupt = false;
    try {
        const id = registerImage(world, new Blob(["corrupt PNG"], { type: "image/png" }));
        await prepareImages(world);
        await prepareImages(world);
        const texture = imageTexture(world, id);
        corrupt =
            texture.width === 1 &&
            texture.height === 1 &&
            JSON.stringify(await bytes(texture, 0)) === "[0,0,0,0]" &&
            warnings.some((warning) =>
                warning.includes(`image ${id} ([object Blob]) failed to load:`),
            );
    } finally {
        console.warn = warn;
    }
    // Each role's first late image prepares its mip pipeline when neither role had images at warm.
    const empty = await createApp({ defaults: false, plugins: [RenderingPlugin] });
    for (const isSrgb of [true, false]) registerImage(empty.world, blob, { isSrgb });
    await prepareImages(empty.world);
    await empty.dispose();

    const decode = globalThis.createImageBitmap;
    let release!: () => void;
    const blocked = new Promise<void>((resolve) => {
        release = resolve;
    });
    let entered!: () => void;
    const decoding = new Promise<void>((resolve) => {
        entered = resolve;
    });
    let finished!: Promise<ImageBitmap>;
    globalThis.createImageBitmap = (async (...args: Parameters<typeof decode>) => {
        entered();
        await blocked;
        finished = decode(...args);
        return finished;
    }) as typeof decode;
    const before = { textures, groups };
    try {
        registerImage(world, blob);
        await decoding;
        const prepared = prepareImages(world);
        await app.dispose();
        await prepared;
        release();
        // Let the uncancellable browser decode finish before checking disposal's guard.
        await blocked;
        await finished;
        await Promise.resolve();
    } finally {
        release();
        globalThis.createImageBitmap = decode;
    }
    const disposed = { textures: textures - before.textures, groups: groups - before.groups };
    document.body.textContent = JSON.stringify({ equal, loaded, steady, disposed, corrupt });
} finally {
    counted.createTexture = createTexture;
    counted.createBindGroup = createBindGroup;
    await app.dispose();
    URL.revokeObjectURL(url);
}
