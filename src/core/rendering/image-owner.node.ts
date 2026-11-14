import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp, probeTexture } from "../../engine";
import { type Image, Images, imageTextures } from "./image";
import {
    imageSampler,
    imageTexture,
    prepareImages,
    RenderingPlugin,
    registerImage,
    replaceImage,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();
const pixel = (value: number): Image => ({
    format: "rgba8unorm",
    width: 1,
    height: 1,
    levels: [new Uint8Array([value, 0, 0, 255])],
});
function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((done) => {
        resolve = done;
    });
    return { promise, resolve };
}

test("replacement bounds native texture ownership, preserves sampler and accepts changed dimensions", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Image owner requires a Dawn adapter");
    console.log("Image owner adapter", {
        vendor: adapter.info.vendor,
        device: adapter.info.device,
        fallback: adapter.info.isFallbackAdapter,
    });
    const device = await adapter.requestDevice();
    const original = device.createTexture;
    const create = spyOn(device, "createTexture");
    let created = 0;
    let destroyed = 0;
    create.mockImplementation((descriptor) => {
        const texture = original.call(device, descriptor);
        if (descriptor.label?.startsWith("image-native:")) {
            created++;
            const destroy = texture.destroy.bind(texture);
            texture.destroy = () => {
                destroyed++;
                destroy();
            };
        }
        return texture;
    });
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin], device, adapter });
    const world = app.world;
    try {
        const id = registerImage(world, () => pixel(0), { sampler: { magFilter: 0 } });
        await prepareImages(world);
        const sampler = imageSampler(world, id);
        for (let n = 1; n <= 40; n++) {
            const previous = imageTexture(world, id);
            replaceImage(world, id, () => ({
                ...pixel(n),
                width: 2,
                levels: [new Uint8Array([n, 0, 0, 255, n, 0, 0, 255])],
            }));
            expect(world.owns(previous)).toBe(false);
            await prepareImages(world);
            expect(imageTexture(world, id).width).toBe(2);
            expect(imageSampler(world, id)).toBe(sampler);
            const probe = await probeTexture(world, imageTexture(world, id));
            expect(Array.from(new Uint8Array(probe.bytes))).toEqual([n, 0, 0, 255, n, 0, 0, 255]);
            expect(created - destroyed).toBe(1);
        }
        expect(world.resource(Images).snapshot().names.length).toBe(1);
        app.dispose();
        expect(created - destroyed).toBe(0);
    } finally {
        create.mockRestore();
        app.dispose();
        device.destroy();
    }
});

test("pending sources settle on replacement and cannot publish after completion or disposal", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Image owner requires a Dawn adapter");
    const device = await adapter.requestDevice();
    const create = spyOn(device, "createTexture");
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin], device, adapter });
    const world = app.world;
    const first = deferred<Image>();
    const id = registerImage(world, () => first.promise);
    const pending = prepareImages(world);
    replaceImage(world, id, () => pixel(17));
    await pending;
    await prepareImages(world);
    const current = imageTexture(world, id);
    first.resolve(pixel(33));
    await Promise.resolve();
    expect(imageTexture(world, id)).toBe(current);
    const second = deferred<Image>();
    replaceImage(world, id, () => second.promise);
    const disposed = prepareImages(world);
    create.mockClear();
    app.dispose();
    await disposed;
    second.resolve(pixel(44));
    await Promise.resolve();
    expect(create).not.toHaveBeenCalled();
    create.mockRestore();
    device.destroy();
});

for (const operation of ["replace", "dispose"] as const)
    test(`pending bitmap decode closes stale bitmap after ${operation} without uploading`, async () => {
        const original = globalThis.createImageBitmap;
        const bitmap = deferred<ImageBitmap>();
        const close = () => {
            closed++;
        };
        let closed = 0;
        globalThis.createImageBitmap = (() => bitmap.promise) as typeof createImageBitmap;
        const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
        try {
            const id = registerImage(app.world, new Blob(["decode"]));
            const pending = prepareImages(app.world);
            const state = app.world.resource(imageTextures);
            if (operation === "dispose") app.dispose();
            else replaceImage(app.world, id, () => pixel(77));
            await pending;
            if (operation === "replace") await prepareImages(app.world);
            const current = state.textures[0][id];
            bitmap.resolve({ width: 1, height: 1, close } as ImageBitmap);
            await Promise.resolve();
            await Promise.resolve();
            expect(closed).toBe(1);
            expect(state.textures[0][id]).toBe(current);
        } finally {
            app.dispose();
            globalThis.createImageBitmap = original;
        }
    });
