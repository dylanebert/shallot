import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp } from "../../engine";
import { type Image, imageTextures } from "./image";
import { prepareImages, RenderingPlugin, registerImage, replaceImage } from "./index";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();

test("invalid replacement rejects preparation without retaining the prior texture; later replacement recovers", async () => {
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
    const world = app.world;
    const pixel = (): Image => ({
        format: "rgba8unorm",
        width: 1,
        height: 1,
        levels: [new Uint8Array(4)],
    });
    try {
        const id = registerImage(world, pixel);
        await prepareImages(world);
        const old = world.resource(imageTextures).textures[0][id];
        replaceImage(world, id, () => ({ ...pixel(), levels: [new Uint8Array(3)] }));
        await expect(prepareImages(world)).rejects.toThrow("invalid image level");
        expect(world.owns(old)).toBe(false);
        expect(world.resource(imageTextures).textures[0][id]).toBeUndefined();
        await prepareImages(world);
        replaceImage(world, id, pixel);
        await prepareImages(world);
        expect(world.resource(imageTextures).textures[0][id]).toBeDefined();
        let reject!: (error: Error) => void;
        replaceImage(
            world,
            id,
            () =>
                new Promise<Image>((_, fail) => {
                    reject = fail;
                }),
        );
        const superseded = prepareImages(world);
        replaceImage(world, id, pixel);
        reject(new Error("superseded failure"));
        await superseded;
        await prepareImages(world);
        expect(world.resource(imageTextures).textures[0][id]).toBeDefined();
    } finally {
        app.dispose();
    }
});
