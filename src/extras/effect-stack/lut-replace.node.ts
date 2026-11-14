import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import {
    attachTexture,
    Camera,
    captureTexture,
    type ImageSource,
    prepareImages,
    registerImage,
    replaceImage,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { createApp, type World } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { ChromaticAberration, EffectStackPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();
const lut =
    (v: number): ImageSource =>
    () => ({
        width: 3,
        height: 1,
        format: "rgba8unorm",
        levels: [new Uint8Array([v, 0, 0, 255, 0, v, 0, 255, 0, 0, v, 255])],
    });

for (const replace of [false, true]) {
    test(`colorLut captures ${replace ? "after replacement at the same id" : "without replacement"}`, async () => {
        let camera = 0;
        let id = -1;
        const app = await createApp({
            defaults: false,
            plugins: [
                ...DEFAULT_PLUGINS,
                EffectStackPlugin,
                {
                    name: "Lut",
                    dependencies: [...DEFAULT_PLUGINS, EffectStackPlugin],
                    initialize(world: World) {
                        camera = world.create();
                        world.add(camera, Transform);
                        world.add(camera, Camera, { antialias: 0, clearColor: 0x808080 });
                        world.add(camera, StandardRenderer);
                        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                        attachTexture(world, camera, { width: 8, height: 8 });
                        id = registerImage(world, lut(255));
                    },
                },
            ],
        });
        const world = app.world;
        const errors: string[] = [];
        const onError = (event: Event) => {
            errors.push((event as GPUUncapturedErrorEvent).error.message);
        };
        world.gpu.device.addEventListener("uncapturederror", onError);
        try {
            await prepareImages(world);
            world.add(camera, ChromaticAberration, { colorLut: id, intensity: 0.2 });
            world.step(1 / 60);
            await captureTexture(world, camera);
            if (replace) replaceImage(world, id, lut(128));
            await prepareImages(world);
            world.step(1 / 60);
            await captureTexture(world, camera);
            expect(errors).toEqual([]);
        } finally {
            world.gpu.device.removeEventListener("uncapturederror", onError);
            app.dispose();
        }
    });
}
