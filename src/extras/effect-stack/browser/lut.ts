import { attachTexture, Camera, captureTexture, registerImage } from "../../../core/rendering";
import { Transform } from "../../../core/transform";
import { createApp } from "../../../engine/app";
import { DEFAULT_PLUGINS } from "../../../standard";
import { StandardRenderer } from "../../../standard/rendering";
import { ChromaticAberration, EffectStackPlugin } from "..";

async function whiteImage(size: number): Promise<Blob> {
    const canvas = new OffscreenCanvas(size, size);
    const context = canvas.getContext("2d")!;
    context.fillStyle = "white";
    context.fillRect(0, 0, size, size);
    return canvas.convertToBlob();
}

const mixed = new URLSearchParams(location.search).has("mixed");
const large = mixed ? await whiteImage(2048) : undefined;
const lut = await whiteImage(mixed ? 4 : 1024);
let camera = 0;
let lutId = -1;
try {
    const app = await createApp({
        defaults: false,
        plugins: [
            ...DEFAULT_PLUGINS,
            EffectStackPlugin,
            {
                name: "RegisteredChromaticLut",
                dependencies: [EffectStackPlugin],
                initialize(world) {
                    if (large) registerImage(world, large);
                    lutId = registerImage(world, lut);
                    camera = world.create();
                    world.add(camera, Transform);
                    world.add(camera, Camera, { clearColor: 0xffffff, antialias: 0 });
                    world.add(camera, StandardRenderer);
                    world.add(camera, ChromaticAberration, { intensity: 1 });
                    attachTexture(world, camera, { width: 64, height: 64 });
                },
            },
        ],
    });
    try {
        app.world.step(0);
        const baseline = await captureTexture(app.world, camera);
        app.world.storage(ChromaticAberration).colorLut.set(camera, lutId);
        app.world.step(0);
        const custom = await captureTexture(app.world, camera);
        const at = (8 * 64 + 8) * 4;
        const expected = baseline.rgba.slice(at, at + 4);
        const actual = custom.rgba.slice(at, at + 4);
        if (expected[0]! < 150 || !actual.every((value, i) => value === expected[i]))
            throw new Error(`LUT ${lutId}: expected ${expected}, received ${actual}`);
        document.body.textContent = `LUT ${lutId} preserves the white input`;
    } finally {
        await app.dispose();
    }
} catch (error) {
    document.body.textContent = String(error);
}
