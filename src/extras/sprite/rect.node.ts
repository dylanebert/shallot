import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    ImageFilterMode,
    registerImage,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { imageTextures } from "../../core/rendering/image";
import { Transform } from "../../core/transform";
import { StandardRenderer } from "../../standard/rendering";
import { Sprite, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
let camera = -1;
let image = -1;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            SpritePlugin,
            {
                name: "SpriteRect",
                dependencies: [SpritePlugin],
                initialize(world) {
                    image = registerImage(world, new Blob(), {
                        sampler: { magFilter: ImageFilterMode.Nearest },
                    });
                    camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5] });
                    world.add(camera, Camera, {
                        mode: CameraMode.Orthographic,
                        size: 2,
                        antialias: 0,
                        clearColor: 0,
                    });
                    world.add(camera, StandardRenderer);
                    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                    attachTexture(world, camera, { width: 128, height: 64 });
                },
            },
        ],
    },
]);

test("a Sprite rect shows that pixel region of its image from the top-left, right way up", async () => {
    const { world } = subjects()[0];
    const device = world.gpu.device;
    // A 4×4 gray sheet whose bottom-right 2×2 frame is red, green over blue, white.
    const gray = [128, 128, 128, 255];
    const frame: Record<string, number[]> = {
        "2,2": [255, 0, 0, 255],
        "3,2": [0, 255, 0, 255],
        "2,3": [0, 0, 255, 255],
        "3,3": [255, 255, 255, 255],
    };
    const texels: number[] = [];
    for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++) texels.push(...(frame[`${x},${y}`] ?? gray));
    const texture = device.createTexture({
        size: [4, 4],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(texture);
    device.queue.writeTexture({ texture }, new Uint8Array(texels), { bytesPerRow: 16 }, [4, 4]);
    world.resource(imageTextures).textures[0][image] = texture;
    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite, { image, size: [2, 2], rect: [2, 2, 2, 2] });
    world.step(0);
    const { rgba } = await captureTexture(world, camera);
    const pixel = (x: number, y: number) =>
        Array.from(rgba.subarray((y * 128 + x) * 4, (y * 128 + x) * 4 + 3));
    // The 2×2 world-unit quad covers pixels 48–80 by 16–48 at 16 pixels per unit.
    expect([pixel(56, 24), pixel(72, 24), pixel(56, 40), pixel(72, 40)]).toEqual([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 255],
    ]);
});
