import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Billboard,
    Camera,
    CameraMode,
    captureTexture,
    ImageAddressMode,
    ImageBindings,
    ImageFilterMode,
    registerImage,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Sprite, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            SpritePlugin,
            {
                name: "ImageSamplerConsumers",
                initialize(world) {
                    for (const mode of [
                        ImageAddressMode.ClampToEdge,
                        ImageAddressMode.Repeat,
                        ImageAddressMode.MirrorRepeat,
                    ]) {
                        for (const filter of [ImageFilterMode.Nearest, ImageFilterMode.Linear]) {
                            registerImage(world, new Blob(), {
                                sampler: {
                                    addressModeU: mode,
                                    addressModeV: mode,
                                    magFilter: filter,
                                    minFilter: filter,
                                },
                            });
                        }
                    }
                },
            },
        ],
    },
]);

test("Sprite and StandardMaterial use the same registered address modes and filters outside [0, 1]", async () => {
    const { world } = subjects()[0];
    const device = world.gpu.device;
    const image = device.createTexture({
        size: [2, 2, 6],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(image);
    const pixels = new Uint8Array(2 * 2 * 6 * 4);
    for (let layer = 0; layer < 6; layer++)
        pixels.set(
            [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255],
            layer * 16,
        );
    device.queue.writeTexture(
        { texture: image },
        pixels,
        { bytesPerRow: 8, rowsPerImage: 2 },
        [2, 2, 6],
    );
    world.resource(ImageBindings).images = image;
    const vertices: number[] = [];
    for (const [x, y] of [
        [-1, -1],
        [2, -1],
        [2, 2],
        [-1, 2],
    ])
        vertices.push(x, y, 0, x, 0, 0, 1, 1 - y);
    const mesh = world.resource(Meshes).add({
        name: "outside-uv-quad",
        vertices: new Float32Array(vertices),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5] });
    world.add(camera, Camera, { mode: CameraMode.Orthographic, size: 2, antialias: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 128, height: 64 });
    const owner = world.create();
    world.add(owner, Transform, { translation: [100, 0, 0] });
    world.add(owner, Sprite, { size: [1, 1], billboard: Billboard.World });
    world.step(0);
    // Reuse Sprite's public material row on a mesh whose local coordinates cross image boundaries.
    const sprite = world.create();
    world.add(sprite, Transform, { translation: [2, 0, 0] });
    world.add(sprite, MeshInstance, { mesh });
    world.add(sprite, MeshMaterial, {
        type: world.storage(MeshMaterial).type.get(owner),
        material: world.storage(MeshMaterial).material.get(owner),
    });
    const row = world.resource(Materials).add(StandardMaterial({ unlit: true }));
    const standard = world.create();
    world.add(standard, Transform, { translation: [-2.5, -0.5, 0] });
    world.add(standard, MeshInstance, { mesh });
    world.add(standard, MeshMaterial, row);
    const frames: Uint8ClampedArray[] = [];
    for (let id = 0; id < 6; id++) {
        world.storage(Sprite).image.set(owner, id);
        world.resource(Materials).update(row, { baseColorTexture: id });
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        frames.push(rgba);
        let maximum = 0;
        for (let y = 10; y < 54; y++)
            for (let x = 10; x < 54; x++)
                for (let c = 0; c < 3; c++) {
                    maximum = Math.max(
                        maximum,
                        Math.abs(rgba[(y * 128 + x) * 4 + c] - rgba[(y * 128 + x + 64) * 4 + c]),
                    );
                }
        expect(maximum, `image ${id}`).toBeLessThanOrEqual(1);
    }
    for (let id = 1; id < 6; id++) expect(frames[id]).not.toEqual(frames[0]);
});
