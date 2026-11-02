import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Billboard,
    Camera,
    CameraMode,
    captureTexture,
    ImageAddressMode,
    ImageFilterMode,
    imageSampler,
    registerImage,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { imageTextures } from "../../core/rendering/image";
import { Transform } from "../../core/transform";
import { rawDevice } from "../../engine/runtime";
import { Draws, MeshMaterial, StandardRenderer } from "../../standard/rendering";
import { MaterialBatches, materialBatch } from "../../standard/rendering/material-batch";
import { Sprite, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
let camera = -1;
const apps = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            SpritePlugin,
            {
                name: "PerImageSprites",
                dependencies: [SpritePlugin],
                initialize(world) {
                    for (const addressModeU of [
                        ImageAddressMode.ClampToEdge,
                        ImageAddressMode.Repeat,
                        ImageAddressMode.Repeat,
                    ])
                        registerImage(world, new Blob(), {
                            sampler: {
                                addressModeU,
                                magFilter: ImageFilterMode.Nearest,
                                minFilter: ImageFilterMode.Nearest,
                            },
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

test("two sprite images split draws and bind their registered hardware addressing; changed images move batches", async () => {
    const { world } = apps()[0];
    const device = world.gpu.device;
    const samplers = spyOn(rawDevice(device), "createSampler");
    const groups = spyOn(rawDevice(device), "createBindGroup");
    try {
        for (let id = 0; id < 2; id++) {
            const texture = device.createTexture({
                size: [2, 1],
                format: "rgba8unorm-srgb",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            world.own(texture);
            device.queue.writeTexture(
                { texture },
                new Uint8Array(
                    id === 0
                        ? [255, 0, 0, 255, 0, 255, 0, 255]
                        : [0, 0, 255, 255, 255, 255, 0, 255],
                ),
                { bytesPerRow: 8 },
                [2, 1],
            );
            world.resource(imageTextures).textures[0][id] = texture;
        }
        const vertices: number[] = [];
        for (const [x, y] of [
            [-1, -1],
            [2, -1],
            [2, 2],
            [-1, 2],
        ])
            vertices.push(x, y, 0, x, 0, 0, 1, 1 - y);
        const mesh = world.resource(Meshes).add({
            name: "outside-image-edges",
            vertices: new Float32Array(vertices),
            indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        });
        const owners: number[] = [];
        for (let id = 0; id < 2; id++) {
            const owner = world.create();
            world.add(owner, Transform, { translation: [100, 0, 0] });
            world.add(owner, Sprite, { image: id, billboard: Billboard.World });
            owners.push(owner);
        }
        world.step(0);
        for (let id = 0; id < 2; id++) {
            const owner = owners[id];
            world.remove(owner, MeshInstance);
            const eid = world.create();
            world.add(eid, Transform, { translation: [id === 0 ? -2 : 2, 0, 0] });
            world.add(eid, MeshInstance, { mesh });
            world.add(eid, MeshMaterial, {
                type: world.storage(MeshMaterial).type.get(owner),
                material: world.storage(MeshMaterial).material.get(owner),
            });
        }
        world.step(0);
        const draws = [...world.resource(Draws)].filter((draw) => draw.mesh === mesh);
        expect(draws.length).toBe(2);
        const modes = samplers.mock.calls.map(([descriptor]) => descriptor?.addressModeU);
        expect(modes).toContain("clamp-to-edge");
        expect(modes).toContain("repeat");
        for (const mode of ["clamp-to-edge", "repeat"]) {
            const index = modes.indexOf(mode as GPUAddressMode);
            const sampler = samplers.mock.results[index].value;
            expect(
                groups.mock.calls.some(([descriptor]) =>
                    Array.from(descriptor.entries).some((entry) => entry.resource === sampler),
                ),
            ).toBe(true);
        }
        expect(imageSampler(world, 1)).toBe(imageSampler(world, 2));
        const first = (await captureTexture(world, camera)).rgba;
        const pixel = (rgba: Uint8ClampedArray, x: number) =>
            Array.from(rgba.subarray((32 * 128 + x) * 4, (32 * 128 + x) * 4 + 3));
        expect(pixel(first, 20)).toEqual([255, 0, 0]);
        expect(pixel(first, 84)).toEqual([255, 255, 0]);
        const material = world.storage(MeshMaterial);
        const type = material.type.get(owners[0]);
        const row = material.material.get(owners[0]);
        expect(materialBatch(world, type, row)).not.toBe(
            materialBatch(world, type, material.material.get(owners[1])),
        );
        world.storage(Sprite).image.set(owners[0], 1);
        world.step(0);
        expect(materialBatch(world, type, row)).toBe(
            materialBatch(world, type, material.material.get(owners[1])),
        );
        expect([...world.resource(Draws)].filter((draw) => draw.mesh === mesh).length).toBe(1);
        expect(pixel((await captureTexture(world, camera)).rgba, 20)).toEqual([255, 255, 0]);
        const late = registerImage(world, new Blob());
        expectWarning(new RegExp(`color image ${late} has no native texture`));
        world.storage(Sprite).image.set(owners[0], late);
        world.step(0);
        expect(pixel((await captureTexture(world, camera)).rgba, 20)).toEqual([0, 0, 0]);
        world.step(0);
        await captureTexture(world, camera);
        const groupCount = groups.mock.calls.length;
        const samplerCount = samplers.mock.calls.length;
        world.step(0);
        await captureTexture(world, camera);
        expect(groups.mock.calls.length).toBe(groupCount);
        expect(samplers.mock.calls.length).toBe(samplerCount);
        const batch = materialBatch(world, type, row);
        expect(world.resource(MaterialBatches).types[batch]).toBe(type);
        const fallback = world.resource(imageTextures).fallback[0];
        expect(fallback.width).toBe(1);
        expect(fallback.height).toBe(1);
    } finally {
        groups.mockRestore();
        samplers.mockRestore();
    }
});
