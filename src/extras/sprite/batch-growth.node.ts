import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, registerImage } from "../../core/rendering";
import { imageTextures } from "../../core/rendering/image";
import { Transform } from "../../core/transform";
import { rawDevice } from "../../engine/runtime";
import { Draws, StandardRenderer } from "../../standard/rendering";
import { MaterialBatches } from "../../standard/rendering/material-batch";
import { Sprite, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const configs: Parameters<typeof gpuApps>[1] = [
    {
        defaults: false,
        plugins: [
            SpritePlugin,
            {
                name: "BatchGrowth",
                dependencies: [SpritePlugin],
                initialize(world) {
                    registerImage(world, new Blob());
                    registerImage(world, new Blob());
                    const camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 5] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    attachTexture(world, camera, { width: 32, height: 32 });
                },
            },
        ],
    },
];
const apps = gpuApps(import.meta.path, [configs[0]!, configs[0]!]);

test("100 animated image rows re-pair only their consumers and reuse freed batch slots and GPU buffers", () => {
    const { world } = apps()[1];
    for (let id = 0; id < 2; id++) {
        const texture = world.gpu.device.createTexture({
            size: [1, 1],
            format: "rgba8unorm-srgb",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        world.own(texture);
        world.resource(imageTextures).textures[0][id] = texture;
    }
    const sprites = new Uint32Array(10000);
    for (let i = 0; i < sprites.length; i++) {
        const eid = world.create();
        sprites[i] = eid;
        world.add(eid, Transform);
        world.add(eid, Sprite, { image: i < 100 ? 1 : 0 });
    }
    world.step(0);
    world.step(0);
    const state = world.resource(MaterialBatches);
    const batches = state.types.length;
    const buffer = state.buffer;
    const probes = spyOn(world, "has");
    const buffers = spyOn(rawDevice(world.gpu.device), "createBuffer");
    try {
        world.step(0);
        const idle = probes.mock.calls.length;
        for (let frame = 0; frame < 4; frame++) {
            probes.mockClear();
            for (let i = 0; i < 100; i++) world.storage(Sprite).image.set(sprites[i], frame & 1);
            world.step(0);
            expect(probes.mock.calls.length).toBeLessThanOrEqual(idle + 300);
            expect(state.types.length).toBe(batches);
            expect(state.buffer).toBe(buffer);
        }
        expect(buffers.mock.calls.length).toBe(0);
    } finally {
        buffers.mockRestore();
        probes.mockRestore();
    }
});

test("growing sprite image batches alone grows indirect arguments and counts without new meshes", async () => {
    const { world } = apps()[0];
    const device = world.gpu.device;
    for (let id = 0; id < 2; id++) {
        const texture = device.createTexture({
            size: [1, 1],
            format: "rgba8unorm-srgb",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        world.own(texture);
        world.resource(imageTextures).textures[0][id] = texture;
    }
    const sprites: number[] = [];
    for (let i = 0; i < 2; i++) {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Sprite);
        sprites.push(eid);
    }
    world.step(0);
    await device.queue.onSubmittedWorkDone();
    const before = [...world.resource(Draws)].map((draw) => draw.mesh);
    device.pushErrorScope("validation");
    world.storage(Sprite).image.set(sprites[1], 1);
    world.step(0);
    await device.queue.onSubmittedWorkDone();
    const error = await device.popErrorScope();
    expect(error?.message ?? null).toBeNull();
    const draws = [...world.resource(Draws)];
    expect(draws.length).toBe(before.length + 1);
    expect(draws.every((draw) => before.includes(draw.mesh))).toBe(true);
    for (const draw of draws)
        expect(world.gpu.root.unwrap(draw.args.indirect).size).toBeGreaterThanOrEqual(
            draw.args.viewStride!,
        );
});
