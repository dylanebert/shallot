import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Billboard, Camera } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import {
    AlphaPipelineKey,
    Draws,
    materialTypeId,
    StandardRenderer,
} from "../../standard/rendering";
import { Sprite, SpriteMaterialType, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [SpritePlugin] },
    { defaults: false, plugins: [SpritePlugin] },
]);

test("a large anchored world Sprite stays in the camera draw when its quad reaches the frustum", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });

    const sprite = world.create();
    world.add(sprite, Transform, { translation: [4, 0, 0, 0] });
    world.add(sprite, Sprite, {
        size: [10, 10],
        anchor: [0.5, 0.5],
        billboard: Billboard.World,
    });

    world.step(0);
    world.step(0);
    const instanceBound = world.storage(MeshInstance).cullBounds;
    expect(instanceBound.w.get(sprite)).toBeGreaterThan(7);
    const type = materialTypeId(world, SpriteMaterialType);
    const mesh = world.storage(MeshInstance).mesh.get(sprite);
    const draw = [...world.resource(Draws)].find(
        (entry) =>
            entry.materialType === type &&
            entry.alphaPipelineKey === AlphaPipelineKey.Mask &&
            entry.mesh === mesh,
    );
    expect(draw).toBeDefined();
    const offset = draw!.args.offset ?? 0;
    const snapshot = await probeBuffer(world, world.gpu.root.unwrap(draw!.args.indirect), {
        offset,
        size: 20,
    });
    expect(new Uint32Array(snapshot.bytes)[1]).toBe(1);
});

test("a Sprite spawned before a frame draws in that frame", async () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });

    const sprite = world.create();
    world.add(sprite, Transform, { translation: [0, 0, 0, 0] });
    world.add(sprite, Sprite, { size: [1, 1] });

    world.step(0);
    const type = materialTypeId(world, SpriteMaterialType);
    const mesh = world.storage(MeshInstance).mesh.get(sprite);
    const draw = [...world.resource(Draws)].find(
        (entry) =>
            entry.materialType === type &&
            entry.alphaPipelineKey === AlphaPipelineKey.Mask &&
            entry.mesh === mesh,
    );
    expect(draw).toBeDefined();
    const snapshot = await probeBuffer(world, world.gpu.root.unwrap(draw!.args.indirect), {
        offset: draw!.args.offset ?? 0,
        size: 20,
    });
    expect(new Uint32Array(snapshot.bytes)[1]).toBe(1);
});

test("a terminal simulation system's Sprite size change reaches that frame's cull bounds", () => {
    const { world } = subjects()[0];
    const sprite = world.create();
    world.add(sprite, Transform);
    world.add(sprite, Sprite, { size: [1, 1] });
    world.step(0);

    world.addSystem({
        name: "sprite-size-terminal",
        group: "simulation",
        terminal: true,
        update: (world) => world.storage(Sprite).size.x.set(sprite, 10),
    });
    world.step(0);
    expect(world.storage(MeshInstance).cullBounds.w.get(sprite)).toBeGreaterThan(5);
});
