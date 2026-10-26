import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { MeshMaterial } from "../../standard/rendering";
import { Sprite, SpritePlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [SpritePlugin] }]);

function held(world: World, eid: number): unknown[] {
    return [
        world.has(eid, Sprite),
        world.has(eid, MeshInstance),
        world.has(eid, MeshMaterial),
        world.storage(MeshInstance).mesh.get(eid),
        world.storage(MeshMaterial).type.get(eid),
        world.storage(MeshMaterial).material.get(eid),
    ];
}

test("a restore between a mesh entity and a published Sprite equals its image", () => {
    const { world } = subjects()[0];
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh: 3 });
    world.add(eid, MeshMaterial, { material: 5 });
    const mesh = world.snapshot();
    const meshHeld = held(world, eid);
    world.remove(eid, MeshInstance);
    world.remove(eid, MeshMaterial);
    world.add(eid, Sprite);
    world.step(0);
    const sprite = world.snapshot();
    const spriteHeld = held(world, eid);
    expect(spriteHeld.slice(0, 3)).toEqual([true, true, true]);

    world.restore(mesh);
    expect(held(world, eid)).toEqual(meshHeld);
    world.step(0);
    expect(held(world, eid)).toEqual(meshHeld);

    world.restore(sprite);
    expect(held(world, eid)).toEqual(spriteHeld);
    world.step(0);
    expect(held(world, eid)).toEqual(spriteHeld);
    world.destroy(eid);
});
