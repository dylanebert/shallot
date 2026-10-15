import { expect, test } from "bun:test";
import { Billboard } from "../../core/rendering";
import { World } from "../../engine/ecs/world";
import { Sprite } from "./pack";

test("sprites default to Screen orientation", () => {
    const world = new World();
    world.registry.register(Sprite);
    const sprite = world.create();
    world.add(sprite, Sprite);
    expect(world.storage(Sprite).billboard.get(sprite)).toBe(Billboard.Screen);
    world.dispose();
});
