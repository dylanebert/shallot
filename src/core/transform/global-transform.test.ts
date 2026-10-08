import { expect, test } from "bun:test";
import { component, f32 } from "../../engine/ecs/component";
import { World } from "../../engine/ecs/world";
import { GlobalTransform, Transform, TransformPlugin } from "./index";

test("TransformPlugin installs processing on a bare World for frames and exact ticks", () => {
    const world = new World();
    for (const entry of TransformPlugin.components ?? []) world.registry.register(entry);
    world.registerRecovery(
        TransformPlugin.name,
        typeof TransformPlugin.recovery === "function"
            ? TransformPlugin.recovery(world)
            : TransformPlugin.recovery,
    );
    for (const system of TransformPlugin.systems ?? [])
        world.addSystem(system, TransformPlugin.name);
    const eid = world.create();
    world.add(eid, Transform);
    const source = world.storage(Transform).translation;
    const global = world.storage(GlobalTransform).translation;
    source.x.set(eid, 3);
    world.step(0);
    expect(global.x.get(eid)).toBe(3);
    source.x.set(eid, 7);
    world.tick();
    expect(global.x.get(eid)).toBe(7);
    expect(() => world.snapshot()).not.toThrow();
    world.dispose();
});

test("required GlobalTransform is inserted when missing and remains after its requirer is removed", () => {
    const Producer = component(
        "PlacementProducer",
        { value: f32 },
        { requires: [GlobalTransform] },
    );
    const world = new World();
    world.registry.register(Producer);
    const eid = world.create();
    world.add(eid, Producer);
    expect(world.has(eid, GlobalTransform)).toBe(true);
    world.storage(GlobalTransform).translation.set(eid, 7, 8, 9, 0);
    world.remove(eid, Producer);
    world.add(eid, Producer);
    expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(7);
    world.remove(eid, Producer);
    world.step(0);
    expect(world.has(eid, GlobalTransform)).toBe(true);
    world.dispose();
});
