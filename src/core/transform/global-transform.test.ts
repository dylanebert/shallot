import { expect, test } from "bun:test";
import { component, f32 } from "../../engine/ecs/component";
import { World } from "../../engine/ecs/world";
import { deriveTransforms, GlobalTransform, Transform, TransformPlugin } from "./index";

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

test("rotation-only and scale-only writes derive their matching GlobalTransform fields", () => {
    const world = new World();
    for (const entry of TransformPlugin.components ?? []) world.registry.register(entry);
    const eid = world.create();
    world.add(eid, Transform);
    deriveTransforms(world);
    world.clearChanges();

    world.storage(Transform).rotation.set(eid, 0, 0, 1, 0);
    deriveTransforms(world);
    expect(
        Array.from(world.storage(GlobalTransform).rotation.column.slice(eid * 4, eid * 4 + 4)),
    ).toEqual([0, 0, 1, 0]);
    world.clearChanges();

    world.storage(Transform).scale.set(eid, 2, 3, 4, 5);
    deriveTransforms(world);
    expect(
        Array.from(world.storage(GlobalTransform).scale.column.slice(eid * 4, eid * 4 + 4)),
    ).toEqual([2, 3, 4, 5]);
    world.dispose();
});

test("deriveTransforms visits marked eids beyond the first dirty word", () => {
    const world = new World();
    for (const entry of TransformPlugin.components ?? []) world.registry.register(entry);
    let eid = 0;
    while (eid < 32) eid = world.create();
    world.add(eid, Transform, { translation: [12, 13, 14, 15] });

    deriveTransforms(world);

    expect(
        Array.from(world.storage(GlobalTransform).translation.column.slice(eid * 4, eid * 4 + 4)),
    ).toEqual([12, 13, 14, 15]);
    world.dispose();
});

test("removing Transform does not derive its cleared fields into GlobalTransform", () => {
    const world = new World();
    for (const entry of TransformPlugin.components ?? []) world.registry.register(entry);
    const eid = world.create();
    world.add(eid, Transform, { translation: [4, 5, 6, 7] });
    deriveTransforms(world);
    const global = world.storage(GlobalTransform);
    global.translation.set(eid, 9, 8, 7, 6);
    global.rotation.set(eid, 1, 2, 3, 4);
    global.scale.set(eid, 5, 6, 7, 8);
    global.linearVelocity.set(eid, 11, 12, 13, 14);
    world.clearChanges();

    world.remove(eid, Transform);
    deriveTransforms(world);

    expect(world.has(eid, GlobalTransform)).toBe(true);
    expect(Array.from(global.translation.column.slice(eid * 4, eid * 4 + 4))).toEqual([9, 8, 7, 6]);
    expect(Array.from(global.rotation.column.slice(eid * 4, eid * 4 + 4))).toEqual([1, 2, 3, 4]);
    expect(Array.from(global.scale.column.slice(eid * 4, eid * 4 + 4))).toEqual([5, 6, 7, 8]);
    expect(Array.from(global.linearVelocity.column.slice(eid * 4, eid * 4 + 4))).toEqual([
        11, 12, 13, 14,
    ]);
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
