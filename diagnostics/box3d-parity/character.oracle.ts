import { test } from "bun:test";
import { Body, PhysicsPlugin, ShapeKind } from "../../src/core/physics";
import { World } from "../../src/engine";
import { Character, CharacterPlugin, StandardPhysicsPlugin } from "../../src/standard/physics";

test("character scene fixed-step time", async () => {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const plugin of [PhysicsPlugin, CharacterPlugin])
        for (const component of plugin.components!) world.registry.register(component);
    for (const system of [...StandardPhysicsPlugin.systems!, ...CharacterPlugin.systems!]) world.addSystem(system);
    const floor = world.create();
    world.add(floor, Body, { type: 0, position: [0, -0.5, 0, 0], halfExtents: [1000, 0.5, 1000, 0] });
    const mover = world.create();
    world.add(mover, Body, { type: 1, shape: ShapeKind.Capsule, position: [0, 1.3, 0, 0], halfExtents: [0, 0.5, 0, 0.3] });
    world.add(mover, Character);
    const velocity = world.storage(Character).velocity;
    const step = (i: number) => {
        velocity.set(mover, i % 120 < 60 ? 3 : -3, -12, 0, 0);
        world.step(1 / 60);
    };
    for (let i = 0; i < 12000; ++i) step(i);
    const samples: number[] = [];
    for (let batch = 0; batch < 21; ++batch) {
        const start = performance.now();
        for (let i = 0; i < 1000; ++i) step(i);
        samples.push((performance.now() - start) / 1000);
    }
    samples.sort((a, b) => a - b);
    console.log(`character median ms/step: ${samples[10]}`);
    world.dispose();
});
