import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, Hulls, Shape } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine/app";
import { PhysicsWorldDefinition, StandardPhysicsPlugin } from ".";
import { physicsCounters } from "./runtime";

setDefaultTimeout(TIMEOUT.node);

// Box3D's Large World at 100 by 100: static boxes naming one Hulls entry, each invoking contact creation.
test("10,000 floor shapes naming one box build one hull, the first tick visits each once and the second visits none", async () => {
    const grid = 100;
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            const definition = world.resource(PhysicsWorldDefinition);
            definition.workerCount = 0;
            definition.capacity = {
                staticShapeCount: grid * grid,
                staticBodyCount: grid * grid,
                dynamicShapeCount: 0,
                dynamicBodyCount: 0,
                contactCount: 1024,
            };
        },
    });
    try {
        const world = app.world;
        const geometry = world.resource(Hulls).add({ name: "floor", halfExtents: [5, 0.25, 5] });
        const built = physicsCounters(world).hullsBuilt;
        for (let i = 0; i < grid; i++)
            for (let j = 0; j < grid; j++) {
                const eid = world.create();
                world.add(eid, Transform, {
                    translation: [-5 * grid + (i + 0.5) * 10, 0, -5 * grid + (j + 0.5) * 10],
                });
                world.add(eid, Body);
                world.add(eid, Shape, { geometry, invokeContactCreation: 1 });
            }
        world.tick();
        expect(physicsCounters(world).hullsBuilt - built).toBe(1);
        expect(physicsCounters(world).shapesVisited).toBe(grid * grid);
        world.tick();
        expect(physicsCounters(world).shapesVisited).toBe(0);
        expect(physicsCounters(world).hullsBuilt - built).toBe(1);
    } finally {
        app.dispose();
    }
});
