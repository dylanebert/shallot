import { expect, setDefaultTimeout, test } from "bun:test";
import { Time } from "@dylanebert/shallot";
import { Body, BodyType } from "@dylanebert/shallot/physics";
import {
    StandardPhysicsPlugin,
    setLinearVelocity,
    setTargetTransform,
} from "@dylanebert/shallot/standard/physics";
import { CEILING } from "@dylanebert/shallot/testing";
import { GlobalTransform } from "@dylanebert/shallot/transform";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";

setDefaultTimeout(CEILING.node);
await setupGlobals();

async function sleepingPlatform() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const platform = world.create();
    world.add(platform, Body, { type: BodyType.Kinematic });
    world.tick();
    setTargetTransform(world, platform, [0, 0, 0], [0, 0, 0, 1]);
    world.tick();

    setLinearVelocity(world, platform, { x: 1, y: 0, z: 0 });
    world.tick();
    setLinearVelocity(world, platform, { x: 0, y: 0, z: 0 });
    for (let tick = 0; tick < 120; tick++) world.tick();

    return {
        app,
        world,
        platform,
        position: world.storage(GlobalTransform).translation,
    };
}

test("setTargetTransform drives a sleeping body back to an unchanged target", async () => {
    const { app, world, platform, position } = await sleepingPlatform();
    try {
        expect(position.x.get(platform)).toBeGreaterThan(0);

        setTargetTransform(world, platform, [0, 0, 0], [0, 0, 0, 1]);
        world.tick();

        expect(position.x.get(platform)).toBeCloseTo(0, 5);
    } finally {
        app.dispose();
    }
});

test("setTargetTransform does not publish a sub-threshold target a sleeping body rejects", async () => {
    const { app, world, platform, position } = await sleepingPlatform();
    try {
        const held = position.x.get(platform);
        const published = held;
        const target = held + Time.FIXED_DT * 0.01;
        setTargetTransform(world, platform, [target, 0, 0], [0, 0, 0, 1]);

        expect(position.x.get(platform)).toBe(published);
        world.tick();
        expect(position.x.get(platform)).toBe(held);
    } finally {
        app.dispose();
    }
});
