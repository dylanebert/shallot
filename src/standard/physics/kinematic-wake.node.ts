import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, Time } from "@dylanebert/shallot";
import { Body, BodyType } from "@dylanebert/shallot/physics";
import {
    physicsWorld,
    StandardPhysicsPlugin,
    setKinematic,
    setVelocity,
} from "@dylanebert/shallot/standard/physics";
import { GlobalTransform } from "@dylanebert/shallot/transform";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
await setupGlobals();

async function sleepingPlatform() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    const platform = world.create();
    world.add(platform, Body, { type: BodyType.Kinematic });
    world.tick();
    setKinematic(world, platform, [0, 0, 0], [0, 0, 0, 1]);
    world.tick();

    setVelocity(world, platform, 1, 0, 0);
    world.tick();
    setVelocity(world, platform, 0, 0, 0);
    for (let tick = 0; tick < 120; tick++) world.tick();

    return {
        app,
        world,
        platform,
        body: physicsWorld(world)!.getBody(platform)!,
        position: world.storage(GlobalTransform).translation,
    };
}

test("setKinematic drives a sleeping body back to an unchanged target", async () => {
    const { app, world, platform, body, position } = await sleepingPlatform();
    try {
        expect(body.isAwake()).toBe(false);
        expect(position.x.get(platform)).toBeGreaterThan(0);

        setKinematic(world, platform, [0, 0, 0], [0, 0, 0, 1]);
        world.tick();

        expect(body.getPosition().x).toBeCloseTo(0, 5);
        expect(position.x.get(platform)).toBeCloseTo(0, 5);
    } finally {
        app.dispose();
    }
});

test("setKinematic does not publish a sub-threshold target a sleeping body rejects", async () => {
    const { app, world, platform, body, position } = await sleepingPlatform();
    try {
        expect(body.isAwake()).toBe(false);
        const held = body.getPosition().x;
        const published = position.x.get(platform);
        const target = held + Time.FIXED_DT * 0.01;
        setKinematic(world, platform, [target, 0, 0], [0, 0, 0, 1]);

        expect(body.isAwake()).toBe(false);
        expect(position.x.get(platform)).toBe(published);
        world.tick();
        expect(body.getPosition().x).toBe(held);
        expect(position.x.get(platform)).toBe(published);
    } finally {
        app.dispose();
    }
});
