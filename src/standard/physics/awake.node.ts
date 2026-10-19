import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType } from "../../core/physics";
import { createApp } from "../../engine";
import { TIMEOUT } from "../../project/testing";
import { isAwake, StandardPhysicsPlugin, setAwake } from ".";

setDefaultTimeout(TIMEOUT.node);

test("isAwake reads live solver state, not the spawn definition", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic });
        expect(isAwake(world, body)).toBeUndefined();
        world.tick();
        expect(isAwake(world, body)).toBe(true);
        setAwake(world, body, false);
        expect(isAwake(world, body)).toBe(false);
        expect(world.storage(Body).isAwake.get(body)).toBe(1);
        setAwake(world, body, true);
        expect(isAwake(world, body)).toBe(true);
        world.destroy(body);
        world.tick();
        expect(isAwake(world, body)).toBeUndefined();
    } finally {
        app.dispose();
    }
});
