import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { Body, BodyType } from "../../core/physics";
import { createApp, GlobalTransform, Time, Transform } from "../../engine";
import { physicsWorld, readBody, StandardPhysicsPlugin } from ".";
import { BodyField, setBodyField } from "./kernel/bodyrecords";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("a body falling asleep publishes zero ECS velocity, and waking publishes its current velocity", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.step(Time.FIXED_DT);
        const physics = physicsWorld(world)!;
        physics.setGravity({ x: 0, y: 0, z: 0 });
        const body = physics.getBody(eid)!;
        setBodyField(physics.state, body.id.index1 - 1, BodyField.sleepThreshold, 1);
        body.setLinearVelocity({ x: 0.01, y: 0, z: 0 });
        let fellAsleep = false;
        for (let tick = 0; tick < 60 && !fellAsleep; ++tick) {
            world.step(Time.FIXED_DT);
            const events = physics.getBodyEvents();
            for (let i = 0; i < events.count; ++i) {
                const event = events.moveEvents[i];
                if (event.body.id.index1 === body.id.index1 && event.fellAsleep) fellAsleep = true;
            }
        }
        expect(fellAsleep).toBe(true);
        expect(body.isAwake()).toBe(false);
        expect(body.getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
        expect(readBody(world, eid)!.linearVelocity).toEqual([0, 0, 0]);
        const velocity = world.storage(GlobalTransform).linearVelocity;
        expect(Array.from(velocity.column.subarray(eid * 4, eid * 4 + 4))).toEqual([0, 0, 0, 0]);

        body.setAwake(true);
        world.step(Time.FIXED_DT);
        expect(body.isAwake()).toBe(true);
        expect(readBody(world, eid)!.linearVelocity).toEqual([0, 0, 0]);
        body.setAwake(false);
        body.setLinearVelocity({ x: 2, y: 3, z: 4 });
        world.step(Time.FIXED_DT);
        expect(body.isAwake()).toBe(true);
        expect(body.getLinearVelocity()).toEqual({ x: 2, y: 3, z: 4 });
        expect(readBody(world, eid)!.linearVelocity).toEqual([2, 3, 4]);
        expect(Array.from(velocity.column.subarray(eid * 4, eid * 4 + 4))).toEqual([2, 3, 4, 0]);
    } finally {
        app.dispose();
    }
});

test("physics warns once per entity carrying Body and Transform, not for either alone", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const warning = spyOn(console, "warn").mockImplementation(() => {});
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body);
        const transform = world.create();
        world.add(transform, Transform);
        world.step(Time.FIXED_DT);
        expect(warning).not.toHaveBeenCalled();
        const both = world.create();
        world.add(both, Body);
        world.add(both, Transform);
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(1);
        expect(warning.mock.calls[0]?.[0]).toBe(
            `physics-sync: entity ${both} carries both Body and Transform; both write GlobalTransform`,
        );
    } finally {
        warning.mockRestore();
        app.dispose();
    }
});
