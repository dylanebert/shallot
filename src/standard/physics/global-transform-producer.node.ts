import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { Body, BodyType } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { Time } from "../../engine";
import { PhysicsWorldDefinition, StandardPhysicsPlugin } from ".";
import { BodyField, setBodyField } from "./kernel/bodyrecords";
import { physicsBodyIndex, physicsState, setAwake, setLinearVelocity } from "./runtime";
import { bodyGetLinearVelocity, bodyIsAwake } from "./world/body";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("a body falling asleep publishes zero ECS velocity, and waking publishes its current velocity", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.step(Time.FIXED_DT);
        const state = physicsState(world)!;
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        const body = physicsBodyIndex(world, eid);
        expect(body).toBeGreaterThanOrEqual(0);
        setBodyField(state, body, BodyField.sleepThreshold, 1);
        setLinearVelocity(world, eid, { x: 0.01, y: 0, z: 0 });
        for (let tick = 0; tick < 60 && bodyIsAwake(state, body); ++tick) world.step(Time.FIXED_DT);
        expect(bodyIsAwake(state, body)).toBe(false);
        expect(bodyGetLinearVelocity(state, body, { x: 0, y: 0, z: 0 })).toEqual({
            x: 0,
            y: 0,
            z: 0,
        });
        const velocity = world.storage(GlobalTransform).linearVelocity;
        expect(Array.from(velocity.column.subarray(eid * 4, eid * 4 + 4))).toEqual([0, 0, 0, 0]);

        setAwake(world, eid, true);
        world.step(Time.FIXED_DT);
        expect(bodyIsAwake(state, body)).toBe(true);
        setAwake(world, eid, false);
        setLinearVelocity(world, eid, { x: 2, y: 3, z: 4 });
        world.step(Time.FIXED_DT);
        expect(bodyIsAwake(state, body)).toBe(true);
        expect(bodyGetLinearVelocity(state, body, { x: 0, y: 0, z: 0 })).toEqual({
            x: 2,
            y: 3,
            z: 4,
        });
        expect(Array.from(velocity.column.subarray(eid * 4, eid * 4 + 4))).toEqual([2, 3, 4, 0]);
    } finally {
        app.dispose();
    }
});

test("physics warns once per entity carrying Body and Transform, not for either alone", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body);
        const transform = world.create();
        world.add(transform, Transform);
        world.step(Time.FIXED_DT);
        const both = world.create();
        world.add(both, Body);
        world.add(both, Transform);
        expectWarning(new RegExp(`physics-sync: entity ${both} carries both Body and Transform`));
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
    } finally {
        app.dispose();
    }
});
