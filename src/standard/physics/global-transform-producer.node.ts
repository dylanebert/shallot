import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createTestGpuApp as createApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { Body, BodyType, Shape } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import { Time } from "../../engine";
import { PhysicsWorldDefinition, StandardPhysicsPlugin } from ".";
import { BodyField, setBodyField } from "./kernel/bodyrecords";
import { physicsBodyIndex, physicsState, setAwake, setLinearVelocity } from "./runtime";
import { bodyGetLinearVelocity, bodyIsAwake } from "./world/body";

setDefaultTimeout(TIMEOUT.node);

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
        await disposeGpuApps([app]);
    }
});

test("a falling body spawns from Transform and writes its pose back each step", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Transform, { translation: [3, 10, 4], rotation: [0, 0, 1, 0] });
        world.add(body, Body, { type: BodyType.Dynamic });
        world.add(body, Shape);
        const pose = world.storage(Transform);
        const global = world.storage(GlobalTransform);
        let previousY = 10;
        for (let tick = 0; tick < 3; tick++) {
            world.tick();
            expect(pose.translation.x.get(body)).toBe(3);
            expect(pose.translation.z.get(body)).toBe(4);
            expect(pose.rotation.z.get(body)).toBe(1);
            expect(pose.translation.y.get(body)).toBeLessThan(previousY);
            expect(pose.translation.y.get(body)).toBe(global.translation.y.get(body));
            expect(pose.rotation.z.get(body)).toBe(global.rotation.z.get(body));
            previousY = pose.translation.y.get(body);
        }
    } finally {
        await disposeGpuApps([app]);
    }
});
