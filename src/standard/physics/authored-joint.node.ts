import { expect, setDefaultTimeout, test } from "bun:test";
import { Time } from "@dylanebert/shallot";
import { Body, BodyType, Shape } from "@dylanebert/shallot/physics";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { GlobalTransform, Transform } from "@dylanebert/shallot/transform";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";
import { physicsBodyIndex, physicsState } from "./runtime";
import { defaultJointDef } from "./solver/joint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "./solver/revoluteJoint";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("the solver world's entity lookup lets a revolute joint constrain two authored bodies", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const anchor = world.create();
        world.add(anchor, Transform, { translation: [0, 3, 0] });
        world.add(anchor, Body);
        world.add(anchor, Shape);
        const bob = world.create();
        world.add(bob, Transform, { translation: [0, 1, 0] });
        world.add(bob, Body, { type: BodyType.Dynamic });
        world.add(bob, Shape);
        const free = world.create();
        world.add(free, Transform, { translation: [4, 1, 0] });
        world.add(free, Body, { type: BodyType.Dynamic });
        world.add(free, Shape);
        const empty = world.create();
        expect(physicsBodyIndex(world, bob)).toBe(-1);
        expect(physicsBodyIndex(world, empty)).toBe(-1);
        world.step(Time.FIXED_DT);
        const state = physicsState(world)!;
        const fixedBody = physicsBodyIndex(world, anchor);
        const movingBody = physicsBodyIndex(world, bob);
        expect(fixedBody).toBeGreaterThanOrEqual(0);
        expect(movingBody).toBeGreaterThanOrEqual(0);
        createRevoluteJoint(
            state,
            defaultRevoluteJointDef({
                ...defaultJointDef(),
                bodyIdA: fixedBody,
                bodyIdB: movingBody,
                localFrameB: {
                    p: { x: 0, y: 2, z: 0 },
                    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
                },
            }),
        );
        for (let tick = 0; tick < 120; tick++) world.step(Time.FIXED_DT);
        expect(world.storage(GlobalTransform).translation.y.get(bob)).toBeCloseTo(1, 1);
        expect(world.storage(GlobalTransform).translation.y.get(free)).toBeLessThan(-10);
        world.remove(bob, Body);
        expect(physicsBodyIndex(world, bob)).toBe(-1);
    } finally {
        app.dispose();
    }
});
