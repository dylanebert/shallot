import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, Time } from "@dylanebert/shallot";
import { Body, BodyType, Shape } from "@dylanebert/shallot/physics";
import { physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { GlobalTransform } from "@dylanebert/shallot/transform";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("the solver world's entity lookup lets a revolute joint constrain two authored bodies", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const anchor = world.create();
        world.add(anchor, Body, { position: [0, 3, 0, 0] });
        world.add(anchor, Shape);
        const bob = world.create();
        world.add(bob, Body, { type: BodyType.Dynamic, position: [0, 1, 0, 0] });
        world.add(bob, Shape);
        const free = world.create();
        world.add(free, Body, { type: BodyType.Dynamic, position: [4, 1, 0, 0] });
        world.add(free, Shape);
        const empty = world.create();
        const solver = physicsWorld(world)!;
        expect(solver.getBody(bob)).toBeNull();
        expect(solver.getBody(empty)).toBeNull();
        world.step(Time.FIXED_DT);
        const fixedBody = solver.getBody(anchor)!;
        const movingBody = solver.getBody(bob)!;
        expect(fixedBody).not.toBeNull();
        expect(movingBody).not.toBeNull();
        solver.createRevoluteJoint(fixedBody, movingBody, {
            localFrameB: { p: { x: 0, y: 2, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        });
        for (let tick = 0; tick < 120; tick++) world.step(Time.FIXED_DT);
        expect(world.storage(GlobalTransform).translation.y.get(bob)).toBeCloseTo(1, 1);
        expect(world.storage(GlobalTransform).translation.y.get(free)).toBeLessThan(-10);
        world.remove(bob, Body);
        expect(solver.getBody(bob)).toBeNull();
    } finally {
        app.dispose();
    }
});
