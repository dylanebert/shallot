import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, Time } from "@dylanebert/shallot";
import { Body } from "@dylanebert/shallot/physics";
import {
    physicsWorld,
    readBody,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

test("the solver world's entity lookup lets a revolute joint constrain two authored bodies", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const anchor = world.create();
        world.add(anchor, Body, { mass: 0, position: [0, 3, 0, 0] });
        const bob = world.create();
        world.add(bob, Body, { position: [0, 1, 0, 0] });
        const free = world.create();
        world.add(free, Body, { position: [4, 1, 0, 0] });
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
        expect(readBody(world, bob)!.position[1]).toBeCloseTo(1, 1);
        expect(readBody(world, free)!.position[1]).toBeLessThan(-10);
        world.remove(bob, Body);
        expect(solver.getBody(bob)).toBeNull();
    } finally {
        app.dispose();
    }
});
