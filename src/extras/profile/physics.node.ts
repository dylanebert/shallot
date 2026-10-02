import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { createApp, type Plugin } from "@dylanebert/shallot";
import { PhysicsProfilePlugin } from "@dylanebert/shallot/extras";
import { Body, ShapeKind } from "@dylanebert/shallot/physics";
import { physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

let live: Awaited<ReturnType<typeof createApp>> | null = null;

afterEach(() => {
    live?.dispose();
    live = null;
});

async function stepFalling(plugins: Plugin[]) {
    live = await createApp({ defaults: false, plugins });
    const { world } = live;
    const eid = world.create();
    world.add(eid, Body);
    world.storage(Body).shape.set(eid, ShapeKind.Box);
    world.storage(Body).halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
    world.storage(Body).position.set(eid, 0, 5, 0, 0);
    world.storage(Body).rotation.set(eid, 0, 0, 0, 1);
    world.storage(Body).mass.set(eid, 1);
    for (let i = 0; i < 10; i++) world.step(1 / 60);
    const solverWorld = physicsWorld(world);
    if (!solverWorld) throw new Error("inconclusive: physics world did not warm");
    return solverWorld.getProfile();
}

test("physics phase timings run only when the profile extra composes its clock: a composed World reads elapsed step time and a default World reads zero for every phase", async () => {
    const plain = await stepFalling([StandardPhysicsPlugin]);
    expect(Object.values(plain).every((ms) => ms === 0)).toBe(true);
    live?.dispose();
    live = null;

    const timed = await stepFalling([StandardPhysicsPlugin, PhysicsProfilePlugin]);
    expect(timed.step).toBeGreaterThan(0);
    expect(timed.solve).toBeGreaterThan(0);
    expect(timed.step).toBeGreaterThanOrEqual(timed.solve);
});
