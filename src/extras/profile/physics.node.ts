import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { build, type Plugin } from "@dylanebert/shallot";
import { PhysicsProfilePlugin } from "@dylanebert/shallot/extras";
import { Body, PhysicsPlugin, physicsWorld, ShapeKind } from "@dylanebert/shallot/physics";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

let live: Awaited<ReturnType<typeof build>> | null = null;

afterEach(() => {
    live?.dispose();
    live = null;
});

async function stepFalling(plugins: Plugin[]) {
    live = await build({ defaults: false, plugins });
    const { state } = live;
    const eid = state.create();
    state.add(eid, Body);
    state.of(Body).shape.set(eid, ShapeKind.Box);
    state.of(Body).halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
    state.of(Body).pos.set(eid, 0, 5, 0, 0);
    state.of(Body).quat.set(eid, 0, 0, 0, 1);
    state.of(Body).mass.set(eid, 1);
    for (let i = 0; i < 10; i++) state.step(1 / 60);
    const world = physicsWorld(state);
    if (!world) throw new Error("inconclusive: physics world did not warm");
    return world.getProfile();
}

test("physics phase timings run only when the profile extra composes its clock: a composed State reads elapsed step time and a default State reads zero for every phase", async () => {
    const plain = await stepFalling([PhysicsPlugin]);
    expect(Object.values(plain).every((ms) => ms === 0)).toBe(true);
    live?.dispose();
    live = null;

    const timed = await stepFalling([PhysicsPlugin, PhysicsProfilePlugin]);
    expect(timed.step).toBeGreaterThan(0);
    expect(timed.solve).toBeGreaterThan(0);
    expect(timed.step).toBeGreaterThanOrEqual(timed.solve);
});
