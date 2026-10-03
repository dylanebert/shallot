import { afterEach, expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { Profile, ProfilePlugin } from "@dylanebert/shallot/extras";
import { Body, ShapeKind } from "@dylanebert/shallot/physics";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";
import { cpuTotal } from "../../extras/profile/cpu";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();
let live: Awaited<ReturnType<typeof createApp>> | null = null;
afterEach(() => {
    live?.dispose();
    live = null;
});

test("composed physics records phases as parts of its scheduler timing", async () => {
    live = await createApp({ defaults: false, plugins: [ProfilePlugin, StandardPhysicsPlugin] });
    const { world } = live;
    const eid = world.create();
    world.add(eid, Body);
    const bodies = world.storage(Body);
    bodies.shape.set(eid, ShapeKind.Box);
    bodies.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
    bodies.position.set(eid, 0, 5, 0, 0);
    bodies.rotation.set(eid, 0, 0, 0, 1);
    bodies.mass.set(eid, 1);
    for (let i = 0; i < 10; i++) world.step(1 / 60);
    const cpu = world.resource(Profile).cpu;
    expect(cpu.get("StandardPhysics/step/collide")).toBeGreaterThan(0);
    expect(cpu.get("StandardPhysics/step/solve")).toBeGreaterThan(0);
    let outer = 0;
    for (const [name, ms] of cpu) if (!name.startsWith("StandardPhysics/step/")) outer += ms;
    expect(cpu.get("StandardPhysics/step")).toBeGreaterThan(0);
    expect(cpuTotal(cpu)).toBe(outer);
});
