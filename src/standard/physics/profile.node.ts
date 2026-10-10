import { afterEach, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { Profile, ProfilePlugin } from "@dylanebert/shallot/extras";
import { Body, BodyType, Shape } from "@dylanebert/shallot/physics";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";
import { cpuTotal } from "../../extras/profile/cpu";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

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
    world.add(eid, Shape);
    const bodies = world.storage(Body);
    bodies.position.set(eid, 0, 5, 0, 0);
    bodies.rotation.set(eid, 0, 0, 0, 1);
    bodies.type.set(eid, BodyType.Dynamic);
    // A phase with no work can read 0 ms on a real clock and go unrecorded; each read advances 1 ms here.
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        for (let i = 0; i < 10; i++) world.step(1 / 60);
    } finally {
        now.mockRestore();
    }
    const cpu = world.resource(Profile).cpu;
    expect(cpu.get("StandardPhysics/step/collide")).toBeGreaterThan(0);
    expect(cpu.get("StandardPhysics/step/solve")).toBeGreaterThan(0);
    let outer = 0;
    for (const [name, ms] of cpu) if (!name.startsWith("StandardPhysics/step/")) outer += ms;
    expect(cpu.get("StandardPhysics/step")).toBeGreaterThan(0);
    expect(cpuTotal(cpu)).toBe(outer);
});
