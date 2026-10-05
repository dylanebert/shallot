import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, Time } from "@dylanebert/shallot";
import { Body, Joint, Spring } from "@dylanebert/shallot/physics";
import { physicsWorld, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

for (const constraint of [Joint, Spring]) {
    test(`${constraint === Joint ? "a joint" : "a spring"} constrains nothing when its endpoint is destroyed and recycled in one update`, async () => {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            const world = app.world;
            const anchor = world.create();
            world.add(anchor, Body, { mass: 0, position: [0, 3, 0, 0] });
            const bob = world.create();
            world.add(bob, Body, { position: [0, 1, 0, 0] });
            const link = world.create();
            world.add(link, constraint, { a: anchor, b: bob });
            world.step(Time.FIXED_DT);
            const solver = physicsWorld(world)!;
            expect(solver.getCounters().jointCount).toBe(1);
            world.destroy(bob);
            const replacement = world.create();
            expect(replacement).toBe(bob);
            world.add(replacement, Body, { position: [0, 1, 0, 0] });
            world.step(Time.FIXED_DT);
            expect(solver.getBody(replacement)).not.toBeNull();
            expect(solver.getCounters().jointCount).toBe(0);
        } finally {
            app.dispose();
        }
    });
}
