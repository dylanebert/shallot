import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp, Transform } from "../../engine";
import { Body, PhysicsPlugin } from "./index";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

for (const [first, second, firstProducer, secondProducer] of [
    [Transform, Body, "transform", "body"],
    [Body, Transform, "body", "transform"],
] as const) {
    test(`a second GlobalTransform producer ${secondProducer} refuses after ${firstProducer} without changing membership`, async () => {
        const app = await createApp({ defaults: false, plugins: [PhysicsPlugin] });
        try {
            const world = app.world;
            const eid = world.create();
            world.add(eid, first);
            expect(() => world.add(eid, second)).toThrow(
                new RegExp(`cannot attach "${secondProducer}".*excluded by "${firstProducer}"`),
            );
            expect(world.has(eid, first)).toBe(true);
            expect(world.has(eid, second)).toBe(false);
        } finally {
            app.dispose();
        }
    });
}
