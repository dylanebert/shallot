import { expect, setDefaultTimeout, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../../scripts/test-tiers";
import { Body, BodyType } from "../../core/physics";
import { GlobalTransform } from "../../core/transform";
import { createApp, Time } from "../../engine";
import { physicsWorld, StandardPhysicsPlugin } from ".";
import { hash } from "./api";

setDefaultTimeout(CEILING.node);
await setupGlobals();

test("exact ticks then draw match frame ticks in physics hash and published fields", async () => {
    const a = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const b = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        for (const app of [a, b]) {
            const eid = app.world.create();
            app.world.add(eid, Body, { type: BodyType.Dynamic, position: [0, 10, 0, 0] });
        }
        for (let i = 0; i < 120; i++) {
            a.world.tick();
            b.world.step(Time.FIXED_DT);
            expect(hash(physicsWorld(a.world)!)).toBe(hash(physicsWorld(b.world)!));
        }
        a.world.step(0);
        expect(Array.from(a.world.storage(GlobalTransform).translation.column)).toEqual(
            Array.from(b.world.storage(GlobalTransform).translation.column),
        );
        expect(a.world.gpu.frame).toBe(1);
        expect(b.world.gpu.frame).toBe(120);
    } finally {
        a.dispose();
        b.dispose();
    }
});
