import { expect, setDefaultTimeout, test } from "bun:test";
import { Transform, type WorldSnapshot } from "@dylanebert/shallot";
import { Body, BodyType, DistanceJoint } from "@dylanebert/shallot/physics";
import {
    hashPhysics,
    StandardPhysicsPlugin,
    setTargetTransform,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createTestGpuApp as createApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { snapshot } from "./api/snapshot";
import { physicsBodyIndex, physicsState } from "./runtime";
import { worldCounters } from "./world/world";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();

for (const mutation of ["spawn", "despawn", "joint"] as const) {
    test(`world restore reconciles physics bindings after ${mutation}`, async () => {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        const world = app.world;
        try {
            const anchor = world.create();
            world.add(anchor, Transform, { translation: [0, 3, 0] });
            world.add(anchor, Body, { type: BodyType.Static });
            const bob = world.create();
            world.add(bob, Transform, { translation: [0, 1, 0] });
            world.add(bob, Body);
            world.tick();
            const saved = world.snapshot();
            const expected = hashPhysics(world);
            if (mutation === "despawn") world.destroy(bob);
            else {
                const eid = world.create();
                if (mutation === "spawn") {
                    world.add(eid, Transform, { translation: [3, 2, 0] });
                    world.add(eid, Body);
                } else
                    world.add(eid, DistanceJoint, {
                        a: anchor,
                        b: bob,
                        enableSpring: 1,
                        length: 2,
                        hertz: 2,
                    });
            }
            world.tick();
            world.restore(saved);
            expect(hashPhysics(world)).toBe(expected);
            expect(worldCounters(physicsState(world)!).bodyCount).toBe(2);
            expect(worldCounters(physicsState(world)!).jointCount).toBe(0);
            expect(physicsBodyIndex(world, bob)).toBeGreaterThanOrEqual(0);
            world.tick();
            expect(worldCounters(physicsState(world)!).bodyCount).toBe(2);
        } finally {
            await disposeGpuApps([app]);
        }
    });
}

test("kinematic targets replay the same physics hashes after world recovery", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    try {
        const platform = world.create();
        world.add(platform, Body, { type: BodyType.Kinematic });
        const bob = world.create();
        world.add(bob, Transform, { translation: [0.2, 1, 0] });
        world.add(bob, Body);
        const place = (tick: number) =>
            setTargetTransform(world, platform, [tick * 0.05, 0, 0], [0, 0, 0, 1]);
        for (let tick = 0; tick < 5; tick++) {
            place(tick);
            world.tick();
        }
        const saved = world.snapshot();
        const advance = () => {
            const hashes: bigint[] = [];
            for (let tick = 5; tick < 9; tick++) {
                place(tick);
                world.tick();
                hashes.push(hashPhysics(world));
            }
            return hashes;
        };
        const expected = advance();
        world.restore(saved);
        expect(advance()).toEqual(expected);
    } finally {
        await disposeGpuApps([app]);
    }
});

test("world recovery refuses a solver-only image before changing simulation", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body);
        world.tick();
        const solverOnly = snapshot(physicsState(world)!);
        world.tick();
        const hash = hashPhysics(world);
        expect(() => world.restore(solverOnly as unknown as WorldSnapshot)).toThrow(
            "invalid snapshot",
        );
        expect(hashPhysics(world)).toBe(hash);
    } finally {
        await disposeGpuApps([app]);
    }
});
