import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, type WorldSnapshot } from "@dylanebert/shallot";
import { Body, BodyType, DistanceJoint } from "@dylanebert/shallot/physics";
import {
    hashPhysics,
    physicsWorld,
    StandardPhysicsPlugin,
    setTargetTransform,
} from "@dylanebert/shallot/standard/physics";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
await setupGlobals();

for (const mutation of ["spawn", "despawn", "joint"] as const) {
    test(`world restore reconciles physics bindings after ${mutation}`, async () => {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        const world = app.world;
        try {
            const anchor = world.create();
            world.add(anchor, Body, { type: BodyType.Static, position: [0, 3, 0, 0] });
            const bob = world.create();
            world.add(bob, Body, { position: [0, 1, 0, 0] });
            world.tick();
            const saved = world.snapshot();
            const expected = hashPhysics(world);
            if (mutation === "despawn") world.destroy(bob);
            else {
                const eid = world.create();
                if (mutation === "spawn") world.add(eid, Body, { position: [3, 2, 0, 0] });
                else
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
            expect(physicsWorld(world)!.getCounters().bodyCount).toBe(2);
            expect(physicsWorld(world)!.getCounters().jointCount).toBe(0);
            expect(physicsWorld(world)!.getBody(bob)?.isValid()).toBe(true);
            world.tick();
            expect(physicsWorld(world)!.getCounters().bodyCount).toBe(2);
        } finally {
            app.dispose();
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
        world.add(bob, Body, { position: [0.2, 1, 0, 0] });
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
        app.dispose();
    }
});

test("world recovery refuses a solver-only image before changing simulation", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const eid = world.create();
        world.add(eid, Body);
        world.tick();
        const solverOnly = physicsWorld(world)!.snapshot();
        world.tick();
        const hash = hashPhysics(world);
        expect(() => world.restore(solverOnly as unknown as WorldSnapshot)).toThrow(
            "invalid snapshot",
        );
        expect(hashPhysics(world)).toBe(hash);
    } finally {
        app.dispose();
    }
});
