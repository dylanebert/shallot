import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../../src/standard/physics/api";
import { kernel } from "../../src/standard/physics/kernel/kernel";
import { assertPublicOracleKernel } from "./oracle-kernel";

test("split acquires component-sized scratch and ids only after the no-split return", async () => {
    await assertPublicOracleKernel();
    for (const redundant of [false, true]) {
        const world = new PhysicsWorld();
        try {
            const bodies = Array.from({ length: 3 }, () => world.createBody({ type: BodyType.Dynamic }));
            world.createFilterJoint(bodies[0], bodies[1]);
            const removed = world.createFilterJoint(bodies[1], bodies[2]);
            if (redundant) world.createFilterJoint(bodies[0], bodies[2]);
            removed.destroy();
            const k = kernel(world.state.ecsState) as unknown as {
                box3dSplitScratch(world: number, body: number, lane: number): number;
            };
            for (let lane = 0; lane < 4; ++lane)
                expect(k.box3dSplitScratch(world.state.worldId, bodies[0].id.index1 - 1, lane)).toBe(redundant ? 0 : 2);
        } finally {
            world.destroy();
        }
    }
});

test("sleeping a seventeen-body island reserves its destination body array once", async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld();
    try {
        const bodies = Array.from({ length: 17 }, (_, i) => world.createBody({
            type: BodyType.Dynamic, position: { x: i, y: 0, z: 0 },
        }));
        for (let i = 1; i < bodies.length; ++i)
            world.createDistanceJoint(bodies[i - 1], bodies[i], { length: 1 });
        bodies[0].setAwake(false);
        const k = kernel(world.state.ecsState) as unknown as {
            box3dSleepingBodyCapacity(world: number, body: number): number;
            box3dSleepingBodyAllocations(world: number, body: number): number;
        };
        const body = bodies[0].id.index1 - 1;
        expect(k.box3dSleepingBodyCapacity(world.state.worldId, body)).toBe(17);
        expect(k.box3dSleepingBodyAllocations(world.state.worldId, body)).toBe(1);
    } finally {
        world.destroy();
    }
});
