import { expect, test } from "bun:test";
import { BodyType } from "../../src/standard/physics/common/types";
import { kernel } from "../../src/standard/physics/kernel/kernel";
import { createDistanceJoint, defaultDistanceJointDef } from "../../src/standard/physics/solver/distanceJoint";
import { createFilterJoint, defaultJointDef, destroyJointInternal } from "../../src/standard/physics/solver/joint";
import { createTestBody, createTestWorld, destroyTestWorld } from "../../src/standard/physics/testing";
import { bodySetAwake } from "../../src/standard/physics/world/body";
import { assertPublicOracleKernel } from "./oracle-kernel";

test("split acquires component-sized scratch and ids only after the no-split return", async () => {
    await assertPublicOracleKernel();
    for (const redundant of [false, true]) {
        const world = createTestWorld();
        try {
            const bodies = Array.from({ length: 3 }, () =>
                createTestBody(world, { type: BodyType.Dynamic }),
            );
            createFilterJoint(world, { ...defaultJointDef(), bodyIdA: bodies[0]!, bodyIdB: bodies[1]! });
            const removed = createFilterJoint(world, {
                ...defaultJointDef(),
                bodyIdA: bodies[1]!,
                bodyIdB: bodies[2]!,
            }).joint;
            if (redundant)
                createFilterJoint(world, {
                    ...defaultJointDef(),
                    bodyIdA: bodies[0]!,
                    bodyIdB: bodies[2]!,
                });
            destroyJointInternal(world, removed, true);
            const k = kernel(world.ecsState) as unknown as {
                box3dSplitScratch(world: number, body: number, lane: number): number;
            };
            for (let lane = 0; lane < 4; ++lane)
                expect(k.box3dSplitScratch(world.worldId, bodies[0]!, lane)).toBe(redundant ? 0 : 2);
        } finally {
            destroyTestWorld(world);
        }
    }
});

test("sleeping a seventeen-body island reserves its destination body array once", async () => {
    await assertPublicOracleKernel();
    const world = createTestWorld();
    try {
        const bodies = Array.from({ length: 17 }, (_, i) =>
            createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i, y: 0, z: 0 },
            }),
        );
        for (let i = 1; i < bodies.length; ++i) {
            const def = defaultDistanceJointDef({
                ...defaultJointDef(),
                bodyIdA: bodies[i - 1]!,
                bodyIdB: bodies[i]!,
            });
            def.length = 1;
            createDistanceJoint(world, def);
        }
        bodySetAwake(world, bodies[0]!, false);
        const k = kernel(world.ecsState) as unknown as {
            box3dSleepingBodyCapacity(world: number, body: number): number;
            box3dSleepingBodyAllocations(world: number, body: number): number;
        };
        expect(k.box3dSleepingBodyCapacity(world.worldId, bodies[0]!)).toBe(17);
        expect(k.box3dSleepingBodyAllocations(world.worldId, bodies[0]!)).toBe(1);
    } finally {
        destroyTestWorld(world);
    }
});
