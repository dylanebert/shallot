import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, Shape, Transform, WheelJoint } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { StandardPhysicsPlugin, StepPhysicsSystem, SyncPhysicsConstraintsSystem } from "./index";
import { WHJ_LOWER_SUSPENSION_IMPULSE, WHJ_UPPER_SUSPENSION_IMPULSE } from "./kernel/joint-layout";
import { readJointFloat, writeJointFloat } from "./kernel/jointcolumns";
import { physicsState } from "./runtime";
import { jointIds } from "./solver/joint.fixture";

setDefaultTimeout(TIMEOUT.node);

test("changing either authored wheel suspension limit clears both accumulated limit impulses before solving", async () => {
    let joint = 0;
    let inspect = false;
    const app = await createApp({
        defaults: false,
        plugins: [
            StandardPhysicsPlugin,
            {
                name: "InspectWheelLimits",
                systems: [
                    {
                        name: "inspect-wheel-limits",
                        group: "fixed",
                        after: [SyncPhysicsConstraintsSystem],
                        before: [StepPhysicsSystem],
                        update(world) {
                            if (!inspect) return;
                            const state = physicsState(world)!;
                            const id = jointIds(state)[0]!;
                            expect(readJointFloat(state, id, WHJ_LOWER_SUSPENSION_IMPULSE)).toBe(0);
                            expect(readJointFloat(state, id, WHJ_UPPER_SUSPENSION_IMPULSE)).toBe(0);
                        },
                    },
                ],
                recovery: "stateless",
            },
        ],
        setup(world) {
            const anchor = world.create();
            world.add(anchor, Body);
            world.add(anchor, Transform);
            const wheel = world.create();
            world.add(wheel, Body, { type: BodyType.Dynamic });
            world.add(wheel, Transform);
            world.add(wheel, Shape);
            joint = world.create();
            world.add(joint, WheelJoint, {
                a: anchor,
                b: wheel,
                enableSuspensionLimit: 1,
                lowerSuspensionLimit: -0.2,
                upperSuspensionLimit: 0.2,
            });
        },
    });
    try {
        const world = app.world;
        world.tick();
        inspect = true;
        for (const field of ["lowerSuspensionLimit", "upperSuspensionLimit"] as const) {
            const state = physicsState(world)!;
            const id = jointIds(state)[0]!;
            writeJointFloat(state, id, WHJ_LOWER_SUSPENSION_IMPULSE, 2);
            writeJointFloat(state, id, WHJ_UPPER_SUSPENSION_IMPULSE, 3);
            world
                .storage(WheelJoint)
                [field].set(joint, field === "lowerSuspensionLimit" ? -0.5 : 0.5);
            world.tick();
        }
    } finally {
        app.dispose();
    }
});
