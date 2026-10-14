import { createApp } from "@dylanebert/shallot";
import { BodyType } from "../common/types";
import { init, shutdown, threads } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { createDistanceJoint, defaultDistanceJointDef } from "./distanceJoint";
import { createFilterJoint, defaultJointDef } from "./joint";
import { createMotorJoint, defaultMotorJointDef } from "./motorJoint";
import { createParallelJoint, defaultParallelJointDef } from "./parallelJoint";
import { createPrismaticJoint, defaultPrismaticJointDef } from "./prismaticJoint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "./revoluteJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "./sphericalJoint";
import { createWeldJoint, defaultWeldJointDef } from "./weldJoint";
import { createWheelJoint, defaultWheelJointDef } from "./wheelJoint";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create(input: string) {
    const count = Number(input);
    await init(undefined, { threads: count });
    if (threads(undefined) !== (count || 1))
        throw new Error("joint allocation requires the requested schedule");
    const state = createTestWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    const hull = makeBoxHull(0.25, 0.25, 0.25);
    const kinds = [
        "Parallel",
        "Distance",
        "Filter",
        "Motor",
        "Prismatic",
        "Revolute",
        "Spherical",
        "Weld",
        "Wheel",
    ] as const;
    for (let i = 0; i < kinds.length; i++) {
        const anchor = createTestBody(state, { position: { x: i * 4, y: 0, z: 0 } });
        const body = createTestBody(state, {
            type: BodyType.Dynamic,
            position: { x: i * 4, y: 1, z: 0 },
        });
        createHullShape(state, body, testShapeDef(), hull);
        const base = { ...defaultJointDef(), bodyIdA: anchor, bodyIdB: body };
        switch (kinds[i]) {
            case "Parallel":
                createParallelJoint(state, defaultParallelJointDef(base));
                break;
            case "Distance":
                createDistanceJoint(state, defaultDistanceJointDef({ ...base }));
                break;
            case "Filter":
                createFilterJoint(state, base);
                break;
            case "Motor":
                createMotorJoint(state, defaultMotorJointDef(base));
                break;
            case "Prismatic":
                createPrismaticJoint(state, defaultPrismaticJointDef(base));
                break;
            case "Revolute":
                createRevoluteJoint(state, defaultRevoluteJointDef(base));
                break;
            case "Spherical":
                createSphericalJoint(state, defaultSphericalJointDef(base));
                break;
            case "Weld":
                createWeldJoint(state, defaultWeldJointDef(base));
                break;
            case "Wheel":
                createWheelJoint(state, defaultWheelJointDef(base));
                break;
        }
    }
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "joint-allocation",
                systems: [
                    {
                        name: "joint-step",
                        group: "fixed",
                        update: () => stepTestWorld(state, 1 / 60, 4),
                    },
                ],
            },
        ],
    });
    return {
        step: () => app.world.step(1 / 60),
        dispose: async () => {
            app.dispose();
            destroyTestWorld(state);
            await shutdown(undefined);
        },
    };
}
