import { expect, test } from "bun:test";
import { BodyType } from "../common/types";
import * as J from "../kernel/joint-layout";
import { kernel } from "../kernel/kernel";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";
import type { WorldState } from "../world/world";
import { createDistanceJoint, defaultDistanceJointDef } from "./distanceJoint";
import { createFilterJoint, defaultJointDef, destroyJointInternal } from "./joint";
import { createMotorJoint, defaultMotorJointDef } from "./motorJoint";
import { createParallelJoint, defaultParallelJointDef } from "./parallelJoint";
import { createPrismaticJoint, defaultPrismaticJointDef } from "./prismaticJoint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "./revoluteJoint";
import { createSphericalJoint, defaultSphericalJointDef } from "./sphericalJoint";
import { createWeldJoint, defaultWeldJointDef } from "./weldJoint";
import { createWheelJoint, defaultWheelJointDef } from "./wheelJoint";

const range = (start: number, count: number) => Array.from({ length: count }, (_, i) => start + i);
const flagWord = (field: number) => (field & ~(1 << 16)) >>> 2;
const kinds: [string, (world: WorldState, a: number, b: number) => number, number[]][] = [
    [
        "distance",
        (w, a, b) =>
            createDistanceJoint(
                w,
                defaultDistanceJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        [...range(J.DJ_LENGTH, 9), flagWord(J.DJ_ENABLE)],
    ],
    [
        "filter",
        (w, a, b) => createFilterJoint(w, { ...defaultJointDef(), bodyIdA: a, bodyIdB: b }).joint,
        [],
    ],
    [
        "motor",
        (w, a, b) =>
            createMotorJoint(
                w,
                defaultMotorJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        range(J.MJ_LINEAR_VELOCITY, 14),
    ],
    [
        "parallel",
        (w, a, b) =>
            createParallelJoint(
                w,
                defaultParallelJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        range(J.PLJ_HERTZ, 3),
    ],
    [
        "prismatic",
        (w, a, b) =>
            createPrismaticJoint(
                w,
                defaultPrismaticJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        [...range(J.PJ_HERTZ, 7), flagWord(J.PJ_ENABLE)],
    ],
    [
        "revolute",
        (w, a, b) =>
            createRevoluteJoint(
                w,
                defaultRevoluteJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        [...range(J.RJ_HERTZ, 7), flagWord(J.RJ_ENABLE)],
    ],
    [
        "spherical",
        (w, a, b) =>
            createSphericalJoint(
                w,
                defaultSphericalJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        [...range(J.SJ_HERTZ, 13), flagWord(J.SJ_ENABLE)],
    ],
    [
        "weld",
        (w, a, b) =>
            createWeldJoint(
                w,
                defaultWeldJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        range(J.WJ_LINEAR_HERTZ, 4),
    ],
    [
        "wheel",
        (w, a, b) =>
            createWheelJoint(
                w,
                defaultWheelJointDef({ ...defaultJointDef(), bodyIdA: a, bodyIdB: b }),
            ).joint,
        [
            ...range(J.WHJ_MAX_SPIN_TORQUE, 2),
            ...range(J.WHJ_LOWER_SUSPENSION_LIMIT, 4),
            ...range(J.WHJ_LOWER_STEERING_LIMIT, 6),
            flagWord(J.WHJ_ENABLE),
            flagWord(J.WHJ_ENABLE) + 1,
        ],
    ],
];

for (const [name, create, fields] of kinds) {
    test(`${name} creation rebuilds its definition and zeroes impulses and solver scratch in a reused slot`, () => {
        const world = createTestWorld();
        try {
            const a = createTestBody(world);
            const b = createTestBody(world, { type: BodyType.Dynamic });
            const k = kernel(world.ecsState);
            const words = (joint: number) => {
                k.bodySetActiveWorld(world.worldId);
                return new Uint32Array(k.memory.buffer, k.jointSimPtr(joint), J.JOINT_STRIDE);
            };
            const scratch = range(J.J_PAYLOAD, J.JOINT_STRIDE - J.J_PAYLOAD).filter(
                (i) => !fields.includes(i),
            );
            const definition = (joint: number) => fields.map((i) => words(joint)[i]);
            const zeroes = (joint: number) => scratch.map((i) => words(joint)[i]);
            const first = create(world, a, b);
            const expectedDefinition = definition(first);
            expect(zeroes(first)).toEqual(Array(scratch.length).fill(0));
            words(first).fill(0x41200000, J.J_PAYLOAD, J.JOINT_STRIDE);
            const index1 = first + 1;
            destroyJointInternal(world, first, true);
            const second = create(world, a, b);
            expect(second + 1).toBe(index1);
            expect(definition(second)).toEqual(expectedDefinition);
            expect(zeroes(second)).toEqual(Array(scratch.length).fill(0));
        } finally {
            destroyTestWorld(world);
        }
    });
}
