import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { BodyType } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    islandArrayCount,
    islandArrayGet,
    islandField,
    islandKernel,
} from "../kernel/islandcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import { defaultJointDef, destroyJointInternal } from "../solver/joint";
import { jointIds } from "../solver/joint.fixture";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";

function distanceJoint(world: ReturnType<typeof createTestWorld>, a: number, b: number) {
    const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
    return createDistanceJoint(world, { ...defaultDistanceJointDef(base), length: 1 }).joint;
}

test("island fix borrowing follows vector relocation and memory growth", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = Array.from({ length: 32 }, () =>
            createTestBody(world, { type: BodyType.Dynamic }),
        );
        for (let i = 1; i < bodies.length; ++i) {
            if (i === 16) islandKernel(world).memory.grow(1);
            distanceJoint(world, bodies[0]!, bodies[i]!);
            for (let j = 0; j <= i; ++j) {
                const body = bodies[j]!;
                expect(bodyField(world, body, BodyField.islandId)).toBe(
                    bodyField(world, bodies[0]!, BodyField.islandId),
                );
                expect(
                    islandArrayGet(
                        world,
                        bodyField(world, body, BodyField.islandId),
                        0,
                        bodyField(world, body, BodyField.islandIndex),
                    ),
                ).toBe(bodyField(world, body, BodyField.id));
            }
            const joint = jointIds(world)[i - 1]!;
            expect(
                islandArrayGet(
                    world,
                    jointField(world, joint, JointField.islandId),
                    2,
                    jointField(world, joint, JointField.islandIndex),
                ),
            ).toBe(jointField(world, joint, JointField.jointId));
        }
    } finally {
        destroyTestWorld(world);
    }
});

test("island split preserves link membership, fixes body and joint slots, and restores the id pool", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = Array.from({ length: 4 }, (_, x) =>
            createTestBody(world, { type: BodyType.Dynamic, position: { x, y: 0, z: 0 } }),
        );
        const joints = bodies.slice(1).map((b, i) => distanceJoint(world, bodies[i]!, b));
        const base = bodyField(world, bodies[0]!, BodyField.islandId);
        expect(islandArrayCount(world, base, 0)).toBe(4);
        destroyJointInternal(world, joints[1]!, true);
        const saved = snapshot(world);
        const verify = () => {
            islandKernel(world).islandSplit(base);
            expect(bodyField(world, bodies[0]!, BodyField.islandId)).toBe(
                bodyField(world, bodies[1]!, BodyField.islandId),
            );
            expect(bodyField(world, bodies[2]!, BodyField.islandId)).toBe(
                bodyField(world, bodies[3]!, BodyField.islandId),
            );
            expect(bodyField(world, bodies[0]!, BodyField.islandId)).not.toBe(
                bodyField(world, bodies[2]!, BodyField.islandId),
            );
            for (const body of bodies) {
                expect(
                    islandArrayGet(
                        world,
                        bodyField(world, body, BodyField.islandId),
                        0,
                        bodyField(world, body, BodyField.islandIndex),
                    ),
                ).toBe(bodyField(world, body, BodyField.id));
                expect(islandField(world, bodyField(world, body, BodyField.islandId), 3)).toBe(0);
            }
            for (const joint of jointIds(world)) {
                if (jointField(world, joint, JointField.islandId) === -1) continue;
                expect(
                    islandArrayGet(
                        world,
                        jointField(world, joint, JointField.islandId),
                        2,
                        jointField(world, joint, JointField.islandIndex),
                    ),
                ).toBe(jointField(world, joint, JointField.jointId));
                expect(
                    islandArrayCount(world, jointField(world, joint, JointField.islandId), 2),
                ).toBe(1);
            }
            return bodies.map((body) => bodyField(world, body, BodyField.islandId));
        };
        const ids = verify();
        restore(world, saved);
        expect(verify()).toEqual(ids);
    } finally {
        destroyTestWorld(world);
    }
});
