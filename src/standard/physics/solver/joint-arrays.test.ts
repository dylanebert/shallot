import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { NULL_INDEX } from "../common/array";
import { GRAPH_COLOR_COUNT, OVERFLOW_INDEX, SetType } from "../common/constants";
import { BodyType } from "../common/types";
import { DJ_IMPULSE, DJ_MOTOR_IMPULSE, J_JOINT_ID, JOINT_STRIDE } from "../kernel/columns";
import { jointArrayCount, jointArrayKey, jointAt, writeJointFloat } from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import { solverSetCount } from "../kernel/solversetcolumns";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";
import { bodySetAwake, bodySetType } from "../world/body";
import type { WorldState } from "../world/world";
import { createDistanceJoint, defaultDistanceJointDef } from "./distanceJoint";
import { defaultJointDef, destroyJointInternal } from "./joint";
import { jointIds } from "./joint.fixture";

function distanceJoint(
    world: WorldState,
    a: number,
    b: number,
    config: Record<string, unknown> = {},
) {
    const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
    const def = { ...defaultDistanceJointDef(base), ...config };
    return createDistanceJoint(world, def).joint;
}

function records(world: WorldState): Map<number, number[]> {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    const result = new Map<number, number[]>();
    for (const joint of jointIds(world)) {
        if (jointField(world, joint, JointField.setIndex) === NULL_INDEX) continue;
        const key = jointArrayKey(world, joint);
        expect(jointField(world, joint, JointField.localIndex)).toBeLessThan(
            jointArrayCount(world, key),
        );
        expect(jointAt(world, key, jointField(world, joint, JointField.localIndex))).toBe(joint);
        const words = new Uint32Array(
            k.memory.buffer,
            k.jointArrayPtr(key) +
                jointField(world, joint, JointField.localIndex) * JOINT_STRIDE * 4,
            JOINT_STRIDE,
        );
        expect(words[J_JOINT_ID]).toBe(jointField(world, joint, JointField.jointId));
        result.set(jointField(world, joint, JointField.jointId), Array.from(words));
    }
    for (let key = 0; key < GRAPH_COLOR_COUNT + solverSetCount(world); ++key) {
        for (let i = 0; i < jointArrayCount(world, key); ++i) {
            const joint = jointAt(world, key, i);
            expect(jointArrayKey(world, joint)).toBe(key);
            expect(jointField(world, joint, JointField.localIndex)).toBe(i);
        }
    }
    return result;
}

test("sleeping joint arrays wake in their original order, including overflow, and restoring one world leaves a sibling's arrays untouched", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const hubs: number[] = [];
        for (const subject of [world, sibling]) {
            const hub = createTestBody(subject, { type: BodyType.Dynamic });
            hubs.push(hub);
            for (let i = 0; i < 28; ++i) {
                const body = createTestBody(subject, {
                    type: BodyType.Dynamic,
                    position: { x: i + 1, y: 0, z: 0 },
                });
                const joint = distanceJoint(subject, hub, body, { length: i + 1, hertz: i + 2 });
                writeJointFloat(subject, joint, DJ_IMPULSE, i + 0.125);
            }
        }
        expect(jointArrayCount(world, OVERFLOW_INDEX)).toBeGreaterThan(0);
        const before = records(world);
        const siblingBefore = records(sibling);
        const hub = hubs[0]!;
        bodySetAwake(world, hub, false);
        expect(records(world)).toEqual(before);
        const setIndex = jointField(world, 0, JointField.setIndex);
        const key = GRAPH_COLOR_COUNT + setIndex;
        const order = Array.from({ length: jointArrayCount(world, key) }, (_, i) =>
            jointAt(world, key, i),
        );
        const saved = snapshot(world);
        bodySetAwake(world, hub, true);
        expect(records(world)).toEqual(before);
        const actual: number[] = [];
        for (let color = 0; color < GRAPH_COLOR_COUNT; ++color)
            for (let i = 0; i < jointArrayCount(world, color); ++i)
                actual.push(jointAt(world, color, i));
        expect(actual).toEqual(order);
        restore(world, saved);
        expect(records(world)).toEqual(before);
        expect(records(sibling)).toEqual(siblingBefore);
        bodySetAwake(world, hub, true);
        expect(records(world)).toEqual(before);
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});

test("joint-array moves preserve every record word and fix local indices through sleep, wake, neighbor removal, recolor and snapshot restore", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const pairs = Array.from({ length: 5 }, (_, i) => {
            const a = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i * 10, y: 0, z: 0 },
            });
            const b = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i * 10 + 1, y: 0, z: 0 },
            });
            const joint = distanceJoint(world, a, b, {
                length: i + 1,
                hertz: i + 2,
                enableMotor: true,
                motorSpeed: i + 3,
            });
            writeJointFloat(world, joint, DJ_IMPULSE, i + 0.25);
            writeJointFloat(world, joint, DJ_MOTOR_IMPULSE, -i - 0.5);
            return { a, b, joint };
        });
        const before = records(world);
        const saved = snapshot(world);
        const assertUnchanged = () => {
            for (const [id, words] of records(world)) expect(words).toEqual(before.get(id)!);
        };
        bodySetAwake(world, pairs[1]!.a, false);
        expect(jointField(world, pairs[1]!.joint, JointField.setIndex)).toBeGreaterThanOrEqual(
            SetType.FirstSleeping,
        );
        assertUnchanged();
        bodySetAwake(world, pairs[1]!.b, true);
        assertUnchanged();
        destroyJointInternal(world, pairs[0]!.joint, false);
        assertUnchanged();
        bodySetType(world, pairs[2]!.a, BodyType.Static);
        assertUnchanged();
        bodySetType(world, pairs[2]!.a, BodyType.Dynamic);
        assertUnchanged();
        restore(world, saved);
        expect(records(world)).toEqual(before);
        bodySetAwake(world, pairs[3]!.a, false);
        assertUnchanged();
        bodySetAwake(world, pairs[3]!.b, true);
        assertUnchanged();
    } finally {
        destroyTestWorld(world);
    }
});
