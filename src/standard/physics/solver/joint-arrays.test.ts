import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api/index";
import { NULL_INDEX } from "../common/array";
import { GRAPH_COLOR_COUNT, OVERFLOW_INDEX, SetType } from "../common/constants";
import { DJ_IMPULSE, DJ_MOTOR_IMPULSE, J_JOINT_ID, JOINT_STRIDE } from "../kernel/columns";
import { jointArrayCount, jointArrayKey, jointAt, writeJointFloat } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import { solverSetCount } from "../kernel/solversetcolumns";
import type { WorldState } from "../world/world";

function records(world: WorldState): Map<number, number[]> {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    const result = new Map<number, number[]>();
    for (const joint of world.joints) {
        if (joint.setIndex === NULL_INDEX) continue;
        const key = jointArrayKey(joint);
        expect(joint.localIndex).toBeLessThan(jointArrayCount(world, key));
        expect(jointAt(world, key, joint.localIndex)).toBe(joint);
        const words = new Uint32Array(
            k.memory.buffer,
            k.jointArrayPtr(key) + joint.localIndex * JOINT_STRIDE * 4,
            JOINT_STRIDE,
        );
        expect(words[J_JOINT_ID]).toBe(joint.jointId);
        result.set(joint.jointId, Array.from(words));
    }
    for (let key = 0; key < GRAPH_COLOR_COUNT + solverSetCount(world); ++key) {
        for (let i = 0; i < jointArrayCount(world, key); ++i) {
            const joint = jointAt(world, key, i);
            expect(jointArrayKey(joint)).toBe(key);
            expect(joint.localIndex).toBe(i);
        }
    }
    return result;
}

test("sleeping joint arrays wake in their original order, including overflow, and restoring one world leaves a sibling's arrays untouched", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const hubs = [];
        for (const subject of [world, sibling]) {
            const hub = subject.createBody({ type: BodyType.Dynamic });
            hubs.push(hub);
            for (let i = 0; i < 28; ++i) {
                const body = subject.createBody({
                    type: BodyType.Dynamic,
                    position: { x: i + 1, y: 0, z: 0 },
                });
                const joint = subject.createDistanceJoint(hub, body, {
                    length: i + 1,
                    hertz: i + 2,
                });
                writeJointFloat(
                    subject.state,
                    subject.state.joints[joint.id.index1 - 1],
                    DJ_IMPULSE,
                    i + 0.125,
                );
            }
        }
        expect(jointArrayCount(world.state, OVERFLOW_INDEX)).toBeGreaterThan(0);
        const before = records(world.state);
        const siblingBefore = records(sibling.state);
        const hub = hubs[0];
        hub.setAwake(false);
        expect(records(world.state)).toEqual(before);
        const setIndex = world.state.joints[0].setIndex;
        const key = GRAPH_COLOR_COUNT + setIndex;
        const order = Array.from(
            { length: jointArrayCount(world.state, key) },
            (_, i) => jointAt(world.state, key, i).jointId,
        );
        const snapshot = world.snapshot();
        hub.setAwake(true);
        expect(records(world.state)).toEqual(before);
        const actual: number[] = [];
        for (let color = 0; color < GRAPH_COLOR_COUNT; ++color) {
            for (let i = 0; i < jointArrayCount(world.state, color); ++i)
                actual.push(jointAt(world.state, color, i).jointId);
        }
        expect(actual).toEqual(order);
        world.restore(snapshot);
        expect(records(world.state)).toEqual(before);
        expect(records(sibling.state)).toEqual(siblingBefore);
        hub.setAwake(true);
        expect(records(world.state)).toEqual(before);
    } finally {
        world.destroy();
        sibling.destroy();
    }
});

test("joint-array moves preserve every record word and fix local indices through sleep, wake, neighbor removal, recolor and snapshot restore", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const pairs = Array.from({ length: 5 }, (_, i) => {
            const a = world.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 10, y: 0, z: 0 },
            });
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 10 + 1, y: 0, z: 0 },
            });
            const joint = world.createDistanceJoint(a, b, {
                length: i + 1,
                hertz: i + 2,
                enableMotor: true,
                motorSpeed: i + 3,
            });
            const record = world.state.joints[joint.id.index1 - 1];
            writeJointFloat(world.state, record, DJ_IMPULSE, i + 0.25);
            writeJointFloat(world.state, record, DJ_MOTOR_IMPULSE, -i - 0.5);
            return { a, b, joint };
        });
        const before = records(world.state);
        const snapshot = world.snapshot();
        const assertUnchanged = () => {
            for (const [id, words] of records(world.state)) expect(words).toEqual(before.get(id)!);
        };
        pairs[1].a.setAwake(false);
        expect(world.state.joints[pairs[1].joint.id.index1 - 1].setIndex).toBeGreaterThanOrEqual(
            SetType.FirstSleeping,
        );
        assertUnchanged();
        pairs[1].b.setAwake(true);
        assertUnchanged();
        pairs[0].joint.destroy(false);
        assertUnchanged();
        pairs[2].a.setType(BodyType.Static);
        assertUnchanged();
        pairs[2].a.setType(BodyType.Dynamic);
        assertUnchanged();
        world.restore(snapshot);
        expect(records(world.state)).toEqual(before);
        pairs[3].a.setAwake(false);
        assertUnchanged();
        pairs[3].b.setAwake(true);
        assertUnchanged();
    } finally {
        world.destroy();
    }
});
