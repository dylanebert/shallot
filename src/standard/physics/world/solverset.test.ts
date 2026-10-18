import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { SetType } from "../common/constants";
import { BodyType } from "../common/types";
import { bodySimSlot, setSimField, simBodyId } from "../kernel/bodycolumns";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { SIM_STRIDE } from "../kernel/columns";
import { islandField } from "../kernel/islandcolumns";
import { jointArrayCount, jointArrayKey, jointAt } from "../kernel/jointcolumns";
import { JointField, jointCapacity, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import {
    setArrayCount,
    setArrayGet,
    setBodyCount,
    solverSetCount,
    solverSetIndex,
} from "../kernel/solversetcolumns";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import { defaultJointDef } from "../solver/joint";
import { jointIds } from "../solver/joint.fixture";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld } from "../testing";
import { bodyGetLinearVelocity, bodyIsAwake, bodySetAwake, destroyBody } from "../world/body";
import type { WorldState } from "./world";

function distanceJoint(world: WorldState, a: number, b: number, length = 1) {
    const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
    return createDistanceJoint(world, { ...defaultDistanceJointDef(base), length }).joint;
}

function sims(world: WorldState) {
    world.bodyStore.refreshViews();
    const result = new Map<number, unknown[]>();
    for (let set = 0; set < solverSetCount(world); ++set) {
        if (solverSetIndex(world, set) === -1) continue;
        for (let i = 0; i < setBodyCount(world, set); ++i) {
            const slot = bodySimSlot(set, i),
                id = simBodyId(world, slot);
            expect(bodyField(world, id, BodyField.setIndex)).toBe(set);
            expect(bodyField(world, id, BodyField.localIndex)).toBe(i);
            const columns = world.bodyStore.simColumns(set);
            result.set(id, [
                Array.from(columns.simF.subarray(i * SIM_STRIDE, (i + 1) * SIM_STRIDE)),
                Array.from(columns.simU.subarray(i * SIM_STRIDE, (i + 1) * SIM_STRIDE)),
            ]);
        }
        for (let i = 0; i < setArrayCount(world, set, 1); ++i) {
            const island = setArrayGet(world, set, 1, i);
            expect(islandField(world, island, 0)).toBe(set);
            expect(islandField(world, island, 1)).toBe(i);
        }
    }
    for (const joint of jointIds(world)) {
        if (jointField(world, joint, JointField.setIndex) === -1) continue;
        expect(jointField(world, joint, JointField.localIndex)).toBeLessThan(
            jointArrayCount(world, jointArrayKey(world, joint)),
        );
        expect(
            jointAt(
                world,
                jointArrayKey(world, joint),
                jointField(world, joint, JointField.localIndex),
            ),
        ).toBe(joint);
    }
    return result;
}

test("solver-set moves preserve all sim fields and fix body/island/joint slots through sleep, wake, id reuse and snapshot restore", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const pairs = Array.from({ length: 6 }, (_, i) => {
            const a = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i * 10, y: i, z: 0 },
            });
            const b = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i * 10 + 1, y: i, z: 0 },
            });
            distanceJoint(world, a, b);
            const slot = bodySimSlot(
                bodyField(world, a, BodyField.setIndex),
                bodyField(world, a, BodyField.localIndex),
            );
            setSimField(world, slot, "force", { x: i + 0.25, y: -i, z: 2 });
            setSimField(world, slot, "torque", { x: 3, y: i + 0.5, z: -1 });
            setSimField(world, slot, "center0", { x: 7, y: 8, z: i + 0.75 });
            return { a, b };
        });
        createTestBody(sibling, { type: BodyType.Static, position: { x: 99, y: 88, z: 77 } });
        const before = sims(world),
            other = sims(sibling),
            saved = snapshot(world);
        bodySetAwake(world, pairs[1]!.a, false);
        expect(sims(world)).toEqual(before);
        const first = bodyField(world, pairs[1]!.a, BodyField.setIndex);
        bodySetAwake(world, pairs[4]!.a, false);
        expect(sims(world)).toEqual(before);
        const asleep = snapshot(world);
        bodySetAwake(world, pairs[1]!.b, true);
        expect(sims(world)).toEqual(before);
        bodySetAwake(world, pairs[2]!.a, false);
        expect(bodyField(world, pairs[2]!.a, BodyField.setIndex)).toBe(first);
        expect(sims(world)).toEqual(before);
        restore(world, asleep);
        expect(sims(world)).toEqual(before);
        expect(sims(sibling)).toEqual(other);
        bodySetAwake(world, pairs[1]!.b, true);
        bodySetAwake(world, pairs[4]!.b, true);
        expect(sims(world)).toEqual(before);
        restore(world, saved);
        expect(sims(world)).toEqual(before);
        destroyBody(world, pairs[0]!.a);
        const remaining = sims(world);
        expect(remaining.size).toBe(before.size - 1);
        for (const [id, sim] of remaining) expect(sim).toEqual(before.get(id)!);
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});

test("body transfers swap-remove awake, static and disabled sim rows and discard the source awake state", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = Array.from({ length: 4 }, () =>
            createTestBody(world, { type: BodyType.Dynamic, linearVelocity: { x: 1, y: 2, z: 3 } }),
        );
        const body = bodies[1]!;
        const before = sims(world);
        kernel(world.ecsState).bodyTransfer(world.worldId, body, SetType.Disabled, true);
        expect(sims(world)).toEqual(before);
        kernel(world.ecsState).bodyTransfer(world.worldId, body, SetType.Static, true);
        expect(sims(world)).toEqual(before);
        kernel(world.ecsState).bodyTransfer(world.worldId, body, SetType.Awake, true);
        expect(sims(world)).toEqual(before);
        expect(bodyGetLinearVelocity(world, body, { x: 0, y: 0, z: 0 })).toEqual({
            x: 0,
            y: 0,
            z: 0,
        });
    } finally {
        destroyTestWorld(world);
    }
});

test("joint creation merges sleeping sets in append order without waking bodies", () => {
    for (const sizes of [
        [2, 3],
        [3, 2],
        [2, 2],
    ]) {
        const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
        try {
            const groups = sizes.map((size, group) => {
                const bodies = Array.from({ length: size }, (_, i) =>
                    createTestBody(world, {
                        type: BodyType.Dynamic,
                        position: { x: group * 10 + i, y: 0, z: 0 },
                    }),
                );
                for (let i = 1; i < size; ++i) distanceJoint(world, bodies[i - 1]!, bodies[i]!);
                bodySetAwake(world, bodies[0]!, false);
                return bodies;
            });
            const sets = groups.map((group) => bodyField(world, group[0]!, BodyField.setIndex));
            expect(sets[0]).not.toBe(sets[1]);
            const before = sims(world);
            const jointOrder = sets.map((set) =>
                jointIds(world)
                    .filter((joint) => jointField(world, joint, JointField.setIndex) === set)
                    .sort(
                        (a, b) =>
                            jointField(world, a, JointField.localIndex) -
                            jointField(world, b, JointField.localIndex),
                    )
                    .map((joint) => jointField(world, joint, JointField.jointId)),
            );
            const newJointId = jointCapacity(world);
            const survivor = sizes[0]! >= sizes[1]! ? 0 : 1;
            distanceJoint(world, groups[0]![0]!, groups[1]![0]!, 10);
            jointOrder[1]!.push(newJointId);
            const expectedJoints = [...jointOrder[survivor]!, ...jointOrder[1 - survivor]!];
            const mergedJoints = jointIds(world)
                .filter((joint) => jointField(world, joint, JointField.setIndex) === sets[survivor])
                .sort(
                    (a, b) =>
                        jointField(world, a, JointField.localIndex) -
                        jointField(world, b, JointField.localIndex),
                )
                .map((joint) => jointField(world, joint, JointField.jointId));
            expect(mergedJoints).toEqual(expectedJoints);
            expect(solverSetIndex(world, sets[1 - survivor]!)).toBe(-1);
            expect(setBodyCount(world, sets[survivor]!)).toBe(sizes[0]! + sizes[1]!);
            const order = [...groups[survivor]!, ...groups[1 - survivor]!];
            for (let i = 0; i < order.length; ++i) {
                const body = order[i]!;
                expect(bodyIsAwake(world, body)).toBe(false);
                expect(simBodyId(world, bodySimSlot(sets[survivor]!, i))).toBe(body);
            }
            expect(sims(world)).toEqual(before);
            expect(setArrayCount(world, sets[survivor]!, 1)).toBe(1);
            stepTestWorld(world, 1 / 60);
            for (const body of order) expect(bodyIsAwake(world, body)).toBe(false);
            bodySetAwake(world, groups[0]![0]!, true);
            for (const body of order) expect(bodyIsAwake(world, body)).toBe(true);
            sims(world);
        } finally {
            destroyTestWorld(world);
        }
    }
});
