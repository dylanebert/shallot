import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { SetType } from "../common/constants";
import { bodySimSlot, setSimField, simBodyId } from "../kernel/bodycolumns";
import { FIN_STRIDE, SIM_STRIDE, SIM2_STRIDE } from "../kernel/columns";
import { jointArrayCount, jointArrayKey, jointAt } from "../kernel/jointcolumns";
import {
    setArrayCount,
    setArrayGet,
    setBodyCount,
    solverSetCount,
    solverSetIndex,
} from "../kernel/solversetcolumns";
import { transferBody } from "./solverset";
import type { WorldState } from "./world";

function sims(world: WorldState) {
    world.bodyStore.refreshViews();
    const result = new Map<number, unknown[]>();
    for (let set = 0; set < solverSetCount(world); ++set) {
        if (solverSetIndex(world, set) === -1) continue;
        for (let i = 0; i < setBodyCount(world, set); ++i) {
            const slot = bodySimSlot(set, i),
                id = simBodyId(world, slot);
            expect(world.bodies[id].setIndex).toBe(set);
            expect(world.bodies[id].localIndex).toBe(i);
            const columns = world.bodyStore.simColumns(set);
            result.set(id, [
                Array.from(columns.simF.subarray(i * SIM_STRIDE, (i + 1) * SIM_STRIDE)),
                Array.from(columns.finF.subarray(i * FIN_STRIDE, (i + 1) * FIN_STRIDE)),
                Array.from(columns.sim2U.subarray(i * SIM2_STRIDE, (i + 1) * SIM2_STRIDE)),
            ]);
        }
        for (let i = 0; i < setArrayCount(world, set, 1); ++i) {
            const island = world.islands[setArrayGet(world, set, 1, i)];
            expect(island.setIndex).toBe(set);
            expect(island.localIndex).toBe(i);
        }
    }
    for (const joint of world.joints) {
        if (joint.setIndex === -1) continue;
        expect(joint.localIndex).toBeLessThan(jointArrayCount(world, jointArrayKey(joint)));
        expect(jointAt(world, jointArrayKey(joint), joint.localIndex)).toBe(joint);
    }
    return result;
}
test("solver-set moves preserve all sim fields and fix body/island/joint slots through sleep, wake, id reuse and snapshot restore", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const pairs = Array.from({ length: 6 }, (_, i) => {
            const a = world.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 10, y: i, z: 0 },
            });
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 10 + 1, y: i, z: 0 },
            });
            world.createDistanceJoint(a, b, { length: 1 });
            const record = world.state.bodies[a.id.index1 - 1];
            const slot = bodySimSlot(record.setIndex, record.localIndex);
            setSimField(world.state, slot, "force", { x: i + 0.25, y: -i, z: 2 });
            setSimField(world.state, slot, "torque", { x: 3, y: i + 0.5, z: -1 });
            setSimField(world.state, slot, "center0", { x: 7, y: 8, z: i + 0.75 });
            return { a, b };
        });
        sibling.createBody({ type: BodyType.Static, position: { x: 99, y: 88, z: 77 } });
        const before = sims(world.state),
            other = sims(sibling.state),
            snapshot = world.snapshot();
        pairs[1].a.setAwake(false);
        expect(sims(world.state)).toEqual(before);
        const first = world.state.bodies[pairs[1].a.id.index1 - 1].setIndex;
        pairs[4].a.setAwake(false);
        expect(sims(world.state)).toEqual(before);
        const asleep = world.snapshot();
        pairs[1].b.setAwake(true);
        expect(sims(world.state)).toEqual(before);
        pairs[2].a.setAwake(false);
        expect(world.state.bodies[pairs[2].a.id.index1 - 1].setIndex).toBe(first);
        expect(sims(world.state)).toEqual(before);
        world.restore(asleep);
        expect(sims(world.state)).toEqual(before);
        expect(sims(sibling.state)).toEqual(other);
        pairs[1].b.setAwake(true);
        pairs[4].b.setAwake(true);
        expect(sims(world.state)).toEqual(before);
        world.restore(snapshot);
        expect(sims(world.state)).toEqual(before);
        pairs[0].a.destroy();
        const remaining = sims(world.state);
        expect(remaining.size).toBe(before.size - 1);
        for (const [id, sim] of remaining) expect(sim).toEqual(before.get(id)!);
    } finally {
        world.destroy();
        sibling.destroy();
    }
});
test("body transfers swap-remove awake, static and disabled sim rows and discard the source awake state", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = Array.from({ length: 4 }, () =>
            world.createBody({ type: BodyType.Dynamic, linearVelocity: { x: 1, y: 2, z: 3 } }),
        );
        const body = world.state.bodies[bodies[1].id.index1 - 1];
        const before = sims(world.state);
        transferBody(world.state, SetType.Disabled, SetType.Awake, body);
        expect(sims(world.state)).toEqual(before);
        transferBody(world.state, SetType.Static, SetType.Disabled, body);
        expect(sims(world.state)).toEqual(before);
        transferBody(world.state, SetType.Awake, SetType.Static, body);
        expect(sims(world.state)).toEqual(before);
        expect(bodies[1].getLinearVelocity()).toEqual({ x: 0, y: 0, z: 0 });
    } finally {
        world.destroy();
    }
});

test("joint creation merges sleeping sets in append order without waking bodies", () => {
    for (const sizes of [
        [2, 3],
        [3, 2],
        [2, 2],
    ]) {
        const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
        try {
            const groups = sizes.map((size, group) => {
                const bodies = Array.from({ length: size }, (_, i) =>
                    world.createBody({
                        type: BodyType.Dynamic,
                        position: { x: group * 10 + i, y: 0, z: 0 },
                    }),
                );
                for (let i = 1; i < size; ++i)
                    world.createDistanceJoint(bodies[i - 1], bodies[i], { length: 1 });
                bodies[0].setAwake(false);
                return bodies;
            });
            const sets = groups.map((group) => world.state.bodies[group[0].id.index1 - 1].setIndex);
            expect(sets[0]).not.toBe(sets[1]);
            const before = sims(world.state);
            const jointOrder = sets.map((set) =>
                world.state.joints
                    .filter((joint) => joint.setIndex === set)
                    .sort((a, b) => a.localIndex - b.localIndex)
                    .map((joint) => joint.jointId),
            );
            const newJointId = world.state.joints.length;
            const survivor = sizes[0] >= sizes[1] ? 0 : 1;
            world.createDistanceJoint(groups[0][0], groups[1][0], { length: 10 });
            jointOrder[1].push(newJointId);
            const expectedJoints = [...jointOrder[survivor], ...jointOrder[1 - survivor]];
            const mergedJoints = world.state.joints
                .filter((joint) => joint.setIndex === sets[survivor])
                .sort((a, b) => a.localIndex - b.localIndex)
                .map((joint) => joint.jointId);
            expect(mergedJoints).toEqual(expectedJoints);
            expect(solverSetIndex(world.state, sets[1 - survivor])).toBe(-1);
            expect(setBodyCount(world.state, sets[survivor])).toBe(sizes[0] + sizes[1]);
            const order = [...groups[survivor], ...groups[1 - survivor]];
            for (let i = 0; i < order.length; ++i) {
                const body = order[i];
                expect(body.isAwake()).toBe(false);
                expect(simBodyId(world.state, bodySimSlot(sets[survivor], i))).toBe(
                    body.id.index1 - 1,
                );
            }
            expect(sims(world.state)).toEqual(before);
            expect(setArrayCount(world.state, sets[survivor], 1)).toBe(1);
            world.step(1 / 60);
            for (const body of order) expect(body.isAwake()).toBe(false);
            groups[0][0].setAwake(true);
            for (const body of order) expect(body.isAwake()).toBe(true);
            sims(world.state);
        } finally {
            world.destroy();
        }
    }
});
