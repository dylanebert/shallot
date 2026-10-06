import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { SetType } from "../common/constants";
import { bodySimSlot, setSimField, simBodyId, simField } from "../kernel/bodycolumns";
import { jointArrayCount, jointArrayKey, jointAt } from "../kernel/jointcolumns";
import {
    setArrayCount,
    setArrayGet,
    setBodyCount,
    solverSetCount,
    solverSetIndex,
} from "../kernel/solversetcolumns";
import type { BodySim } from "./body";
import { transferBody } from "./solverset";
import type { WorldState } from "./world";

const fields: (keyof BodySim)[] = [
    "transform",
    "center",
    "rotation0",
    "center0",
    "localCenter",
    "force",
    "torque",
    "invMass",
    "invInertiaLocal",
    "invInertiaWorld",
    "minExtent",
    "maxExtent",
    "maxAngularVelocity",
    "linearDamping",
    "angularDamping",
    "gravityScale",
    "bodyId",
    "flags",
];
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
            result.set(
                id,
                fields.map((field) => simField(world, slot, field)),
            );
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
