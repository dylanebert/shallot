import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { ContactField, contactField } from "../collision/contact";
import { setArraySnapshot, solverSetIndex } from "../kernel/solversetcolumns";

test("sleeping-set merge appends contacts and fixes their set and local indices without waking", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const groups = [0, 10].map((x) =>
            [0, 1.9].map((offset) => {
                const body = world.createBody({
                    type: BodyType.Dynamic,
                    position: { x: x + offset, y: 0, z: 0 },
                });
                body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
                return body;
            }),
        );
        world.step(1 / 60);
        for (const group of groups) group[0].setAwake(false);
        const sets = groups.map((group) => world.state.bodies[group[0].id.index1 - 1].setIndex);
        const contacts = sets.map((set) => setArraySnapshot(world.state, set, 0));
        expect(contacts.map((ids) => ids.length)).toEqual([1, 1]);
        world.createDistanceJoint(groups[0][0], groups[1][0], { length: 10 });
        const expected = contacts.flat();
        expect(solverSetIndex(world.state, sets[1])).toBe(-1);
        expect(setArraySnapshot(world.state, sets[0], 0)).toEqual(expected);
        for (let i = 0; i < expected.length; ++i) {
            expect(contactField(world.state, expected[i], ContactField.setIndex)).toBe(sets[0]);
            expect(contactField(world.state, expected[i], ContactField.localIndex)).toBe(i);
        }
        for (const body of groups.flat()) expect(body.isAwake()).toBe(false);
        world.step(1 / 60);
        for (const body of groups.flat()) expect(body.isAwake()).toBe(false);
    } finally {
        world.destroy();
    }
});
