import { expect, test } from "bun:test";
import { ContactField, contactField } from "../collision/contact";
import { BodyType } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { setArraySnapshot, solverSetIndex } from "../kernel/solversetcolumns";
import { createSphereShape } from "../shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import { defaultJointDef } from "../solver/joint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodyIsAwake, bodySetAwake } from "./body";

test("sleeping-set merge appends contacts and fixes their set and local indices without waking", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const groups = [0, 10].map((x) =>
            [0, 1.9].map((offset) => {
                const body = createTestBody(world, {
                    type: BodyType.Dynamic,
                    position: { x: x + offset, y: 0, z: 0 },
                });
                createSphereShape(world, body, testShapeDef(), {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 1,
                });
                return body;
            }),
        );
        stepTestWorld(world, 1 / 60);
        for (const group of groups) bodySetAwake(world, group[0]!, false);
        const sets = groups.map((group) => bodyField(world, group[0]!, BodyField.setIndex));
        const contacts = sets.map((set) => setArraySnapshot(world, set, 0));
        expect(contacts.map((ids) => ids.length)).toEqual([1, 1]);
        const base = { ...defaultJointDef(), bodyIdA: groups[0]![0]!, bodyIdB: groups[1]![0]! };
        createDistanceJoint(world, { ...defaultDistanceJointDef(base), length: 10 });
        const expected = contacts.flat();
        expect(solverSetIndex(world, sets[1]!)).toBe(-1);
        expect(setArraySnapshot(world, sets[0]!, 0)).toEqual(expected);
        for (let i = 0; i < expected.length; ++i) {
            expect(contactField(world, expected[i]!, ContactField.setIndex)).toBe(sets[0]);
            expect(contactField(world, expected[i]!, ContactField.localIndex)).toBe(i);
        }
        for (const body of groups.flat()) expect(bodyIsAwake(world, body)).toBe(false);
        stepTestWorld(world, 1 / 60);
        for (const body of groups.flat()) expect(bodyIsAwake(world, body)).toBe(false);
    } finally {
        destroyTestWorld(world);
    }
});
