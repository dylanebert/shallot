import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { SetType } from "../common/constants";
import { BodyType } from "../common/types";
import { makeBoxHull } from "../shapes/hull";
import { createHullShape } from "../shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import { defaultJointDef } from "../solver/joint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodySetAwake, bodySetType, destroyBody, getMassData } from "../world/body";
import { BODY_RECORD_STRIDE, BodyField, bodyField, setBodyField } from "./bodyrecords";
import { kernel } from "./kernel";

function words(world: ReturnType<typeof createTestWorld>): number[] {
    const k = kernel(world.ecsState);
    const length = k.bodyLength(world.worldId) * BODY_RECORD_STRIDE;
    return Array.from(world.bodyStore.recordU.subarray(0, length));
}

test("body records and their LIFO pool restore every word, including free-slot generation, topology and mass", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sibling = createTestWorld();
    try {
        const hull = makeBoxHull(0.5, 0.5, 0.5);
        const a = createTestBody(world, { type: BodyType.Dynamic, sleepThreshold: 0.125 });
        const b = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0.9, y: 0, z: 0 },
        });
        createHullShape(world, a, testShapeDef({ density: 2 }), hull);
        createHullShape(world, b, testShapeDef({ density: 3 }), hull);
        const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b, collideConnected: true };
        createDistanceJoint(world, { ...defaultDistanceJointDef(base), length: 0.9 });
        stepTestWorld(world, 1 / 60);
        const c = createTestBody(world);
        const generation = bodyField(world, c, BodyField.generation);
        const free = c;
        destroyBody(world, c);
        setBodyField(world, a, BodyField.sleepTime, 0.125);
        setBodyField(world, a, BodyField.sleepVelocity, 0.25);
        const firstMass = getMassData(world, a);
        const secondMass = getMassData(world, b);
        expect(firstMass.mass).toBe(2);
        expect(secondMass.mass).toBe(3);
        expect(firstMass.inertia.cx.x).toBeCloseTo(1 / 3, 6);
        const before = words(world);
        const saved = snapshot(world);
        createTestBody(sibling, { type: BodyType.Dynamic });
        const other = words(sibling);
        bodySetAwake(world, a, false);
        expect(bodyField(world, a, BodyField.setIndex)).toBeGreaterThanOrEqual(
            SetType.FirstSleeping,
        );
        bodySetAwake(world, a, true);
        bodySetType(world, b, BodyType.Static);
        createTestBody(world, { type: BodyType.Dynamic });
        restore(world, saved);
        expect(words(world)).toEqual(before);
        expect(words(sibling)).toEqual(other);
        const reused = createTestBody(world);
        expect(reused).toBe(free);
        expect(bodyField(world, reused, BodyField.generation)).toBe((generation + 1) & 0xffff);
        expect(getMassData(world, a)).toEqual(firstMass);
        expect(getMassData(world, b)).toEqual(secondMass);
    } finally {
        destroyTestWorld(sibling);
        destroyTestWorld(world);
    }
});
