import { expect, test } from "bun:test";
import { ContactField, contactCount, contactIsValid, setContactField } from "../collision/contact";
import { updateBroadPhasePairs } from "../collision/pairs";
import { BodyType } from "../common/types";
import { EventKind, eventCount, eventId } from "../kernel/eventbuffers";
import { kernel } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { destroyBody } from "../world/body";

test("contact begin, hit and end events retain a full u32 generation and the solved begin impulse", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 }, enableContinuous: false });
    try {
        const ground = createTestBody(world, { position: { x: 0, y: -0.5, z: 0 } });
        createHullShape(world, ground, testShapeDef(), makeBoxHull(10, 0.5, 10));
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 0.5, z: 0 },
            linearVelocity: { x: 0, y: -10, z: 0 },
        });
        createHullShape(
            world,
            body,
            testShapeDef({ enableContactEvents: true, enableHitEvents: true }),
            makeBoxHull(0.5, 0.5, 0.5),
        );
        updateBroadPhasePairs(world);
        expect(contactCount(world)).toBe(1);
        const generation = 65536;
        setContactField(world, 0, ContactField.generation, generation);
        stepTestWorld(world, 1 / 60, 4);
        expect(eventCount(world, EventKind.ContactBegin)).toBe(1);
        const beginContact = eventId(world, EventKind.ContactBegin, 0, 4);
        expect(beginContact.generation).toBe(generation);
        expect(contactIsValid(world, beginContact)).toBe(true);
        const impulse = kernel(world.ecsState).eventFloat(
            world.worldId,
            EventKind.ContactBegin,
            0,
            7,
        );
        expect(impulse).toBeGreaterThan(0);
        expect(eventCount(world, EventKind.ContactHit)).toBe(1);
        const hitContact = eventId(world, EventKind.ContactHit, 0, 4);
        expect(hitContact.generation).toBe(generation);
        expect(contactIsValid(world, hitContact)).toBe(true);
        destroyBody(world, body);
        expect(kernel(world.ecsState).eventFloat(world.worldId, EventKind.ContactBegin, 0, 7)).toBe(
            impulse,
        );
        stepTestWorld(world, 1 / 60, 4);
        expect(eventCount(world, EventKind.ContactEnd)).toBe(1);
        const endContact = eventId(world, EventKind.ContactEnd, 0, 4);
        expect(endContact.generation).toBe(generation);
        expect(contactIsValid(world, endContact)).toBe(false);
    } finally {
        destroyTestWorld(world);
    }
});
