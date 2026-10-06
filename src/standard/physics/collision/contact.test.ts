import { ContactField, contactCapacity, contactField } from "./contact";
// contact machinery: create/destroy and the solver-set placement it picks. Contacts are normally
// born from the broad-phase collide phase (solver stage); here they're driven directly so the
// create/destroy path is exercised now.

import { expect, test } from "bun:test";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { BodyType, defaultBodyDef, defaultShapeDef, defaultWorldDef } from "../common/types";
import { createSphereShape } from "../shapes/shape";
import { createBody } from "../world/body";
import { createWorld, getWorld, type WorldState } from "../world/world";
import { createContact, destroyContact } from "./contact";
import { addKey, removeKey } from "./table";

function dynamicSphere(world: WorldState, radius: number) {
    const bodyId = createBody(world, { ...defaultBodyDef(), type: BodyType.Dynamic });
    const body = world.bodies[bodyId];
    const shape = createSphereShape(world, body, defaultShapeDef(), {
        center: { x: 0, y: 0, z: 0 },
        radius,
    });
    if (shape === null) {
        throw new Error("shape creation failed");
    }
    return { bodyId, body, shape };
}

test("createContact leaves a body edge, an awake-set row or a broad-phase pair entry behind after destroyContact, leaking a contact into the next step", () => {
    const world = getWorld(createWorld(undefined, defaultWorldDef())) as WorldState;
    const a = dynamicSphere(world, 1);
    const b = dynamicSphere(world, 1);

    createContact(world, a.shape, b.shape, 0);

    expect(contactCapacity(world)).toBe(1);
    const contact = 0;
    // Both bodies awake → the contact lives in the awake set as non-touching.
    expect(contactField(world, contact, ContactField.setIndex)).toBe(SetType.Awake);
    expect(world.solverSets[SetType.Awake].contactIndices).toContain(0);
    // Edge list threaded through both bodies.
    expect(a.body.contactCount, "body a contactCount").toBe(1);
    expect(b.body.contactCount, "body b contactCount").toBe(1);
    expect(a.body.headContactKey).not.toBe(NULL_INDEX);
    expect(contactField(world, contact, ContactField.bodyIdA)).toBe(a.bodyId);
    expect(contactField(world, contact, ContactField.bodyIdB)).toBe(b.bodyId);
    // Pair recorded so it isn't turned into a second contact.
    expect(addKey(world.broadPhase.pairSet, a.shape.id, b.shape.id, 0)).toBe(true);

    destroyContact(world, contact, false);

    expect(a.body.contactCount, "body a contactCount after destroy").toBe(0);
    expect(b.body.contactCount, "body b contactCount after destroy").toBe(0);
    expect(a.body.headContactKey).toBe(NULL_INDEX);
    expect(b.body.headContactKey).toBe(NULL_INDEX);
    expect(world.solverSets[SetType.Awake].contactIndices.length).toBe(0);
    expect(removeKey(world.broadPhase.pairSet, a.shape.id, b.shape.id, 0)).toBe(false);
});

test("createContact files a non-touching contact between two asleep bodies into the awake set, so sleeping islands pay for contacts nothing is simulating", () => {
    // A body that starts asleep lands in a sleeping set; a contact where neither body is awake
    // parks in the disabled set (the non-touching parking lot).
    const world = getWorld(createWorld(undefined, defaultWorldDef())) as WorldState;

    const mk = (radius: number) => {
        const bodyId = createBody(world, {
            ...defaultBodyDef(),
            type: BodyType.Dynamic,
            isAwake: false,
        });
        const body = world.bodies[bodyId];
        const shape = createSphereShape(world, body, defaultShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius,
        });
        return { body, shape: shape as NonNullable<typeof shape> };
    };
    const a = mk(1);
    const b = mk(1);
    expect(a.body.setIndex).toBeGreaterThanOrEqual(SetType.FirstSleeping);

    createContact(world, a.shape, b.shape, 0);
    expect(contactField(world, 0, ContactField.setIndex)).toBe(SetType.Disabled);
});
