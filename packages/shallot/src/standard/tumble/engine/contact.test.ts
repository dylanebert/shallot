// Stage 7 contact machinery: create/destroy and the deterministic shape-order canonicalization
// (the register dispatch). Contacts are normally born from the broad-phase collide phase (solver
// stage); here they're driven directly so the create/destroy path is exercised now.

import { describe, expect, test } from "bun:test";
import { NULL_INDEX } from "./array";
import { createBody } from "./body";
import { createContact, destroyContact } from "./contact";
import { SetType } from "./core";
import { createSphereShape } from "./shape";
import { containsKey } from "./table";
import { BodyType, defaultBodyDef, defaultShapeDef, defaultWorldDef } from "./types";
import { createWorld, getWorld, type WorldState } from "./world";

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

describe("contact create / destroy", () => {
    test("links both bodies, joins the awake set, and cleans up on destroy", () => {
        const world = getWorld(createWorld(defaultWorldDef())) as WorldState;
        const a = dynamicSphere(world, 1);
        const b = dynamicSphere(world, 1);

        createContact(world, a.shape, b.shape, 0);

        expect(world.contacts.length).toBe(1);
        const contact = world.contacts[0];
        // Both bodies awake → the contact lives in the awake set as non-touching.
        expect(contact.setIndex).toBe(SetType.Awake);
        expect(world.solverSets[SetType.Awake].contactIndices).toContain(0);
        // Edge list threaded through both bodies.
        expect(a.body.contactCount).toBe(1);
        expect(b.body.contactCount).toBe(1);
        expect(a.body.headContactKey).not.toBe(NULL_INDEX);
        expect(contact.edges[0].bodyId).toBe(a.bodyId);
        expect(contact.edges[1].bodyId).toBe(b.bodyId);
        // Pair recorded so it isn't turned into a second contact.
        expect(containsKey(world.broadPhase.pairSet, a.shape.id, b.shape.id, 0)).toBe(true);

        destroyContact(world, contact, false);

        expect(a.body.contactCount).toBe(0);
        expect(b.body.contactCount).toBe(0);
        expect(a.body.headContactKey).toBe(NULL_INDEX);
        expect(b.body.headContactKey).toBe(NULL_INDEX);
        expect(world.solverSets[SetType.Awake].contactIndices.length).toBe(0);
        expect(containsKey(world.broadPhase.pairSet, a.shape.id, b.shape.id, 0)).toBe(false);
    });

    test("a non-touching contact between sleeping-adjacent bodies parks in the disabled set", () => {
        // A body that starts asleep lands in a sleeping set; a contact where neither body is awake
        // parks in the disabled set (the non-touching parking lot).
        const world = getWorld(createWorld(defaultWorldDef())) as WorldState;

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
        expect(world.contacts[0].setIndex).toBe(SetType.Disabled);
    });
});
