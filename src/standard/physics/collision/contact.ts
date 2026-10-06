// Contact identity lives in the kernel directory. Body lists, sets, graph and island links remain
// with their TypeScript owners until those owners move.
import { NULL_INDEX } from "../common/array";
import type { Vec3 } from "../common/math";
import { ShapeType } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { kernel } from "../kernel/kernel";
import { getCompoundChild } from "../shapes/compound";
import type { Shape } from "../shapes/shape";
import { wakeBody } from "../world/body";
import type { WorldState } from "../world/world";
import { DIR_STRIDE } from "./manifoldstore";
import { addKey, removeKey } from "./table";

export const ContactFlags = {
    contactTouchingFlag: 0x00000001,
    contactHitEventFlag: 0x00000002,
    contactEnableContactEvents: 0x00000004,
    contactStaticFlag: 0x00000008,
    contactRecycleFlag: 0x00000010,
    simTouchingFlag: 0x00010000,
    simDisjoint: 0x00020000,
    simStartedTouching: 0x00040000,
    simStoppedTouching: 0x00080000,
    simEnableHitEvent: 0x00100000,
    simEnablePreSolveEvents: 0x00200000,
    simMeshContact: 0x00400000,
    relativeTransformValid: 0x00800000,
} as const;

export const ContactField = {
    flags: 6,
    manifoldCount: 7,
    bodySimIndexA: 9,
    bodySimIndexB: 10,
    setIndex: 37,
    colorIndex: 38,
    localIndex: 39,
    bodyIdA: 40,
    prevKeyA: 41,
    nextKeyA: 42,
    bodyIdB: 43,
    prevKeyB: 44,
    nextKeyB: 45,
    shapeIdA: 46,
    shapeIdB: 47,
    childIndex: 48,
    islandId: 49,
    islandIndex: 50,
    contactId: 51,
    generation: 52,
    collideIndex: 53,
} as const;

/** Read a field at a contact id; signed null indices remain -1. */
export function contactField(world: WorldState, id: number, field: number): number {
    const value = world.manifoldStore.dirU[id * DIR_STRIDE + field];
    return field === ContactField.generation ? value : value | 0;
}
export function setContactField(world: WorldState, id: number, field: number, value: number): void {
    world.manifoldStore.dirU[id * DIR_STRIDE + field] = value;
}
export function contactBodyId(world: WorldState, id: number, side: number): number {
    return contactField(world, id, ContactField.bodyIdA + 3 * side);
}
export function contactNextKey(world: WorldState, key: number): number {
    return contactField(world, key >> 1, ContactField.nextKeyA + 3 * (key & 1));
}
export function contactCapacity(world: WorldState): number {
    const k = kernel(world.ecsState);
    return k.contactCapacity(world.worldId);
}
/** Independent list of live contact ids for requested observation. */
export function contactIds(world: WorldState): number[] {
    const ids: number[] = [];
    for (let id = 0, count = contactCapacity(world); id < count; ++id)
        if (contactField(world, id, ContactField.contactId) !== NULL_INDEX) ids.push(id);
    return ids;
}
export function contactCount(world: WorldState): number {
    const k = kernel(world.ecsState);
    return k.contactCount(world.worldId);
}

export type ManifoldPoint = {
    anchorA: Vec3;
    anchorB: Vec3;
    separation: number;
    baseSeparation: number;
    normalImpulse: number;
    totalNormalImpulse: number;
    normalVelocity: number;
    featureId: number;
    triangleIndex: number;
    persisted: boolean;
};
export type Manifold = {
    points: ManifoldPoint[];
    normal: Vec3;
    twistImpulse: number;
    frictionImpulse: Vec3;
    rollingImpulse: Vec3;
    pointCount: number;
};

export function awakeContactCount(world: WorldState): number {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k.awakeContactCount();
}
export function awakeContactGet(world: WorldState, index: number): number {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k.awakeContactGet(index);
}
export function updateAwakeContact(world: WorldState, id: number): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.awakeContactUpdate(id);
}
function removeAwakeContact(world: WorldState, id: number): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.awakeContactRemove(id);
}
export function writeBodySimIndex(world: WorldState, body: number): void {
    kernel(world.ecsState).bodySyncContacts(world.worldId, body);
}
export function reclassifyBodyContacts(world: WorldState, body: number): void {
    writeBodySimIndex(world, body);
    for (
        let key = bodyField(world, body, BodyField.headContactKey);
        key !== NULL_INDEX;
        key = contactNextKey(world, key)
    )
        updateAwakeContact(world, key >> 1);
}
export function createContact(
    world: WorldState,
    shapeA: Shape,
    shapeB: Shape,
    childIndex: number,
): void {
    const k = kernel(world.ecsState);
    const order = k.contactPairOrder(shapeA.type, shapeB.type);
    if (order === 0) return;
    if (order === 2) {
        createContact(world, shapeB, shapeA, childIndex);
        return;
    }
    let flags = 0;
    if (
        shapeA.type === ShapeType.Mesh ||
        shapeA.type === ShapeType.HeightField ||
        (shapeA.type === ShapeType.Compound &&
            getCompoundChild(shapeA.compound!, childIndex).type === ShapeType.Mesh)
    )
        flags |= ContactFlags.simMeshContact;
    if (shapeA.enableContactEvents || shapeB.enableContactEvents)
        flags |= ContactFlags.contactEnableContactEvents;
    if (shapeA.enablePreSolveEvents || shapeB.enablePreSolveEvents)
        flags |= ContactFlags.simEnablePreSolveEvents;
    const id = k.bodyCreateContact(world.worldId, shapeA.id, shapeB.id, childIndex, flags);
    addKey(world.broadPhase.pairSet, shapeA.id, shapeB.id, childIndex);
    updateAwakeContact(world, id);
}

export function destroyContact(world: WorldState, id: number, wakeBodies: boolean): void {
    removeAwakeContact(world, id);
    const shapeIdA = contactField(world, id, ContactField.shapeIdA);
    const shapeIdB = contactField(world, id, ContactField.shapeIdB);
    removeKey(
        world.broadPhase.pairSet,
        shapeIdA,
        shapeIdB,
        contactField(world, id, ContactField.childIndex),
    );
    world.manifoldStore.freeSlot(id);
    const bodyA = contactBodyId(world, id, 0);
    const bodyB = contactBodyId(world, id, 1);
    const flags = contactField(world, id, ContactField.flags);
    const touching = (flags & ContactFlags.contactTouchingFlag) !== 0;
    if (touching && flags & ContactFlags.contactEnableContactEvents) {
        const a = world.shapes[shapeIdA],
            b = world.shapes[shapeIdB];
        world.contactEndEvents[world.endEventArrayIndex].push({
            shapeIdA: { index1: a.id + 1, world0: world.worldId, generation: a.generation },
            shapeIdB: { index1: b.id + 1, world0: world.worldId, generation: b.generation },
            contactId: {
                index1: id + 1,
                world0: world.worldId,
                generation: contactField(world, id, ContactField.generation),
            },
            normalImpulse: 0,
        });
    }
    kernel(world.ecsState).bodyDestroyContact(world.worldId, id);
    if (wakeBodies && touching) {
        wakeBody(world, bodyA);
        wakeBody(world, bodyB);
    }
}
