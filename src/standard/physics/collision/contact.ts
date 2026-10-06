// contact.c bindings (Box3D, Erin Catto, MIT).
import { NULL_INDEX } from "../common/array";
import type { Vec3 } from "../common/math";
import { kernel } from "../kernel/kernel";
import type { Shape } from "../shapes/shape";
import type { WorldState } from "../world/world";
import { DIR_STRIDE } from "./manifoldstore";

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
export function writeBodySimIndex(world: WorldState, body: number): void {
    kernel(world.ecsState).bodySyncContacts(world.worldId, body);
}
export function reclassifyBodyContacts(world: WorldState, body: number): void {
    kernel(world.ecsState).bodyReclassifyContacts(world.worldId, body);
}
export function createContact(
    world: WorldState,
    shapeA: Shape,
    shapeB: Shape,
    childIndex: number,
): void {
    kernel(world.ecsState).contactCreateWorld(world.worldId, shapeA, shapeB, childIndex);
}

export function destroyContact(world: WorldState, id: number, wakeBodies: boolean): void {
    kernel(world.ecsState).contactDestroyWorld(world.worldId, id, wakeBodies);
}
