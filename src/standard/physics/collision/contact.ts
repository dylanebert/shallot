// Contact identity lives in the kernel directory. Body lists, sets, graph and island links remain
// with their TypeScript owners until those owners move.
import { NULL_INDEX, swapRemove } from "../common/array";
import { SetType } from "../common/constants";
import type { Vec3 } from "../common/math";
import { BodyType, ShapeType } from "../common/types";
import { bodyType, shapeBodyId } from "../kernel/filtercolumns";
import { kernel } from "../kernel/kernel";
import {
    setArrayCount,
    setArrayGet,
    setArrayPush,
    setArrayRemove,
} from "../kernel/solversetcolumns";
import { getCompoundChild } from "../shapes/compound";
import type { Shape } from "../shapes/shape";
import { removeContactFromGraph } from "../solver/graph";
import { type Body, BodyFlags, wakeBody } from "../world/body";
import { unlinkContact } from "../world/island";
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

export function updateAwakeContact(world: WorldState, id: number): void {
    if (contactField(world, id, ContactField.setIndex) === SetType.Awake) {
        if (contactField(world, id, ContactField.collideIndex) !== NULL_INDEX) return;
        setContactField(world, id, ContactField.collideIndex, world.awakeContacts.length);
        world.awakeContacts.push(id);
    } else removeAwakeContact(world, id);
}
function removeAwakeContact(world: WorldState, id: number): void {
    const index = contactField(world, id, ContactField.collideIndex);
    if (index === NULL_INDEX) return;
    if (swapRemove(world.awakeContacts, index) !== NULL_INDEX)
        setContactField(world, world.awakeContacts[index], ContactField.collideIndex, index);
    setContactField(world, id, ContactField.collideIndex, NULL_INDEX);
}
export function writeBodySimIndex(world: WorldState, body: Body): void {
    const index = bodyType(world, body.id) === BodyType.Static ? NULL_INDEX : body.localIndex;
    for (let key = body.headContactKey; key !== NULL_INDEX; key = contactNextKey(world, key))
        setContactField(world, key >> 1, ContactField.bodySimIndexA + (key & 1), index);
}
export function reclassifyBodyContacts(world: WorldState, body: Body): void {
    writeBodySimIndex(world, body);
    for (let key = body.headContactKey; key !== NULL_INDEX; key = contactNextKey(world, key))
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
    const bodyA = world.bodies[shapeBodyId(world, shapeA.id)];
    const bodyB = world.bodies[shapeBodyId(world, shapeB.id)];
    const setIndex =
        bodyA.setIndex === SetType.Awake || bodyB.setIndex === SetType.Awake
            ? SetType.Awake
            : SetType.Disabled;
    const set = setIndex;
    k.bodySetActiveWorld(world.worldId);
    const id = k.allocContact();
    setContactField(world, id, ContactField.setIndex, setIndex);
    setContactField(world, id, ContactField.localIndex, setArrayCount(world, set, 0));
    setContactField(world, id, ContactField.shapeIdA, shapeA.id);
    setContactField(world, id, ContactField.shapeIdB, shapeB.id);
    setContactField(world, id, ContactField.childIndex, childIndex);
    let flags = 0;
    if (
        bodyA.flags & BodyFlags.enableContactRecycling &&
        bodyB.flags & BodyFlags.enableContactRecycling
    )
        flags |= ContactFlags.contactRecycleFlag;
    if (
        shapeA.type === ShapeType.Mesh ||
        shapeA.type === ShapeType.HeightField ||
        (shapeA.type === ShapeType.Compound &&
            getCompoundChild(shapeA.compound!, childIndex).type === ShapeType.Mesh)
    )
        flags |= ContactFlags.simMeshContact;
    if (
        bodyType(world, bodyA.id) === BodyType.Static ||
        bodyType(world, bodyB.id) === BodyType.Static
    )
        flags |= ContactFlags.contactStaticFlag;
    if (shapeA.enableContactEvents || shapeB.enableContactEvents)
        flags |= ContactFlags.contactEnableContactEvents;
    if (shapeA.enablePreSolveEvents || shapeB.enablePreSolveEvents)
        flags |= ContactFlags.simEnablePreSolveEvents;
    setContactField(world, id, ContactField.flags, flags);
    if (flags & ContactFlags.simMeshContact) k.ensureMeshCache(id);
    for (let side = 0; side < 2; ++side) {
        const body = side === 0 ? bodyA : bodyB;
        setContactField(world, id, ContactField.bodyIdA + 3 * side, body.id);
        setContactField(world, id, ContactField.nextKeyA + 3 * side, body.headContactKey);
        const key = (id << 1) | side;
        if (body.headContactKey !== NULL_INDEX)
            setContactField(
                world,
                body.headContactKey >> 1,
                ContactField.prevKeyA + 3 * (body.headContactKey & 1),
                key,
            );
        body.headContactKey = key;
        body.contactCount += 1;
    }
    addKey(world.broadPhase.pairSet, shapeA.id, shapeB.id, childIndex);
    setArrayPush(world, set, 0, id);
    updateAwakeContact(world, id);
    setContactField(
        world,
        id,
        ContactField.bodySimIndexA,
        bodyType(world, bodyA.id) === BodyType.Static ? NULL_INDEX : bodyA.localIndex,
    );
    setContactField(
        world,
        id,
        ContactField.bodySimIndexB,
        bodyType(world, bodyB.id) === BodyType.Static ? NULL_INDEX : bodyB.localIndex,
    );
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
    const bodyA = world.bodies[contactBodyId(world, id, 0)];
    const bodyB = world.bodies[contactBodyId(world, id, 1)];
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
    for (let side = 0; side < 2; ++side) {
        const body = side === 0 ? bodyA : bodyB;
        const prev = contactField(world, id, ContactField.prevKeyA + 3 * side);
        const next = contactField(world, id, ContactField.nextKeyA + 3 * side);
        if (prev !== NULL_INDEX)
            setContactField(world, prev >> 1, ContactField.nextKeyA + 3 * (prev & 1), next);
        if (next !== NULL_INDEX)
            setContactField(world, next >> 1, ContactField.prevKeyA + 3 * (next & 1), prev);
        if (body.headContactKey === ((id << 1) | side)) body.headContactKey = next;
        body.contactCount -= 1;
    }
    if (contactField(world, id, ContactField.islandId) !== NULL_INDEX) unlinkContact(world, id);
    const colorIndex = contactField(world, id, ContactField.colorIndex);
    const localIndex = contactField(world, id, ContactField.localIndex);
    if (colorIndex !== NULL_INDEX) {
        removeContactFromGraph(
            world,
            bodyA.id,
            bodyB.id,
            colorIndex,
            localIndex,
            (flags & ContactFlags.simMeshContact) !== 0,
        );
    } else {
        const set = contactField(world, id, ContactField.setIndex);
        if (setArrayRemove(world, set, 0, localIndex) !== NULL_INDEX)
            setContactField(
                world,
                setArrayGet(world, set, 0, localIndex),
                ContactField.localIndex,
                localIndex,
            );
    }
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.freeContact(id);
    if (wakeBodies && touching) {
        wakeBody(world, bodyA);
        wakeBody(world, bodyB);
    }
}
