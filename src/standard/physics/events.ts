import type { EntityRef, Resource } from "../../engine";

export interface PhysicsSensorEvent {
    sensorShape: EntityRef;
    sensorBody: EntityRef;
    visitorShape: EntityRef;
    visitorBody: EntityRef;
}

export interface PhysicsContactPair {
    shapeA: EntityRef;
    bodyA: EntityRef;
    shapeB: EntityRef;
    bodyB: EntityRef;
}

export interface PhysicsContactEvent extends PhysicsContactPair {
    normalImpulse: number;
}

export interface PhysicsHitEvent extends PhysicsContactPair {
    point: { x: number; y: number; z: number };
    normal: { x: number; y: number; z: number };
    approachSpeed: number;
    userMaterialIdA: bigint;
    userMaterialIdB: bigint;
}

export interface PhysicsJointEvent {
    joint: EntityRef;
}

export interface PhysicsEventRecords<T> {
    /** The first `count` records are valid for the last fixed step; storage keeps its high-water mark. */
    records: T[];
    count: number;
}

export interface PhysicsEventsState {
    sensorBegin: PhysicsEventRecords<PhysicsSensorEvent>;
    sensorEnd: PhysicsEventRecords<PhysicsSensorEvent>;
    contactBegin: PhysicsEventRecords<PhysicsContactEvent>;
    contactEnd: PhysicsEventRecords<PhysicsContactEvent>;
    contactHit: PhysicsEventRecords<PhysicsHitEvent>;
    joints: PhysicsEventRecords<PhysicsJointEvent>;
}

function records<T>(): PhysicsEventRecords<T> {
    return { records: [], count: 0 };
}

/**
 * Box3D events keyed by generation-aware EntityRefs. Resolve them on their owning World; a dead
 * entity resolves to 0 instead of aliasing a reused eid. Records rebuild after every fixed step.
 */
export const PhysicsEvents: Resource<PhysicsEventsState> = {
    create: () => ({
        sensorBegin: records<PhysicsSensorEvent>(),
        sensorEnd: records<PhysicsSensorEvent>(),
        contactBegin: records<PhysicsContactEvent>(),
        contactEnd: records<PhysicsContactEvent>(),
        contactHit: records<PhysicsHitEvent>(),
        joints: records<PhysicsJointEvent>(),
    }),
};

export function clearPhysicsEvents(events: PhysicsEventsState): void {
    events.sensorBegin.count = 0;
    events.sensorEnd.count = 0;
    events.contactBegin.count = 0;
    events.contactEnd.count = 0;
    events.contactHit.count = 0;
    events.joints.count = 0;
}
