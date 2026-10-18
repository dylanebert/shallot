// body.c bindings (Box3D, Erin Catto, MIT).

import { NULL_INDEX } from "../common/array";
import { BODY_NAME_LENGTH, SetType } from "../common/constants";
import type { EntityId } from "../common/ids";
import type { Mat3, Pos, Quat, Vec3, WorldTransform } from "../common/records";

const f32 = Math.fround;

import type { BodyDef, BodyType } from "../common/types";
import {
    readSimLocalCenter,
    readSimTransform,
    readStateLinearVelocity,
} from "../kernel/bodycolumns";
import { BodyField, bodyField, bodyInertia } from "../kernel/bodyrecords";
import { bodyType } from "../kernel/filtercolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import { ShapeField, shapeField } from "../kernel/shaperecords";
import type { MassData } from "../shapes/geometry";
import type { WorldState } from "./world";

export const BodyFlags = {
    lockLinearX: 0x00000001,
    lockLinearY: 0x00000002,
    lockLinearZ: 0x00000004,
    lockAngularX: 0x00000008,
    lockAngularY: 0x00000010,
    lockAngularZ: 0x00000020,
    isFast: 0x00000040,
    isBullet: 0x00000080,
    isSpeedCapped: 0x00000100,
    hadTimeOfImpact: 0x00000200,
    allowFastRotation: 0x00000400,
    enlargeBounds: 0x00000800,
    dynamicFlag: 0x00001000,
    enableSleep: 0x00002000,
    enableContactRecycling: 0x00004000,
} as const;
export type BodySim = {
    readonly transform: WorldTransform;
    center: Pos;
    readonly rotation0: Quat;
    center0: Pos;
    localCenter: Vec3;
    force: Vec3;
    torque: Vec3;
    invMass: number;
    invInertiaLocal: Mat3;
    invInertiaWorld: Mat3;
    minExtent: number;
    maxExtent: Vec3;
    linearDamping: number;
    angularDamping: number;
    gravityScale: number;
    bodyId: number;
    flags: number;
};
export function getBodySim(_world: WorldState, body: number): number {
    return -body - 1;
}
export function getBodyState(world: WorldState, body: number): number | null {
    const index = kernel(world.ecsState).bodyStateIndex(world.worldId, body);
    return index < 0 ? null : index;
}
export function bodyIsValid(world: WorldState, id: EntityId): boolean {
    if (!world.inUse) return false;
    const body = id.index1 - 1;
    const k = kernel(world.ecsState);
    return (
        body >= 0 &&
        body < k.bodyLength(world.worldId) &&
        k.bodyAlive(world.worldId, body) !== 0 &&
        k.bodyGeneration(world.worldId, body) === id.generation
    );
}

export function bodyGetType(world: WorldState, body: number): BodyType {
    return bodyType(world, bodyField(world, body, BodyField.id));
}

export function bodyGetTransform(
    world: WorldState,
    body: number,
    out: WorldTransform,
): WorldTransform {
    return readSimTransform(world, getBodySim(world, body), out);
}

export function bodyGetLinearVelocity(world: WorldState, body: number, out: Vec3): Vec3 {
    const state = getBodyState(world, body);
    if (state !== null) return readStateLinearVelocity(world, state, out);
    out.x = 0;
    out.y = 0;
    out.z = 0;
    return out;
}

export function bodyIsAwake(world: WorldState, body: number): boolean {
    return bodyField(world, body, BodyField.setIndex) === SetType.Awake;
}

export function makeBodyId(world: WorldState, bodyId: number): EntityId {
    if (bodyId === NULL_INDEX) return { index1: 0, world0: 0, generation: 0 };
    return {
        index1: bodyId + 1,
        world0: world.worldId,
        generation: kernel(world.ecsState).bodyGeneration(world.worldId, bodyId),
    };
}
export function readBodyTransform(
    world: WorldState,
    body: number,
    out: WorldTransform,
): WorldTransform {
    return readSimTransform(world, getBodySim(world, body), out);
}
export function bodySetLinearVelocity(world: WorldState, body: number, v: Vec3): void {
    kernel(world.ecsState).bodyVelocitySet(
        world.worldId,
        body,
        false,
        f32(v.x),
        f32(v.y),
        f32(v.z),
    );
}
export function bodySetAngularVelocity(world: WorldState, body: number, v: Vec3): void {
    kernel(world.ecsState).bodyVelocitySet(world.worldId, body, true, f32(v.x), f32(v.y), f32(v.z));
}
export function bodySetTargetTransform(
    world: WorldState,
    body: number,
    target: WorldTransform,
    timeStep: number,
    wake: boolean,
): void {
    const p = target.p,
        q = target.q;
    kernel(world.ecsState).bodyTargetVelocity(
        world.worldId,
        body,
        p.x,
        p.y,
        p.z,
        f32(q.v.x),
        f32(q.v.y),
        f32(q.v.z),
        f32(q.s),
        timeStep,
        wake,
    );
}
export function bodyApplyForce(
    world: WorldState,
    body: number,
    force: Vec3,
    point: Pos,
    wake: boolean,
): void {
    kernel(world.ecsState).bodyApply(
        world.worldId,
        body,
        0,
        f32(force.x),
        f32(force.y),
        f32(force.z),
        f32(point.x),
        f32(point.y),
        f32(point.z),
        world.maxLinearSpeed,
        wake,
    );
}
export function bodyApplyForceToCenter(
    world: WorldState,
    body: number,
    force: Vec3,
    wake: boolean,
): void {
    kernel(world.ecsState).bodyApply(
        world.worldId,
        body,
        1,
        f32(force.x),
        f32(force.y),
        f32(force.z),
        0,
        0,
        0,
        world.maxLinearSpeed,
        wake,
    );
}
export function bodyApplyTorque(
    world: WorldState,
    body: number,
    torque: Vec3,
    wake: boolean,
): void {
    kernel(world.ecsState).bodyApply(
        world.worldId,
        body,
        2,
        f32(torque.x),
        f32(torque.y),
        f32(torque.z),
        0,
        0,
        0,
        world.maxLinearSpeed,
        wake,
    );
}
export function bodyApplyLinearImpulse(
    world: WorldState,
    body: number,
    impulse: Vec3,
    point: Pos,
    wake: boolean,
): void {
    kernel(world.ecsState).bodyApply(
        world.worldId,
        body,
        3,
        f32(impulse.x),
        f32(impulse.y),
        f32(impulse.z),
        f32(point.x),
        f32(point.y),
        f32(point.z),
        world.maxLinearSpeed,
        wake,
    );
}
export function bodyApplyLinearImpulseToCenter(
    world: WorldState,
    body: number,
    impulse: Vec3,
    wake: boolean,
): void {
    kernel(world.ecsState).bodyApply(
        world.worldId,
        body,
        4,
        f32(impulse.x),
        f32(impulse.y),
        f32(impulse.z),
        0,
        0,
        0,
        world.maxLinearSpeed,
        wake,
    );
}
export function bodyApplyAngularImpulse(
    world: WorldState,
    body: number,
    impulse: Vec3,
    wake: boolean,
): void {
    kernel(world.ecsState).bodyApply(
        world.worldId,
        body,
        5,
        f32(impulse.x),
        f32(impulse.y),
        f32(impulse.z),
        0,
        0,
        0,
        world.maxLinearSpeed,
        wake,
    );
}
export function bodySetTransform(
    world: WorldState,
    body: number,
    position: Pos,
    rotation: Quat,
): void {
    kernel(world.ecsState).bodySetPose(
        world.worldId,
        body,
        f32(position.x),
        f32(position.y),
        f32(position.z),
        f32(rotation.v.x),
        f32(rotation.v.y),
        f32(rotation.v.z),
        f32(rotation.s),
    );
}
export const BodyProperty = {
    linearDamping: 0,
    angularDamping: 1,
    gravityScale: 2,
    sleepThreshold: 3,
    enableSleep: 4,
    motionLocks: 5,
    isBullet: 6,
    allowFastRotation: 7,
    enableContactRecycling: 8,
    isEnabled: 9,
    isAwake: 10,
} as const;

export function bodyGetProperty(world: WorldState, body: number, property: number): number {
    return kernel(world.ecsState).bodyGetProperty(world.worldId, body, property);
}
export function bodySetProperty(
    world: WorldState,
    body: number,
    property: number,
    value: number,
): void {
    if (world.locked) return;
    world.locked = true;
    kernel(world.ecsState).bodySetProperty(world.worldId, body, property, f32(value));
    world.bodyStore.refreshViews();
    world.locked = false;
}
export function bodySetType(world: WorldState, body: number, type: BodyType): void {
    world.locked = true;
    world.broadPhase.store.initialize();
    kernel(world.ecsState).bodySetType(world.worldId, body, type);
    world.broadPhase.store.refreshViews();
    world.locked = false;
}
export function bodySetAwake(world: WorldState, body: number, awake: boolean): void {
    world.locked = true;
    kernel(world.ecsState).bodySetAwake(world.worldId, body, awake);
    world.locked = false;
}
export function createBody(world: WorldState, def: BodyDef): number {
    world.locked = true;
    const flags =
        Number(def.motionLocks.linearX) |
        (Number(def.motionLocks.linearY) << 1) |
        (Number(def.motionLocks.linearZ) << 2) |
        (Number(def.motionLocks.angularX) << 3) |
        (Number(def.motionLocks.angularY) << 4) |
        (Number(def.motionLocks.angularZ) << 5) |
        (Number(def.isBullet) << 7) |
        (Number(def.allowFastRotation) << 10) |
        (Number(def.enableSleep) << 13) |
        (Number(def.enableContactRecycling) << 14);
    const p = def.position,
        q = def.rotation,
        v = def.linearVelocity,
        w = def.angularVelocity;
    const bodyId = kernel(world.ecsState).bodyCreateSim(
        world.worldId,
        def.type,
        flags,
        def.isAwake,
        def.isEnabled,
        def.sleepThreshold,
        p.x,
        p.y,
        p.z,
        q.v.x,
        q.v.y,
        q.v.z,
        q.s,
        v.x,
        v.y,
        v.z,
        w.x,
        w.y,
        w.z,
        def.linearDamping,
        def.angularDamping,
        def.gravityScale,
    );
    world.bodyStore.refreshViews();
    world.bodyNames[bodyId] = def.name ? def.name.slice(0, BODY_NAME_LENGTH) : "";
    world.bodyUserData[bodyId] = def.userData;
    world.locked = false;
    return bodyId;
}
export function destroyBody(world: WorldState, body: number): void {
    world.locked = true;
    let key = bodyField(world, body, BodyField.headJointKey);
    while (key !== NULL_INDEX) {
        const joint = key >> 1;
        const edge = key & 1;
        key = jointField(world, joint, JointField.nextKeyA + 3 * edge);
        world.jointUserData[joint] = null;
    }
    let shape = bodyField(world, body, BodyField.headShapeId);
    while (shape !== NULL_INDEX) {
        world.shapeUserData[shape] = undefined;
        world.shapeNames[shape] = "";
        shape = shapeField(world, shape, ShapeField.nextShapeId);
    }
    kernel(world.ecsState).bodyDestroyWorld(world.worldId, body);
    releaseGeometryIdentities(world);
    world.bodyUserData[body] = undefined;
    world.bodyNames[body] = "";
    world.locked = false;
}
export function releaseGeometryIdentities(world: WorldState): void {
    world.geometryIdentityValues.forEach(releaseGeometryIdentity, world);
}
function releaseGeometryIdentity(this: WorldState, _value: unknown, identity: number): void {
    const k = kernel(this.ecsState);
    if (
        k.geometryDatabaseLookup(this.worldId, 4, identity) === 0 &&
        k.geometryDatabaseLookup(this.worldId, 2, identity) === 0 &&
        k.geometryDatabaseLookup(this.worldId, 1, identity) === 0
    )
        this.geometryIdentityValues.delete(identity);
}
export function bodyDisable(world: WorldState, body: number): void {
    if (world.locked) return;
    world.locked = true;
    kernel(world.ecsState).bodyDisable(world.worldId, body);
    world.locked = false;
}
export function bodyEnable(world: WorldState, body: number): void {
    if (world.locked) return;
    world.locked = true;
    kernel(world.ecsState).bodyEnable(world.worldId, body);
    world.broadPhase.store.refreshViews();
    world.locked = false;
}
export function updateBodyMassData(world: WorldState, body: number): void {
    kernel(world.ecsState).bodyUpdateMass(world.worldId, body);
}
export function getMassData(world: WorldState, body: number): MassData {
    return {
        mass: bodyField(world, body, BodyField.mass),
        center: readSimLocalCenter(world, getBodySim(world, body), { x: 0, y: 0, z: 0 }),
        inertia: bodyInertia(world, body, {
            cx: { x: 0, y: 0, z: 0 },
            cy: { x: 0, y: 0, z: 0 },
            cz: { x: 0, y: 0, z: 0 },
        }),
    };
}
