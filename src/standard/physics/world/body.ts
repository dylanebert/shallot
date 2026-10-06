// body.c bindings (Box3D, Erin Catto, MIT).
import { BODY_NAME_LENGTH } from "../common/constants";
import { NULL_INDEX } from "../common/array";
import type { EntityId } from "../common/ids";
import {
    mat3,
    type Mat3,
    type Pos,
    type Quat,
    type Vec3,
    type WorldTransform,
} from "../common/math";
import type { BodyDef, BodyType } from "../common/types";
import { readSimLocalCenter, readSimTransform } from "../kernel/bodycolumns";
import { BodyField, bodyField, bodyInertia } from "../kernel/bodyrecords";
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
export const FIXED_ROTATION =
    BodyFlags.lockAngularX | BodyFlags.lockAngularY | BodyFlags.lockAngularZ;
export const BODY_TRANSIENT_FLAGS =
    BodyFlags.isFast | BodyFlags.isSpeedCapped | BodyFlags.hadTimeOfImpact;
export type BodyState = {
    linearVelocity: Vec3;
    angularVelocity: Vec3;
    deltaPosition: Vec3;
    deltaRotation: Quat;
    flags: number;
};
export function identityBodyState(): BodyState {
    return {
        linearVelocity: { x: 0, y: 0, z: 0 },
        angularVelocity: { x: 0, y: 0, z: 0 },
        deltaPosition: { x: 0, y: 0, z: 0 },
        deltaRotation: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        flags: 0,
    };
}
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
    maxAngularVelocity: number;
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
export function syncBodyFlags(world: WorldState, body: number): void {
    kernel(world.ecsState).bodySyncFlags(world.worldId, body);
}
export function bodySetLinearVelocity(world: WorldState, body: number, v: Vec3): void {
    kernel(world.ecsState).bodyVelocitySet(world.worldId, body, false, v.x, v.y, v.z);
}
export function bodySetAngularVelocity(world: WorldState, body: number, v: Vec3): void {
    kernel(world.ecsState).bodyVelocitySet(world.worldId, body, true, v.x, v.y, v.z);
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
        q.v.x,
        q.v.y,
        q.v.z,
        q.s,
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
        force.x,
        force.y,
        force.z,
        point.x,
        point.y,
        point.z,
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
        force.x,
        force.y,
        force.z,
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
        torque.x,
        torque.y,
        torque.z,
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
        impulse.x,
        impulse.y,
        impulse.z,
        point.x,
        point.y,
        point.z,
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
        impulse.x,
        impulse.y,
        impulse.z,
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
        impulse.x,
        impulse.y,
        impulse.z,
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
        position.x,
        position.y,
        position.z,
        rotation.v.x,
        rotation.v.y,
        rotation.v.z,
        rotation.s,
    );
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
export function wakeBody(world: WorldState, body: number): boolean {
    return !!kernel(world.ecsState).bodyWakeWorld(world.worldId, body);
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
    world.geometryIdentityValues.forEach(releaseGeometryIdentity, world);
    world.bodyUserData[body] = undefined;
    world.bodyNames[body] = "";
    world.locked = false;
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
    kernel(world.ecsState).bodyEnable(world.worldId, body);
    world.broadPhase.store.refreshViews();
}
export function updateBodyMassData(world: WorldState, body: number): void {
    kernel(world.ecsState).bodyUpdateMass(world.worldId, body);
}
export function getMassData(world: WorldState, body: number): MassData {
    return {
        mass: bodyField(world, body, BodyField.mass),
        center: readSimLocalCenter(world, getBodySim(world, body), { x: 0, y: 0, z: 0 }),
        inertia: bodyInertia(world, body, mat3.zero()),
    };
}
