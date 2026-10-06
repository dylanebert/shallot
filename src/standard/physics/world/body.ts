import { ContactField, contactField } from "../collision/contact";
import {
    bodySimSlot,
    readSimInvInertiaLocal,
    readSimInvInertiaWorld,
    readSimMaxExtent,
    setSimField,
    setStateField,
    simBodyId,
    simFlags,
    simInvMass,
    simMinExtent,
} from "../kernel/bodycolumns";
import { bodyType, setBodyType, shapeSensorIndex } from "../kernel/filtercolumns";
import {
    createSolverSet,
    setBodyCount,
    setBodyPush,
    setBodyRemove,
} from "../kernel/solversetcolumns";
// Rigid body lifecycle and the 3-way body split, ported from Box3D's body.c (Erin Catto, MIT).
// A body is stored as three records: the cold organizational handle (b3Body, in world.bodies,
// id-indexed), the hot simulation payload (b3BodySim, in a solver set's bodySims column), and the
// solver velocity/delta state (b3BodyState, only in the awake set's bodyStates column). Static and
// sleeping bodies have a sim but no state.
//
// fround discipline (every f32 result rounds through Math.fround, keeping bit-exact f32 parity). This file holds the types + accessors; the
// create/destroy/setType/mass machinery is appended below.

import { moveProxy as bpMoveProxy } from "../collision/broadphase";
import { destroyContact, writeBodySimIndex } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { BODY_NAME_LENGTH, HUGE, SetType, SPECULATIVE_DISTANCE } from "../common/constants";
import type { EntityId } from "../common/ids";
import {
    aabb,
    FLT_MIN,
    f32,
    froundConfig,
    type Mat3,
    mat3,
    minf,
    type Pos,
    type Quat,
    quat,
    steiner,
    transformWorldPoint,
    type Vec3,
    vec3,
    type WorldTransform,
} from "../common/math";
import { type BodyDef, BodyType, ShapeType } from "../common/types";
import {
    addSimForce,
    addSimForceTorque,
    addSimTorque,
    readSimCenter,
    readSimLocalCenter,
    readSimTransform,
    readStateAngularVelocity,
    readStateLinearVelocity,
    reserveBodies,
    residentPush,
    residentRemove,
    writeSimRotation0,
    writeSimTransform,
} from "../kernel/bodycolumns";
import {
    addIslandBody,
    islandArrayCount,
    islandField,
    removeIslandBody,
} from "../kernel/islandcolumns";
import { kernel } from "../kernel/kernel";
import {
    destroyShapeSlot,
    readFatAabb,
    syncBodyQuery,
    writeFatAabb,
    writeTightAabb,
} from "../kernel/shapecolumns";
import type { MassData } from "../shapes/geometry";
import {
    computeFatShapeAABBOut,
    computeShapeExtent,
    computeShapeMass,
    createShapeProxy,
    destroyShapeAllocations,
    destroyShapeProxy,
} from "../shapes/shape";
import { destroyJointInternal } from "../solver/joint";
import { createIsland, destroyIsland, linkJoint, splitIsland, unlinkJoint } from "./island";
import { destroySensor } from "./sensor";
import {
    destroySolverSet,
    transferBody,
    transferJoint,
    trySleepIsland,
    wakeSolverSet,
} from "./solverset";
import type { WorldState } from "./world";

/** Body flags (b3BodyFlags). Lock bits, transient per-step markers, and the dynamic/sleep bits. */
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
    // The solver may write to this body (dynamic). Kept off kinematic bodies to avoid cross-worker
    // cache thrash on shared state.
    dynamicFlag: 0x00001000,
    enableSleep: 0x00002000,
    enableContactRecycling: 0x00004000,
} as const;

/** The three angular lock bits: set together they mean fixed rotation (b3_fixedRotation). */
export const FIXED_ROTATION =
    BodyFlags.lockAngularX | BodyFlags.lockAngularY | BodyFlags.lockAngularZ;

/** Flags reset on every solver-set transfer (b3_bodyTransientFlags). */
export const BODY_TRANSIENT_FLAGS =
    BodyFlags.isFast | BodyFlags.isSpeedCapped | BodyFlags.hadTimeOfImpact;

/**
 * Solver velocity/delta state (b3BodyState). Only awake dynamic/kinematic bodies have one. Delta
 * position/rotation keep the solver in float precision far from the origin; static bodies use the
 * identity state so the solver never writes them.
 */
export type BodyState = {
    linearVelocity: Vec3;
    angularVelocity: Vec3;
    deltaPosition: Vec3;
    deltaRotation: Quat;
    flags: number;
};

/** @returns the canonical zero/identity body state (b3_identityBodyState). */
export function identityBodyState(): BodyState {
    return {
        linearVelocity: { x: 0, y: 0, z: 0 },
        angularVelocity: { x: 0, y: 0, z: 0 },
        deltaPosition: { x: 0, y: 0, z: 0 },
        deltaRotation: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        flags: 0,
    };
}

/** Body integration + collision payload (b3BodySim). Lives in every set's bodySims column. */
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

/** Cold body handle, not touched by the solver (b3Body). Indexed by body id in world.bodies. */
export type Body = {
    userData: unknown;
    setIndex: number;
    localIndex: number;
    // [31: contactId | 1: edgeIndex]
    headContactKey: number;
    contactCount: number;
    headShapeId: number;
    shapeCount: number;
    headChainId: number;
    // [31: jointId | 1: edgeIndex]
    headJointKey: number;
    jointCount: number;
    islandId: number;
    islandIndex: number;
    sleepThreshold: number;
    sleepTime: number;
    mass: number;
    inertia: Mat3;
    bodyMoveIndex: number;
    id: number;
    flags: number;
    name: string;
};

/** @returns the sim slot addressed by the body record (b3GetBodySim). */
export function getBodySim(_world: WorldState, body: Body): number {
    return -body.id - 1;
}

/** @returns the awake state's local index, or null (b3GetBodyState). */
export function getBodyState(_world: WorldState, body: Body): number | null {
    if (body.setIndex === SetType.Awake) {
        return body.localIndex;
    }
    return null;
}

/** @returns a public body id for a raw body index (b3MakeBodyId). */
export function makeBodyId(world: WorldState, bodyId: number): EntityId {
    if (bodyId === NULL_INDEX) {
        return { index1: 0, world0: 0, generation: 0 };
    }
    return {
        index1: bodyId + 1,
        world0: world.worldId,
        generation: kernel(world.ecsState).bodyGeneration(world.worldId, bodyId),
    };
}

/** Copy the body's world transform into caller-owned output (b3GetBodyTransformQuick). */
export function readBodyTransform(
    world: WorldState,
    body: Body,
    out: WorldTransform,
): WorldTransform {
    return readSimTransform(world, getBodySim(world, body), out);
}

/**
 * Copy a body's persistent (non-transient) flags into its sim and, when awake, its state
 * (b3SyncBodyFlags). Called after any change to body.flags that the solver reads (type, locks, bullet).
 */
export function syncBodyFlags(world: WorldState, body: Body): void {
    const flags = body.flags & ~BODY_TRANSIENT_FLAGS;
    setSimField(world, getBodySim(world, body), "flags", flags);
    const state = getBodyState(world, body);
    if (state !== null) setStateField(world, state, "flags", flags);
}

/** Set a body's linear velocity, waking it when the velocity is nonzero (b3Body_SetLinearVelocity). */
export function bodySetLinearVelocity(world: WorldState, body: Body, linearVelocity: Vec3): void {
    const wake = kernel(world.ecsState).bodyVelocityWake(
        world.worldId,
        body.id,
        linearVelocity.x,
        linearVelocity.y,
        linearVelocity.z,
    );
    if (wake < 0) return;
    if (wake > 0) wakeBody(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;
    setStateField(world, state, "linearVelocity", linearVelocity);
}

/**
 * Set a body's angular velocity, masking out locked angular axes and waking it when the result is
 * nonzero (b3Body_SetAngularVelocity).
 */
export function bodySetAngularVelocity(world: WorldState, body: Body, angularVelocity: Vec3): void {
    if (bodyType(world, body.id) === BodyType.Static) return;
    const w: Vec3 = {
        x: body.flags & BodyFlags.lockAngularX ? 0 : angularVelocity.x,
        y: body.flags & BodyFlags.lockAngularY ? 0 : angularVelocity.y,
        z: body.flags & BodyFlags.lockAngularZ ? 0 : angularVelocity.z,
    };
    if (vec3.lengthSq(w) !== 0) wakeBody(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;
    setStateField(world, state, "angularVelocity", w);
}

const targetCenter = vec3.zero();
const targetLocalCenter = vec3.zero();
const targetPose: WorldTransform = { p: vec3.zero(), q: quat.identity() };
const targetExtent = vec3.zero();
const targetLinear = vec3.zero();
const targetAngular = vec3.zero();
const targetDifference = vec3.zero();
const targetConjugate = vec3.zero();

/**
 * Drive a (typically kinematic) body toward a target transform over one time step by setting the
 * linear and angular velocity that reaches it (b3Body_SetTargetTransform). Used to animate a kinematic
 * pusher along a path.
 */
export function bodySetTargetTransform(
    world: WorldState,
    body: Body,
    target: WorldTransform,
    timeStep: number,
    wake: boolean,
): void {
    if (body.setIndex === SetType.Disabled) return;
    if (bodyType(world, body.id) === BodyType.Static || timeStep <= 0) return;
    if (body.setIndex !== SetType.Awake && wake === false) return;

    const sim = getBodySim(world, body);

    // Linear velocity from the world-space center difference, demoted to f32.
    readSimCenter(world, sim, targetCenter);
    readSimLocalCenter(world, sim, targetLocalCenter);
    quat.rotateOut(target.q, targetLocalCenter, targetLinear);
    vec3.addOut(targetLinear, target.p, targetLinear);
    vec3.subOut(targetLinear, targetCenter, targetLinear);
    const invTimeStep = f32(1 / timeStep);
    vec3.scaleOut(invTimeStep, targetLinear, targetLinear);

    // Angular velocity: w = 2 * (q2 - q1) * conj(q1) / dt, using the shortest-arc quaternion.
    const q1 = readSimTransform(world, sim, targetPose).q;
    const q2 = target.q;
    const xx = f32(q1.v.x * q2.v.x);
    const yy = f32(q1.v.y * q2.v.y);
    const zz = f32(q1.v.z * q2.v.z);
    const ss = f32(q1.s * q2.s);
    const sign = f32(f32(f32(xx + yy) + zz) + ss) < 0 ? -1 : 1;
    targetDifference.x = f32(sign * q2.v.x - q1.v.x);
    targetDifference.y = f32(sign * q2.v.y - q1.v.y);
    targetDifference.z = f32(sign * q2.v.z - q1.v.z);
    const ds = f32(sign * q2.s - q1.s);
    targetConjugate.x = -q1.v.x;
    targetConjugate.y = -q1.v.y;
    targetConjugate.z = -q1.v.z;
    vec3.crossOut(targetDifference, targetConjugate, targetAngular);
    targetAngular.x = f32(
        f32(targetAngular.x + f32(ds * targetConjugate.x)) + f32(q1.s * targetDifference.x),
    );
    targetAngular.y = f32(
        f32(targetAngular.y + f32(ds * targetConjugate.y)) + f32(q1.s * targetDifference.y),
    );
    targetAngular.z = f32(
        f32(targetAngular.z + f32(ds * targetConjugate.z)) + f32(q1.s * targetDifference.z),
    );
    vec3.scaleOut(f32(2 * invTimeStep), targetAngular, targetAngular);

    // If the body is asleep, wake only when the target motion exceeds the sleep threshold.
    if (body.setIndex !== SetType.Awake) {
        readSimMaxExtent(world, sim, targetExtent);
        targetExtent.x = f32(targetAngular.x * targetExtent.x);
        targetExtent.y = f32(targetAngular.y * targetExtent.y);
        targetExtent.z = f32(targetAngular.z * targetExtent.z);
        const maxVelocity = f32(vec3.length(targetLinear) + vec3.length(targetExtent));
        if (maxVelocity < body.sleepThreshold) return;
        wakeBody(world, body);
    }

    const state = getBodyState(world, body);
    if (state === null) return;
    setStateField(world, state, "linearVelocity", targetLinear);
    setStateField(world, state, "angularVelocity", targetAngular);
}

// --- forces + impulses -----------------------------------------------------------------------

/**
 * Accumulate a world-space force at a world-space point, waking the body when `wake` (b3Body_ApplyForce).
 * The force integrates over the next step; an off-center point also produces a torque.
 */
export function bodyApplyForce(
    world: WorldState,
    body: Body,
    force: Vec3,
    point: Pos,
    wake: boolean,
): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    addSimForce(world, sim, force);
    addSimForceTorque(world, sim, force, point);
}

/** Accumulate a world-space force at the center of mass (b3Body_ApplyForceToCenter). No torque. */
export function bodyApplyForceToCenter(
    world: WorldState,
    body: Body,
    force: Vec3,
    wake: boolean,
): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    addSimForce(world, sim, force);
}

/** Accumulate a torque about the center of mass (b3Body_ApplyTorque). */
export function bodyApplyTorque(world: WorldState, body: Body, torque: Vec3, wake: boolean): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    addSimTorque(world, sim, torque);
}

// Clamp a linear velocity to the world's max linear speed (the shared tail of the impulse setters).
function clampLinearSpeed(world: WorldState, v: Vec3): Vec3 {
    const maxLinearSpeed = world.maxLinearSpeed;
    const lengthSq = vec3.lengthSq(v);
    if (lengthSq > f32(maxLinearSpeed * maxLinearSpeed)) {
        if (lengthSq > f32(1000 * FLT_MIN)) {
            vec3.scaleOut(f32(1 / f32(Math.sqrt(lengthSq))), v, v);
        } else {
            v.x = 0;
            v.y = 0;
            v.z = 0;
        }
        vec3.scaleOut(maxLinearSpeed, v, v);
    }
    return v;
}

const impulseScratch = {
    linear: vec3.zero(),
    angular: vec3.zero(),
    center: vec3.zero(),
    r: vec3.zero(),
    mrn: vec3.zero(),
    inertia: mat3.zero(),
};

/**
 * Apply an instantaneous world-space impulse at a world-space point, changing velocity immediately
 * (b3Body_ApplyLinearImpulse). An off-center point also changes angular velocity. Linear speed is
 * clamped to the world's max. Caller-owned scratch is borrowed for this call, never retained by the body.
 */
export function bodyApplyLinearImpulse(
    world: WorldState,
    body: Body,
    impulse: Vec3,
    point: Pos,
    wake: boolean,
    scratch = impulseScratch,
): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;

    const v = readStateLinearVelocity(world, state, scratch.linear);
    vec3.mulAddOut(v, simInvMass(world, sim), impulse, v);
    const lengthSq = vec3.lengthSq(v);
    const max = world.maxLinearSpeed;
    if (lengthSq > f32(max * max)) {
        if (lengthSq > f32(1000 * FLT_MIN)) {
            const scale = f32(1 / f32(Math.sqrt(lengthSq)));
            v.x = f32(max * f32(v.x * scale));
            v.y = f32(max * f32(v.y * scale));
            v.z = f32(max * f32(v.z * scale));
        } else {
            v.x = 0;
            v.y = 0;
            v.z = 0;
        }
    }
    setStateField(world, state, "linearVelocity", v);

    readSimCenter(world, sim, scratch.center);
    vec3.subOut(point, scratch.center, scratch.r);
    vec3.crossOut(scratch.r, impulse, scratch.r);
    readSimInvInertiaWorld(world, sim, scratch.inertia);
    mat3.mulVOut(scratch.inertia, scratch.r, scratch.mrn);
    const angular = readStateAngularVelocity(world, state, scratch.angular);
    vec3.addOut(angular, scratch.mrn, angular);
    setStateField(world, state, "angularVelocity", angular);
}

/** Apply an instantaneous impulse at the center of mass (b3Body_ApplyLinearImpulseToCenter). */
export function bodyApplyLinearImpulseToCenter(
    world: WorldState,
    body: Body,
    impulse: Vec3,
    wake: boolean,
): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;

    const v = readStateLinearVelocity(world, state, impulseScratch.linear);
    vec3.mulAddOut(v, simInvMass(world, sim), impulse, v);
    setStateField(world, state, "linearVelocity", clampLinearSpeed(world, v));
}

const angularImpulsePose: WorldTransform = { p: vec3.zero(), q: quat.identity() };
const angularImpulseInertia = mat3.zero();
const angularImpulseLocal = vec3.zero();
const angularImpulseDelta = vec3.zero();
const angularImpulseVelocity = vec3.zero();

/** Apply an instantaneous angular impulse, changing angular velocity immediately (b3Body_ApplyAngularImpulse). */
export function bodyApplyAngularImpulse(
    world: WorldState,
    body: Body,
    impulse: Vec3,
    wake: boolean,
): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;

    // Rotate the impulse into the body frame, apply the local inverse inertia, rotate back.
    const q = readSimTransform(world, sim, angularImpulsePose).q;
    quat.invRotateOut(q, impulse, angularImpulseLocal);
    readSimInvInertiaLocal(world, sim, angularImpulseInertia);
    mat3.mulVOut(angularImpulseInertia, angularImpulseLocal, angularImpulseDelta);
    quat.rotateOut(q, angularImpulseDelta, angularImpulseDelta);
    readStateAngularVelocity(world, state, angularImpulseVelocity);
    vec3.addOut(angularImpulseVelocity, angularImpulseDelta, angularImpulseVelocity);
    setStateField(world, state, "angularVelocity", angularImpulseVelocity);
}

// --- transform + type + awake ----------------------------------------------------------------

// Registers for bodySetTransform's column-view path; never live across calls.
const setPose: WorldTransform = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
const setBox = { lowerBound: vec3.zero(), upperBound: vec3.zero() };
const setFat = { lowerBound: vec3.zero(), upperBound: vec3.zero() };
const setCenter: Vec3 = { x: 0, y: 0, z: 0 };
const setLocalCenter: Vec3 = { x: 0, y: 0, z: 0 };
const setRotation = mat3.zero();
const setRotationT = mat3.zero();
const setInertiaTmp = mat3.zero();
const setInvILocal = mat3.zero();
const setInvIWorld = mat3.zero();

/**
 * Teleport a body to a new pose (b3Body_SetTransform), recomputing its center of mass, world inverse
 * inertia, and shape broadphase proxies. Does not change velocity; the body keeps moving from the new
 * pose. Prefer setTargetTransform for kinematic path animation.
 */
export function bodySetTransform(
    world: WorldState,
    body: Body,
    position: Pos,
    rotation: Quat,
): void {
    const sim = getBodySim(world, body);

    // Read back the f32 column write before deriving the center and inertia.
    const transform = setPose;
    transform.p.x = position.x;
    transform.p.y = position.y;
    transform.p.z = position.z;
    transform.q.v.x = rotation.v.x;
    transform.q.v.y = rotation.v.y;
    transform.q.v.z = rotation.v.z;
    transform.q.s = rotation.s;
    writeSimTransform(world, sim, transform);
    readSimTransform(world, sim, transform);

    readSimLocalCenter(world, sim, setLocalCenter);
    quat.rotateOut(transform.q, setLocalCenter, setCenter);
    vec3.addOut(setCenter, transform.p, setCenter);

    mat3.fromQuatOut(transform.q, setRotation);
    readSimInvInertiaLocal(world, sim, setInvILocal);
    mat3.mulOut(setRotation, setInvILocal, setInertiaTmp);
    mat3.transposeOut(setRotation, setRotationT);
    mat3.mulOut(setInertiaTmp, setRotationT, setInvIWorld);

    setSimField(world, sim, "center", setCenter);
    setSimField(world, sim, "invInertiaWorld", setInvIWorld);
    writeSimRotation0(world, sim, transform.q);
    setSimField(world, sim, "center0", setCenter);

    const broadPhase = world.broadPhase;
    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        const box = computeFatShapeAABBOut(shape, transform, SPECULATIVE_DISTANCE, setBox);
        world.shapeStore.refreshViews();
        writeTightAabb(world.shapeStore.shapeF, shape.id, box);
        const fatAABB = readFatAabb(world, shape.id, setFat);
        if (aabb.contains(fatAABB, box) === false) {
            const margin = shape.aabbMargin;
            fatAABB.lowerBound.x = f32(box.lowerBound.x - margin);
            fatAABB.lowerBound.y = f32(box.lowerBound.y - margin);
            fatAABB.lowerBound.z = f32(box.lowerBound.z - margin);
            fatAABB.upperBound.x = f32(box.upperBound.x + margin);
            fatAABB.upperBound.y = f32(box.upperBound.y + margin);
            fatAABB.upperBound.z = f32(box.upperBound.z + margin);
            writeFatAabb(world, shape.id, fatAABB);

            // The body could be disabled, in which case it has no proxy.
            if (shape.proxyKey !== NULL_INDEX) {
                bpMoveProxy(broadPhase, shape.proxyKey, fatAABB);
            }
        }

        shapeId = shape.nextShapeId;
    }
    syncBodyQuery(world, body);
}

/**
 * Change a body's type (static / kinematic / dynamic), moving it between solver sets and rebuilding its
 * island participation, contacts, joints, and broadphase proxies (b3Body_SetType). Not supported for
 * bodies carrying a compound or height-field shape when the target type is non-static.
 */
export function bodySetType(world: WorldState, body: Body, type: BodyType): void {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    world.locked = true;

    const originalType = bodyType(world, body.id);
    if (originalType === type) {
        world.locked = false;
        return;
    }

    if (type !== BodyType.Static) {
        let shapeId = body.headShapeId;
        while (shapeId !== NULL_INDEX) {
            const shape = world.shapes[shapeId];
            if (shape.type === ShapeType.Compound || shape.type === ShapeType.HeightField) {
                // Setting the body type is not supported for bodies with compound/height-field shapes.
                // (Deviation: the C returns here without unlocking — a lock leak; the port unlocks.)
                world.locked = false;
                return;
            }
            shapeId = shape.nextShapeId;
        }
    }

    // Disabled bodies don't change solver sets or islands when they change type.
    if (body.setIndex === SetType.Disabled) {
        setBodyType(world, body.id, type);
        if (type === BodyType.Dynamic) body.flags |= BodyFlags.dynamicFlag;
        else body.flags &= ~BodyFlags.dynamicFlag;
        syncBodyFlags(world, body);
        updateBodyMassData(world, body);
        world.locked = false;
        return;
    }

    // Stage 2: destroy all contacts but don't wake bodies (we don't need to).
    destroyBodyContacts(world, body, false);

    // Stage 3: wake this body (a no-op for a static body).
    wakeBody(world, body);

    // Stage 4: move all live joints to the static set so they can re-acquire consistent colors below.
    const staticSet = SetType.Static;
    let jointKey = body.headJointKey;
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;

        if (joint.setIndex === SetType.Disabled) continue;

        // Wake attached bodies: wakeBody above does not wake bodies attached to a static body.
        wakeBody(world, world.bodies[joint.edges[0].bodyId]);
        wakeBody(world, world.bodies[joint.edges[1].bodyId]);

        unlinkJoint(world, joint);
        transferJoint(world, staticSet, joint.setIndex, joint);
    }

    // Stage 5: change the type and transfer the body between solver sets.
    setBodyType(world, body.id, type);
    if (type === BodyType.Dynamic) body.flags |= BodyFlags.dynamicFlag;
    else body.flags &= ~BodyFlags.dynamicFlag;

    const awakeSet = SetType.Awake;
    const sourceSet = body.setIndex;
    const targetSet = type === BodyType.Static ? staticSet : awakeSet;
    transferBody(world, targetSet, sourceSet, body);

    // Stage 6: update island participation.
    if (originalType === BodyType.Static) {
        createIslandForBody(world, SetType.Awake, body);
    } else if (type === BodyType.Static) {
        removeBodyFromIsland(world, body);
    }

    // Stage 7: transfer joints back to the awake set when either attached body is now dynamic.
    jointKey = body.headJointKey;
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;

        if (joint.setIndex === SetType.Disabled) continue;

        const bodyA = world.bodies[joint.edges[0].bodyId];
        const bodyB = world.bodies[joint.edges[1].bodyId];
        if (
            bodyType(world, bodyA.id) === BodyType.Dynamic ||
            bodyType(world, bodyB.id) === BodyType.Dynamic
        ) {
            transferJoint(world, awakeSet, staticSet, joint);
        }
    }

    // Recreate shape proxies in the broadphase against the new body type.
    const transform = readBodyTransform(world, body, bodyPoseScratch1);
    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;
        destroyShapeProxy(shape, world.broadPhase);
        createShapeProxy(shape, world.broadPhase, type, transform, true);
    }

    // Relink joints where at least one attached body is dynamic and enabled.
    jointKey = body.headJointKey;
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;

        const otherBodyId = joint.edges[edgeIndex ^ 1].bodyId;
        const otherBody = world.bodies[otherBodyId];
        if (otherBody.setIndex === SetType.Disabled) continue;
        if (
            bodyType(world, body.id) !== BodyType.Dynamic &&
            bodyType(world, otherBody.id) !== BodyType.Dynamic
        )
            continue;

        linkJoint(world, joint);
    }

    syncBodyFlags(world, body);
    updateBodyMassData(world, body);

    world.locked = false;
}

/**
 * Force a body awake or asleep (b3Body_SetAwake). Sleeping puts the body's whole island to sleep,
 * splitting it first if pending constraint removals left it separable.
 */
export function bodySetAwake(world: WorldState, body: Body, awake: boolean): void {
    world.locked = true;

    if (awake && body.setIndex >= SetType.FirstSleeping) {
        wakeBody(world, body);
    } else if (awake === false && body.setIndex === SetType.Awake) {
        if (islandField(world, body.islandId, 3) > 0) {
            // Must split the island before sleeping. This is expensive.
            splitIsland(world, body.islandId);
        }
        trySleepIsland(world, body.islandId);
    }

    world.locked = false;
}

// --- lifecycle -------------------------------------------------------------------------------

function emptyBodySim(): BodySim {
    return {
        transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        center: { x: 0, y: 0, z: 0 },
        rotation0: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        center0: { x: 0, y: 0, z: 0 },
        localCenter: { x: 0, y: 0, z: 0 },
        force: { x: 0, y: 0, z: 0 },
        torque: { x: 0, y: 0, z: 0 },
        invMass: 0,
        invInertiaLocal: mat3.zero(),
        invInertiaWorld: mat3.zero(),
        minExtent: 0,
        maxExtent: { x: 0, y: 0, z: 0 },
        maxAngularVelocity: 0,
        linearDamping: 0,
        angularDamping: 0,
        gravityScale: 0,
        bodyId: NULL_INDEX,
        flags: 0,
    };
}

function emptyBody(): Body {
    return {
        userData: undefined,
        setIndex: NULL_INDEX,
        localIndex: NULL_INDEX,
        headContactKey: NULL_INDEX,
        contactCount: 0,
        headShapeId: NULL_INDEX,
        shapeCount: 0,
        headChainId: NULL_INDEX,
        headJointKey: NULL_INDEX,
        jointCount: 0,
        islandId: NULL_INDEX,
        islandIndex: NULL_INDEX,
        sleepThreshold: 0,
        sleepTime: 0,
        mass: 0,
        inertia: mat3.zero(),
        bodyMoveIndex: NULL_INDEX,
        id: NULL_INDEX,
        flags: 0,
        name: "",
    };
}

function createIslandForBody(world: WorldState, setIndex: number, body: Body): void {
    const island = createIsland(world, setIndex);
    addIslandBody(world, island, body.id);
}

function removeBodyFromIsland(world: WorldState, body: Body): void {
    if (body.islandId === NULL_INDEX) {
        return;
    }

    const islandId = body.islandId;
    removeIslandBody(world, islandId, body.islandIndex);
    if (islandArrayCount(world, islandId, 0) === 0) {
        destroyIsland(world, islandId);
    }

    body.islandId = NULL_INDEX;
    body.islandIndex = NULL_INDEX;
}

function destroyBodyContacts(world: WorldState, body: Body, wakeBodies: boolean): void {
    let edgeKey = body.headContactKey;
    while (edgeKey !== NULL_INDEX) {
        const contactId = edgeKey >> 1;
        const edgeIndex = edgeKey & 1;
        const contact = contactId;
        edgeKey = contactField(world, contact, ContactField.nextKeyA + 3 * edgeIndex);
        destroyContact(world, contact, wakeBodies);
    }
}

/** Create a body from a definition (b3CreateBody). @returns the raw body id. */
export function createBody(world: WorldState, def: BodyDef): number {
    world.locked = true;

    // Round every user float to f32 once at ingress (position/rotation/velocity/damping/…); the C def is
    // f32, so an unrounded f64 scalar would reach the solver and break bit-exact parity.
    def = froundConfig(def);

    const isAwake = (def.isAwake || def.enableSleep === false) && def.isEnabled;

    // determine the solver set
    let setId: number;
    if (def.isEnabled === false) {
        setId = SetType.Disabled;
    } else if (def.type === BodyType.Static) {
        setId = SetType.Static;
    } else if (isAwake) {
        setId = SetType.Awake;
    } else {
        // new set for a sleeping body in its own island
        setId = createSolverSet(world);
    }

    // The cold record remains the world-local authoring/handle bridge; the lifecycle fields are
    // registered in the kernel record columns before any solver path can observe the body.
    const bodyId = kernel(world.ecsState).bodyCreate(world.worldId);

    let lockFlags = 0;
    lockFlags |= def.motionLocks.linearX ? BodyFlags.lockLinearX : 0;
    lockFlags |= def.motionLocks.linearY ? BodyFlags.lockLinearY : 0;
    lockFlags |= def.motionLocks.linearZ ? BodyFlags.lockLinearZ : 0;
    lockFlags |= def.motionLocks.angularX ? BodyFlags.lockAngularX : 0;
    lockFlags |= def.motionLocks.angularY ? BodyFlags.lockAngularY : 0;
    lockFlags |= def.motionLocks.angularZ ? BodyFlags.lockAngularZ : 0;

    const set = setId;
    const bodySim = emptyBodySim();
    bodySim.transform.p = { ...def.position };
    bodySim.transform.q = { v: { ...def.rotation.v }, s: def.rotation.s };
    bodySim.center = { ...def.position };
    bodySim.rotation0.v = { ...def.rotation.v };
    bodySim.rotation0.s = def.rotation.s;
    bodySim.center0 = { ...def.position };
    bodySim.minExtent = HUGE;
    bodySim.linearDamping = def.linearDamping;
    bodySim.angularDamping = def.angularDamping;
    bodySim.gravityScale = def.gravityScale;
    bodySim.bodyId = bodyId;
    let flags = lockFlags;
    flags |= def.isBullet ? BodyFlags.isBullet : 0;
    flags |= def.allowFastRotation ? BodyFlags.allowFastRotation : 0;
    flags |= def.type === BodyType.Dynamic ? BodyFlags.dynamicFlag : 0;
    flags |= def.enableSleep ? BodyFlags.enableSleep : 0;
    flags |= def.enableContactRecycling ? BodyFlags.enableContactRecycling : 0;
    bodySim.flags = flags;
    // Defer the awake append until its state columns have been reserved.
    if (setId !== SetType.Awake) setBodyPush(world, set, bodySim);

    // The awake body's solver state is column-resident; its initial values are written into the region
    // once it's sized to the new total-body high-water (below, after the body id is registered).
    let awakeState: BodyState | null = null;
    if (setId === SetType.Awake) {
        awakeState = identityBodyState();
        awakeState.linearVelocity = { ...def.linearVelocity };
        awakeState.angularVelocity = { ...def.angularVelocity };
        awakeState.flags = bodySim.flags;
        bodySim.maxAngularVelocity = f32(vec3.length(def.angularVelocity) + f32(5.0));
    }

    if (bodyId === world.bodies.length) {
        world.bodies.push(emptyBody());
    }
    const body = world.bodies[bodyId];

    body.name = def.name ? def.name.slice(0, BODY_NAME_LENGTH) : "";
    body.userData = def.userData;
    body.setIndex = setId;
    // Awake: the sim view is pushed below, so its index is the current (pre-push) length; every other
    // set already pushed at line above, so its index is length - 1.
    body.localIndex =
        setId === SetType.Awake ? setBodyCount(world, set) : setBodyCount(world, set) - 1;

    body.headShapeId = NULL_INDEX;
    body.shapeCount = 0;
    body.headChainId = NULL_INDEX;
    body.headContactKey = NULL_INDEX;
    body.contactCount = 0;
    body.headJointKey = NULL_INDEX;
    body.jointCount = 0;
    body.islandId = NULL_INDEX;
    body.islandIndex = NULL_INDEX;
    body.bodyMoveIndex = NULL_INDEX;
    body.id = bodyId;
    body.sleepThreshold = def.sleepThreshold;
    body.sleepTime = 0;
    body.mass = 0;
    body.inertia = mat3.zero();
    setBodyType(world, body.id, def.type);
    body.flags = bodySim.flags;

    // enabled dynamic and kinematic bodies need an island
    if (setId >= SetType.Awake) {
        createIslandForBody(world, setId, body);
    }

    // Size the resident body region to the new total-body high-water (grow-only), so the region always
    // covers every body — a later mid-step wake can't outgrow it. Refresh views after reserves.
    if (reserveBodies(world.ecsState, world.bodies.length)) {
        world.manifoldStore.refreshViews();
    }
    // Write the awake body's initial state + sim into its resident record and append the views (refresh
    // first — a prior grow may have left the store's views detached, and the writes go through them).
    if (awakeState !== null) {
        world.bodyStore.refreshViews();
        residentPush(world, awakeState, bodySim, body.headShapeId);
    }

    world.locked = false;
    return bodyId;
}

/** Wake a sleeping body's set (b3WakeBody). @returns whether the body was sleeping. */
export function wakeBody(world: WorldState, body: Body): boolean {
    if (body.setIndex >= SetType.FirstSleeping) {
        wakeSolverSet(world, body.setIndex);
        return true;
    }
    return false;
}

/** Destroy a body and everything attached to it (b3DestroyBody). */
export function destroyBody(world: WorldState, body: Body): void {
    world.locked = true;

    const wakeBodies = true;

    // Destroy attached joints
    let jointKey = body.headJointKey;
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;
        destroyJointInternal(world, joint, wakeBodies);
    }

    destroyBodyContacts(world, body, wakeBodies);

    // Destroy attached shapes and their proxies
    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        if (shapeSensorIndex(world, shape.id) !== NULL_INDEX) {
            destroySensor(world, shape);
        }
        destroyShapeProxy(shape, world.broadPhase);
        destroyShapeAllocations(world, shape);
        destroyShapeSlot(world, shapeId);
        shape.id = NULL_INDEX;
        shapeId = shape.nextShapeId;
    }

    removeBodyFromIsland(world, body);

    // Remove body sim from the solver set that owns it.
    const set = body.setIndex;
    if (body.setIndex === SetType.Awake) {
        // Awake: sim + state are resident views. Migrate the tail record (sim + world) into the freed
        // slot, drop the tail views, and fix the moved body's localIndex. (No refresh needed — unlike
        // the in-step sleep/wake/transfer paths, destroyBody runs outside step(), so no manifold/geo
        // grow has detached the store's views since the last create.)
        const movedBodyId = residentRemove(world, body.localIndex);
        if (movedBodyId !== NULL_INDEX) {
            const movedBody = world.bodies[movedBodyId];
            movedBody.localIndex = body.localIndex;
            // The moved body stays awake — refresh its contacts' bodySimIndex to the new localIndex.
            writeBodySimIndex(world, movedBody);
            syncBodyQuery(world, movedBody);
        }
    } else {
        const movedIndex = setBodyRemove(world, set, body.localIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedSim = bodySimSlot(set, body.localIndex);
            world.bodies[simBodyId(world, movedSim)].localIndex = body.localIndex;
        }
        if (set >= SetType.FirstSleeping && setBodyCount(world, set) === 0) {
            // Remove the solver set if it is now an orphan
            destroySolverSet(world, set);
        }
    }

    kernel(world.ecsState).bodyDestroy(world.worldId, body.id);
    body.setIndex = NULL_INDEX;
    body.localIndex = NULL_INDEX;
    body.id = NULL_INDEX;

    world.locked = false;
}

/** Recompute mass, center of mass, and inertia from the body's shapes (b3UpdateBodyMassData). */
export function updateBodyMassData(world: WorldState, body: Body): void {
    const transformScratch1 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const centerScratch2 = { x: 0, y: 0, z: 0 };
    const maxExtentScratch3 = { x: 0, y: 0, z: 0 };
    const transformScratch4 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const invInertiaLocalScratch5 = {
        cx: { x: 0, y: 0, z: 0 },
        cy: { x: 0, y: 0, z: 0 },
        cz: { x: 0, y: 0, z: 0 },
    };
    const centerScratch6 = { x: 0, y: 0, z: 0 };
    const transformScratch7 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const localCenterScratch8 = { x: 0, y: 0, z: 0 };
    const centerScratch9 = { x: 0, y: 0, z: 0 };
    const angularVelocityScratch10 = { x: 0, y: 0, z: 0 };
    const centerScratch11 = { x: 0, y: 0, z: 0 };
    const linearVelocityScratch12 = { x: 0, y: 0, z: 0 };
    const maxExtentScratch13 = { x: 0, y: 0, z: 0 };

    const bodySim = getBodySim(world, body);

    body.mass = 0;
    body.inertia = mat3.zero();
    setSimField(world, bodySim, "invMass", 0);
    setSimField(world, bodySim, "invInertiaLocal", mat3.zero());
    setSimField(world, bodySim, "invInertiaWorld", mat3.zero());
    setSimField(world, bodySim, "localCenter", { x: 0, y: 0, z: 0 });
    setSimField(world, bodySim, "minExtent", HUGE);
    setSimField(world, bodySim, "maxExtent", { x: 0, y: 0, z: 0 });

    if (body.headShapeId === NULL_INDEX) {
        return;
    }

    // Static and kinematic sims have zero mass.
    if (bodyType(world, body.id) !== BodyType.Dynamic) {
        setSimField(world, bodySim, "center", {
            ...readSimTransform(world, bodySim, transformScratch1).p,
        });
        setSimField(world, bodySim, "center0", {
            ...readSimCenter(world, bodySim, centerScratch2),
        });

        if (bodyType(world, body.id) === BodyType.Kinematic) {
            let shapeId = body.headShapeId;
            while (shapeId !== NULL_INDEX) {
                const s = world.shapes[shapeId];
                const extent = computeShapeExtent(s, { x: 0, y: 0, z: 0 });
                setSimField(
                    world,
                    bodySim,
                    "minExtent",
                    minf(simMinExtent(world, bodySim), extent.minExtent),
                );
                setSimField(
                    world,
                    bodySim,
                    "maxExtent",
                    vec3.max(readSimMaxExtent(world, bodySim, maxExtentScratch3), extent.maxExtent),
                );
                shapeId = s.nextShapeId;
            }
        }
        syncBodyQuery(world, body);
        return;
    }

    const masses: MassData[] = [];

    let localCenter: Vec3 = { x: 0, y: 0, z: 0 };
    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const s = world.shapes[shapeId];
        shapeId = s.nextShapeId;

        if (s.density === 0) {
            masses.push({ mass: 0, center: { x: 0, y: 0, z: 0 }, inertia: mat3.zero() });
            continue;
        }

        const massData = computeShapeMass(s);
        body.mass = f32(body.mass + massData.mass);
        localCenter = vec3.mulAdd(localCenter, massData.mass, massData.center);
        masses.push(massData);
    }

    if (body.mass > 0) {
        setSimField(world, bodySim, "invMass", f32(1 / body.mass));
        localCenter = vec3.scale(simInvMass(world, bodySim), localCenter);
    }

    for (let shapeIndex = 0; shapeIndex < masses.length; ++shapeIndex) {
        const massData = masses[shapeIndex];
        if (massData.mass === 0) {
            continue;
        }
        const offset = vec3.sub(localCenter, massData.center);
        const inertia = mat3.add(massData.inertia, steiner(massData.mass, offset));
        body.inertia = mat3.add(body.inertia, inertia);
    }

    const det = mat3.det(body.inertia);
    if (det > 0) {
        setSimField(world, bodySim, "invInertiaLocal", mat3.invertT(body.inertia));
        const rotationMatrix = mat3.fromQuat(readSimTransform(world, bodySim, transformScratch4).q);
        setSimField(
            world,
            bodySim,
            "invInertiaWorld",
            mat3.mul(
                mat3.mul(
                    rotationMatrix,
                    readSimInvInertiaLocal(world, bodySim, invInertiaLocalScratch5),
                ),
                mat3.transpose(rotationMatrix),
            ),
        );
    }

    const oldCenter = readSimCenter(world, bodySim, centerScratch6);
    setSimField(world, bodySim, "localCenter", localCenter);
    setSimField(
        world,
        bodySim,
        "center",
        transformWorldPoint(
            readSimTransform(world, bodySim, transformScratch7),
            readSimLocalCenter(world, bodySim, localCenterScratch8),
        ),
    );
    setSimField(world, bodySim, "center0", {
        ...readSimCenter(world, bodySim, centerScratch9),
    });

    const state = getBodyState(world, body);
    if (state !== null) {
        const deltaLinear = vec3.cross(
            readStateAngularVelocity(world, state, angularVelocityScratch10),
            vec3.sub(readSimCenter(world, bodySim, centerScratch11), oldCenter),
        );
        setStateField(
            world,
            state,
            "linearVelocity",
            vec3.add(readStateLinearVelocity(world, state, linearVelocityScratch12), deltaLinear),
        );
    }

    let extentShapeId = body.headShapeId;
    while (extentShapeId !== NULL_INDEX) {
        const s = world.shapes[extentShapeId];
        const extent = computeShapeExtent(s, localCenter);
        setSimField(
            world,
            bodySim,
            "minExtent",
            minf(simMinExtent(world, bodySim), extent.minExtent),
        );
        setSimField(
            world,
            bodySim,
            "maxExtent",
            vec3.max(readSimMaxExtent(world, bodySim, maxExtentScratch13), extent.maxExtent),
        );
        extentShapeId = s.nextShapeId;
    }

    // Apply fixed rotation
    if ((simFlags(world, bodySim) & FIXED_ROTATION) === FIXED_ROTATION) {
        body.inertia = mat3.zero();
        setSimField(world, bodySim, "invInertiaLocal", mat3.zero());
        setSimField(world, bodySim, "invInertiaWorld", mat3.zero());
    }
    syncBodyQuery(world, body);
}

/** @returns the body's mass, local center of mass, and rotational inertia (b3Body_GetMassData). */
export function getMassData(world: WorldState, body: Body): MassData {
    const localCenterScratch1 = { x: 0, y: 0, z: 0 };

    const bodySim = getBodySim(world, body);
    return {
        mass: body.mass,
        center: readSimLocalCenter(world, bodySim, localCenterScratch1),
        inertia: body.inertia,
    };
}
