// Rigid body lifecycle and the 3-way body split, ported from Box3D's body.c (Erin Catto, MIT).
// A body is stored as three records: the cold organizational handle (b3Body, in world.bodies,
// id-indexed), the hot simulation payload (b3BodySim, in a solver set's bodySims column), and the
// solver velocity/delta state (b3BodyState, only in the awake set's bodyStates column). Static and
// sleeping bodies have a sim but no state.
//
// fround discipline per the README. This file holds the types + accessors; the
// create/destroy/setType/mass machinery is appended below.

import { NULL_INDEX, swapRemove } from "./array";
import { reserveBodies, residentPush, residentRemove } from "./bodycolumns";
import { moveProxy as bpMoveProxy } from "./broadphase";
import { destroyContact, writeBodySimIndex } from "./contact";
import { BODY_NAME_LENGTH, HUGE, SetType, SPECULATIVE_DISTANCE } from "./core";
import {
    type DistanceInput,
    emptyCache,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeProxy,
    shapeDistance,
} from "./distance";
import { writeFatAabb } from "./fataabbcolumns";
import type { Capsule, MassData } from "./geometry";
import { allocId, type EntityId, freeId } from "./ids";
import { createIsland, destroyIsland, linkJoint, splitIsland, unlinkJoint } from "./island";
import { destroyJointInternal } from "./joint";
import {
    type AABB,
    aabb,
    clampInt,
    FLT_MAX,
    f32,
    froundConfig,
    type Mat3,
    mat3,
    minf,
    offsetPos,
    type Pos,
    type Quat,
    quat,
    steiner,
    subPos,
    toRelativeTransform,
    transformWorldPoint,
    type Vec3,
    vec3,
    type WorldTransform,
} from "./math";
import type { PlaneResult } from "./mover";
import { destroySensor } from "./sensor";
import {
    collideMover,
    computeFatShapeAABB,
    computeShapeExtent,
    computeShapeMass,
    createShapeProxy,
    destroyShapeAllocations,
    destroyShapeProxy,
    getShapeMaterials,
    makeShapeProxy,
    overlapShape,
    rayCastShape,
    type Shape,
    shapeCastShape,
} from "./shape";
import {
    destroySolverSet,
    emptySolverSet,
    transferBody,
    transferJoint,
    trySleepIsland,
    wakeSolverSet,
} from "./solverset";
import {
    type BodyDef,
    BodyType,
    type QueryFilter,
    ShapeType,
    shouldQueryCollide,
    toQueryFilterBits,
} from "./types";
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
    transform: WorldTransform;
    center: Pos;
    rotation0: Quat;
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
    type: BodyType;
    generation: number;
    name: string;
};

const cloneVec = (v: Vec3): Vec3 => ({ x: v.x, y: v.y, z: v.z });
const cloneQuat = (q: Quat): Quat => ({ v: cloneVec(q.v), s: q.s });
const cloneMat3 = (m: Mat3): Mat3 => ({
    cx: cloneVec(m.cx),
    cy: cloneVec(m.cy),
    cz: cloneVec(m.cz),
});

/**
 * Deep-copy a body sim (the C `memcpy(simDst, simSrc)` in every solver-set transfer). The target
 * must not alias the source's Vec3/Quat/Mat3 sub-objects — the solver mutates them in place.
 */
export function cloneBodySim(s: BodySim): BodySim {
    return {
        transform: { p: cloneVec(s.transform.p), q: cloneQuat(s.transform.q) },
        center: cloneVec(s.center),
        rotation0: cloneQuat(s.rotation0),
        center0: cloneVec(s.center0),
        localCenter: cloneVec(s.localCenter),
        force: cloneVec(s.force),
        torque: cloneVec(s.torque),
        invMass: s.invMass,
        invInertiaLocal: cloneMat3(s.invInertiaLocal),
        invInertiaWorld: cloneMat3(s.invInertiaWorld),
        minExtent: s.minExtent,
        maxExtent: cloneVec(s.maxExtent),
        maxAngularVelocity: s.maxAngularVelocity,
        linearDamping: s.linearDamping,
        angularDamping: s.angularDamping,
        gravityScale: s.gravityScale,
        bodyId: s.bodyId,
        flags: s.flags,
    };
}

/** @returns the body's simulation payload from whichever solver set owns it (b3GetBodySim). */
export function getBodySim(world: WorldState, body: Body): BodySim {
    return world.solverSets[body.setIndex].bodySims[body.localIndex];
}

/** @returns the body's solver state, or null when the body is not awake (b3GetBodyState). */
export function getBodyState(world: WorldState, body: Body): BodyState | null {
    if (body.setIndex === SetType.Awake) {
        return world.solverSets[SetType.Awake].bodyStates[body.localIndex];
    }
    return null;
}

/** @returns a public body id for a raw body index (b3MakeBodyId). */
export function makeBodyId(world: WorldState, bodyId: number): EntityId {
    if (bodyId === NULL_INDEX) {
        return { index1: 0, world0: 0, generation: 0 };
    }
    const body = world.bodies[bodyId];
    return { index1: bodyId + 1, world0: world.worldId, generation: body.generation };
}

/** @returns the body's world transform (b3GetBodyTransformQuick). */
export function getBodyTransformQuick(world: WorldState, body: Body): WorldTransform {
    return getBodySim(world, body).transform;
}

/**
 * Copy a body's persistent (non-transient) flags into its sim and, when awake, its state
 * (b3SyncBodyFlags). Called after any change to body.flags that the solver reads (type, locks, bullet).
 */
export function syncBodyFlags(world: WorldState, body: Body): void {
    const flags = body.flags & ~BODY_TRANSIENT_FLAGS;
    getBodySim(world, body).flags = flags;
    const state = getBodyState(world, body);
    if (state !== null) state.flags = flags;
}

/** Set a body's linear velocity, waking it when the velocity is nonzero (b3Body_SetLinearVelocity). */
export function bodySetLinearVelocity(world: WorldState, body: Body, linearVelocity: Vec3): void {
    if (body.type === BodyType.Static) return;
    if (vec3.lengthSq(linearVelocity) > 0) wakeBody(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;
    // Copy, don't store the caller's object: finalize mutates state.linearVelocity in place.
    state.linearVelocity = { x: linearVelocity.x, y: linearVelocity.y, z: linearVelocity.z };
}

/**
 * Set a body's angular velocity, masking out locked angular axes and waking it when the result is
 * nonzero (b3Body_SetAngularVelocity).
 */
export function bodySetAngularVelocity(world: WorldState, body: Body, angularVelocity: Vec3): void {
    if (body.type === BodyType.Static) return;
    const w: Vec3 = {
        x: body.flags & BodyFlags.lockAngularX ? 0 : angularVelocity.x,
        y: body.flags & BodyFlags.lockAngularY ? 0 : angularVelocity.y,
        z: body.flags & BodyFlags.lockAngularZ ? 0 : angularVelocity.z,
    };
    if (vec3.lengthSq(w) !== 0) wakeBody(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;
    state.angularVelocity = w;
}

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
    if (body.type === BodyType.Static || timeStep <= 0) return;
    if (body.setIndex !== SetType.Awake && wake === false) return;

    const sim = getBodySim(world, body);

    // Linear velocity from the world-space center difference, demoted to f32.
    const center1 = sim.center;
    const center2 = transformWorldPoint(target, sim.localCenter);
    const invTimeStep = f32(1 / timeStep);
    const linearVelocity = vec3.scale(invTimeStep, subPos(center2, center1));

    // Angular velocity: w = 2 * (q2 - q1) * conj(q1) / dt, using the shortest-arc quaternion.
    const q1 = sim.transform.q;
    let q2 = target.q;
    if (quat.dot(q1, q2) < 0) q2 = quat.negate(q2);
    const dq: Quat = { v: vec3.sub(q2.v, q1.v), s: f32(q2.s - q1.s) };
    const omega = quat.mul(dq, quat.conjugate(q1));
    const angularVelocity = vec3.scale(f32(2 * invTimeStep), omega.v);

    // If the body is asleep, wake only when the target motion exceeds the sleep threshold.
    if (body.setIndex !== SetType.Awake) {
        const maxVelocity = f32(
            vec3.length(linearVelocity) + vec3.length(vec3.mul(angularVelocity, sim.maxExtent)),
        );
        if (maxVelocity < body.sleepThreshold) return;
        wakeBody(world, body);
    }

    const state = getBodyState(world, body);
    if (state === null) return;
    state.linearVelocity = linearVelocity;
    state.angularVelocity = angularVelocity;
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
    sim.force = vec3.add(sim.force, force);
    sim.torque = vec3.add(sim.torque, vec3.cross(subPos(point, sim.center), force));
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
    sim.force = vec3.add(sim.force, force);
}

/** Accumulate a torque about the center of mass (b3Body_ApplyTorque). */
export function bodyApplyTorque(world: WorldState, body: Body, torque: Vec3, wake: boolean): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    sim.torque = vec3.add(sim.torque, torque);
}

// Clamp a linear velocity to the world's max linear speed (the shared tail of the impulse setters).
function clampLinearSpeed(world: WorldState, v: Vec3): Vec3 {
    const maxLinearSpeed = world.maxLinearSpeed;
    if (vec3.lengthSq(v) > f32(maxLinearSpeed * maxLinearSpeed)) {
        return vec3.scale(maxLinearSpeed, vec3.normalize(v));
    }
    return v;
}

/**
 * Apply an instantaneous world-space impulse at a world-space point, changing velocity immediately
 * (b3Body_ApplyLinearImpulse). An off-center point also changes angular velocity. Linear speed is
 * clamped to the world's max.
 */
export function bodyApplyLinearImpulse(
    world: WorldState,
    body: Body,
    impulse: Vec3,
    point: Pos,
    wake: boolean,
): void {
    if (wake && body.setIndex >= SetType.FirstSleeping) wakeBody(world, body);
    if (body.setIndex !== SetType.Awake) return;
    const sim = getBodySim(world, body);
    const state = getBodyState(world, body);
    if (state === null) return;

    state.linearVelocity = clampLinearSpeed(
        world,
        vec3.mulAdd(state.linearVelocity, sim.invMass, impulse),
    );

    const delta = mat3.mulV(sim.invInertiaWorld, vec3.cross(subPos(point, sim.center), impulse));
    state.angularVelocity = vec3.add(state.angularVelocity, delta);
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

    state.linearVelocity = clampLinearSpeed(
        world,
        vec3.mulAdd(state.linearVelocity, sim.invMass, impulse),
    );
}

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
    const localImpulse = quat.invRotate(sim.transform.q, impulse);
    const localDelta = mat3.mulV(sim.invInertiaLocal, localImpulse);
    state.angularVelocity = vec3.add(
        state.angularVelocity,
        quat.rotate(sim.transform.q, localDelta),
    );
}

// --- transform + type + awake ----------------------------------------------------------------

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

    sim.transform = { p: cloneVec(position), q: cloneQuat(rotation) };
    sim.center = transformWorldPoint(sim.transform, sim.localCenter);

    const rot = mat3.fromQuat(sim.transform.q);
    sim.invInertiaWorld = mat3.mul(mat3.mul(rot, sim.invInertiaLocal), mat3.transpose(rot));

    sim.rotation0 = cloneQuat(sim.transform.q);
    sim.center0 = cloneVec(sim.center);

    const broadPhase = world.broadPhase;
    const transform = sim.transform;
    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        const box = computeFatShapeAABB(shape, transform, SPECULATIVE_DISTANCE);
        shape.aabb = box;

        if (aabb.contains(shape.fatAABB, box) === false) {
            const margin = shape.aabbMargin;
            const fatAABB: AABB = {
                lowerBound: {
                    x: f32(box.lowerBound.x - margin),
                    y: f32(box.lowerBound.y - margin),
                    z: f32(box.lowerBound.z - margin),
                },
                upperBound: {
                    x: f32(box.upperBound.x + margin),
                    y: f32(box.upperBound.y + margin),
                    z: f32(box.upperBound.z + margin),
                },
            };
            shape.fatAABB = fatAABB;
            writeFatAabb(world, shape);

            // The body could be disabled, in which case it has no proxy.
            if (shape.proxyKey !== NULL_INDEX) {
                bpMoveProxy(broadPhase, shape.proxyKey, fatAABB);
            }
        }

        shapeId = shape.nextShapeId;
    }
}

/**
 * Change a body's type (static / kinematic / dynamic), moving it between solver sets and rebuilding its
 * island participation, contacts, joints, and broadphase proxies (b3Body_SetType). Not supported for
 * bodies carrying a compound or height-field shape when the target type is non-static.
 */
export function bodySetType(world: WorldState, body: Body, type: BodyType): void {
    world.locked = true;

    const originalType = body.type;
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
        body.type = type;
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
    const staticSet = world.solverSets[SetType.Static];
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
        transferJoint(world, staticSet, world.solverSets[joint.setIndex], joint);
    }

    // Stage 5: change the type and transfer the body between solver sets.
    body.type = type;
    if (type === BodyType.Dynamic) body.flags |= BodyFlags.dynamicFlag;
    else body.flags &= ~BodyFlags.dynamicFlag;

    const awakeSet = world.solverSets[SetType.Awake];
    const sourceSet = world.solverSets[body.setIndex];
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
        if (bodyA.type === BodyType.Dynamic || bodyB.type === BodyType.Dynamic) {
            transferJoint(world, awakeSet, staticSet, joint);
        }
    }

    // Recreate shape proxies in the broadphase against the new body type.
    const transform = getBodyTransformQuick(world, body);
    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;
        destroyShapeProxy(shape, world.broadPhase);
        createShapeProxy(shape, world.broadPhase, type, transform, true);
        writeFatAabb(world, shape);
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
        if (body.type !== BodyType.Dynamic && otherBody.type !== BodyType.Dynamic) continue;

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
        const island = world.islands[body.islandId];
        if (island.constraintRemoveCount > 0) {
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
        type: BodyType.Static,
        generation: 0,
        name: "",
    };
}

function createIslandForBody(world: WorldState, setIndex: number, body: Body): void {
    const island = createIsland(world, setIndex);
    island.bodies.push(body.id);
    body.islandId = island.islandId;
    body.islandIndex = 0;
}

function removeBodyFromIsland(world: WorldState, body: Body): void {
    if (body.islandId === NULL_INDEX) {
        return;
    }

    const islandId = body.islandId;
    const island = world.islands[islandId];
    {
        const localIndex = body.islandIndex;
        const movedBodyId = island.bodies[island.bodies.length - 1];
        island.bodies[localIndex] = movedBodyId;
        world.bodies[movedBodyId].islandIndex = localIndex;
        island.bodies.pop();
    }

    if (island.bodies.length === 0) {
        destroyIsland(world, island.islandId);
    }

    body.islandId = NULL_INDEX;
    body.islandIndex = NULL_INDEX;
}

function destroyBodyContacts(world: WorldState, body: Body, wakeBodies: boolean): void {
    let edgeKey = body.headContactKey;
    while (edgeKey !== NULL_INDEX) {
        const contactId = edgeKey >> 1;
        const edgeIndex = edgeKey & 1;
        const contact = world.contacts[contactId];
        edgeKey = contact.edges[edgeIndex].nextKey;
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
        setId = allocId(world.solverSetIdPool);
        if (setId === world.solverSets.length) {
            world.solverSets.push(emptySolverSet());
        }
        world.solverSets[setId].setIndex = setId;
    }

    const bodyId = allocId(world.bodyIdPool);

    let lockFlags = 0;
    lockFlags |= def.motionLocks.linearX ? BodyFlags.lockLinearX : 0;
    lockFlags |= def.motionLocks.linearY ? BodyFlags.lockLinearY : 0;
    lockFlags |= def.motionLocks.linearZ ? BodyFlags.lockLinearZ : 0;
    lockFlags |= def.motionLocks.angularX ? BodyFlags.lockAngularX : 0;
    lockFlags |= def.motionLocks.angularY ? BodyFlags.lockAngularY : 0;
    lockFlags |= def.motionLocks.angularZ ? BodyFlags.lockAngularZ : 0;

    const set = world.solverSets[setId];
    const bodySim = emptyBodySim();
    bodySim.transform.p = { ...def.position };
    bodySim.transform.q = { v: { ...def.rotation.v }, s: def.rotation.s };
    bodySim.center = { ...def.position };
    bodySim.rotation0 = { v: { ...bodySim.transform.q.v }, s: bodySim.transform.q.s };
    bodySim.center0 = { ...bodySim.center };
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
    // Awake bodies hold a column-backed `ResidentBodySim` view (pushed via `residentPush` below, once
    // the region is sized); every other set holds the plain sim. Defer the awake push so the view lands
    // over the resident record rather than a plain object being replaced.
    if (setId !== SetType.Awake) set.bodySims.push(bodySim);

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
    body.localIndex = setId === SetType.Awake ? set.bodySims.length : set.bodySims.length - 1;
    body.generation += 1;
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
    body.type = def.type;
    body.flags = bodySim.flags;

    // enabled dynamic and kinematic bodies need an island
    if (setId >= SetType.Awake) {
        createIslandForBody(world, setId, body);
    }

    // Size the resident body region to the new total-body high-water (grow-only), so the region always
    // covers every body — a later mid-step wake can't outgrow it. A grow relocates the manifold +
    // geometry regions and detaches views, so refresh those before anything reads through them.
    if (reserveBodies(world.bodies.length)) {
        world.manifoldStore.refreshViews();
    }
    // Write the awake body's initial state + sim into its resident record and append the views (refresh
    // first — a prior grow may have left the store's views detached, and the writes go through them).
    if (awakeState !== null) {
        world.bodyStore.refreshViews();
        residentPush(
            world.bodyStore,
            set.bodyStates,
            set.bodySims,
            awakeState,
            bodySim,
            body.headShapeId,
        );
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
        if (shape.sensorIndex !== NULL_INDEX) {
            destroySensor(world, shape);
        }
        destroyShapeProxy(shape, world.broadPhase);
        destroyShapeAllocations(world, shape);
        freeId(world.shapeIdPool, shapeId);
        shape.id = NULL_INDEX;
        shapeId = shape.nextShapeId;
    }

    removeBodyFromIsland(world, body);

    // Remove body sim from the solver set that owns it.
    const set = world.solverSets[body.setIndex];
    if (body.setIndex === SetType.Awake) {
        // Awake: sim + state are resident views. Migrate the tail record (sim + state) into the freed
        // slot, drop the tail views, and fix the moved body's localIndex. (No refresh needed — unlike
        // the in-step sleep/wake/transfer paths, destroyBody runs outside step(), so no manifold/geo
        // grow has detached the store's views since the last create.)
        const movedBodyId = residentRemove(
            world.bodyStore,
            set.bodyStates,
            set.bodySims,
            body.localIndex,
        );
        if (movedBodyId !== NULL_INDEX) {
            const movedBody = world.bodies[movedBodyId];
            movedBody.localIndex = body.localIndex;
            // The moved body stays awake — refresh its contacts' bodySimIndex to the new localIndex.
            writeBodySimIndex(world, movedBody);
        }
    } else {
        const movedIndex = swapRemove(set.bodySims, body.localIndex);
        if (movedIndex !== NULL_INDEX) {
            const movedSim = set.bodySims[body.localIndex];
            world.bodies[movedSim.bodyId].localIndex = body.localIndex;
        }
        if (set.setIndex >= SetType.FirstSleeping && set.bodySims.length === 0) {
            // Remove the solver set if it is now an orphan
            destroySolverSet(world, set.setIndex);
        }
    }

    freeId(world.bodyIdPool, body.id);
    body.setIndex = NULL_INDEX;
    body.localIndex = NULL_INDEX;
    body.id = NULL_INDEX;

    world.locked = false;
}

/** Recompute mass, center of mass, and inertia from the body's shapes (b3UpdateBodyMassData). */
export function updateBodyMassData(world: WorldState, body: Body): void {
    const bodySim = getBodySim(world, body);

    body.mass = 0;
    body.inertia = mat3.zero();
    bodySim.invMass = 0;
    bodySim.invInertiaLocal = mat3.zero();
    bodySim.invInertiaWorld = mat3.zero();
    bodySim.localCenter = { x: 0, y: 0, z: 0 };
    bodySim.minExtent = HUGE;
    bodySim.maxExtent = { x: 0, y: 0, z: 0 };

    if (body.headShapeId === NULL_INDEX) {
        return;
    }

    // Static and kinematic sims have zero mass.
    if (body.type !== BodyType.Dynamic) {
        bodySim.center = { ...bodySim.transform.p };
        bodySim.center0 = { ...bodySim.center };

        if (body.type === BodyType.Kinematic) {
            let shapeId = body.headShapeId;
            while (shapeId !== NULL_INDEX) {
                const s = world.shapes[shapeId];
                const extent = computeShapeExtent(s, { x: 0, y: 0, z: 0 });
                bodySim.minExtent = minf(bodySim.minExtent, extent.minExtent);
                bodySim.maxExtent = vec3.max(bodySim.maxExtent, extent.maxExtent);
                shapeId = s.nextShapeId;
            }
        }
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
        bodySim.invMass = f32(1 / body.mass);
        localCenter = vec3.scale(bodySim.invMass, localCenter);
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
        bodySim.invInertiaLocal = mat3.invertT(body.inertia);
        const rotationMatrix = mat3.fromQuat(bodySim.transform.q);
        bodySim.invInertiaWorld = mat3.mul(
            mat3.mul(rotationMatrix, bodySim.invInertiaLocal),
            mat3.transpose(rotationMatrix),
        );
    }

    const oldCenter = bodySim.center;
    bodySim.localCenter = localCenter;
    bodySim.center = transformWorldPoint(bodySim.transform, bodySim.localCenter);
    bodySim.center0 = { ...bodySim.center };

    const state = getBodyState(world, body);
    if (state !== null) {
        const deltaLinear = vec3.cross(state.angularVelocity, vec3.sub(bodySim.center, oldCenter));
        state.linearVelocity = vec3.add(state.linearVelocity, deltaLinear);
    }

    let extentShapeId = body.headShapeId;
    while (extentShapeId !== NULL_INDEX) {
        const s = world.shapes[extentShapeId];
        const extent = computeShapeExtent(s, localCenter);
        bodySim.minExtent = minf(bodySim.minExtent, extent.minExtent);
        bodySim.maxExtent = vec3.max(bodySim.maxExtent, extent.maxExtent);
        extentShapeId = s.nextShapeId;
    }

    // Apply fixed rotation
    if ((bodySim.flags & FIXED_ROTATION) === FIXED_ROTATION) {
        body.inertia = mat3.zero();
        bodySim.invInertiaLocal = mat3.zero();
        bodySim.invInertiaWorld = mat3.zero();
    }
}

/** @returns the body's mass, local center of mass, and rotational inertia (b3Body_GetMassData). */
export function getMassData(world: WorldState, body: Body): MassData {
    const bodySim = getBodySim(world, body);
    return { mass: body.mass, center: bodySim.localCenter, inertia: body.inertia };
}

// --- per-body queries -----------------------------------------------------------------------

/** A single ray/shape-cast hit against a body's shapes (b3BodyCastResult). */
export type BodyCastResult = {
    shapeId: EntityId;
    point: Pos;
    normal: Vec3;
    fraction: number;
    triangleIndex: number;
    userMaterialId: bigint;
    iterations: number;
    hit: boolean;
};

const emptyBodyCastResult = (): BodyCastResult => ({
    shapeId: { index1: 0, world0: 0, generation: 0 },
    point: { x: 0, y: 0, z: 0 },
    normal: { x: 0, y: 0, z: 0 },
    fraction: 0,
    triangleIndex: 0,
    userMaterialId: 0n,
    iterations: 0,
    hit: false,
});

/**
 * Cast a ray at one body's shapes, using the supplied world transform (b3Body_CastRay). Everything
 * is re-centered on `origin` so the collision math stays exact far out; the closest hit wins.
 */
export function bodyCastRay(
    world: WorldState,
    body: Body,
    origin: Pos,
    translation: Vec3,
    filter: QueryFilter,
    maxFraction: number,
    bodyTransform: WorldTransform,
): BodyCastResult {
    const filterBits = toQueryFilterBits(filter);
    let result = emptyBodyCastResult();

    // The consistent framing is to center on the ray origin.
    const shapeInput: RayCastInput = { origin: { x: 0, y: 0, z: 0 }, translation, maxFraction };
    const transform = toRelativeTransform(bodyTransform, origin);

    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;

        if (shouldQueryCollide(shape.filter, filterBits) === false) {
            continue;
        }

        const shapeOutput = rayCastShape(shape, transform, shapeInput);
        if (shapeOutput.hit === false) {
            continue;
        }
        if (shapeOutput.fraction > shapeInput.maxFraction) {
            continue;
        }

        const materialIndex = clampInt(shapeOutput.materialIndex, 0, shape.materialCount - 1);
        result = {
            shapeId: { index1: shape.id + 1, world0: world.worldId, generation: shape.generation },
            point: offsetPos(origin, shapeOutput.point),
            normal: shapeOutput.normal,
            fraction: shapeOutput.fraction,
            triangleIndex: shapeOutput.triangleIndex,
            userMaterialId: getShapeMaterials(shape)[materialIndex].userMaterialId,
            iterations: shapeOutput.iterations,
            hit: true,
        };
        shapeInput.maxFraction = shapeOutput.fraction;
    }

    return result;
}

/** Cast a convex proxy at one body's shapes, using the supplied world transform (b3Body_CastShape). */
export function bodyCastShape(
    world: WorldState,
    body: Body,
    origin: Pos,
    proxy: ShapeProxy,
    translation: Vec3,
    filter: QueryFilter,
    maxFraction: number,
    canEncroach: boolean,
    bodyTransform: WorldTransform,
): BodyCastResult {
    const filterBits = toQueryFilterBits(filter);
    let result = emptyBodyCastResult();
    const transform = toRelativeTransform(bodyTransform, origin);

    const shapeInput: ShapeCastInput = { proxy, translation, maxFraction, canEncroach };

    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;

        if (shouldQueryCollide(shape.filter, filterBits) === false) {
            continue;
        }

        const shapeOutput = shapeCastShape(shape, transform, shapeInput);
        if (shapeOutput.hit === false) {
            continue;
        }
        if (shapeOutput.fraction > shapeInput.maxFraction) {
            continue;
        }

        const materialIndex = clampInt(shapeOutput.materialIndex, 0, shape.materialCount - 1);
        result = {
            shapeId: { index1: shape.id + 1, world0: world.worldId, generation: shape.generation },
            point: offsetPos(origin, shapeOutput.point),
            normal: shapeOutput.normal,
            fraction: shapeOutput.fraction,
            triangleIndex: shapeOutput.triangleIndex,
            userMaterialId: getShapeMaterials(shape)[materialIndex].userMaterialId,
            iterations: shapeOutput.iterations,
            hit: true,
        };
        shapeInput.maxFraction = shapeOutput.fraction;
    }

    return result;
}

/** True if `proxy` overlaps any of one body's shapes, using the supplied transform (b3Body_OverlapShape). */
export function bodyOverlapShape(
    world: WorldState,
    body: Body,
    origin: Pos,
    proxy: ShapeProxy,
    filter: QueryFilter,
    bodyTransform: WorldTransform,
): boolean {
    const filterBits = toQueryFilterBits(filter);
    const transform = toRelativeTransform(bodyTransform, origin);

    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;

        if (shouldQueryCollide(shape.filter, filterBits) === false) {
            continue;
        }
        if (overlapShape(shape, transform, proxy)) {
            return true;
        }
    }

    return false;
}

/**
 * Closest point on a body's convex shapes to `target`, and its distance (b3Body_GetClosestPoint).
 * Uses the body's stored transform. Mesh/height/compound shapes are skipped.
 */
export function bodyGetClosestPoint(
    world: WorldState,
    body: Body,
    target: Vec3,
): { point: Vec3; distance: number } {
    const transform = toRelativeTransform(getBodyTransformQuick(world, body), { x: 0, y: 0, z: 0 });

    let closestDistance = FLT_MAX;
    let closestPoint = transform.p;

    // Target rides in frame A at the origin, so the shape's relative pose in A is the body transform.
    const input: DistanceInput = {
        proxyA: { points: [target], count: 1, radius: 0 },
        proxyB: { points: [], count: 0, radius: 0 },
        transform,
        useRadii: false,
    };

    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape: Shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;

        const type = shape.type;
        if (type !== ShapeType.Sphere && type !== ShapeType.Capsule && type !== ShapeType.Hull) {
            continue;
        }

        input.proxyB = makeShapeProxy(shape);
        const output = shapeDistance(input, emptyCache());
        if (output.distance < closestDistance) {
            closestDistance = output.distance;
            closestPoint = output.pointB;
        }
    }

    return { point: closestPoint, distance: closestDistance };
}

/** One collision plane between a mover and a specific shape of a body (b3BodyPlaneResult). */
export type BodyPlaneResult = {
    shapeId: EntityId;
    result: PlaneResult;
};

/**
 * Collide a capsule `mover` (at `origin`) against one body's convex shapes, using the supplied world
 * transform (b3Body_CollideMover). One plane per touched sphere/capsule/hull shape (mesh/height/
 * compound are skipped); stops at `capacity` planes. Everything is re-centered on `origin`.
 */
export function bodyCollideMover(
    world: WorldState,
    body: Body,
    capacity: number,
    origin: Pos,
    mover: Capsule,
    filter: QueryFilter,
    bodyTransform: WorldTransform,
): BodyPlaneResult[] {
    const filterBits = toQueryFilterBits(filter);
    const results: BodyPlaneResult[] = [];
    if (capacity === 0) {
        return results;
    }

    const transform = toRelativeTransform(bodyTransform, origin);

    let shapeId = body.headShapeId;
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;

        if (shouldQueryCollide(shape.filter, filterBits) === false) {
            continue;
        }

        const type = shape.type;
        if (type !== ShapeType.Sphere && type !== ShapeType.Capsule && type !== ShapeType.Hull) {
            continue;
        }

        const planes = collideMover(shape, transform, mover, 1);
        if (planes.length > 0) {
            results.push({
                shapeId: {
                    index1: shape.id + 1,
                    world0: world.worldId,
                    generation: shape.generation,
                },
                result: planes[0],
            });
            if (results.length === capacity) {
                return results;
            }
        }
    }

    return results;
}
