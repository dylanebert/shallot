import { ContactField, contactField } from "../collision/contact";
import { BodyField, bodyField, bodyInertia } from "../kernel/bodyrecords";
import { bodyType, shapeSensorIndex } from "../kernel/filtercolumns";
import { reserveProxy } from "../kernel/treecolumns";
// body.c bindings (Box3D, Erin Catto, MIT). Body records, solver-set sims and awake states
// belong to the kernel. Walks over joint and shape records remain here until those records move.

import { destroyContact } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { BODY_NAME_LENGTH, SetType } from "../common/constants";
import type { EntityId } from "../common/ids";
import {
    type Mat3,
    mat3,
    type Pos,
    type Quat,
    type Vec3,
    vec3,
    type WorldTransform,
} from "../common/math";
import { type BodyDef, BodyType, ShapeType } from "../common/types";
import { readSimLocalCenter, readSimTransform } from "../kernel/bodycolumns";
import { islandField } from "../kernel/islandcolumns";
import { kernel } from "../kernel/kernel";
import { destroyShapeSlot, syncBodyQuery } from "../kernel/shapecolumns";
import type { MassData } from "../shapes/geometry";
import {
    computeShapeExtent,
    computeShapeMass,
    destroyShapeAllocations,
    destroyShapeProxy,
} from "../shapes/shape";
import { destroyJointInternal } from "../solver/joint";
import { linkJoint, splitIsland, unlinkJoint } from "./island";
import { destroySensor } from "./sensor";
import { transferBody, transferJoint, trySleepIsland, wakeSolverSet } from "./solverset";
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

/** @returns the sim slot addressed by the body record (b3GetBodySim). */
export function getBodySim(_world: WorldState, body: number): number {
    return -body - 1;
}

/** @returns the awake state's local index, or null (b3GetBodyState). */
export function getBodyState(_world: WorldState, body: number): number | null {
    const index = kernel(_world.ecsState).bodyStateIndex(_world.worldId, body);
    return index < 0 ? null : index;
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
    body: number,
    out: WorldTransform,
): WorldTransform {
    return readSimTransform(world, getBodySim(world, body), out);
}

/**
 * Copy a body's persistent (non-transient) flags into its sim and, when awake, its state
 * (b3SyncBodyFlags). Called after any change to body.flags that the solver reads (type, locks, bullet).
 */
export function syncBodyFlags(world: WorldState, body: number): void {
    kernel(world.ecsState).bodySyncFlags(world.worldId, body);
}

/** Set a body's linear velocity, waking it when the velocity is nonzero (b3Body_SetLinearVelocity). */
export function bodySetLinearVelocity(world: WorldState, body: number, linearVelocity: Vec3): void {
    const k = kernel(world.ecsState);
    if (
        k.bodyVelocitySet(
            world.worldId,
            body,
            false,
            linearVelocity.x,
            linearVelocity.y,
            linearVelocity.z,
        )
    ) {
        wakeBody(world, body);
        k.bodyVelocitySet(
            world.worldId,
            body,
            false,
            linearVelocity.x,
            linearVelocity.y,
            linearVelocity.z,
        );
    }
}

/**
 * Set a body's angular velocity, masking out locked angular axes and waking it when the result is
 * nonzero (b3Body_SetAngularVelocity).
 */
export function bodySetAngularVelocity(
    world: WorldState,
    body: number,
    angularVelocity: Vec3,
): void {
    const k = kernel(world.ecsState);
    if (
        k.bodyVelocitySet(
            world.worldId,
            body,
            true,
            angularVelocity.x,
            angularVelocity.y,
            angularVelocity.z,
        )
    ) {
        wakeBody(world, body);
        k.bodyVelocitySet(
            world.worldId,
            body,
            true,
            angularVelocity.x,
            angularVelocity.y,
            angularVelocity.z,
        );
    }
}

/**
 * Drive a (typically kinematic) body toward a target transform over one time step by setting the
 * linear and angular velocity that reaches it (b3Body_SetTargetTransform). Used to animate a kinematic
 * pusher along a path.
 */
export function bodySetTargetTransform(
    world: WorldState,
    body: number,
    target: WorldTransform,
    timeStep: number,
    wake: boolean,
): void {
    const k = kernel(world.ecsState);
    if (
        k.bodyTargetVelocity(
            world.worldId,
            body,
            target.p.x,
            target.p.y,
            target.p.z,
            target.q.v.x,
            target.q.v.y,
            target.q.v.z,
            target.q.s,
            timeStep,
            wake,
        )
    ) {
        wakeBody(world, body);
        k.bodyTargetVelocity(
            world.worldId,
            body,
            target.p.x,
            target.p.y,
            target.p.z,
            target.q.v.x,
            target.q.v.y,
            target.q.v.z,
            target.q.s,
            timeStep,
            false,
        );
    }
}

// --- forces + impulses -----------------------------------------------------------------------

/**
 * Accumulate a world-space force at a world-space point, waking the body when `wake` (b3Body_ApplyForce).
 * The force integrates over the next step; an off-center point also produces a torque.
 */
export function bodyApplyForce(
    world: WorldState,
    body: number,
    force: Vec3,
    point: Pos,
    wake: boolean,
): void {
    if (wake && bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping)
        wakeBody(world, body);
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
    );
}

/** Accumulate a world-space force at the center of mass (b3Body_ApplyForceToCenter). No torque. */
export function bodyApplyForceToCenter(
    world: WorldState,
    body: number,
    force: Vec3,
    wake: boolean,
): void {
    if (wake && bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping)
        wakeBody(world, body);
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
    );
}

/** Accumulate a torque about the center of mass (b3Body_ApplyTorque). */
export function bodyApplyTorque(
    world: WorldState,
    body: number,
    torque: Vec3,
    wake: boolean,
): void {
    if (wake && bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping)
        wakeBody(world, body);
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
    );
}

/**
 * Apply an instantaneous world-space impulse at a world-space point, changing velocity immediately
 * (b3Body_ApplyLinearImpulse). An off-center point also changes angular velocity. Linear speed is
 * clamped to the world's max.
 */
export function bodyApplyLinearImpulse(
    world: WorldState,
    body: number,
    impulse: Vec3,
    point: Pos,
    wake: boolean,
): void {
    if (wake && bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping)
        wakeBody(world, body);
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
    );
}

/** Apply an instantaneous impulse at the center of mass (b3Body_ApplyLinearImpulseToCenter). */
export function bodyApplyLinearImpulseToCenter(
    world: WorldState,
    body: number,
    impulse: Vec3,
    wake: boolean,
): void {
    if (wake && bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping)
        wakeBody(world, body);
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
    );
}

/** Apply an instantaneous angular impulse, changing angular velocity immediately (b3Body_ApplyAngularImpulse). */
export function bodyApplyAngularImpulse(
    world: WorldState,
    body: number,
    impulse: Vec3,
    wake: boolean,
): void {
    if (wake && bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping)
        wakeBody(world, body);
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
    let shapeId = bodyField(world, body, BodyField.headShapeId);
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        kernel(world.ecsState).bodyShapeBounds(world.worldId, body, shapeId);
        shapeId = shape.nextShapeId;
    }
    syncBodyQuery(world, body);
}

/**
 * Change a body's type (static / kinematic / dynamic), moving it between solver sets and rebuilding its
 * island participation, contacts, joints, and broadphase proxies (b3Body_SetType). Not supported for
 * bodies carrying a compound or height-field shape when the target type is non-static.
 */
export function bodySetType(world: WorldState, body: number, type: BodyType): void {
    world.locked = true;

    const originalType = bodyType(world, bodyField(world, body, BodyField.id));
    if (originalType === type) {
        world.locked = false;
        return;
    }

    if (type !== BodyType.Static) {
        let shapeId = bodyField(world, body, BodyField.headShapeId);
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
    if (bodyField(world, body, BodyField.setIndex) === SetType.Disabled) {
        kernel(world.ecsState).bodyChangeType(world.worldId, body, type);
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
    let jointKey = bodyField(world, body, BodyField.headJointKey);
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;

        if (joint.setIndex === SetType.Disabled) continue;

        // Wake attached bodies: wakeBody above does not wake bodies attached to a static body.
        wakeBody(world, joint.edges[0].bodyId);
        wakeBody(world, joint.edges[1].bodyId);

        unlinkJoint(world, joint);
        transferJoint(world, staticSet, joint.setIndex, joint);
    }

    // Stage 5: change the type and transfer the body between solver sets.
    kernel(world.ecsState).bodyChangeType(world.worldId, body, type);

    const awakeSet = SetType.Awake;
    const sourceSet = bodyField(world, body, BodyField.setIndex);
    const targetSet = type === BodyType.Static ? staticSet : awakeSet;
    transferBody(world, targetSet, sourceSet, body);

    // Stage 6: update island participation.
    if (originalType === BodyType.Static) {
        createIslandForBody(world, body);
    } else if (type === BodyType.Static) {
        removeBodyFromIsland(world, body);
    }

    // Stage 7: transfer joints back to the awake set when either attached body is now dynamic.
    jointKey = bodyField(world, body, BodyField.headJointKey);
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;

        if (joint.setIndex === SetType.Disabled) continue;

        const bodyA = joint.edges[0].bodyId;
        const bodyB = joint.edges[1].bodyId;
        if (
            bodyType(world, bodyField(world, bodyA, BodyField.id)) === BodyType.Dynamic ||
            bodyType(world, bodyField(world, bodyB, BodyField.id)) === BodyType.Dynamic
        ) {
            transferJoint(world, awakeSet, staticSet, joint);
        }
    }

    // Recreate shape proxies in the broadphase against the new body type.
    let shapeId = bodyField(world, body, BodyField.headShapeId);
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        shapeId = shape.nextShapeId;
        destroyShapeProxy(shape, world.broadPhase);
        createBodyProxy(world, body, shape.id);
    }

    // Relink joints where at least one attached body is dynamic and enabled.
    jointKey = bodyField(world, body, BodyField.headJointKey);
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;

        const otherBodyId = joint.edges[edgeIndex ^ 1].bodyId;
        const otherBody = otherBodyId;
        if (bodyField(world, otherBody, BodyField.setIndex) === SetType.Disabled) continue;
        if (
            bodyType(world, bodyField(world, body, BodyField.id)) !== BodyType.Dynamic &&
            bodyType(world, bodyField(world, otherBody, BodyField.id)) !== BodyType.Dynamic
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
export function bodySetAwake(world: WorldState, body: number, awake: boolean): void {
    world.locked = true;

    if (awake && bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping) {
        wakeBody(world, body);
    } else if (awake === false && bodyField(world, body, BodyField.setIndex) === SetType.Awake) {
        if (islandField(world, bodyField(world, body, BodyField.islandId), 3) > 0) {
            // Must split the island before sleeping. This is expensive.
            splitIsland(world, bodyField(world, body, BodyField.islandId));
        }
        trySleepIsland(world, bodyField(world, body, BodyField.islandId));
    }

    world.locked = false;
}

// --- lifecycle -------------------------------------------------------------------------------

function createIslandForBody(world: WorldState, body: number): void {
    kernel(world.ecsState).bodyCreateIsland(world.worldId, body);
}

function removeBodyFromIsland(world: WorldState, body: number): void {
    kernel(world.ecsState).bodyRemoveIsland(world.worldId, body);
}

function createBodyProxy(world: WorldState, body: number, shapeId: number): void {
    const type = bodyType(world, body);
    world.broadPhase.store.refreshIfStale();
    reserveProxy(world.broadPhase.trees[type]);
    world.shapes[shapeId].proxyKey = kernel(world.ecsState).bodyCreateProxy(
        world.worldId,
        body,
        shapeId,
    );
}

function destroyBodyContacts(world: WorldState, body: number, wakeBodies: boolean): void {
    let edgeKey = bodyField(world, body, BodyField.headContactKey);
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

    let lockFlags = 0;
    lockFlags |= def.motionLocks.linearX ? BodyFlags.lockLinearX : 0;
    lockFlags |= def.motionLocks.linearY ? BodyFlags.lockLinearY : 0;
    lockFlags |= def.motionLocks.linearZ ? BodyFlags.lockLinearZ : 0;
    lockFlags |= def.motionLocks.angularX ? BodyFlags.lockAngularX : 0;
    lockFlags |= def.motionLocks.angularY ? BodyFlags.lockAngularY : 0;
    lockFlags |= def.motionLocks.angularZ ? BodyFlags.lockAngularZ : 0;

    let flags = lockFlags;
    flags |= def.isBullet ? BodyFlags.isBullet : 0;
    flags |= def.allowFastRotation ? BodyFlags.allowFastRotation : 0;
    flags |= def.enableSleep ? BodyFlags.enableSleep : 0;
    flags |= def.enableContactRecycling ? BodyFlags.enableContactRecycling : 0;
    const k = kernel(world.ecsState);
    const bodyId = k.bodyCreateSim(
        world.worldId,
        def.type,
        flags,
        def.isAwake,
        def.isEnabled,
        def.sleepThreshold,
        def.position.x,
        def.position.y,
        def.position.z,
        def.rotation.v.x,
        def.rotation.v.y,
        def.rotation.v.z,
        def.rotation.s,
        def.linearVelocity.x,
        def.linearVelocity.y,
        def.linearVelocity.z,
        def.angularVelocity.x,
        def.angularVelocity.y,
        def.angularVelocity.z,
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

/** Wake a sleeping body's set (b3WakeBody). @returns whether the body was sleeping. */
export function wakeBody(world: WorldState, body: number): boolean {
    if (bodyField(world, body, BodyField.setIndex) >= SetType.FirstSleeping) {
        wakeSolverSet(world, bodyField(world, body, BodyField.setIndex));
        return true;
    }
    return false;
}

/** Destroy a body and everything attached to it (b3DestroyBody). */
export function destroyBody(world: WorldState, body: number): void {
    world.locked = true;

    const wakeBodies = true;

    // Destroy attached joints
    let jointKey = bodyField(world, body, BodyField.headJointKey);
    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const joint = world.joints[jointId];
        jointKey = joint.edges[edgeIndex].nextKey;
        destroyJointInternal(world, joint, wakeBodies);
    }

    destroyBodyContacts(world, body, wakeBodies);

    // Destroy attached shapes and their proxies
    let shapeId = bodyField(world, body, BodyField.headShapeId);
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

    const moved = kernel(world.ecsState).bodyDestroy(world.worldId, body) | 0;
    if (moved !== NULL_INDEX) syncBodyQuery(world, moved);
    world.bodyUserData[body] = undefined;
    world.bodyNames[body] = "";

    world.locked = false;
}

export function bodyDisable(world: WorldState, body: number): void {
    if (world.locked) return;
    if (bodyField(world, body, BodyField.setIndex) === SetType.Disabled) return;
    world.locked = true;
    destroyBodyContacts(world, body, true);
    let key = bodyField(world, body, BodyField.headJointKey);
    while (key !== NULL_INDEX) {
        const joint = world.joints[key >> 1];
        const edge = key & 1;
        key = joint.edges[edge].nextKey;
        if (joint.setIndex === SetType.Disabled) continue;
        unlinkJoint(world, joint);
        transferJoint(world, SetType.Disabled, joint.setIndex, joint);
    }
    let shapeId = bodyField(world, body, BodyField.headShapeId);
    while (shapeId !== NULL_INDEX) {
        const shape = world.shapes[shapeId];
        destroyShapeProxy(shape, world.broadPhase);
        shapeId = shape.nextShapeId;
    }
    removeBodyFromIsland(world, body);
    transferBody(world, SetType.Disabled, bodyField(world, body, BodyField.setIndex), body);
    world.locked = false;
}

export function bodyEnable(world: WorldState, body: number): void {
    if (world.locked) return;
    if (bodyField(world, body, BodyField.setIndex) !== SetType.Disabled) return;
    const target = bodyType(world, body) === BodyType.Static ? SetType.Static : SetType.Awake;
    transferBody(world, target, SetType.Disabled, body);
    let shapeId = bodyField(world, body, BodyField.headShapeId);
    while (shapeId !== NULL_INDEX) {
        createBodyProxy(world, body, shapeId);
        shapeId = world.shapes[shapeId].nextShapeId;
    }
    if (target !== SetType.Static) createIslandForBody(world, body);
    let key = bodyField(world, body, BodyField.headJointKey);
    while (key !== NULL_INDEX) {
        const joint = world.joints[key >> 1];
        const edge = key & 1;
        key = joint.edges[edge].nextKey;
        const a = bodyField(world, joint.edges[0].bodyId, BodyField.setIndex);
        const b = bodyField(world, joint.edges[1].bodyId, BodyField.setIndex);
        if (a === SetType.Disabled || b === SetType.Disabled) continue;
        const set = a === SetType.Static ? b : a;
        transferJoint(world, set, SetType.Disabled, joint);
        if (set !== SetType.Static) linkJoint(world, joint);
    }
}

/** Recompute mass, center of mass, and inertia from the body's shapes (b3UpdateBodyMassData). */
const massExtentCenter = vec3.zero();
export function updateBodyMassData(world: WorldState, body: number): void {
    const k = kernel(world.ecsState);
    k.bodyMassBegin(world.worldId, body);
    const type = bodyType(world, body);
    if (type === BodyType.Dynamic) {
        let shapeId = bodyField(world, body, BodyField.headShapeId);
        while (shapeId !== NULL_INDEX) {
            const shape = world.shapes[shapeId];
            if (shape.density === 0) {
                k.bodyMassInput(world.worldId, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            } else {
                const data = computeShapeMass(shape);
                const m = data.inertia;
                k.bodyMassInput(
                    world.worldId,
                    data.mass,
                    data.center.x,
                    data.center.y,
                    data.center.z,
                    m.cx.x,
                    m.cx.y,
                    m.cx.z,
                    m.cy.x,
                    m.cy.y,
                    m.cy.z,
                    m.cz.x,
                    m.cz.y,
                    m.cz.z,
                );
            }
            shapeId = shape.nextShapeId;
        }
    }
    k.bodyMassFinish(world.worldId, body);
    if (type !== BodyType.Static) {
        readSimLocalCenter(world, getBodySim(world, body), massExtentCenter);
        let shapeId = bodyField(world, body, BodyField.headShapeId);
        while (shapeId !== NULL_INDEX) {
            const shape = world.shapes[shapeId];
            const extent = computeShapeExtent(shape, massExtentCenter);
            k.bodyMassExtent(
                world.worldId,
                body,
                extent.minExtent,
                extent.maxExtent.x,
                extent.maxExtent.y,
                extent.maxExtent.z,
            );
            shapeId = shape.nextShapeId;
        }
    }
    if (bodyField(world, body, BodyField.shapeCount) > 0) syncBodyQuery(world, body);
}

/** @returns the body's mass, local center of mass, and rotational inertia (b3Body_GetMassData). */
export function getMassData(world: WorldState, body: number): MassData {
    const localCenterScratch1 = { x: 0, y: 0, z: 0 };

    const bodySim = getBodySim(world, body);
    return {
        mass: bodyField(world, body, BodyField.mass),
        center: readSimLocalCenter(world, bodySim, localCenterScratch1),
        inertia: bodyInertia(world, body, mat3.zero()),
    };
}
