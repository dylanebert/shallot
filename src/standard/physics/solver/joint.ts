import { ContactField, contactField } from "../collision/contact";
import {
    DJ_ENABLE,
    DJ_ENABLE_LIMIT,
    DJ_ENABLE_SPRING,
    DJ_LENGTH,
    DJ_MAX_LENGTH,
    DJ_MIN_LENGTH,
    J_CONSTRAINT_DAMPING,
    J_CONSTRAINT_HERTZ,
    J_FORCE_THRESHOLD,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    J_TORQUE_THRESHOLD,
    PJ_ENABLE,
    PJ_ENABLE_LIMIT,
    PJ_LOWER_TRANSLATION,
    PJ_UPPER_TRANSLATION,
    RJ_ENABLE,
    RJ_ENABLE_LIMIT,
    RJ_LOWER_ANGLE,
    RJ_UPPER_ANGLE,
    SJ_CONE_ANGLE,
    SJ_ENABLE,
    SJ_ENABLE_CONE_LIMIT,
    SJ_ENABLE_TWIST_LIMIT,
    SJ_LOWER_TWIST_ANGLE,
    SJ_UPPER_TWIST_ANGLE,
    WHJ_ENABLE,
    WHJ_ENABLE_SUSPENSION_LIMIT,
    WHJ_LOWER_SUSPENSION_LIMIT,
    WHJ_UPPER_SUSPENSION_LIMIT,
    WJ_ANGULAR_HERTZ,
    WJ_LINEAR_HERTZ,
} from "../kernel/columns";
import { bodyType } from "../kernel/filtercolumns";
import {
    readJointFlag,
    readJointFloat,
    writeJointFloat,
    writeJointTransform,
} from "../kernel/jointcolumns";
// Joints — the common machinery from Box3D's joint.c (Erin Catto, MIT). A joint constrains two
// bodies. The organizational handle (b3Joint) lives in world.joints and threads two doubly-linked
// edges through the attached bodies; each simulation record (b3JointSim) lives in the kernel array
// for its graph color or non-awake solver set. This hub holds create/destroy and the common queries;
// per-type authoring and queries live beside their definitions.
//
// The port uses canonical graph coloring; joints that cannot fit a real color use the overflow color.
// Every arithmetic op is fround-wrapped in the per-type files; see the README.

import { changeBodyFilter } from "../collision/bodyfilter";
import { bufferMove } from "../collision/broadphase";
import { destroyContact } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { GRAPH_COLOR_COUNT, SetType } from "../common/constants";
import { allocId, freeId } from "../common/ids";
import {
    absf,
    FLT_MAX,
    f32,
    maxf,
    maxInt,
    quat,
    type Transform,
    transformWorldPoint,
    type Vec3,
    vec3,
} from "../common/math";
import { BodyType } from "../common/types";
import { J_JOINT_ID, J_TYPE } from "../kernel/columns";
import {
    appendJointRecord,
    readJointVec3,
    removeJointRecord,
    writeJointWord,
} from "../kernel/jointcolumns";
import { type Body, readBodyTransform, wakeBody } from "../world/body";
import { linkJoint, unlinkJoint } from "../world/island";
import { wakeSolverSet } from "../world/solverset";
import type { WorldState } from "../world/world";
import { getDistanceJointForce } from "./distanceJoint";
import { createJointInGraph, removeJointFromGraph } from "./graph";
import { getMotorJointForce, getMotorJointTorque } from "./motorJoint";
import { getParallelJointTorque } from "./parallelJoint";
import { getPrismaticJointForce, getPrismaticJointTorque } from "./prismaticJoint";
import { getRevoluteJointForce, getRevoluteJointTorque } from "./revoluteJoint";
import { getSphericalJointForce, getSphericalJointTorque } from "./sphericalJoint";
import { getWeldJointForce, getWeldJointTorque } from "./weldJoint";
import { getWheelJointForce, getWheelJointTorque } from "./wheelJoint";

/** Joint kind (b3JointType). Numeric values mirror the C enum order (parallel = 0 … wheel = 8). */
export const JointType = {
    Parallel: 0,
    Distance: 1,
    Filter: 2,
    Motor: 3,
    Prismatic: 4,
    Revolute: 5,
    Spherical: 6,
    Weld: 7,
    Wheel: 8,
} as const;
export type JointType = (typeof JointType)[keyof typeof JointType];

/** One end of a joint in a body's doubly-linked joint list (b3JointEdge). */
export type JointEdge = {
    bodyId: number;
    prevKey: number;
    nextKey: number;
};

/** The organizational joint handle stored in world.joints (b3Joint). */
export type Joint = {
    userData: unknown;
    setIndex: number;
    colorIndex: number;
    localIndex: number;
    edges: [JointEdge, JointEdge];
    jointId: number;
    islandId: number;
    islandIndex: number;
    drawScale: number;
    type: JointType;
    generation: number;
    collideConnected: boolean;
};

/** The resolved base joint definition (b3JointDef, body handles already resolved to ids). */
export type JointDef = {
    bodyIdA: number;
    bodyIdB: number;
    localFrameA: Transform;
    localFrameB: Transform;
    forceThreshold: number;
    torqueThreshold: number;
    constraintHertz: number;
    constraintDampingRatio: number;
    drawScale: number;
    collideConnected: boolean;
    userData: unknown;
};
const identityTransform = (): Transform => ({
    p: {
        x: 0,
        y: 0,
        z: 0,
    },
    q: {
        v: {
            x: 0,
            y: 0,
            z: 0,
        },
        s: 1,
    },
});

/** @returns the ported base joint definition defaults (b3DefaultJointDef). */
export function defaultJointDef(): JointDef {
    return {
        bodyIdA: NULL_INDEX,
        bodyIdB: NULL_INDEX,
        localFrameA: identityTransform(),
        localFrameB: identityTransform(),
        forceThreshold: FLT_MAX,
        torqueThreshold: FLT_MAX,
        constraintHertz: 60,
        constraintDampingRatio: 2,
        drawScale: 1,
        collideConnected: false,
        userData: null,
    };
}

/** @returns a fresh zeroed joint handle slot. */
export function emptyJoint(): Joint {
    return {
        userData: null,
        setIndex: NULL_INDEX,
        colorIndex: NULL_INDEX,
        localIndex: NULL_INDEX,
        edges: [
            {
                bodyId: NULL_INDEX,
                prevKey: NULL_INDEX,
                nextKey: NULL_INDEX,
            },
            {
                bodyId: NULL_INDEX,
                prevKey: NULL_INDEX,
                nextKey: NULL_INDEX,
            },
        ],
        jointId: NULL_INDEX,
        islandId: NULL_INDEX,
        islandIndex: NULL_INDEX,
        drawScale: 0,
        type: JointType.Parallel,
        generation: 0,
        collideConnected: false,
    };
}

/** Create the common joint handle + sim and place it in the right solver set (b3CreateJoint). */
export function createJoint(
    world: WorldState,
    def: JointDef,
    type: JointType,
): {
    joint: Joint;
} {
    const bodyA = world.bodies[def.bodyIdA];
    const bodyB = world.bodies[def.bodyIdB];
    const bodyIdA = bodyA.id;
    const bodyIdB = bodyB.id;
    const maxSetIndex = maxInt(bodyA.setIndex, bodyB.setIndex);
    const jointId = allocId(world.jointIdPool);
    if (jointId === world.joints.length) {
        world.joints.push(emptyJoint());
    }
    const joint = world.joints[jointId];
    joint.jointId = jointId;
    joint.userData = def.userData;
    joint.generation += 1;
    joint.setIndex = NULL_INDEX;
    joint.colorIndex = NULL_INDEX;
    joint.localIndex = NULL_INDEX;
    joint.islandId = NULL_INDEX;
    joint.islandIndex = NULL_INDEX;
    joint.drawScale = def.drawScale;
    joint.type = type;
    joint.collideConnected = def.collideConnected;
    if (!joint.collideConnected) changeBodyFilter(world, bodyIdA, bodyIdB, 1);

    // Doubly linked list on bodyA
    joint.edges[0] = {
        bodyId: bodyIdA,
        prevKey: NULL_INDEX,
        nextKey: bodyA.headJointKey,
    };
    const keyA = (jointId << 1) | 0;
    if (bodyA.headJointKey !== NULL_INDEX) {
        const jointA = world.joints[bodyA.headJointKey >> 1];
        jointA.edges[bodyA.headJointKey & 1].prevKey = keyA;
    }
    bodyA.headJointKey = keyA;
    bodyA.jointCount += 1;

    // Doubly linked list on bodyB
    joint.edges[1] = {
        bodyId: bodyIdB,
        prevKey: NULL_INDEX,
        nextKey: bodyB.headJointKey,
    };
    const keyB = (jointId << 1) | 1;
    if (bodyB.headJointKey !== NULL_INDEX) {
        const jointB = world.joints[bodyB.headJointKey >> 1];
        jointB.edges[bodyB.headJointKey & 1].prevKey = keyB;
    }
    bodyB.headJointKey = keyB;
    bodyB.jointCount += 1;
    const sim = joint;
    if (bodyA.setIndex === SetType.Disabled || bodyB.setIndex === SetType.Disabled) {
        // if either body is disabled, create in disabled set
        joint.setIndex = SetType.Disabled;
        joint.localIndex = appendJointRecord(world, GRAPH_COLOR_COUNT + joint.setIndex);
    } else if (
        bodyType(world, bodyA.id) !== BodyType.Dynamic &&
        bodyType(world, bodyB.id) !== BodyType.Dynamic
    ) {
        // joint is not attached to a dynamic body
        joint.setIndex = SetType.Static;
        joint.localIndex = appendJointRecord(world, GRAPH_COLOR_COUNT + joint.setIndex);
    } else if (bodyA.setIndex === SetType.Awake || bodyB.setIndex === SetType.Awake) {
        // if either body is sleeping, wake it
        if (maxSetIndex >= SetType.FirstSleeping) {
            wakeSolverSet(world, maxSetIndex);
        }
        joint.setIndex = SetType.Awake;
        createJointInGraph(world, joint);
    } else {
        // joint connected between sleeping and/or static bodies
        if (
            bodyA.setIndex !== bodyB.setIndex &&
            bodyA.setIndex >= SetType.FirstSleeping &&
            bodyB.setIndex >= SetType.FirstSleeping
        ) {
            // Merging two sleeping sets needs b3MergeSolverSets — no ported path / fixture reaches it.
            throw new Error(
                "physics: joint between two separate sleeping sets (needs mergeSolverSets)",
            );
        }
        const setIndex = maxSetIndex;
        joint.setIndex = setIndex;
        joint.localIndex = appendJointRecord(world, GRAPH_COLOR_COUNT + setIndex);
    }
    writeJointWord(world, joint, J_JOINT_ID, jointId);
    writeJointWord(world, joint, J_TYPE, type);
    writeJointTransform(world, sim, J_LOCAL_FRAME_A, def.localFrameA);
    writeJointTransform(world, sim, J_LOCAL_FRAME_B, def.localFrameB);
    writeJointFloat(world, sim, J_CONSTRAINT_HERTZ, def.constraintHertz);
    writeJointFloat(world, sim, J_CONSTRAINT_DAMPING, def.constraintDampingRatio);
    writeJointFloat(world, sim, J_FORCE_THRESHOLD, def.forceThreshold);
    writeJointFloat(world, sim, J_TORQUE_THRESHOLD, def.torqueThreshold);
    if (joint.setIndex > SetType.Disabled) {
        linkJoint(world, joint);
    }
    return { joint };
}

/**
 * Create a filter joint (b3CreateFilterJoint): connects two bodies solely to suppress collision
 * between them (collideConnected defaults false). It carries no constraint — prepare/warm-start/solve
 * are no-ops.
 */
export function createFilterJoint(
    world: WorldState,
    def: JointDef,
): {
    joint: Joint;
} {
    return createJoint(world, def, JointType.Filter);
}

/** Destroy a joint (b3DestroyJointInternal): unlink the body edges, the island, and the solver set. */
export function destroyJointInternal(world: WorldState, joint: Joint, wakeBodies: boolean): void {
    const jointId = joint.jointId;
    const edgeA = joint.edges[0];
    const edgeB = joint.edges[1];
    const idA = edgeA.bodyId;
    const idB = edgeB.bodyId;
    const bodyA = world.bodies[idA];
    const bodyB = world.bodies[idB];
    if (!joint.collideConnected) changeBodyFilter(world, idA, idB, -1);

    // Remove from body A
    if (edgeA.prevKey !== NULL_INDEX) {
        const prevJoint = world.joints[edgeA.prevKey >> 1];
        prevJoint.edges[edgeA.prevKey & 1].nextKey = edgeA.nextKey;
    }
    if (edgeA.nextKey !== NULL_INDEX) {
        const nextJoint = world.joints[edgeA.nextKey >> 1];
        nextJoint.edges[edgeA.nextKey & 1].prevKey = edgeA.prevKey;
    }
    const edgeKeyA = (jointId << 1) | 0;
    if (bodyA.headJointKey === edgeKeyA) {
        bodyA.headJointKey = edgeA.nextKey;
    }
    bodyA.jointCount -= 1;

    // Remove from body B
    if (edgeB.prevKey !== NULL_INDEX) {
        const prevJoint = world.joints[edgeB.prevKey >> 1];
        prevJoint.edges[edgeB.prevKey & 1].nextKey = edgeB.nextKey;
    }
    if (edgeB.nextKey !== NULL_INDEX) {
        const nextJoint = world.joints[edgeB.nextKey >> 1];
        nextJoint.edges[edgeB.nextKey & 1].prevKey = edgeB.prevKey;
    }
    const edgeKeyB = (jointId << 1) | 1;
    if (bodyB.headJointKey === edgeKeyB) {
        bodyB.headJointKey = edgeB.nextKey;
    }
    bodyB.jointCount -= 1;
    if (joint.islandId !== NULL_INDEX) {
        unlinkJoint(world, joint);
    }

    // Remove joint from the solver set that owns it
    const setIndex = joint.setIndex;
    const localIndex = joint.localIndex;
    if (setIndex === SetType.Awake) {
        removeJointFromGraph(
            world,
            joint.edges[0].bodyId,
            joint.edges[1].bodyId,
            joint.colorIndex,
            localIndex,
        );
    } else {
        removeJointRecord(world, GRAPH_COLOR_COUNT + setIndex, localIndex);
    }

    // Free joint and id (preserve joint generation)
    joint.setIndex = NULL_INDEX;
    joint.localIndex = NULL_INDEX;
    joint.colorIndex = NULL_INDEX;
    joint.jointId = NULL_INDEX;
    freeId(world.jointIdPool, jointId);
    if (wakeBodies) {
        wakeBody(world, bodyA);
        wakeBody(world, bodyB);
    }
}

// --- Dispatch ---------------------------------------------------------------------------------

/** The constraint force this joint applies (b3GetJointConstraintForce). */
export function getJointConstraintForce(world: WorldState, sim: Joint): Vec3 {
    switch (sim.type) {
        case JointType.Distance:
            return getDistanceJointForce(world, sim);
        case JointType.Motor:
            return getMotorJointForce(world, sim);
        case JointType.Prismatic:
            return getPrismaticJointForce(world, sim);
        case JointType.Revolute:
            return getRevoluteJointForce(world, sim);
        case JointType.Spherical:
            return getSphericalJointForce(world, sim);
        case JointType.Weld:
            return getWeldJointForce(world, sim);
        case JointType.Wheel:
            return getWheelJointForce(world, sim);
        case JointType.Parallel:
        case JointType.Filter:
            return {
                x: 0,
                y: 0,
                z: 0,
            };
    }
}

/** The constraint torque this joint applies (b3GetJointConstraintTorque). */
export function getJointConstraintTorque(world: WorldState, sim: Joint): Vec3 {
    switch (sim.type) {
        case JointType.Parallel:
            return getParallelJointTorque(world, sim);
        case JointType.Motor:
            return getMotorJointTorque(world, sim);
        case JointType.Prismatic:
            return getPrismaticJointTorque(world, sim);
        case JointType.Revolute:
            return getRevoluteJointTorque(world, sim);
        case JointType.Spherical:
            return getSphericalJointTorque(world, sim);
        case JointType.Weld:
            return getWeldJointTorque(world, sim);
        case JointType.Wheel:
            return getWheelJointTorque(world, sim);
        case JointType.Distance:
        case JointType.Filter:
            return {
                x: 0,
                y: 0,
                z: 0,
            };
    }
}

/** Toggle whether the two connected bodies collide, updating the broad-phase (b3Joint_SetCollideConnected). */
export function setJointCollideConnected(
    world: WorldState,
    joint: Joint,
    shouldCollide: boolean,
): void {
    if (joint.collideConnected === shouldCollide) {
        return;
    }
    changeBodyFilter(world, joint.edges[0].bodyId, joint.edges[1].bodyId, shouldCollide ? -1 : 1);
    joint.collideConnected = shouldCollide;
    const bodyA = world.bodies[joint.edges[0].bodyId];
    const bodyB = world.bodies[joint.edges[1].bodyId];
    if (shouldCollide) {
        // Tell the broad-phase to look for new pairs on the body with fewest shapes.
        let shapeId = bodyA.shapeCount < bodyB.shapeCount ? bodyA.headShapeId : bodyB.headShapeId;
        while (shapeId !== NULL_INDEX) {
            const shape = world.shapes[shapeId];
            if (shape.proxyKey !== NULL_INDEX) {
                bufferMove(world.broadPhase, shape.proxyKey);
            }
            shapeId = shape.nextShapeId;
        }
    } else {
        destroyContactsBetweenBodies(world, bodyA, bodyB);
    }
}

/** Destroy any contacts between two bodies (b3DestroyContactsBetweenBodies) — walk the shorter list. */
function destroyContactsBetweenBodies(world: WorldState, bodyA: Body, bodyB: Body): void {
    let contactKey: number;
    let otherBodyId: number;
    if (bodyA.contactCount < bodyB.contactCount) {
        contactKey = bodyA.headContactKey;
        otherBodyId = bodyB.id;
    } else {
        contactKey = bodyB.headContactKey;
        otherBodyId = bodyA.id;
    }

    // No need to wake bodies when a joint removes collision between them.
    while (contactKey !== NULL_INDEX) {
        const contactId = contactKey >> 1;
        const edgeIndex = contactKey & 1;
        const contact = contactId;
        contactKey = contactField(world, contact, ContactField.nextKeyA + 3 * edgeIndex);
        const otherEdgeIndex = edgeIndex ^ 1;
        if (
            contactField(world, contact, ContactField.bodyIdA + 3 * otherEdgeIndex) === otherBodyId
        ) {
            // Careful: this removes the contact from the list we are walking.
            destroyContact(world, contact, false);
        }
    }
}

/** Wake both bodies attached to a joint (b3Joint_WakeBodies). */
export function wakeJointBodies(world: WorldState, joint: Joint): void {
    world.locked = true;
    wakeBody(world, world.bodies[joint.edges[0].bodyId]);
    wakeBody(world, world.bodies[joint.edges[1].bodyId]);
    world.locked = false;
}

/** The linear separation error at the joint anchors (b3Joint_GetLinearSeparation). */
export function getJointLinearSeparation(world: WorldState, joint: Joint): number {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const sim = joint;
    const xfA = readBodyTransform(world, world.bodies[joint.edges[0].bodyId], bodyPoseScratch1);
    const xfB = readBodyTransform(world, world.bodies[joint.edges[1].bodyId], bodyPoseScratch2);
    const pA = transformWorldPoint(xfA, readJointVec3(world, sim, J_LOCAL_FRAME_A));
    const pB = transformWorldPoint(xfB, readJointVec3(world, sim, J_LOCAL_FRAME_B));
    const dp = vec3.sub(pB, pA);
    switch (joint.type) {
        case JointType.Parallel:
        case JointType.Motor:
        case JointType.Filter:
            return 0;
        case JointType.Distance: {
            const length = vec3.length(dp);
            if (readJointFlag(world, sim, DJ_ENABLE, DJ_ENABLE_SPRING)) {
                if (readJointFlag(world, sim, DJ_ENABLE, DJ_ENABLE_LIMIT)) {
                    if (length < readJointFloat(world, sim, DJ_MIN_LENGTH))
                        return f32(readJointFloat(world, sim, DJ_MIN_LENGTH) - length);
                    if (length > readJointFloat(world, sim, DJ_MAX_LENGTH))
                        return f32(length - readJointFloat(world, sim, DJ_MAX_LENGTH));
                    return 0;
                }
                return 0;
            }
            return absf(f32(length - readJointFloat(world, sim, DJ_LENGTH)));
        }
        case JointType.Revolute:
        case JointType.Spherical:
            return vec3.length(dp);
        case JointType.Weld: {
            return readJointFloat(world, sim, WJ_LINEAR_HERTZ) === 0 ? vec3.length(dp) : 0;
        }
        case JointType.Prismatic: {
            return axisSeparation(
                xfA,
                dp,
                readJointFlag(world, sim, PJ_ENABLE, PJ_ENABLE_LIMIT),
                readJointFloat(world, sim, PJ_LOWER_TRANSLATION),
                readJointFloat(world, sim, PJ_UPPER_TRANSLATION),
            );
        }
        case JointType.Wheel: {
            return axisSeparation(
                xfA,
                dp,
                readJointFlag(world, sim, WHJ_ENABLE, WHJ_ENABLE_SUSPENSION_LIMIT),
                readJointFloat(world, sim, WHJ_LOWER_SUSPENSION_LIMIT),
                readJointFloat(world, sim, WHJ_UPPER_SUSPENSION_LIMIT),
            );
        }
    }
}

/** Shared perpendicular + axial-limit separation for prismatic/wheel (their b3Joint_GetLinearSeparation arms). */
function axisSeparation(
    xfA: Transform,
    dp: Vec3,
    enableLimit: boolean,
    lower: number,
    upper: number,
): number {
    const axisA = quat.rotate(xfA.q, vec3.axisX());
    const perpA = vec3.perp(axisA);
    const perpendicular = absf(vec3.dot(perpA, dp));
    let limit = 0;
    if (enableLimit) {
        const translation = vec3.dot(axisA, dp);
        if (translation < lower) limit = f32(lower - translation);
        if (upper < translation) limit = f32(translation - upper);
    }
    return f32(Math.sqrt(f32(f32(perpendicular * perpendicular) + f32(limit * limit))));
}

/** The angular separation error at the joint (b3Joint_GetAngularSeparation). */
export function getJointAngularSeparation(world: WorldState, joint: Joint): number {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const sim = joint;
    const xfA = readBodyTransform(world, world.bodies[joint.edges[0].bodyId], bodyPoseScratch1);
    const xfB = readBodyTransform(world, world.bodies[joint.edges[1].bodyId], bodyPoseScratch2);
    const relQ = quat.invMul(xfA.q, xfB.q);
    switch (joint.type) {
        case JointType.Distance:
        case JointType.Motor:
        case JointType.Filter:
            return 0;
        case JointType.Parallel:
            // Remove the hinge angle before measuring.
            relQ.v.z = 0;
            return quat.getAngle(relQ);
        case JointType.Prismatic:
            return quat.getAngle(relQ);
        case JointType.Revolute: {
            if (readJointFlag(world, sim, RJ_ENABLE, RJ_ENABLE_LIMIT)) {
                const angle = quat.getTwistAngle(relQ);
                if (angle < readJointFloat(world, sim, RJ_LOWER_ANGLE)) return quat.getAngle(relQ);
                if (readJointFloat(world, sim, RJ_UPPER_ANGLE) < angle) return quat.getAngle(relQ);
            }
            // Remove the hinge angle.
            relQ.v.z = 0;
            return quat.getAngle(relQ);
        }
        case JointType.Spherical: {
            let sum = 0;
            if (readJointFlag(world, sim, SJ_ENABLE, SJ_ENABLE_CONE_LIMIT)) {
                const swingAngle = quat.getSwingAngle(relQ);
                sum = f32(
                    sum + maxf(0, f32(swingAngle - readJointFloat(world, sim, SJ_CONE_ANGLE))),
                );
            }
            if (readJointFlag(world, sim, SJ_ENABLE, SJ_ENABLE_TWIST_LIMIT)) {
                const twistAngle = quat.getTwistAngle(relQ);
                sum = f32(
                    sum +
                        maxf(0, f32(readJointFloat(world, sim, SJ_LOWER_TWIST_ANGLE) - twistAngle)),
                );
                sum = f32(
                    sum +
                        maxf(0, f32(twistAngle - readJointFloat(world, sim, SJ_UPPER_TWIST_ANGLE))),
                );
            }
            return sum;
        }
        case JointType.Weld: {
            return readJointFloat(world, sim, WJ_ANGULAR_HERTZ) === 0 ? quat.getAngle(relQ) : 0;
        }
        case JointType.Wheel:
            // Wheel joints do not constrain a single angular separation. The C public getter's
            // release path reports zero for this unconstrained diagnostic.
            return 0;
    }
}
