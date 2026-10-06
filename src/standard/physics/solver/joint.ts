// Joint definitions and public queries use the kernel's identity and simulation records by id.
import { bufferMove } from "../collision/broadphase";
import { ContactField, contactField, destroyContact } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import {
    absf,
    FLT_MAX,
    f32,
    maxf,
    quat,
    type Transform,
    transformWorldPoint,
    type Vec3,
    vec3,
} from "../common/math";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    DJ_ENABLE,
    DJ_ENABLE_LIMIT,
    DJ_ENABLE_SPRING,
    DJ_LENGTH,
    DJ_MAX_LENGTH,
    DJ_MIN_LENGTH,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
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
import { readJointFlag, readJointFloat, readJointVec3 } from "../kernel/jointcolumns";
import { JointField, jointField, setJointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import { ShapeField, shapeField } from "../kernel/shaperecords";
import { readBodyTransform, wakeBody } from "../world/body";
import type { WorldState } from "../world/world";
import { getDistanceJointForce } from "./distanceJoint";
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

/** Kernel joint identity index. */
export type Joint = number;

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

/** Create a joint identity and sim in Box3D's solver-set and island order. */
export function createJointRecord(world: WorldState, def: JointDef, type: JointType): Joint {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    const joint = k.jointCreate(
        def.bodyIdA,
        def.bodyIdB,
        type,
        def.drawScale,
        +def.collideConnected,
        def.localFrameA.p.x,
        def.localFrameA.p.y,
        def.localFrameA.p.z,
        def.localFrameA.q.v.x,
        def.localFrameA.q.v.y,
        def.localFrameA.q.v.z,
        def.localFrameA.q.s,
        def.localFrameB.p.x,
        def.localFrameB.p.y,
        def.localFrameB.p.z,
        def.localFrameB.q.v.x,
        def.localFrameB.q.v.y,
        def.localFrameB.q.v.z,
        def.localFrameB.q.s,
        def.forceThreshold,
        def.torqueThreshold,
        def.constraintHertz,
        def.constraintDampingRatio,
    );
    world.jointUserData[joint] = def.userData;
    return joint;
}

export function createJoint(world: WorldState, def: JointDef, type: JointType): { joint: Joint } {
    return { joint: createJointRecord(world, def, type) };
}

/** A filter joint suppresses collision and carries no solver constraint. */
export function createFilterJoint(world: WorldState, def: JointDef): { joint: Joint } {
    return createJoint(world, def, JointType.Filter);
}

/** Unlink and free a joint, optionally waking both attached solver sets. */
export function destroyJointInternal(world: WorldState, joint: Joint, wakeBodies: boolean): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.jointDestroy(joint, +wakeBodies);
    world.jointUserData[joint] = null;
}

// --- Dispatch ---------------------------------------------------------------------------------

/** The constraint force this joint applies (b3GetJointConstraintForce). */
export function getJointConstraintForce(world: WorldState, sim: Joint): Vec3 {
    switch (jointField(world, sim, JointField.type) as JointType) {
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
    switch (jointField(world, sim, JointField.type) as JointType) {
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
    if (!!jointField(world, joint, JointField.collideConnected) === shouldCollide) {
        return;
    }
    setJointField(world, joint, JointField.collideConnected, +shouldCollide);
    const bodyA = jointField(world, joint, JointField.bodyIdA + 3 * 0);
    const bodyB = jointField(world, joint, JointField.bodyIdA + 3 * 1);
    if (shouldCollide) {
        // Tell the broad-phase to look for new pairs on the body with fewest shapes.
        let shapeId =
            bodyField(world, bodyA, BodyField.shapeCount) <
            bodyField(world, bodyB, BodyField.shapeCount)
                ? bodyField(world, bodyA, BodyField.headShapeId)
                : bodyField(world, bodyB, BodyField.headShapeId);
        while (shapeId !== NULL_INDEX) {
            const shape = shapeId;
            if (shapeField(world, shape, ShapeField.proxyKey) !== NULL_INDEX) {
                bufferMove(world.broadPhase, shapeField(world, shape, ShapeField.proxyKey));
            }
            shapeId = shapeField(world, shape, ShapeField.nextShapeId);
        }
    } else {
        destroyContactsBetweenBodies(world, bodyA, bodyB);
    }
}

/** Destroy any contacts between two bodies (b3DestroyContactsBetweenBodies) — walk the shorter list. */
function destroyContactsBetweenBodies(world: WorldState, bodyA: number, bodyB: number): void {
    let contactKey: number;
    let otherBodyId: number;
    if (
        bodyField(world, bodyA, BodyField.contactCount) <
        bodyField(world, bodyB, BodyField.contactCount)
    ) {
        contactKey = bodyField(world, bodyA, BodyField.headContactKey);
        otherBodyId = bodyField(world, bodyB, BodyField.id);
    } else {
        contactKey = bodyField(world, bodyB, BodyField.headContactKey);
        otherBodyId = bodyField(world, bodyA, BodyField.id);
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
    wakeBody(world, jointField(world, joint, JointField.bodyIdA + 3 * 0));
    wakeBody(world, jointField(world, joint, JointField.bodyIdA + 3 * 1));
    world.locked = false;
}

/** The linear separation error at the joint anchors (b3Joint_GetLinearSeparation). */
export function getJointLinearSeparation(world: WorldState, joint: Joint): number {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const sim = joint;
    const xfA = readBodyTransform(
        world,
        jointField(world, joint, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );
    const xfB = readBodyTransform(
        world,
        jointField(world, joint, JointField.bodyIdA + 3 * 1),
        bodyPoseScratch2,
    );
    const pA = transformWorldPoint(xfA, readJointVec3(world, sim, J_LOCAL_FRAME_A));
    const pB = transformWorldPoint(xfB, readJointVec3(world, sim, J_LOCAL_FRAME_B));
    const dp = vec3.sub(pB, pA);
    switch (jointField(world, joint, JointField.type) as JointType) {
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
    const xfA = readBodyTransform(
        world,
        jointField(world, joint, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );
    const xfB = readBodyTransform(
        world,
        jointField(world, joint, JointField.bodyIdA + 3 * 1),
        bodyPoseScratch2,
    );
    const relQ = quat.invMul(xfA.q, xfB.q);
    switch (jointField(world, joint, JointField.type) as JointType) {
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
