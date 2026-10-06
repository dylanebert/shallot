import { f32, quat, type Vec3, vec3 } from "../common/math";
import {
    readSimCenter,
    readSimLocalCenter,
    readSimTransform,
    readStateAngularVelocity,
    readStateLinearVelocity,
} from "../kernel/bodycolumns";
import {
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    PJ_ANGULAR_IMPULSE,
    PJ_DAMPING_RATIO,
    PJ_ENABLE,
    PJ_ENABLE_LIMIT,
    PJ_ENABLE_MOTOR,
    PJ_ENABLE_SPRING,
    PJ_HERTZ,
    PJ_LOWER_IMPULSE,
    PJ_LOWER_TRANSLATION,
    PJ_MAX_MOTOR_FORCE,
    PJ_MOTOR_IMPULSE,
    PJ_MOTOR_SPEED,
    PJ_PERP_IMPULSE,
    PJ_SPRING_IMPULSE,
    PJ_TARGET_TRANSLATION,
    PJ_UPPER_IMPULSE,
    PJ_UPPER_TRANSLATION,
} from "../kernel/columns";
import {
    readJointFloat,
    readJointQuat,
    readJointVec2,
    readJointVec3,
    writeJointFlag,
    writeJointFloat,
    writeJointVec2,
    writeJointVec3,
} from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { getBodySim, getBodyState, readBodyTransform } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Prismatic joint payload (b3PrismaticJoint). Impulses persist across steps for warm starting. */

/** Prismatic joint definition (b3PrismaticJointDef), body handles resolved to a base JointDef. */
export type PrismaticJointDef = {
    base: JointDef;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    targetTranslation: number;
    enableLimit: boolean;
    lowerTranslation: number;
    upperTranslation: number;
    enableMotor: boolean;
    maxMotorForce: number;
    motorSpeed: number;
};

/** @returns the ported prismatic joint definition defaults (b3DefaultPrismaticJointDef). */
export function defaultPrismaticJointDef(base: JointDef): PrismaticJointDef {
    return {
        base,
        enableSpring: false,
        hertz: 0,
        dampingRatio: 0,
        targetTranslation: 0,
        enableLimit: false,
        lowerTranslation: 0,
        upperTranslation: 0,
        enableMotor: false,
        maxMotorForce: 0,
        motorSpeed: 0,
    };
}

/** Create a prismatic joint (b3CreatePrismaticJoint). @returns the joint handle. */
export function createPrismaticJoint(
    world: WorldState,
    def: PrismaticJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Prismatic);
    writeJointVec2(world, pair.joint, PJ_PERP_IMPULSE, {
        x: 0,
        y: 0,
    });
    writeJointVec3(world, pair.joint, PJ_ANGULAR_IMPULSE, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointFloat(world, pair.joint, PJ_SPRING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_MOTOR_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_LOWER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_UPPER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_HERTZ, def.hertz);
    writeJointFloat(world, pair.joint, PJ_DAMPING_RATIO, def.dampingRatio);
    writeJointFloat(world, pair.joint, PJ_MAX_MOTOR_FORCE, def.maxMotorForce);
    writeJointFloat(world, pair.joint, PJ_MOTOR_SPEED, def.motorSpeed);
    writeJointFloat(world, pair.joint, PJ_TARGET_TRANSLATION, def.targetTranslation);
    writeJointFloat(world, pair.joint, PJ_LOWER_TRANSLATION, def.lowerTranslation);
    writeJointFloat(world, pair.joint, PJ_UPPER_TRANSLATION, def.upperTranslation);
    writeJointFlag(world, pair.joint, PJ_ENABLE, PJ_ENABLE_SPRING, def.enableSpring);
    writeJointFlag(world, pair.joint, PJ_ENABLE, PJ_ENABLE_LIMIT, def.enableLimit);
    writeJointFlag(world, pair.joint, PJ_ENABLE, PJ_ENABLE_MOTOR, def.enableMotor);
    return pair;
}
export function getPrismaticJointForce(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );

    // impulse in joint space
    const impulse: Vec3 = {
        x: readJointVec2(world, sim, PJ_PERP_IMPULSE).x,
        y: readJointVec2(world, sim, PJ_PERP_IMPULSE).y,
        z: f32(
            f32(
                f32(
                    readJointFloat(world, sim, PJ_MOTOR_IMPULSE) +
                        readJointFloat(world, sim, PJ_LOWER_IMPULSE),
                ) + readJointFloat(world, sim, PJ_UPPER_IMPULSE),
            ) + readJointFloat(world, sim, PJ_SPRING_IMPULSE),
        ),
    };
    let force = vec3.scale(world.invH, impulse);
    force = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), force);
    force = quat.rotate(transformA.q, force);
    return force;
}

/** The reaction torque this joint applies (b3GetPrismaticJointTorque). */
export function getPrismaticJointTorque(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );
    let torque = vec3.scale(world.invH, readJointVec3(world, sim, PJ_ANGULAR_IMPULSE));
    torque = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), torque);
    torque = quat.rotate(transformA.q, torque);
    return torque;
}

/** The current translation along the joint axis (b3PrismaticJoint_GetTranslation). */
export function prismaticJointTranslation(world: WorldState, sim: Joint): number {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );
    const transformB = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 1),
        bodyPoseScratch2,
    );
    let jointAxis = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), vec3.axisX());
    jointAxis = quat.rotate(transformA.q, jointAxis);
    const anchorA = quat.rotate(transformA.q, readJointVec3(world, sim, J_LOCAL_FRAME_A));
    const anchorB = quat.rotate(transformB.q, readJointVec3(world, sim, J_LOCAL_FRAME_B));
    const d = vec3.add(vec3.sub(transformB.p, transformA.p), vec3.sub(anchorB, anchorA));
    return vec3.dot(d, jointAxis);
}

const speedPoseA = { p: vec3.zero(), q: quat.identity() };
const speedPoseB = { p: vec3.zero(), q: quat.identity() };
const speedFrame = quat.identity();
const speedAxisX = { x: 1, y: 0, z: 0 };
const speedAxis = vec3.zero();
const speedLocalCenter = vec3.zero();
const speedCenterA = vec3.zero();
const speedCenterB = vec3.zero();
const speedRA = vec3.zero();
const speedRB = vec3.zero();
const speedD = vec3.zero();
const speedLinearA = vec3.zero();
const speedLinearB = vec3.zero();
const speedAngularA = vec3.zero();
const speedAngularB = vec3.zero();
const speedRelative = vec3.zero();
const speedTmp = vec3.zero();
const speedZero = vec3.zero();

/** The current translation speed along the joint axis (b3PrismaticJoint_GetSpeed). */
export function prismaticJointSpeed(world: WorldState, sim: Joint): number {
    const bodyA = jointField(world, sim, JointField.bodyIdA + 3 * 0);
    const bodyB = jointField(world, sim, JointField.bodyIdA + 3 * 1);
    const bodySimA = getBodySim(world, bodyA);
    const bodySimB = getBodySim(world, bodyB);
    const stateA = getBodyState(world, bodyA);
    const stateB = getBodyState(world, bodyB);
    const qA = readSimTransform(world, bodySimA, speedPoseA).q;
    const qB = readSimTransform(world, bodySimB, speedPoseB).q;
    readJointQuat(world, sim, J_LOCAL_FRAME_A + 3, speedFrame);
    quat.rotateOut(speedFrame, speedAxisX, speedAxis);
    quat.rotateOut(qA, speedAxis, speedAxis);
    readJointVec3(world, sim, J_LOCAL_FRAME_A, speedRA);
    readSimLocalCenter(world, bodySimA, speedLocalCenter);
    vec3.subOut(speedRA, speedLocalCenter, speedRA);
    quat.rotateOut(qA, speedRA, speedRA);
    readJointVec3(world, sim, J_LOCAL_FRAME_B, speedRB);
    readSimLocalCenter(world, bodySimB, speedLocalCenter);
    vec3.subOut(speedRB, speedLocalCenter, speedRB);
    quat.rotateOut(qB, speedRB, speedRB);

    // Difference the centers directly; positions are f32 in the single-precision build.
    readSimCenter(world, bodySimA, speedCenterA);
    readSimCenter(world, bodySimB, speedCenterB);
    vec3.subOut(speedCenterB, speedCenterA, speedD);
    vec3.subOut(speedRB, speedRA, speedTmp);
    vec3.addOut(speedD, speedTmp, speedD);
    const vA = stateA !== null ? readStateLinearVelocity(world, stateA, speedLinearA) : speedZero;
    const vB = stateB !== null ? readStateLinearVelocity(world, stateB, speedLinearB) : speedZero;
    const wA = stateA !== null ? readStateAngularVelocity(world, stateA, speedAngularA) : speedZero;
    const wB = stateB !== null ? readStateAngularVelocity(world, stateB, speedAngularB) : speedZero;
    vec3.crossOut(wB, speedRB, speedRelative);
    vec3.addOut(vB, speedRelative, speedRelative);
    vec3.crossOut(wA, speedRA, speedTmp);
    vec3.addOut(vA, speedTmp, speedTmp);
    vec3.subOut(speedRelative, speedTmp, speedRelative);

    // The axis moves with body A, so account for its rotation.
    vec3.crossOut(wA, speedAxis, speedTmp);
    return f32(vec3.dot(speedD, speedTmp) + vec3.dot(speedAxis, speedRelative));
}
