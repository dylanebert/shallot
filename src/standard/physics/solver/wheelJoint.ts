import { atan2, f32, mat3, quat, type Vec3, vec3 } from "../common/math";
import { bodySimSlot, readSimTransform, readStateAngularVelocity } from "../kernel/bodycolumns";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    WHJ_ANGULAR_IMPULSE,
    WHJ_ENABLE,
    WHJ_ENABLE_SPIN_MOTOR,
    WHJ_ENABLE_STEERING,
    WHJ_ENABLE_STEERING_LIMIT,
    WHJ_ENABLE_SUSPENSION_LIMIT,
    WHJ_ENABLE_SUSPENSION_SPRING,
    WHJ_LINEAR_IMPULSE,
    WHJ_LOWER_STEERING_IMPULSE,
    WHJ_LOWER_STEERING_LIMIT,
    WHJ_LOWER_SUSPENSION_IMPULSE,
    WHJ_LOWER_SUSPENSION_LIMIT,
    WHJ_MAX_SPIN_TORQUE,
    WHJ_MAX_STEERING_TORQUE,
    WHJ_SPIN_IMPULSE,
    WHJ_SPIN_SPEED,
    WHJ_STEERING_DAMPING_RATIO,
    WHJ_STEERING_HERTZ,
    WHJ_STEERING_SPRING_IMPULSE,
    WHJ_SUSPENSION_DAMPING_RATIO,
    WHJ_SUSPENSION_HERTZ,
    WHJ_SUSPENSION_SPRING_IMPULSE,
    WHJ_TARGET_STEERING_ANGLE,
    WHJ_UPPER_STEERING_IMPULSE,
    WHJ_UPPER_STEERING_LIMIT,
    WHJ_UPPER_SUSPENSION_IMPULSE,
    WHJ_UPPER_SUSPENSION_LIMIT,
} from "../kernel/columns";
import {
    readJointFloat,
    readJointQuat,
    readJointVec2,
    writeJointFlag,
    writeJointFloat,
    writeJointVec2,
} from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { getBodyState, readBodyTransform } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Wheel joint payload (b3WheelJoint). Impulses persist across steps for warm starting. */

/** Wheel joint definition (b3WheelJointDef), body handles resolved to a base JointDef. */
export type WheelJointDef = {
    base: JointDef;
    enableSuspensionSpring: boolean;
    suspensionHertz: number;
    suspensionDampingRatio: number;
    enableSuspensionLimit: boolean;
    lowerSuspensionLimit: number;
    upperSuspensionLimit: number;
    enableSpinMotor: boolean;
    maxSpinTorque: number;
    spinSpeed: number;
    enableSteering: boolean;
    steeringHertz: number;
    steeringDampingRatio: number;
    targetSteeringAngle: number;
    maxSteeringTorque: number;
    enableSteeringLimit: boolean;
    lowerSteeringLimit: number;
    upperSteeringLimit: number;
};

/** @returns the ported wheel joint definition defaults (b3DefaultWheelJointDef). */
export function defaultWheelJointDef(base: JointDef): WheelJointDef {
    return {
        base,
        enableSuspensionSpring: true,
        suspensionHertz: 1,
        suspensionDampingRatio: f32(0.7),
        enableSuspensionLimit: false,
        lowerSuspensionLimit: 0,
        upperSuspensionLimit: 0,
        enableSpinMotor: false,
        maxSpinTorque: 0,
        spinSpeed: 0,
        enableSteering: false,
        steeringHertz: 1,
        steeringDampingRatio: f32(0.7),
        targetSteeringAngle: 0,
        maxSteeringTorque: 0,
        enableSteeringLimit: false,
        lowerSteeringLimit: 0,
        upperSteeringLimit: 0,
    };
}

/** Create a wheel joint (b3CreateWheelJoint). @returns the joint handle. */
export function createWheelJoint(
    world: WorldState,
    def: WheelJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Wheel);
    writeJointVec2(world, pair.joint, WHJ_LINEAR_IMPULSE, {
        x: 0,
        y: 0,
    });
    writeJointVec2(world, pair.joint, WHJ_ANGULAR_IMPULSE, {
        x: 0,
        y: 0,
    });
    writeJointFloat(world, pair.joint, WHJ_SPIN_IMPULSE, 0);
    writeJointFloat(world, pair.joint, WHJ_MAX_SPIN_TORQUE, def.maxSpinTorque);
    writeJointFloat(world, pair.joint, WHJ_SPIN_SPEED, def.spinSpeed);
    writeJointFloat(world, pair.joint, WHJ_SUSPENSION_SPRING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, WHJ_LOWER_SUSPENSION_IMPULSE, 0);
    writeJointFloat(world, pair.joint, WHJ_UPPER_SUSPENSION_IMPULSE, 0);
    writeJointFloat(world, pair.joint, WHJ_LOWER_SUSPENSION_LIMIT, def.lowerSuspensionLimit);
    writeJointFloat(world, pair.joint, WHJ_UPPER_SUSPENSION_LIMIT, def.upperSuspensionLimit);
    writeJointFloat(world, pair.joint, WHJ_SUSPENSION_HERTZ, def.suspensionHertz);
    writeJointFloat(world, pair.joint, WHJ_SUSPENSION_DAMPING_RATIO, def.suspensionDampingRatio);
    writeJointFloat(world, pair.joint, WHJ_STEERING_SPRING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, WHJ_LOWER_STEERING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, WHJ_UPPER_STEERING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, WHJ_LOWER_STEERING_LIMIT, def.lowerSteeringLimit);
    writeJointFloat(world, pair.joint, WHJ_UPPER_STEERING_LIMIT, def.upperSteeringLimit);
    writeJointFloat(world, pair.joint, WHJ_TARGET_STEERING_ANGLE, def.targetSteeringAngle);
    writeJointFloat(world, pair.joint, WHJ_MAX_STEERING_TORQUE, def.maxSteeringTorque);
    writeJointFloat(world, pair.joint, WHJ_STEERING_HERTZ, def.steeringHertz);
    writeJointFloat(world, pair.joint, WHJ_STEERING_DAMPING_RATIO, def.steeringDampingRatio);
    writeJointFlag(world, pair.joint, WHJ_ENABLE, WHJ_ENABLE_SPIN_MOTOR, def.enableSpinMotor);
    writeJointFlag(
        world,
        pair.joint,
        WHJ_ENABLE,
        WHJ_ENABLE_SUSPENSION_SPRING,
        def.enableSuspensionSpring,
    );
    writeJointFlag(
        world,
        pair.joint,
        WHJ_ENABLE,
        WHJ_ENABLE_SUSPENSION_LIMIT,
        def.enableSuspensionLimit,
    );
    writeJointFlag(world, pair.joint, WHJ_ENABLE, WHJ_ENABLE_STEERING, def.enableSteering);
    writeJointFlag(
        world,
        pair.joint,
        WHJ_ENABLE,
        WHJ_ENABLE_STEERING_LIMIT,
        def.enableSteeringLimit,
    );
    return pair;
}
export function getWheelJointForce(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );

    // impulse in joint space. The z term reads lowerSuspensionLimit (a config value, not an impulse) —
    // an upstream quirk in b3GetWheelJointForce, kept verbatim so this accessor matches C. Not "fixed"
    // to lowerSuspensionImpulse: force accessors aren't hashed, but the port stays faithful to the C API.
    const impulse: Vec3 = {
        x: readJointVec2(world, sim, WHJ_LINEAR_IMPULSE).x,
        y: readJointVec2(world, sim, WHJ_LINEAR_IMPULSE).y,
        z: f32(
            f32(
                readJointFloat(world, sim, WHJ_LOWER_SUSPENSION_LIMIT) +
                    readJointFloat(world, sim, WHJ_UPPER_SUSPENSION_IMPULSE),
            ) + readJointFloat(world, sim, WHJ_SUSPENSION_SPRING_IMPULSE),
        ),
    };
    let force = vec3.scale(world.invH, impulse);
    force = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), force);
    force = quat.rotate(transformA.q, force);
    return force;
}

/** The reaction torque this joint applies (b3GetWheelJointTorque). */
export function getWheelJointTorque(world: WorldState, sim: Joint): Vec3 {
    const transformScratch1 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };

    const bodyA = jointField(world, sim, JointField.bodyIdA + 3 * 0);
    const setA = bodyField(world, bodyA, BodyField.setIndex);
    const bodySimA = bodySimSlot(setA, bodyField(world, bodyA, BodyField.localIndex));
    const qA = quat.mul(
        readSimTransform(world, bodySimA, transformScratch1).q,
        readJointQuat(world, sim, J_LOCAL_FRAME_A + 3),
    );
    const matrixA = mat3.fromQuat(qA);
    return vec3.scale(f32(world.invH * readJointFloat(world, sim, WHJ_SPIN_IMPULSE)), matrixA.cz);
}

/** The spin speed of the wheel about its spin axis (b3WheelJoint_GetSpinSpeed). */
export function wheelJointSpinSpeed(world: WorldState, sim: Joint): number {
    const transformScratch1 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const angularVelocityScratch2 = { x: 0, y: 0, z: 0 };
    const angularVelocityScratch3 = { x: 0, y: 0, z: 0 };

    const bodyA = jointField(world, sim, JointField.bodyIdA + 3 * 0);
    const bodyB = jointField(world, sim, JointField.bodyIdA + 3 * 1);
    const setB = bodyField(world, bodyB, BodyField.setIndex);
    const bodySimB = bodySimSlot(setB, bodyField(world, bodyB, BodyField.localIndex));
    const quatB = quat.mul(
        readSimTransform(world, bodySimB, transformScratch1).q,
        readJointQuat(world, sim, J_LOCAL_FRAME_B + 3),
    );
    const spinAxis = quat.rotate(quatB, vec3.axisZ());
    const zero: Vec3 = {
        x: 0,
        y: 0,
        z: 0,
    };
    const stateA = getBodyState(world, bodyA);
    const stateB = getBodyState(world, bodyB);
    const wA =
        stateA !== null ? readStateAngularVelocity(world, stateA, angularVelocityScratch2) : zero;
    const wB =
        stateB !== null ? readStateAngularVelocity(world, stateB, angularVelocityScratch3) : zero;
    return vec3.dot(vec3.sub(wB, wA), spinAxis);
}

/** The current steering angle about body A's x-axis (b3WheelJoint_GetSteeringAngle). */
export function wheelJointSteeringAngle(world: WorldState, sim: Joint): number {
    const transformScratch1 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const transformScratch2 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };

    const bodyA = jointField(world, sim, JointField.bodyIdA + 3 * 0);
    const bodyB = jointField(world, sim, JointField.bodyIdA + 3 * 1);
    const setA = bodyField(world, bodyA, BodyField.setIndex);
    const setB = bodyField(world, bodyB, BodyField.setIndex);
    const bodySimA = bodySimSlot(setA, bodyField(world, bodyA, BodyField.localIndex));
    const bodySimB = bodySimSlot(setB, bodyField(world, bodyB, BodyField.localIndex));
    const quatA = quat.mul(
        readSimTransform(world, bodySimA, transformScratch1).q,
        readJointQuat(world, sim, J_LOCAL_FRAME_A + 3),
    );
    const quatB = quat.mul(
        readSimTransform(world, bodySimB, transformScratch2).q,
        readJointQuat(world, sim, J_LOCAL_FRAME_B + 3),
    );
    const matrixA = mat3.fromQuat(quatA);
    const matrixB = mat3.fromQuat(quatB);

    // Twist around the x-axis.
    const cs = vec3.dot(matrixB.cz, matrixA.cz);
    const ss = f32(-vec3.dot(matrixB.cz, matrixA.cy));
    return atan2(ss, cs);
}
