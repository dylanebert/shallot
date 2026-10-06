import { clampf, f32, maxf, minf, PI, type Quat, quat, type Vec3, vec3 } from "../common/math";
import {
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    SJ_CONE_ANGLE,
    SJ_DAMPING_RATIO,
    SJ_ENABLE,
    SJ_ENABLE_CONE_LIMIT,
    SJ_ENABLE_MOTOR,
    SJ_ENABLE_SPRING,
    SJ_ENABLE_TWIST_LIMIT,
    SJ_HERTZ,
    SJ_LINEAR_IMPULSE,
    SJ_LOWER_TWIST_ANGLE,
    SJ_LOWER_TWIST_IMPULSE,
    SJ_MAX_MOTOR_TORQUE,
    SJ_MOTOR_IMPULSE,
    SJ_MOTOR_VELOCITY,
    SJ_SPRING_IMPULSE,
    SJ_SWING_IMPULSE,
    SJ_TARGET_ROTATION,
    SJ_UPPER_TWIST_ANGLE,
    SJ_UPPER_TWIST_IMPULSE,
} from "../kernel/columns";
import {
    readJointFloat,
    readJointQuat,
    readJointVec3,
    writeJointFlag,
    writeJointFloat,
    writeJointQuat,
    writeJointVec3,
} from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { readBodyTransform } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Spherical joint payload (b3SphericalJoint). Impulses persist across steps for warm starting. */

/** Spherical joint definition (b3SphericalJointDef), body handles resolved to a base JointDef. */
export type SphericalJointDef = {
    base: JointDef;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    targetRotation: Quat;
    enableConeLimit: boolean;
    coneAngle: number;
    enableTwistLimit: boolean;
    lowerTwistAngle: number;
    upperTwistAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorVelocity: Vec3;
};

/** @returns the ported spherical joint definition defaults (b3DefaultSphericalJointDef). */
export function defaultSphericalJointDef(base: JointDef): SphericalJointDef {
    return {
        base,
        enableSpring: false,
        hertz: 0,
        dampingRatio: 0,
        targetRotation: {
            v: {
                x: 0,
                y: 0,
                z: 0,
            },
            s: 1,
        },
        enableConeLimit: false,
        coneAngle: 0,
        enableTwistLimit: false,
        lowerTwistAngle: 0,
        upperTwistAngle: 0,
        enableMotor: false,
        maxMotorTorque: 0,
        motorVelocity: {
            x: 0,
            y: 0,
            z: 0,
        },
    };
}
const zeroVec3 = (): Vec3 => ({
    x: 0,
    y: 0,
    z: 0,
});

/** Create a spherical joint (b3CreateSphericalJoint). @returns the joint handle. */
export function createSphericalJoint(
    world: WorldState,
    def: SphericalJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Spherical);
    const lowerLimit = f32(f32(-0.99) * PI);
    const upperLimit = f32(f32(0.99) * PI);
    const lowerAngle = minf(def.lowerTwistAngle, def.upperTwistAngle);
    const upperAngle = maxf(def.lowerTwistAngle, def.upperTwistAngle);
    writeJointVec3(world, pair.joint, SJ_LINEAR_IMPULSE, zeroVec3());
    writeJointVec3(world, pair.joint, SJ_SPRING_IMPULSE, zeroVec3());
    writeJointVec3(world, pair.joint, SJ_MOTOR_IMPULSE, zeroVec3());
    writeJointFloat(world, pair.joint, SJ_LOWER_TWIST_IMPULSE, 0);
    writeJointFloat(world, pair.joint, SJ_UPPER_TWIST_IMPULSE, 0);
    writeJointFloat(world, pair.joint, SJ_SWING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, SJ_HERTZ, def.hertz);
    writeJointFloat(world, pair.joint, SJ_DAMPING_RATIO, def.dampingRatio);
    writeJointFloat(world, pair.joint, SJ_MAX_MOTOR_TORQUE, def.maxMotorTorque);
    writeJointVec3(world, pair.joint, SJ_MOTOR_VELOCITY, {
        ...def.motorVelocity,
    });
    writeJointFloat(
        world,
        pair.joint,
        SJ_LOWER_TWIST_ANGLE,
        clampf(lowerAngle, lowerLimit, upperLimit),
    );
    writeJointFloat(
        world,
        pair.joint,
        SJ_UPPER_TWIST_ANGLE,
        clampf(upperAngle, lowerLimit, upperLimit),
    );
    writeJointFloat(world, pair.joint, SJ_CONE_ANGLE, clampf(def.coneAngle, 0, f32(f32(0.5) * PI)));
    writeJointQuat(world, pair.joint, SJ_TARGET_ROTATION, {
        v: {
            ...def.targetRotation.v,
        },
        s: def.targetRotation.s,
    });
    writeJointFlag(world, pair.joint, SJ_ENABLE, SJ_ENABLE_SPRING, def.enableSpring);
    writeJointFlag(world, pair.joint, SJ_ENABLE, SJ_ENABLE_MOTOR, def.enableMotor);
    writeJointFlag(world, pair.joint, SJ_ENABLE, SJ_ENABLE_CONE_LIMIT, def.enableConeLimit);
    writeJointFlag(world, pair.joint, SJ_ENABLE, SJ_ENABLE_TWIST_LIMIT, def.enableTwistLimit);
    return pair;
}
export function getSphericalJointForce(world: WorldState, sim: Joint): Vec3 {
    return vec3.scale(world.invH, readJointVec3(world, sim, SJ_LINEAR_IMPULSE));
}

/** The reaction torque this joint applies (b3GetSphericalJointTorque). */
export function getSphericalJointTorque(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const xfA = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );
    const xfB = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 1),
        bodyPoseScratch2,
    );
    const qA = quat.mul(xfA.q, readJointQuat(world, sim, J_LOCAL_FRAME_A + 3));
    const qB = quat.mul(xfB.q, readJointQuat(world, sim, J_LOCAL_FRAME_B + 3));
    const coneAxis = quat.rotate(qA, vec3.axisZ());
    const twistAxis = quat.rotate(qB, vec3.axisZ());
    const swingAxis = vec3.normalize(vec3.cross(coneAxis, twistAxis));
    let impulse = vec3.add(
        readJointVec3(world, sim, SJ_SPRING_IMPULSE),
        readJointVec3(world, sim, SJ_MOTOR_IMPULSE),
    );
    impulse = vec3.mulAdd(
        impulse,
        f32(
            readJointFloat(world, sim, SJ_LOWER_TWIST_IMPULSE) -
                readJointFloat(world, sim, SJ_UPPER_TWIST_IMPULSE),
        ),
        twistAxis,
    );
    impulse = vec3.mulAdd(impulse, readJointFloat(world, sim, SJ_SWING_IMPULSE), swingAxis);
    return vec3.scale(world.invH, impulse);
}

/** @returns the relative rotation of the two joint frames, twist-adjusted (shared by cone/twist getters). */
function relativeFrameRotation(world: WorldState, sim: Joint): Quat {
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
    const quatA = quat.mul(transformA.q, readJointQuat(world, sim, J_LOCAL_FRAME_A + 3));
    let quatB = quat.mul(transformB.q, readJointQuat(world, sim, J_LOCAL_FRAME_B + 3));
    if (quat.dot(quatA, quatB) < 0) {
        quatB = quat.negate(quatB);
    }
    return quat.invMul(quatA, quatB);
}

/** The current swing (cone) angle (b3SphericalJoint_GetConeAngle). */
export function sphericalJointConeAngle(world: WorldState, sim: Joint): number {
    return quat.getSwingAngle(relativeFrameRotation(world, sim));
}

/** The current twist angle (b3SphericalJoint_GetTwistAngle). */
export function sphericalJointTwistAngle(world: WorldState, sim: Joint): number {
    return quat.getTwistAngle(relativeFrameRotation(world, sim));
}
