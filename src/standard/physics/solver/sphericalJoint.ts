import { type Quat, quat, type Vec3 } from "../common/math";
import { J_LOCAL_FRAME_A, J_LOCAL_FRAME_B } from "../kernel/columns";
import { readJointQuat, readJointReaction } from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
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
/** Create a spherical joint (b3CreateSphericalJoint). @returns the joint handle. */
export function createSphericalJoint(
    world: WorldState,
    def: SphericalJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Spherical);
    kernel(world.ecsState).jointInitSpherical(
        world.worldId,
        pair.joint,
        def.hertz,
        def.dampingRatio,
        def.targetRotation.v.x,
        def.targetRotation.v.y,
        def.targetRotation.v.z,
        def.targetRotation.s,
        def.coneAngle,
        def.lowerTwistAngle,
        def.upperTwistAngle,
        def.maxMotorTorque,
        def.motorVelocity.x,
        def.motorVelocity.y,
        def.motorVelocity.z,
        def.enableSpring,
        def.enableConeLimit,
        def.enableTwistLimit,
        def.enableMotor,
    );
    return pair;
}
export function getSphericalJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetSphericalJointTorque). */
export function getSphericalJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
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
