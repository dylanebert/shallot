import type { Quat, Vec3 } from "../common/records";
import { readJointReaction } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

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
    const joint = kernel(world.ecsState).jointCreateSpherical(
        world.worldId,
        writeJointDefinition(world, def.base),
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
    return finishJointCreation(world, def.base, joint);
}
export function getSphericalJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetSphericalJointTorque). */
export function getSphericalJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}

/** The current swing (cone) angle (b3SphericalJoint_GetConeAngle). */
export function sphericalJointConeAngle(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 1);
}

/** The current twist angle (b3SphericalJoint_GetTwistAngle). */
export function sphericalJointTwistAngle(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 2);
}
