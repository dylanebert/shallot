import type { Vec3 } from "../common/math";
import {
    MJ_ANGULAR_DAMPING_RATIO,
    MJ_ANGULAR_HERTZ,
    MJ_ANGULAR_SPRING_IMPULSE,
    MJ_ANGULAR_VELOCITY,
    MJ_ANGULAR_VELOCITY_IMPULSE,
    MJ_LINEAR_DAMPING_RATIO,
    MJ_LINEAR_HERTZ,
    MJ_LINEAR_SPRING_IMPULSE,
    MJ_LINEAR_VELOCITY,
    MJ_LINEAR_VELOCITY_IMPULSE,
    MJ_MAX_SPRING_FORCE,
    MJ_MAX_SPRING_TORQUE,
    MJ_MAX_VELOCITY_FORCE,
    MJ_MAX_VELOCITY_TORQUE,
} from "../kernel/columns";
import { readJointReaction, writeJointFloat, writeJointVec3 } from "../kernel/jointcolumns";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Motor joint payload (b3MotorJoint). Impulses persist across steps for warm starting. */

/** Motor joint definition (b3MotorJointDef), body handles resolved to a base JointDef. */
export type MotorJointDef = {
    base: JointDef;
    linearVelocity: Vec3;
    maxVelocityForce: number;
    angularVelocity: Vec3;
    maxVelocityTorque: number;
    linearHertz: number;
    linearDampingRatio: number;
    maxSpringForce: number;
    angularHertz: number;
    angularDampingRatio: number;
    maxSpringTorque: number;
};

/** @returns the ported motor joint definition defaults (b3DefaultMotorJointDef). */
export function defaultMotorJointDef(base: JointDef): MotorJointDef {
    return {
        base,
        linearVelocity: {
            x: 0,
            y: 0,
            z: 0,
        },
        maxVelocityForce: 0,
        angularVelocity: {
            x: 0,
            y: 0,
            z: 0,
        },
        maxVelocityTorque: 0,
        linearHertz: 0,
        linearDampingRatio: 0,
        maxSpringForce: 0,
        angularHertz: 0,
        angularDampingRatio: 0,
        maxSpringTorque: 0,
    };
}
const zeroVec3 = (): Vec3 => ({
    x: 0,
    y: 0,
    z: 0,
});

/** Create a motor joint (b3CreateMotorJoint). @returns the joint handle. */
export function createMotorJoint(
    world: WorldState,
    def: MotorJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Motor);
    writeJointVec3(world, pair.joint, MJ_LINEAR_VELOCITY, {
        ...def.linearVelocity,
    });
    writeJointVec3(world, pair.joint, MJ_ANGULAR_VELOCITY, {
        ...def.angularVelocity,
    });
    writeJointFloat(world, pair.joint, MJ_MAX_VELOCITY_FORCE, def.maxVelocityForce);
    writeJointFloat(world, pair.joint, MJ_MAX_VELOCITY_TORQUE, def.maxVelocityTorque);
    writeJointFloat(world, pair.joint, MJ_LINEAR_HERTZ, def.linearHertz);
    writeJointFloat(world, pair.joint, MJ_LINEAR_DAMPING_RATIO, def.linearDampingRatio);
    writeJointFloat(world, pair.joint, MJ_ANGULAR_HERTZ, def.angularHertz);
    writeJointFloat(world, pair.joint, MJ_ANGULAR_DAMPING_RATIO, def.angularDampingRatio);
    writeJointFloat(world, pair.joint, MJ_MAX_SPRING_FORCE, def.maxSpringForce);
    writeJointFloat(world, pair.joint, MJ_MAX_SPRING_TORQUE, def.maxSpringTorque);
    writeJointVec3(world, pair.joint, MJ_LINEAR_VELOCITY_IMPULSE, zeroVec3());
    writeJointVec3(world, pair.joint, MJ_ANGULAR_VELOCITY_IMPULSE, zeroVec3());
    writeJointVec3(world, pair.joint, MJ_LINEAR_SPRING_IMPULSE, zeroVec3());
    writeJointVec3(world, pair.joint, MJ_ANGULAR_SPRING_IMPULSE, zeroVec3());
    return pair;
}
export function getMotorJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetMotorJointTorque). */
export function getMotorJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}
