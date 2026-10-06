import type { Vec3 } from "../common/math";
import { readJointReaction } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
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
/** Create a motor joint (b3CreateMotorJoint). @returns the joint handle. */
export function createMotorJoint(
    world: WorldState,
    def: MotorJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Motor);
    kernel(world.ecsState).jointInitMotor(
        world.worldId,
        pair.joint,
        def.linearVelocity.x,
        def.linearVelocity.y,
        def.linearVelocity.z,
        def.maxVelocityForce,
        def.angularVelocity.x,
        def.angularVelocity.y,
        def.angularVelocity.z,
        def.maxVelocityTorque,
        def.linearHertz,
        def.linearDampingRatio,
        def.maxSpringForce,
        def.angularHertz,
        def.angularDampingRatio,
        def.maxSpringTorque,
    );
    return pair;
}
export function getMotorJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetMotorJointTorque). */
export function getMotorJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}
