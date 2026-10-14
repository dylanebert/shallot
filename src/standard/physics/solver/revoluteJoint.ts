import type { Vec3 } from "../common/math";
import { readJointReaction } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

/** Revolute joint payload (b3RevoluteJoint). Impulses persist across steps for warm starting. */

/** Revolute joint definition (b3RevoluteJointDef), body handles resolved to a base JointDef. */
export type RevoluteJointDef = {
    base: JointDef;
    targetAngle: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    enableLimit: boolean;
    lowerAngle: number;
    upperAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorSpeed: number;
};

/** @returns the ported revolute joint definition defaults (b3DefaultRevoluteJointDef). */
export function defaultRevoluteJointDef(base: JointDef): RevoluteJointDef {
    return {
        base,
        targetAngle: 0,
        enableSpring: false,
        hertz: 0,
        dampingRatio: 0,
        enableLimit: false,
        lowerAngle: 0,
        upperAngle: 0,
        enableMotor: false,
        maxMotorTorque: 0,
        motorSpeed: 0,
    };
}
/** Create a revolute joint (b3CreateRevoluteJoint). @returns the joint handle. */
export function createRevoluteJoint(
    world: WorldState,
    def: RevoluteJointDef,
): {
    joint: Joint;
} {
    const joint = kernel(world.ecsState).jointCreateRevolute(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.hertz,
        def.dampingRatio,
        def.targetAngle,
        def.lowerAngle,
        def.upperAngle,
        def.maxMotorTorque,
        def.motorSpeed,
        def.enableSpring,
        def.enableLimit,
        def.enableMotor,
    );
    return finishJointCreation(world, def.base, joint);
}
export function getRevoluteJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetRevoluteJointTorque). */
export function getRevoluteJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}

/** The current hinge angle (b3RevoluteJoint_GetAngle): relative twist of the two joint frames. */
export function revoluteJointAngle(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 0);
}
