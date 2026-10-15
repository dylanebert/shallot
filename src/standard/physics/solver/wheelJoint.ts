import type { Vec3 } from "../common/records";

const f32 = Math.fround;

import { readJointReaction } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

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
    const joint = kernel(world.ecsState).jointCreateWheel(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.enableSuspensionSpring,
        def.suspensionHertz,
        def.suspensionDampingRatio,
        def.enableSuspensionLimit,
        def.lowerSuspensionLimit,
        def.upperSuspensionLimit,
        def.enableSpinMotor,
        def.maxSpinTorque,
        def.spinSpeed,
        def.enableSteering,
        def.steeringHertz,
        def.steeringDampingRatio,
        def.targetSteeringAngle,
        def.maxSteeringTorque,
        def.enableSteeringLimit,
        def.lowerSteeringLimit,
        def.upperSteeringLimit,
    );
    return finishJointCreation(world, def.base, joint);
}
export function getWheelJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetWheelJointTorque). */
export function getWheelJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}

/** The spin speed of the wheel about its spin axis (b3WheelJoint_GetSpinSpeed). */
export function wheelJointSpinSpeed(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 9);
}

/** The current steering angle about body A's x-axis (b3WheelJoint_GetSteeringAngle). */
export function wheelJointSteeringAngle(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 3);
}
