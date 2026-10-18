import { FLT_MAX, HUGE } from "../common/constants";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

/** Distance joint definition (b3DistanceJointDef), body handles resolved to a base JointDef. */
export type DistanceJointDef = {
    base: JointDef;
    length: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    lowerSpringForce: number;
    upperSpringForce: number;
    enableLimit: boolean;
    minLength: number;
    maxLength: number;
    enableMotor: boolean;
    maxMotorForce: number;
    motorSpeed: number;
};

/** @returns the ported distance joint definition defaults (b3DefaultDistanceJointDef). */
export function defaultDistanceJointDef(base: JointDef): DistanceJointDef {
    return {
        base,
        length: 1,
        enableSpring: false,
        hertz: 0,
        dampingRatio: 0,
        lowerSpringForce: -FLT_MAX,
        upperSpringForce: FLT_MAX,
        enableLimit: false,
        minLength: 0,
        maxLength: HUGE,
        enableMotor: false,
        maxMotorForce: 0,
        motorSpeed: 0,
    };
}

/** Create a distance joint (b3CreateDistanceJoint). @returns the joint handle. */
export function createDistanceJoint(
    world: WorldState,
    def: DistanceJointDef,
): {
    joint: Joint;
} {
    const joint = kernel(world.ecsState).jointCreateDistance(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.length,
        def.hertz,
        def.dampingRatio,
        def.lowerSpringForce,
        def.upperSpringForce,
        def.minLength,
        def.maxLength,
        def.maxMotorForce,
        def.motorSpeed,
        def.enableSpring,
        def.enableLimit,
        def.enableMotor,
    );
    return finishJointCreation(world, def.base, joint);
}

/** The current distance between the two anchor points (b3DistanceJoint_GetCurrentLength). */
export function distanceJointCurrentLength(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 7);
}
