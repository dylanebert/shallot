import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

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
export function createPrismaticJoint(world: WorldState, def: PrismaticJointDef): { joint: Joint } {
    const joint = kernel(world.ecsState).jointCreatePrismatic(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.hertz,
        def.dampingRatio,
        def.targetTranslation,
        def.lowerTranslation,
        def.upperTranslation,
        def.maxMotorForce,
        def.motorSpeed,
        def.enableSpring,
        def.enableLimit,
        def.enableMotor,
    );
    return finishJointCreation(world, def.base, joint);
}

/** The current translation along the joint axis (b3PrismaticJoint_GetTranslation). */
export function prismaticJointTranslation(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 6);
}

/** The current translation speed along the joint axis (b3PrismaticJoint_GetSpeed). */
export function prismaticJointSpeed(world: WorldState, sim: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, sim, 8);
}
