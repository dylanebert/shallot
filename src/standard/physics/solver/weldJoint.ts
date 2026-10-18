import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

/** Weld joint definition (b3WeldJointDef), body handles resolved to a base JointDef. */
export type WeldJointDef = {
    base: JointDef;
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
};

/** @returns the ported weld joint definition defaults (b3DefaultWeldJointDef). */
export function defaultWeldJointDef(base: JointDef): WeldJointDef {
    return {
        base,
        linearHertz: 0,
        linearDampingRatio: 0,
        angularHertz: 0,
        angularDampingRatio: 0,
    };
}

/** Create a weld joint (b3CreateWeldJoint). @returns the joint handle. */
export function createWeldJoint(
    world: WorldState,
    def: WeldJointDef,
): {
    joint: Joint;
} {
    const joint = kernel(world.ecsState).jointCreateWeld(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.linearHertz,
        def.linearDampingRatio,
        def.angularHertz,
        def.angularDampingRatio,
    );
    return finishJointCreation(world, def.base, joint);
}
