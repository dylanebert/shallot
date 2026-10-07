import { FLT_MAX, type Vec3 } from "../common/math";
import { readJointReaction } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

/** Parallel joint payload (b3ParallelJoint). Impulse persists across steps for warm starting. */

/** Parallel joint definition (b3ParallelJointDef), body handles resolved to a base JointDef. */
export type ParallelJointDef = {
    base: JointDef;
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
};

/** @returns the ported parallel joint definition defaults (b3DefaultParallelJointDef). */
export function defaultParallelJointDef(base: JointDef): ParallelJointDef {
    return {
        base,
        hertz: 1,
        dampingRatio: 1,
        maxTorque: FLT_MAX,
    };
}
/** Create a parallel joint (b3CreateParallelJoint). @returns the joint handle. */
export function createParallelJoint(
    world: WorldState,
    def: ParallelJointDef,
): {
    joint: Joint;
} {
    const joint = kernel(world.ecsState).jointCreateParallel(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.hertz,
        def.dampingRatio,
        def.maxTorque,
    );
    return finishJointCreation(world, def.base, joint);
}

export function getParallelJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}
