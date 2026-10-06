import { FLT_MAX, type Vec3 } from "../common/math";
import { readJointReaction } from "../kernel/jointcolumns";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

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
    const pair = createJoint(world, def.base, JointType.Parallel);
    kernel(world.ecsState).jointInitParallel(
        world.worldId,
        pair.joint,
        def.hertz,
        def.dampingRatio,
        def.maxTorque,
    );
    return pair;
}

export function getParallelJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}
