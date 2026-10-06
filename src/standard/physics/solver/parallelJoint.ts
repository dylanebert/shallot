import { FLT_MAX, type Quat, type Vec3 } from "../common/math";
import {
    PLJ_DAMPING_RATIO,
    PLJ_HERTZ,
    PLJ_MAX_TORQUE,
    PLJ_PERP_AXIS_X,
    PLJ_PERP_AXIS_Y,
    PLJ_PERP_IMPULSE,
    PLJ_QUAT_A,
    PLJ_QUAT_B,
} from "../kernel/columns";
import {
    readJointReaction,
    writeJointFloat,
    writeJointQuat,
    writeJointVec2,
    writeJointVec3,
} from "../kernel/jointcolumns";
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
const identityQuat = (): Quat => ({
    v: {
        x: 0,
        y: 0,
        z: 0,
    },
    s: 1,
});

/** Create a parallel joint (b3CreateParallelJoint). @returns the joint handle. */
export function createParallelJoint(
    world: WorldState,
    def: ParallelJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Parallel);
    writeJointVec2(world, pair.joint, PLJ_PERP_IMPULSE, {
        x: 0,
        y: 0,
    });
    writeJointFloat(world, pair.joint, PLJ_HERTZ, def.hertz);
    writeJointFloat(world, pair.joint, PLJ_DAMPING_RATIO, def.dampingRatio);
    writeJointFloat(world, pair.joint, PLJ_MAX_TORQUE, def.maxTorque);
    writeJointQuat(world, pair.joint, PLJ_QUAT_A, identityQuat());
    writeJointQuat(world, pair.joint, PLJ_QUAT_B, identityQuat());
    writeJointVec3(world, pair.joint, PLJ_PERP_AXIS_X, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointVec3(world, pair.joint, PLJ_PERP_AXIS_Y, {
        x: 0,
        y: 0,
        z: 0,
    });
    return pair;
}

export function getParallelJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}
