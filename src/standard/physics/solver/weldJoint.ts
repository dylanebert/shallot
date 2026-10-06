import type { Vec3 } from "../common/math";
import {
    WJ_ANGULAR_DAMPING_RATIO,
    WJ_ANGULAR_HERTZ,
    WJ_ANGULAR_IMPULSE,
    WJ_LINEAR_DAMPING_RATIO,
    WJ_LINEAR_HERTZ,
    WJ_LINEAR_IMPULSE,
} from "../kernel/columns";
import { readJointReaction, writeJointFloat, writeJointVec3 } from "../kernel/jointcolumns";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Weld joint payload (b3WeldJoint). Impulses persist across steps for warm starting. */

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
    const pair = createJoint(world, def.base, JointType.Weld);
    writeJointVec3(world, pair.joint, WJ_LINEAR_IMPULSE, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointVec3(world, pair.joint, WJ_ANGULAR_IMPULSE, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointFloat(world, pair.joint, WJ_LINEAR_HERTZ, def.linearHertz);
    writeJointFloat(world, pair.joint, WJ_LINEAR_DAMPING_RATIO, def.linearDampingRatio);
    writeJointFloat(world, pair.joint, WJ_ANGULAR_HERTZ, def.angularHertz);
    writeJointFloat(world, pair.joint, WJ_ANGULAR_DAMPING_RATIO, def.angularDampingRatio);
    return pair;
}
export function getWeldJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetWeldJointTorque). */
export function getWeldJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}
