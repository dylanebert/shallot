import { type Vec3, vec3 } from "../common/math";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Weld joint payload (b3WeldJoint). Impulses persist across steps for warm starting. */
export type WeldJoint = {
    linearImpulse: Vec3;
    angularImpulse: Vec3;
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
};

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

/** Create a weld joint (b3CreateWeldJoint). @returns the joint handle + sim. */
export function createWeldJoint(
    world: WorldState,
    def: WeldJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Weld);
    const data: WeldJoint = {
        linearImpulse: { x: 0, y: 0, z: 0 },
        angularImpulse: { x: 0, y: 0, z: 0 },
        linearHertz: def.linearHertz,
        linearDampingRatio: def.linearDampingRatio,
        angularHertz: def.angularHertz,
        angularDampingRatio: def.angularDampingRatio,
    };
    pair.sim.data = data;
    return pair;
}

export function getWeldJointForce(world: WorldState, sim: JointSim): Vec3 {
    return vec3.scale(world.invH, (sim.data as WeldJoint).linearImpulse);
}

/** The reaction torque this joint applies (b3GetWeldJointTorque). */
export function getWeldJointTorque(world: WorldState, sim: JointSim): Vec3 {
    return vec3.scale(world.invH, (sim.data as WeldJoint).angularImpulse);
}
