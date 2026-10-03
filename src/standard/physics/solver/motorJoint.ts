import { type Vec3, vec3 } from "../common/math";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Motor joint payload (b3MotorJoint). Impulses persist across steps for warm starting. */
export type MotorJoint = {
    linearVelocity: Vec3;
    angularVelocity: Vec3;
    maxVelocityForce: number;
    maxVelocityTorque: number;
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
    maxSpringForce: number;
    maxSpringTorque: number;
    linearVelocityImpulse: Vec3;
    angularVelocityImpulse: Vec3;
    linearSpringImpulse: Vec3;
    angularSpringImpulse: Vec3;
};

/** Motor joint definition (b3MotorJointDef), body handles resolved to a base JointDef. */
export type MotorJointDef = {
    base: JointDef;
    linearVelocity: Vec3;
    maxVelocityForce: number;
    angularVelocity: Vec3;
    maxVelocityTorque: number;
    linearHertz: number;
    linearDampingRatio: number;
    maxSpringForce: number;
    angularHertz: number;
    angularDampingRatio: number;
    maxSpringTorque: number;
};

/** @returns the ported motor joint definition defaults (b3DefaultMotorJointDef). */
export function defaultMotorJointDef(base: JointDef): MotorJointDef {
    return {
        base,
        linearVelocity: { x: 0, y: 0, z: 0 },
        maxVelocityForce: 0,
        angularVelocity: { x: 0, y: 0, z: 0 },
        maxVelocityTorque: 0,
        linearHertz: 0,
        linearDampingRatio: 0,
        maxSpringForce: 0,
        angularHertz: 0,
        angularDampingRatio: 0,
        maxSpringTorque: 0,
    };
}

const zeroVec3 = (): Vec3 => ({ x: 0, y: 0, z: 0 });

/** Create a motor joint (b3CreateMotorJoint). @returns the joint handle + sim. */
export function createMotorJoint(
    world: WorldState,
    def: MotorJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Motor);
    const data: MotorJoint = {
        linearVelocity: { ...def.linearVelocity },
        angularVelocity: { ...def.angularVelocity },
        maxVelocityForce: def.maxVelocityForce,
        maxVelocityTorque: def.maxVelocityTorque,
        linearHertz: def.linearHertz,
        linearDampingRatio: def.linearDampingRatio,
        angularHertz: def.angularHertz,
        angularDampingRatio: def.angularDampingRatio,
        maxSpringForce: def.maxSpringForce,
        maxSpringTorque: def.maxSpringTorque,
        linearVelocityImpulse: zeroVec3(),
        angularVelocityImpulse: zeroVec3(),
        linearSpringImpulse: zeroVec3(),
        angularSpringImpulse: zeroVec3(),
    };
    pair.sim.data = data;
    return pair;
}

export function getMotorJointForce(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as MotorJoint;
    return vec3.scale(world.invH, vec3.add(joint.linearVelocityImpulse, joint.linearSpringImpulse));
}

/** The reaction torque this joint applies (b3GetMotorJointTorque). */
export function getMotorJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as MotorJoint;
    return vec3.scale(
        world.invH,
        vec3.add(joint.angularVelocityImpulse, joint.angularSpringImpulse),
    );
}
