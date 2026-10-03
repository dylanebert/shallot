import { f32, quat, type Vec2, type Vec3, vec3 } from "../common/math";
import { getBodySim, getBodyState, getBodyTransformQuick } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Prismatic joint payload (b3PrismaticJoint). Impulses persist across steps for warm starting. */
export type PrismaticJoint = {
    perpImpulse: Vec2;
    angularImpulse: Vec3;
    springImpulse: number;
    motorImpulse: number;
    lowerImpulse: number;
    upperImpulse: number;
    hertz: number;
    dampingRatio: number;
    maxMotorForce: number;
    motorSpeed: number;
    targetTranslation: number;
    lowerTranslation: number;
    upperTranslation: number;
    enableSpring: boolean;
    enableLimit: boolean;
    enableMotor: boolean;
};

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

/** Create a prismatic joint (b3CreatePrismaticJoint). @returns the joint handle + sim. */
export function createPrismaticJoint(
    world: WorldState,
    def: PrismaticJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Prismatic);
    const data: PrismaticJoint = {
        perpImpulse: { x: 0, y: 0 },
        angularImpulse: { x: 0, y: 0, z: 0 },
        springImpulse: 0,
        motorImpulse: 0,
        lowerImpulse: 0,
        upperImpulse: 0,
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        maxMotorForce: def.maxMotorForce,
        motorSpeed: def.motorSpeed,
        targetTranslation: def.targetTranslation,
        lowerTranslation: def.lowerTranslation,
        upperTranslation: def.upperTranslation,
        enableSpring: def.enableSpring,
        enableLimit: def.enableLimit,
        enableMotor: def.enableMotor,
    };
    pair.sim.data = data;
    return pair;
}

export function getPrismaticJointForce(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as PrismaticJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);

    // impulse in joint space
    const impulse: Vec3 = {
        x: joint.perpImpulse.x,
        y: joint.perpImpulse.y,
        z: f32(
            f32(f32(joint.motorImpulse + joint.lowerImpulse) + joint.upperImpulse) +
                joint.springImpulse,
        ),
    };

    let force = vec3.scale(world.invH, impulse);
    force = quat.rotate(sim.localFrameA.q, force);
    force = quat.rotate(transformA.q, force);
    return force;
}

/** The reaction torque this joint applies (b3GetPrismaticJointTorque). */
export function getPrismaticJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as PrismaticJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);

    let torque = vec3.scale(world.invH, joint.angularImpulse);
    torque = quat.rotate(sim.localFrameA.q, torque);
    torque = quat.rotate(transformA.q, torque);
    return torque;
}

/** The current translation along the joint axis (b3PrismaticJoint_GetTranslation). */
export function prismaticJointTranslation(world: WorldState, sim: JointSim): number {
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);

    let jointAxis = quat.rotate(sim.localFrameA.q, vec3.axisX());
    jointAxis = quat.rotate(transformA.q, jointAxis);

    const anchorA = quat.rotate(transformA.q, sim.localFrameA.p);
    const anchorB = quat.rotate(transformB.q, sim.localFrameB.p);
    const d = vec3.add(vec3.sub(transformB.p, transformA.p), vec3.sub(anchorB, anchorA));
    return vec3.dot(d, jointAxis);
}

/** The current translation speed along the joint axis (b3PrismaticJoint_GetSpeed). */
export function prismaticJointSpeed(world: WorldState, sim: JointSim): number {
    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];
    const bodySimA = getBodySim(world, bodyA);
    const bodySimB = getBodySim(world, bodyB);
    const stateA = getBodyState(world, bodyA);
    const stateB = getBodyState(world, bodyB);

    const qA = bodySimA.transform.q;
    const qB = bodySimB.transform.q;

    const axisA = quat.rotate(qA, quat.rotate(sim.localFrameA.q, vec3.axisX()));
    const rA = quat.rotate(qA, vec3.sub(sim.localFrameA.p, bodySimA.localCenter));
    const rB = quat.rotate(qB, vec3.sub(sim.localFrameB.p, bodySimB.localCenter));

    // Difference the centers directly; positions are f32 in the single-precision build.
    const d = vec3.add(vec3.sub(bodySimB.center, bodySimA.center), vec3.sub(rB, rA));

    const zero: Vec3 = { x: 0, y: 0, z: 0 };
    const vA = stateA ? stateA.linearVelocity : zero;
    const vB = stateB ? stateB.linearVelocity : zero;
    const wA = stateA ? stateA.angularVelocity : zero;
    const wB = stateB ? stateB.angularVelocity : zero;

    const vRel = vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vec3.add(vA, vec3.cross(wA, rA)));

    // The axis moves with body A, so account for its rotation.
    return f32(vec3.dot(d, vec3.cross(wA, axisA)) + vec3.dot(axisA, vRel));
}
