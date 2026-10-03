import { HUGE, LINEAR_SLOP } from "../common/constants";
import { FLT_MAX, f32, maxf, quat, type Vec3, vec3 } from "../common/math";
import { getBodyTransformQuick } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Distance joint payload (b3DistanceJoint). Impulses persist across steps for warm starting. */
export type DistanceJoint = {
    length: number;
    hertz: number;
    dampingRatio: number;
    lowerSpringForce: number;
    upperSpringForce: number;
    minLength: number;
    maxLength: number;
    maxMotorForce: number;
    motorSpeed: number;
    impulse: number;
    lowerImpulse: number;
    upperImpulse: number;
    motorImpulse: number;
    enableSpring: boolean;
    enableLimit: boolean;
    enableMotor: boolean;
};

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

/** Create a distance joint (b3CreateDistanceJoint). @returns the joint handle + sim. */
export function createDistanceJoint(
    world: WorldState,
    def: DistanceJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Distance);
    const data: DistanceJoint = {
        length: maxf(def.length, LINEAR_SLOP),
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        lowerSpringForce: def.lowerSpringForce,
        upperSpringForce: def.upperSpringForce,
        minLength: maxf(def.minLength, LINEAR_SLOP),
        maxLength: maxf(def.minLength, def.maxLength),
        maxMotorForce: def.maxMotorForce,
        motorSpeed: def.motorSpeed,
        impulse: 0,
        lowerImpulse: 0,
        upperImpulse: 0,
        motorImpulse: 0,
        enableSpring: def.enableSpring,
        enableLimit: def.enableLimit,
        enableMotor: def.enableMotor,
    };
    pair.sim.data = data;
    return pair;
}

export function getDistanceJointForce(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as DistanceJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);

    const pA = vec3.add(quat.rotate(transformA.q, sim.localFrameA.p), transformA.p);
    const pB = vec3.add(quat.rotate(transformB.q, sim.localFrameB.p), transformB.p);
    const d = vec3.sub(pB, pA);
    const axis = vec3.normalize(d);
    const force = f32(
        f32(
            f32(f32(joint.impulse + joint.lowerImpulse) - joint.upperImpulse) + joint.motorImpulse,
        ) * world.invH,
    );
    return vec3.scale(force, axis);
}

/** The current distance between the two anchor points (b3DistanceJoint_GetCurrentLength). */
export function distanceJointCurrentLength(world: WorldState, sim: JointSim): number {
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);
    const pA = vec3.add(quat.rotate(transformA.q, sim.localFrameA.p), transformA.p);
    const pB = vec3.add(quat.rotate(transformB.q, sim.localFrameB.p), transformB.p);
    const d = vec3.sub(pB, pA);
    return vec3.length(d);
}
