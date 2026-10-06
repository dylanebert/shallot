import { HUGE, LINEAR_SLOP } from "../common/constants";
import { FLT_MAX, f32, maxf, quat, type Vec3, vec3 } from "../common/math";
import {
    DJ_DAMPING_RATIO,
    DJ_ENABLE,
    DJ_ENABLE_LIMIT,
    DJ_ENABLE_MOTOR,
    DJ_ENABLE_SPRING,
    DJ_HERTZ,
    DJ_IMPULSE,
    DJ_LENGTH,
    DJ_LOWER_IMPULSE,
    DJ_LOWER_SPRING_FORCE,
    DJ_MAX_LENGTH,
    DJ_MAX_MOTOR_FORCE,
    DJ_MIN_LENGTH,
    DJ_MOTOR_IMPULSE,
    DJ_MOTOR_SPEED,
    DJ_UPPER_IMPULSE,
    DJ_UPPER_SPRING_FORCE,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
} from "../kernel/columns";
import {
    readJointFloat,
    readJointVec3,
    writeJointFlag,
    writeJointFloat,
} from "../kernel/jointcolumns";
import { readBodyTransform } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Distance joint payload (b3DistanceJoint). Impulses persist across steps for warm starting. */

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

/** Create a distance joint (b3CreateDistanceJoint). @returns the joint handle. */
export function createDistanceJoint(
    world: WorldState,
    def: DistanceJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Distance);
    writeJointFloat(world, pair.joint, DJ_LENGTH, maxf(def.length, LINEAR_SLOP));
    writeJointFloat(world, pair.joint, DJ_HERTZ, def.hertz);
    writeJointFloat(world, pair.joint, DJ_DAMPING_RATIO, def.dampingRatio);
    writeJointFloat(world, pair.joint, DJ_LOWER_SPRING_FORCE, def.lowerSpringForce);
    writeJointFloat(world, pair.joint, DJ_UPPER_SPRING_FORCE, def.upperSpringForce);
    writeJointFloat(world, pair.joint, DJ_MIN_LENGTH, maxf(def.minLength, LINEAR_SLOP));
    writeJointFloat(world, pair.joint, DJ_MAX_LENGTH, maxf(def.minLength, def.maxLength));
    writeJointFloat(world, pair.joint, DJ_MAX_MOTOR_FORCE, def.maxMotorForce);
    writeJointFloat(world, pair.joint, DJ_MOTOR_SPEED, def.motorSpeed);
    writeJointFloat(world, pair.joint, DJ_IMPULSE, 0);
    writeJointFloat(world, pair.joint, DJ_LOWER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, DJ_UPPER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, DJ_MOTOR_IMPULSE, 0);
    writeJointFlag(world, pair.joint, DJ_ENABLE, DJ_ENABLE_SPRING, def.enableSpring);
    writeJointFlag(world, pair.joint, DJ_ENABLE, DJ_ENABLE_LIMIT, def.enableLimit);
    writeJointFlag(world, pair.joint, DJ_ENABLE, DJ_ENABLE_MOTOR, def.enableMotor);
    return pair;
}
export function getDistanceJointForce(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(world, sim.edges[0].bodyId, bodyPoseScratch1);
    const transformB = readBodyTransform(world, sim.edges[1].bodyId, bodyPoseScratch2);
    const pA = vec3.add(
        quat.rotate(transformA.q, readJointVec3(world, sim, J_LOCAL_FRAME_A)),
        transformA.p,
    );
    const pB = vec3.add(
        quat.rotate(transformB.q, readJointVec3(world, sim, J_LOCAL_FRAME_B)),
        transformB.p,
    );
    const d = vec3.sub(pB, pA);
    const axis = vec3.normalize(d);
    const force = f32(
        f32(
            f32(
                f32(
                    readJointFloat(world, sim, DJ_IMPULSE) +
                        readJointFloat(world, sim, DJ_LOWER_IMPULSE),
                ) - readJointFloat(world, sim, DJ_UPPER_IMPULSE),
            ) + readJointFloat(world, sim, DJ_MOTOR_IMPULSE),
        ) * world.invH,
    );
    return vec3.scale(force, axis);
}

/** The current distance between the two anchor points (b3DistanceJoint_GetCurrentLength). */
export function distanceJointCurrentLength(world: WorldState, sim: Joint): number {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(world, sim.edges[0].bodyId, bodyPoseScratch1);
    const transformB = readBodyTransform(world, sim.edges[1].bodyId, bodyPoseScratch2);
    const pA = vec3.add(
        quat.rotate(transformA.q, readJointVec3(world, sim, J_LOCAL_FRAME_A)),
        transformA.p,
    );
    const pB = vec3.add(
        quat.rotate(transformB.q, readJointVec3(world, sim, J_LOCAL_FRAME_B)),
        transformB.p,
    );
    const d = vec3.sub(pB, pA);
    return vec3.length(d);
}
