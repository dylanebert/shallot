import { HUGE } from "../common/constants";
import { FLT_MAX, quat, type Vec3, vec3 } from "../common/math";
import { J_LOCAL_FRAME_A, J_LOCAL_FRAME_B } from "../kernel/columns";
import { readJointReaction, readJointVec3 } from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
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
    kernel(world.ecsState).jointInitDistance(
        world.worldId,
        pair.joint,
        def.length,
        def.hertz,
        def.dampingRatio,
        def.lowerSpringForce,
        def.upperSpringForce,
        def.minLength,
        def.maxLength,
        def.maxMotorForce,
        def.motorSpeed,
        def.enableSpring,
        def.enableLimit,
        def.enableMotor,
    );
    return pair;
}
export function getDistanceJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The current distance between the two anchor points (b3DistanceJoint_GetCurrentLength). */
export function distanceJointCurrentLength(world: WorldState, sim: Joint): number {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );
    const transformB = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 1),
        bodyPoseScratch2,
    );
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
