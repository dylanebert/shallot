import { quat, type Vec3 } from "../common/math";
import { J_LOCAL_FRAME_A, J_LOCAL_FRAME_B } from "../kernel/columns";
import { readJointQuat, readJointReaction } from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import { readBodyTransform } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Revolute joint payload (b3RevoluteJoint). Impulses persist across steps for warm starting. */

/** Revolute joint definition (b3RevoluteJointDef), body handles resolved to a base JointDef. */
export type RevoluteJointDef = {
    base: JointDef;
    targetAngle: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    enableLimit: boolean;
    lowerAngle: number;
    upperAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorSpeed: number;
};

/** @returns the ported revolute joint definition defaults (b3DefaultRevoluteJointDef). */
export function defaultRevoluteJointDef(base: JointDef): RevoluteJointDef {
    return {
        base,
        targetAngle: 0,
        enableSpring: false,
        hertz: 0,
        dampingRatio: 0,
        enableLimit: false,
        lowerAngle: 0,
        upperAngle: 0,
        enableMotor: false,
        maxMotorTorque: 0,
        motorSpeed: 0,
    };
}
/** Create a revolute joint (b3CreateRevoluteJoint). @returns the joint handle. */
export function createRevoluteJoint(
    world: WorldState,
    def: RevoluteJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Revolute);
    kernel(world.ecsState).jointInitRevolute(
        world.worldId,
        pair.joint,
        def.hertz,
        def.dampingRatio,
        def.targetAngle,
        def.lowerAngle,
        def.upperAngle,
        def.maxMotorTorque,
        def.motorSpeed,
        def.enableSpring,
        def.enableLimit,
        def.enableMotor,
    );
    return pair;
}
export function getRevoluteJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetRevoluteJointTorque). */
export function getRevoluteJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}

/** The current hinge angle (b3RevoluteJoint_GetAngle): relative twist of the two joint frames. */
export function revoluteJointAngle(world: WorldState, sim: Joint): number {
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
    const quatA = quat.mul(transformA.q, readJointQuat(world, sim, J_LOCAL_FRAME_A + 3));
    let quatB = quat.mul(transformB.q, readJointQuat(world, sim, J_LOCAL_FRAME_B + 3));
    if (quat.dot(quatA, quatB) < 0) {
        // keeps the twist angle in [-pi, pi]
        quatB = quat.negate(quatB);
    }
    const relQ = quat.invMul(quatA, quatB);
    return quat.getTwistAngle(relQ);
}
