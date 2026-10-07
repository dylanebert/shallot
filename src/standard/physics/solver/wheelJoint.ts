import { atan2, f32, mat3, quat, type Vec3, vec3 } from "../common/math";
import { bodySimSlot, readSimTransform, readStateAngularVelocity } from "../kernel/bodycolumns";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { J_LOCAL_FRAME_A, J_LOCAL_FRAME_B } from "../kernel/columns";
import { readJointQuat, readJointReaction } from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import { getBodyState } from "../world/body";
import type { WorldState } from "../world/world";
import { finishJointCreation, type Joint, type JointDef, writeJointDefinition } from "./joint";

/** Wheel joint payload (b3WheelJoint). Impulses persist across steps for warm starting. */

/** Wheel joint definition (b3WheelJointDef), body handles resolved to a base JointDef. */
export type WheelJointDef = {
    base: JointDef;
    enableSuspensionSpring: boolean;
    suspensionHertz: number;
    suspensionDampingRatio: number;
    enableSuspensionLimit: boolean;
    lowerSuspensionLimit: number;
    upperSuspensionLimit: number;
    enableSpinMotor: boolean;
    maxSpinTorque: number;
    spinSpeed: number;
    enableSteering: boolean;
    steeringHertz: number;
    steeringDampingRatio: number;
    targetSteeringAngle: number;
    maxSteeringTorque: number;
    enableSteeringLimit: boolean;
    lowerSteeringLimit: number;
    upperSteeringLimit: number;
};

/** @returns the ported wheel joint definition defaults (b3DefaultWheelJointDef). */
export function defaultWheelJointDef(base: JointDef): WheelJointDef {
    return {
        base,
        enableSuspensionSpring: true,
        suspensionHertz: 1,
        suspensionDampingRatio: f32(0.7),
        enableSuspensionLimit: false,
        lowerSuspensionLimit: 0,
        upperSuspensionLimit: 0,
        enableSpinMotor: false,
        maxSpinTorque: 0,
        spinSpeed: 0,
        enableSteering: false,
        steeringHertz: 1,
        steeringDampingRatio: f32(0.7),
        targetSteeringAngle: 0,
        maxSteeringTorque: 0,
        enableSteeringLimit: false,
        lowerSteeringLimit: 0,
        upperSteeringLimit: 0,
    };
}

/** Create a wheel joint (b3CreateWheelJoint). @returns the joint handle. */
export function createWheelJoint(
    world: WorldState,
    def: WheelJointDef,
): {
    joint: Joint;
} {
    const joint = kernel(world.ecsState).jointCreateWheel(
        world.worldId,
        writeJointDefinition(world, def.base),
        def.enableSuspensionSpring,
        def.suspensionHertz,
        def.suspensionDampingRatio,
        def.enableSuspensionLimit,
        def.lowerSuspensionLimit,
        def.upperSuspensionLimit,
        def.enableSpinMotor,
        def.maxSpinTorque,
        def.spinSpeed,
        def.enableSteering,
        def.steeringHertz,
        def.steeringDampingRatio,
        def.targetSteeringAngle,
        def.maxSteeringTorque,
        def.enableSteeringLimit,
        def.lowerSteeringLimit,
        def.upperSteeringLimit,
    );
    return finishJointCreation(world, def.base, joint);
}
export function getWheelJointForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}

/** The reaction torque this joint applies (b3GetWheelJointTorque). */
export function getWheelJointTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}

/** The spin speed of the wheel about its spin axis (b3WheelJoint_GetSpinSpeed). */
export function wheelJointSpinSpeed(world: WorldState, sim: Joint): number {
    const transformScratch1 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const angularVelocityScratch2 = { x: 0, y: 0, z: 0 };
    const angularVelocityScratch3 = { x: 0, y: 0, z: 0 };

    const bodyA = jointField(world, sim, JointField.bodyIdA + 3 * 0);
    const bodyB = jointField(world, sim, JointField.bodyIdA + 3 * 1);
    const setB = bodyField(world, bodyB, BodyField.setIndex);
    const bodySimB = bodySimSlot(setB, bodyField(world, bodyB, BodyField.localIndex));
    const quatB = quat.mul(
        readSimTransform(world, bodySimB, transformScratch1).q,
        readJointQuat(world, sim, J_LOCAL_FRAME_B + 3),
    );
    const spinAxis = quat.rotate(quatB, vec3.axisZ());
    const zero: Vec3 = {
        x: 0,
        y: 0,
        z: 0,
    };
    const stateA = getBodyState(world, bodyA);
    const stateB = getBodyState(world, bodyB);
    const wA =
        stateA !== null ? readStateAngularVelocity(world, stateA, angularVelocityScratch2) : zero;
    const wB =
        stateB !== null ? readStateAngularVelocity(world, stateB, angularVelocityScratch3) : zero;
    return vec3.dot(vec3.sub(wB, wA), spinAxis);
}

/** The current steering angle about body A's x-axis (b3WheelJoint_GetSteeringAngle). */
export function wheelJointSteeringAngle(world: WorldState, sim: Joint): number {
    const transformScratch1 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const transformScratch2 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };

    const bodyA = jointField(world, sim, JointField.bodyIdA + 3 * 0);
    const bodyB = jointField(world, sim, JointField.bodyIdA + 3 * 1);
    const setA = bodyField(world, bodyA, BodyField.setIndex);
    const setB = bodyField(world, bodyB, BodyField.setIndex);
    const bodySimA = bodySimSlot(setA, bodyField(world, bodyA, BodyField.localIndex));
    const bodySimB = bodySimSlot(setB, bodyField(world, bodyB, BodyField.localIndex));
    const quatA = quat.mul(
        readSimTransform(world, bodySimA, transformScratch1).q,
        readJointQuat(world, sim, J_LOCAL_FRAME_A + 3),
    );
    const quatB = quat.mul(
        readSimTransform(world, bodySimB, transformScratch2).q,
        readJointQuat(world, sim, J_LOCAL_FRAME_B + 3),
    );
    const matrixA = mat3.fromQuat(quatA);
    const matrixB = mat3.fromQuat(quatB);

    // Twist around the x-axis.
    const cs = vec3.dot(matrixB.cz, matrixA.cz);
    const ss = f32(-vec3.dot(matrixB.cz, matrixA.cy));
    return atan2(ss, cs);
}
