import { atan2, f32, mat3, quat, type Vec2, type Vec3, vec3 } from "../common/math";
import { getBodyState, getBodyTransformQuick } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Wheel joint payload (b3WheelJoint). Impulses persist across steps for warm starting. */
export type WheelJoint = {
    linearImpulse: Vec2;
    angularImpulse: Vec2;
    spinImpulse: number;
    maxSpinTorque: number;
    spinSpeed: number;
    suspensionSpringImpulse: number;
    lowerSuspensionImpulse: number;
    upperSuspensionImpulse: number;
    lowerSuspensionLimit: number;
    upperSuspensionLimit: number;
    suspensionHertz: number;
    suspensionDampingRatio: number;
    steeringSpringImpulse: number;
    lowerSteeringImpulse: number;
    upperSteeringImpulse: number;
    lowerSteeringLimit: number;
    upperSteeringLimit: number;
    targetSteeringAngle: number;
    maxSteeringTorque: number;
    steeringHertz: number;
    steeringDampingRatio: number;
    enableSpinMotor: boolean;
    enableSuspensionSpring: boolean;
    enableSuspensionLimit: boolean;
    enableSteering: boolean;
    enableSteeringLimit: boolean;
};

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

/** Create a wheel joint (b3CreateWheelJoint). @returns the joint handle + sim. */
export function createWheelJoint(
    world: WorldState,
    def: WheelJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Wheel);
    const data: WheelJoint = {
        linearImpulse: { x: 0, y: 0 },
        angularImpulse: { x: 0, y: 0 },
        spinImpulse: 0,
        maxSpinTorque: def.maxSpinTorque,
        spinSpeed: def.spinSpeed,
        suspensionSpringImpulse: 0,
        lowerSuspensionImpulse: 0,
        upperSuspensionImpulse: 0,
        lowerSuspensionLimit: def.lowerSuspensionLimit,
        upperSuspensionLimit: def.upperSuspensionLimit,
        suspensionHertz: def.suspensionHertz,
        suspensionDampingRatio: def.suspensionDampingRatio,
        steeringSpringImpulse: 0,
        lowerSteeringImpulse: 0,
        upperSteeringImpulse: 0,
        lowerSteeringLimit: def.lowerSteeringLimit,
        upperSteeringLimit: def.upperSteeringLimit,
        targetSteeringAngle: def.targetSteeringAngle,
        maxSteeringTorque: def.maxSteeringTorque,
        steeringHertz: def.steeringHertz,
        steeringDampingRatio: def.steeringDampingRatio,
        enableSpinMotor: def.enableSpinMotor,
        enableSuspensionSpring: def.enableSuspensionSpring,
        enableSuspensionLimit: def.enableSuspensionLimit,
        enableSteering: def.enableSteering,
        enableSteeringLimit: def.enableSteeringLimit,
    };
    pair.sim.data = data;
    return pair;
}

export function getWheelJointForce(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as WheelJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);

    // impulse in joint space. The z term reads lowerSuspensionLimit (a config value, not an impulse) —
    // an upstream quirk in b3GetWheelJointForce, kept verbatim so this accessor matches C. Not "fixed"
    // to lowerSuspensionImpulse: force accessors aren't hashed, but the port stays faithful to the C API.
    const impulse: Vec3 = {
        x: joint.linearImpulse.x,
        y: joint.linearImpulse.y,
        z: f32(
            f32(joint.lowerSuspensionLimit + joint.upperSuspensionImpulse) +
                joint.suspensionSpringImpulse,
        ),
    };

    let force = vec3.scale(world.invH, impulse);
    force = quat.rotate(sim.localFrameA.q, force);
    force = quat.rotate(transformA.q, force);
    return force;
}

/** The reaction torque this joint applies (b3GetWheelJointTorque). */
export function getWheelJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as WheelJoint;
    const bodyA = world.bodies[sim.bodyIdA];
    const setA = world.solverSets[bodyA.setIndex];
    const bodySimA = setA.bodySims[bodyA.localIndex];

    const qA = quat.mul(bodySimA.transform.q, sim.localFrameA.q);
    const matrixA = mat3.fromQuat(qA);
    return vec3.scale(f32(world.invH * joint.spinImpulse), matrixA.cz);
}

/** The spin speed of the wheel about its spin axis (b3WheelJoint_GetSpinSpeed). */
export function wheelJointSpinSpeed(world: WorldState, sim: JointSim): number {
    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];
    const setB = world.solverSets[bodyB.setIndex];
    const bodySimB = setB.bodySims[bodyB.localIndex];

    const quatB = quat.mul(bodySimB.transform.q, sim.localFrameB.q);
    const spinAxis = quat.rotate(quatB, vec3.axisZ());

    const zero: Vec3 = { x: 0, y: 0, z: 0 };
    const stateA = getBodyState(world, bodyA);
    const stateB = getBodyState(world, bodyB);
    const wA = stateA ? stateA.angularVelocity : zero;
    const wB = stateB ? stateB.angularVelocity : zero;

    return vec3.dot(vec3.sub(wB, wA), spinAxis);
}

/** The current steering angle about body A's x-axis (b3WheelJoint_GetSteeringAngle). */
export function wheelJointSteeringAngle(world: WorldState, sim: JointSim): number {
    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];
    const setA = world.solverSets[bodyA.setIndex];
    const setB = world.solverSets[bodyB.setIndex];
    const bodySimA = setA.bodySims[bodyA.localIndex];
    const bodySimB = setB.bodySims[bodyB.localIndex];

    const quatA = quat.mul(bodySimA.transform.q, sim.localFrameA.q);
    const quatB = quat.mul(bodySimB.transform.q, sim.localFrameB.q);

    const matrixA = mat3.fromQuat(quatA);
    const matrixB = mat3.fromQuat(quatB);

    // Twist around the x-axis.
    const cs = vec3.dot(matrixB.cz, matrixA.cz);
    const ss = f32(-vec3.dot(matrixB.cz, matrixA.cy));
    return atan2(ss, cs);
}
