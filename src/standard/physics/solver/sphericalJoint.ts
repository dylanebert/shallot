import { clampf, f32, maxf, minf, PI, type Quat, quat, type Vec3, vec3 } from "../common/math";
import { getBodyTransformQuick } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Spherical joint payload (b3SphericalJoint). Impulses persist across steps for warm starting. */
export type SphericalJoint = {
    linearImpulse: Vec3;
    springImpulse: Vec3;
    motorImpulse: Vec3;
    lowerTwistImpulse: number;
    upperTwistImpulse: number;
    swingImpulse: number;
    hertz: number;
    dampingRatio: number;
    maxMotorTorque: number;
    motorVelocity: Vec3;
    lowerTwistAngle: number;
    upperTwistAngle: number;
    coneAngle: number;
    targetRotation: Quat;
    enableSpring: boolean;
    enableMotor: boolean;
    enableConeLimit: boolean;
    enableTwistLimit: boolean;
};

/** Spherical joint definition (b3SphericalJointDef), body handles resolved to a base JointDef. */
export type SphericalJointDef = {
    base: JointDef;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    targetRotation: Quat;
    enableConeLimit: boolean;
    coneAngle: number;
    enableTwistLimit: boolean;
    lowerTwistAngle: number;
    upperTwistAngle: number;
    enableMotor: boolean;
    maxMotorTorque: number;
    motorVelocity: Vec3;
};

/** @returns the ported spherical joint definition defaults (b3DefaultSphericalJointDef). */
export function defaultSphericalJointDef(base: JointDef): SphericalJointDef {
    return {
        base,
        enableSpring: false,
        hertz: 0,
        dampingRatio: 0,
        targetRotation: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        enableConeLimit: false,
        coneAngle: 0,
        enableTwistLimit: false,
        lowerTwistAngle: 0,
        upperTwistAngle: 0,
        enableMotor: false,
        maxMotorTorque: 0,
        motorVelocity: { x: 0, y: 0, z: 0 },
    };
}

const zeroVec3 = (): Vec3 => ({ x: 0, y: 0, z: 0 });

/** Create a spherical joint (b3CreateSphericalJoint). @returns the joint handle + sim. */
export function createSphericalJoint(
    world: WorldState,
    def: SphericalJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Spherical);

    const lowerLimit = f32(f32(-0.99) * PI);
    const upperLimit = f32(f32(0.99) * PI);
    const lowerAngle = minf(def.lowerTwistAngle, def.upperTwistAngle);
    const upperAngle = maxf(def.lowerTwistAngle, def.upperTwistAngle);

    const data: SphericalJoint = {
        linearImpulse: zeroVec3(),
        springImpulse: zeroVec3(),
        motorImpulse: zeroVec3(),
        lowerTwistImpulse: 0,
        upperTwistImpulse: 0,
        swingImpulse: 0,
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        maxMotorTorque: def.maxMotorTorque,
        motorVelocity: { ...def.motorVelocity },
        lowerTwistAngle: clampf(lowerAngle, lowerLimit, upperLimit),
        upperTwistAngle: clampf(upperAngle, lowerLimit, upperLimit),
        coneAngle: clampf(def.coneAngle, 0, f32(f32(0.5) * PI)),
        targetRotation: { v: { ...def.targetRotation.v }, s: def.targetRotation.s },
        enableSpring: def.enableSpring,
        enableMotor: def.enableMotor,
        enableConeLimit: def.enableConeLimit,
        enableTwistLimit: def.enableTwistLimit,
    };
    pair.sim.data = data;
    return pair;
}

export function getSphericalJointForce(world: WorldState, sim: JointSim): Vec3 {
    return vec3.scale(world.invH, (sim.data as SphericalJoint).linearImpulse);
}

/** The reaction torque this joint applies (b3GetSphericalJointTorque). */
export function getSphericalJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as SphericalJoint;
    const xfA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const xfB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);
    const qA = quat.mul(xfA.q, sim.localFrameA.q);
    const qB = quat.mul(xfB.q, sim.localFrameB.q);

    const coneAxis = quat.rotate(qA, vec3.axisZ());
    const twistAxis = quat.rotate(qB, vec3.axisZ());
    const swingAxis = vec3.normalize(vec3.cross(coneAxis, twistAxis));

    let impulse = vec3.add(joint.springImpulse, joint.motorImpulse);
    impulse = vec3.mulAdd(
        impulse,
        f32(joint.lowerTwistImpulse - joint.upperTwistImpulse),
        twistAxis,
    );
    impulse = vec3.mulAdd(impulse, joint.swingImpulse, swingAxis);
    return vec3.scale(world.invH, impulse);
}

/** @returns the relative rotation of the two joint frames, twist-adjusted (shared by cone/twist getters). */
function relativeFrameRotation(world: WorldState, sim: JointSim): Quat {
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);
    const quatA = quat.mul(transformA.q, sim.localFrameA.q);
    let quatB = quat.mul(transformB.q, sim.localFrameB.q);
    if (quat.dot(quatA, quatB) < 0) {
        quatB = quat.negate(quatB);
    }
    return quat.invMul(quatA, quatB);
}

/** The current swing (cone) angle (b3SphericalJoint_GetConeAngle). */
export function sphericalJointConeAngle(world: WorldState, sim: JointSim): number {
    return quat.getSwingAngle(relativeFrameRotation(world, sim));
}

/** The current twist angle (b3SphericalJoint_GetTwistAngle). */
export function sphericalJointTwistAngle(world: WorldState, sim: JointSim): number {
    return quat.getTwistAngle(relativeFrameRotation(world, sim));
}
