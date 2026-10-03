import {
    clampf,
    f32,
    maxf,
    minf,
    PI,
    type Quat,
    quat,
    type Vec2,
    type Vec3,
    vec3,
} from "../common/math";
import { getBodyTransformQuick } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";

/** Revolute joint payload (b3RevoluteJoint). Impulses persist across steps for warm starting. */
export type RevoluteJoint = {
    linearImpulse: Vec3;
    perpImpulse: Vec2;
    springImpulse: number;
    motorImpulse: number;
    lowerImpulse: number;
    upperImpulse: number;
    hertz: number;
    dampingRatio: number;
    maxMotorTorque: number;
    motorSpeed: number;
    targetAngle: number;
    lowerAngle: number;
    upperAngle: number;
    frameA: { q: Quat };
    frameB: { q: Quat };
    rotationAxisZ: Vec3;
    perpAxisX: Vec3;
    perpAxisY: Vec3;
    enableSpring: boolean;
    enableMotor: boolean;
    enableLimit: boolean;
};

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

const identityFrame = (): { q: Quat } => ({
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});

/** Create a revolute joint (b3CreateRevoluteJoint). @returns the joint handle + sim. */
export function createRevoluteJoint(
    world: WorldState,
    def: RevoluteJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Revolute);
    const sim = pair.sim;

    const lowerLimit = f32(f32(-0.99) * PI);
    const upperLimit = f32(f32(0.99) * PI);
    const lowerAngle = minf(def.lowerAngle, def.upperAngle);
    const upperAngle = maxf(def.lowerAngle, def.upperAngle);

    const data: RevoluteJoint = {
        linearImpulse: { x: 0, y: 0, z: 0 },
        perpImpulse: { x: 0, y: 0 },
        springImpulse: 0,
        motorImpulse: 0,
        lowerImpulse: 0,
        upperImpulse: 0,
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        maxMotorTorque: def.maxMotorTorque,
        motorSpeed: def.motorSpeed,
        targetAngle: clampf(def.targetAngle, -PI, PI),
        lowerAngle: clampf(lowerAngle, lowerLimit, upperLimit),
        upperAngle: clampf(upperAngle, lowerLimit, upperLimit),
        frameA: identityFrame(),
        frameB: identityFrame(),
        rotationAxisZ: { x: 0, y: 0, z: 0 },
        perpAxisX: { x: 0, y: 0, z: 0 },
        perpAxisY: { x: 0, y: 0, z: 0 },
        enableSpring: def.enableSpring,
        enableLimit: def.enableLimit,
        enableMotor: def.enableMotor,
    };
    sim.data = data;
    return pair;
}

export function getRevoluteJointForce(world: WorldState, sim: JointSim): Vec3 {
    return vec3.scale(world.invH, (sim.data as RevoluteJoint).linearImpulse);
}

/** The reaction torque this joint applies (b3GetRevoluteJointTorque). */
export function getRevoluteJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as RevoluteJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    let axis = quat.rotate(sim.localFrameA.q, vec3.axisZ());
    axis = quat.rotate(transformA.q, axis);

    const relQ = quat.invMul(joint.frameA.q, joint.frameB.q);

    joint.perpAxisX = vec3.scale(
        f32(0.5),
        quat.rotate(
            joint.frameA.q,
            vec3.add(vec3.scale(relQ.s, vec3.axisX()), vec3.cross(relQ.v, vec3.axisX())),
        ),
    );
    joint.perpAxisY = vec3.scale(
        f32(0.5),
        quat.rotate(
            joint.frameA.q,
            vec3.add(vec3.scale(relQ.s, vec3.axisY()), vec3.cross(relQ.v, vec3.axisY())),
        ),
    );

    const axialImpulse = f32(
        f32(f32(joint.springImpulse + joint.motorImpulse) + joint.lowerImpulse) -
            joint.upperImpulse,
    );
    let angularImpulse = vec3.add(
        vec3.scale(joint.perpImpulse.x, joint.perpAxisX),
        vec3.scale(joint.perpImpulse.y, joint.perpAxisY),
    );
    angularImpulse = vec3.mulAdd(angularImpulse, axialImpulse, joint.rotationAxisZ);

    const impulse = vec3.mulAdd(angularImpulse, axialImpulse, axis);
    return vec3.scale(world.invH, impulse);
}

/** The current hinge angle (b3RevoluteJoint_GetAngle): relative twist of the two joint frames. */
export function revoluteJointAngle(world: WorldState, sim: JointSim): number {
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);
    const quatA = quat.mul(transformA.q, sim.localFrameA.q);
    let quatB = quat.mul(transformB.q, sim.localFrameB.q);
    if (quat.dot(quatA, quatB) < 0) {
        // keeps the twist angle in [-pi, pi]
        quatB = quat.negate(quatB);
    }
    const relQ = quat.invMul(quatA, quatB);
    return quat.getTwistAngle(relQ);
}
