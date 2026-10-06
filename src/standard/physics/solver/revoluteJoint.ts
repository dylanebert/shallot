import { clampf, f32, maxf, minf, PI, type Quat, quat, type Vec3, vec3 } from "../common/math";
import {
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    RJ_DAMPING_RATIO,
    RJ_ENABLE,
    RJ_ENABLE_LIMIT,
    RJ_ENABLE_MOTOR,
    RJ_ENABLE_SPRING,
    RJ_FRAME_A,
    RJ_FRAME_B,
    RJ_HERTZ,
    RJ_LINEAR_IMPULSE,
    RJ_LOWER_ANGLE,
    RJ_LOWER_IMPULSE,
    RJ_MAX_MOTOR_TORQUE,
    RJ_MOTOR_IMPULSE,
    RJ_MOTOR_SPEED,
    RJ_PERP_AXIS_X,
    RJ_PERP_AXIS_Y,
    RJ_PERP_IMPULSE,
    RJ_ROTATION_AXIS_Z,
    RJ_SPRING_IMPULSE,
    RJ_TARGET_ANGLE,
    RJ_UPPER_ANGLE,
    RJ_UPPER_IMPULSE,
} from "../kernel/columns";
import {
    readJointFloat,
    readJointQuat,
    readJointVec2,
    readJointVec3,
    writeJointFlag,
    writeJointFloat,
    writeJointQuat,
    writeJointVec2,
    writeJointVec3,
} from "../kernel/jointcolumns";
import { JointField, jointField } from "../kernel/jointrecords";
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
const identityFrame = (): {
    q: Quat;
} => ({
    q: {
        v: {
            x: 0,
            y: 0,
            z: 0,
        },
        s: 1,
    },
});

/** Create a revolute joint (b3CreateRevoluteJoint). @returns the joint handle. */
export function createRevoluteJoint(
    world: WorldState,
    def: RevoluteJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Revolute);
    const lowerLimit = f32(f32(-0.99) * PI);
    const upperLimit = f32(f32(0.99) * PI);
    const lowerAngle = minf(def.lowerAngle, def.upperAngle);
    const upperAngle = maxf(def.lowerAngle, def.upperAngle);
    writeJointVec3(world, pair.joint, RJ_LINEAR_IMPULSE, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointVec2(world, pair.joint, RJ_PERP_IMPULSE, {
        x: 0,
        y: 0,
    });
    writeJointFloat(world, pair.joint, RJ_SPRING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, RJ_MOTOR_IMPULSE, 0);
    writeJointFloat(world, pair.joint, RJ_LOWER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, RJ_UPPER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, RJ_HERTZ, def.hertz);
    writeJointFloat(world, pair.joint, RJ_DAMPING_RATIO, def.dampingRatio);
    writeJointFloat(world, pair.joint, RJ_MAX_MOTOR_TORQUE, def.maxMotorTorque);
    writeJointFloat(world, pair.joint, RJ_MOTOR_SPEED, def.motorSpeed);
    writeJointFloat(world, pair.joint, RJ_TARGET_ANGLE, clampf(def.targetAngle, -PI, PI));
    writeJointFloat(world, pair.joint, RJ_LOWER_ANGLE, clampf(lowerAngle, lowerLimit, upperLimit));
    writeJointFloat(world, pair.joint, RJ_UPPER_ANGLE, clampf(upperAngle, lowerLimit, upperLimit));
    writeJointQuat(world, pair.joint, RJ_FRAME_A + 3, identityFrame().q);
    writeJointQuat(world, pair.joint, RJ_FRAME_B + 3, identityFrame().q);
    writeJointVec3(world, pair.joint, RJ_ROTATION_AXIS_Z, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointVec3(world, pair.joint, RJ_PERP_AXIS_X, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointVec3(world, pair.joint, RJ_PERP_AXIS_Y, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointFlag(world, pair.joint, RJ_ENABLE, RJ_ENABLE_SPRING, def.enableSpring);
    writeJointFlag(world, pair.joint, RJ_ENABLE, RJ_ENABLE_LIMIT, def.enableLimit);
    writeJointFlag(world, pair.joint, RJ_ENABLE, RJ_ENABLE_MOTOR, def.enableMotor);
    return pair;
}
export function getRevoluteJointForce(world: WorldState, sim: Joint): Vec3 {
    return vec3.scale(world.invH, readJointVec3(world, sim, RJ_LINEAR_IMPULSE));
}

/** The reaction torque this joint applies (b3GetRevoluteJointTorque). */
export function getRevoluteJointTorque(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        jointField(world, sim, JointField.bodyIdA + 3 * 0),
        bodyPoseScratch1,
    );
    let axis = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), vec3.axisZ());
    axis = quat.rotate(transformA.q, axis);
    const relQ = quat.invMul(
        readJointQuat(world, sim, RJ_FRAME_A + 3),
        readJointQuat(world, sim, RJ_FRAME_B + 3),
    );
    writeJointVec3(
        world,
        sim,
        RJ_PERP_AXIS_X,
        vec3.scale(
            f32(0.5),
            quat.rotate(
                readJointQuat(world, sim, RJ_FRAME_A + 3),
                vec3.add(vec3.scale(relQ.s, vec3.axisX()), vec3.cross(relQ.v, vec3.axisX())),
            ),
        ),
    );
    writeJointVec3(
        world,
        sim,
        RJ_PERP_AXIS_Y,
        vec3.scale(
            f32(0.5),
            quat.rotate(
                readJointQuat(world, sim, RJ_FRAME_A + 3),
                vec3.add(vec3.scale(relQ.s, vec3.axisY()), vec3.cross(relQ.v, vec3.axisY())),
            ),
        ),
    );
    const axialImpulse = f32(
        f32(
            f32(
                readJointFloat(world, sim, RJ_SPRING_IMPULSE) +
                    readJointFloat(world, sim, RJ_MOTOR_IMPULSE),
            ) + readJointFloat(world, sim, RJ_LOWER_IMPULSE),
        ) - readJointFloat(world, sim, RJ_UPPER_IMPULSE),
    );
    let angularImpulse = vec3.add(
        vec3.scale(
            readJointVec2(world, sim, RJ_PERP_IMPULSE).x,
            readJointVec3(world, sim, RJ_PERP_AXIS_X),
        ),
        vec3.scale(
            readJointVec2(world, sim, RJ_PERP_IMPULSE).y,
            readJointVec3(world, sim, RJ_PERP_AXIS_Y),
        ),
    );
    angularImpulse = vec3.mulAdd(
        angularImpulse,
        axialImpulse,
        readJointVec3(world, sim, RJ_ROTATION_AXIS_Z),
    );
    const impulse = vec3.mulAdd(angularImpulse, axialImpulse, axis);
    return vec3.scale(world.invH, impulse);
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
