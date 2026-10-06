import { f32, quat, type Vec3, vec3 } from "../common/math";
import {
    readSimCenter,
    readSimLocalCenter,
    readSimTransform,
    readStateAngularVelocity,
    readStateLinearVelocity,
} from "../kernel/bodycolumns";
import {
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    PJ_ANGULAR_IMPULSE,
    PJ_DAMPING_RATIO,
    PJ_ENABLE,
    PJ_ENABLE_LIMIT,
    PJ_ENABLE_MOTOR,
    PJ_ENABLE_SPRING,
    PJ_HERTZ,
    PJ_LOWER_IMPULSE,
    PJ_LOWER_TRANSLATION,
    PJ_MAX_MOTOR_FORCE,
    PJ_MOTOR_IMPULSE,
    PJ_MOTOR_SPEED,
    PJ_PERP_IMPULSE,
    PJ_SPRING_IMPULSE,
    PJ_TARGET_TRANSLATION,
    PJ_UPPER_IMPULSE,
    PJ_UPPER_TRANSLATION,
} from "../kernel/columns";
import {
    readJointFloat,
    readJointQuat,
    readJointVec2,
    readJointVec3,
    writeJointFlag,
    writeJointFloat,
    writeJointVec2,
    writeJointVec3,
} from "../kernel/jointcolumns";
import { getBodySim, getBodyState, readBodyTransform } from "../world/body";
import type { WorldState } from "../world/world";
import { createJoint, type Joint, type JointDef, JointType } from "./joint";

/** Prismatic joint payload (b3PrismaticJoint). Impulses persist across steps for warm starting. */

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

/** Create a prismatic joint (b3CreatePrismaticJoint). @returns the joint handle. */
export function createPrismaticJoint(
    world: WorldState,
    def: PrismaticJointDef,
): {
    joint: Joint;
} {
    const pair = createJoint(world, def.base, JointType.Prismatic);
    writeJointVec2(world, pair.joint, PJ_PERP_IMPULSE, {
        x: 0,
        y: 0,
    });
    writeJointVec3(world, pair.joint, PJ_ANGULAR_IMPULSE, {
        x: 0,
        y: 0,
        z: 0,
    });
    writeJointFloat(world, pair.joint, PJ_SPRING_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_MOTOR_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_LOWER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_UPPER_IMPULSE, 0);
    writeJointFloat(world, pair.joint, PJ_HERTZ, def.hertz);
    writeJointFloat(world, pair.joint, PJ_DAMPING_RATIO, def.dampingRatio);
    writeJointFloat(world, pair.joint, PJ_MAX_MOTOR_FORCE, def.maxMotorForce);
    writeJointFloat(world, pair.joint, PJ_MOTOR_SPEED, def.motorSpeed);
    writeJointFloat(world, pair.joint, PJ_TARGET_TRANSLATION, def.targetTranslation);
    writeJointFloat(world, pair.joint, PJ_LOWER_TRANSLATION, def.lowerTranslation);
    writeJointFloat(world, pair.joint, PJ_UPPER_TRANSLATION, def.upperTranslation);
    writeJointFlag(world, pair.joint, PJ_ENABLE, PJ_ENABLE_SPRING, def.enableSpring);
    writeJointFlag(world, pair.joint, PJ_ENABLE, PJ_ENABLE_LIMIT, def.enableLimit);
    writeJointFlag(world, pair.joint, PJ_ENABLE, PJ_ENABLE_MOTOR, def.enableMotor);
    return pair;
}
export function getPrismaticJointForce(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        world.bodies[sim.edges[0].bodyId],
        bodyPoseScratch1,
    );

    // impulse in joint space
    const impulse: Vec3 = {
        x: readJointVec2(world, sim, PJ_PERP_IMPULSE).x,
        y: readJointVec2(world, sim, PJ_PERP_IMPULSE).y,
        z: f32(
            f32(
                f32(
                    readJointFloat(world, sim, PJ_MOTOR_IMPULSE) +
                        readJointFloat(world, sim, PJ_LOWER_IMPULSE),
                ) + readJointFloat(world, sim, PJ_UPPER_IMPULSE),
            ) + readJointFloat(world, sim, PJ_SPRING_IMPULSE),
        ),
    };
    let force = vec3.scale(world.invH, impulse);
    force = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), force);
    force = quat.rotate(transformA.q, force);
    return force;
}

/** The reaction torque this joint applies (b3GetPrismaticJointTorque). */
export function getPrismaticJointTorque(world: WorldState, sim: Joint): Vec3 {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        world.bodies[sim.edges[0].bodyId],
        bodyPoseScratch1,
    );
    let torque = vec3.scale(world.invH, readJointVec3(world, sim, PJ_ANGULAR_IMPULSE));
    torque = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), torque);
    torque = quat.rotate(transformA.q, torque);
    return torque;
}

/** The current translation along the joint axis (b3PrismaticJoint_GetTranslation). */
export function prismaticJointTranslation(world: WorldState, sim: Joint): number {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const transformA = readBodyTransform(
        world,
        world.bodies[sim.edges[0].bodyId],
        bodyPoseScratch1,
    );
    const transformB = readBodyTransform(
        world,
        world.bodies[sim.edges[1].bodyId],
        bodyPoseScratch2,
    );
    let jointAxis = quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), vec3.axisX());
    jointAxis = quat.rotate(transformA.q, jointAxis);
    const anchorA = quat.rotate(transformA.q, readJointVec3(world, sim, J_LOCAL_FRAME_A));
    const anchorB = quat.rotate(transformB.q, readJointVec3(world, sim, J_LOCAL_FRAME_B));
    const d = vec3.add(vec3.sub(transformB.p, transformA.p), vec3.sub(anchorB, anchorA));
    return vec3.dot(d, jointAxis);
}

/** The current translation speed along the joint axis (b3PrismaticJoint_GetSpeed). */
export function prismaticJointSpeed(world: WorldState, sim: Joint): number {
    const transformScratch1 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const transformScratch2 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };
    const localCenterScratch3 = { x: 0, y: 0, z: 0 };
    const localCenterScratch4 = { x: 0, y: 0, z: 0 };
    const centerScratch5 = { x: 0, y: 0, z: 0 };
    const centerScratch6 = { x: 0, y: 0, z: 0 };
    const linearVelocityScratch7 = { x: 0, y: 0, z: 0 };
    const linearVelocityScratch8 = { x: 0, y: 0, z: 0 };
    const angularVelocityScratch9 = { x: 0, y: 0, z: 0 };
    const angularVelocityScratch10 = { x: 0, y: 0, z: 0 };

    const bodyA = world.bodies[sim.edges[0].bodyId];
    const bodyB = world.bodies[sim.edges[1].bodyId];
    const bodySimA = getBodySim(world, bodyA);
    const bodySimB = getBodySim(world, bodyB);
    const stateA = getBodyState(world, bodyA);
    const stateB = getBodyState(world, bodyB);
    const qA = readSimTransform(world, bodySimA, transformScratch1).q;
    const qB = readSimTransform(world, bodySimB, transformScratch2).q;
    const axisA = quat.rotate(
        qA,
        quat.rotate(readJointQuat(world, sim, J_LOCAL_FRAME_A + 3), vec3.axisX()),
    );
    const rA = quat.rotate(
        qA,
        vec3.sub(
            readJointVec3(world, sim, J_LOCAL_FRAME_A),
            readSimLocalCenter(world, bodySimA, localCenterScratch3),
        ),
    );
    const rB = quat.rotate(
        qB,
        vec3.sub(
            readJointVec3(world, sim, J_LOCAL_FRAME_B),
            readSimLocalCenter(world, bodySimB, localCenterScratch4),
        ),
    );

    // Difference the centers directly; positions are f32 in the single-precision build.
    const d = vec3.add(
        vec3.sub(
            readSimCenter(world, bodySimB, centerScratch5),
            readSimCenter(world, bodySimA, centerScratch6),
        ),
        vec3.sub(rB, rA),
    );
    const zero: Vec3 = {
        x: 0,
        y: 0,
        z: 0,
    };
    const vA =
        stateA !== null ? readStateLinearVelocity(world, stateA, linearVelocityScratch7) : zero;
    const vB =
        stateB !== null ? readStateLinearVelocity(world, stateB, linearVelocityScratch8) : zero;
    const wA =
        stateA !== null ? readStateAngularVelocity(world, stateA, angularVelocityScratch9) : zero;
    const wB =
        stateB !== null ? readStateAngularVelocity(world, stateB, angularVelocityScratch10) : zero;
    const vRel = vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vec3.add(vA, vec3.cross(wA, rA)));

    // The axis moves with body A, so account for its rotation.
    return f32(vec3.dot(d, vec3.cross(wA, axisA)) + vec3.dot(axisA, vRel));
}
