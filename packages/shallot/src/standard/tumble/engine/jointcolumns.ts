// Marshal jointed scenes into the kernel joint column (kernel/src/joint_abi.rs) and read the solved
// impulses back. Joints are few, so — unlike contacts — the whole record marshals in each step and
// reads back out; the impulses live authoritatively in `JointSim.data`. The kernel derives the anchors
// + softness in `prepare`, so this only copies the raw body-sim inputs + config + the persistent
// impulses (no arithmetic — bit-exact by construction against the serial prepare's identical inputs).
//
// All eight solver joint types are wired (3b template + 3c). Filter joints carry a header-only no-op
// record (a collision filter has no solve), so every awake joint type is kernel-resident and
// `writeRecord`'s throw is unreachable for an awake joint.

import { NULL_INDEX } from "./array";
import {
    type Columns,
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
    J_CENTER_A,
    J_CENTER_B,
    J_CONSTRAINT_DAMPING,
    J_CONSTRAINT_HERTZ,
    J_INV_IA,
    J_INV_IB,
    J_INV_MASS_A,
    J_INV_MASS_B,
    J_LOCAL_CENTER_A,
    J_LOCAL_CENTER_B,
    J_LOCAL_FRAME_A,
    J_LOCAL_FRAME_B,
    J_QA,
    J_QB,
    J_SIM_INDEX_A,
    J_SIM_INDEX_B,
    J_TYPE,
    JOINT_STRIDE,
    MJ_ANGULAR_DAMPING_RATIO,
    MJ_ANGULAR_HERTZ,
    MJ_ANGULAR_SPRING_IMPULSE,
    MJ_ANGULAR_VELOCITY,
    MJ_ANGULAR_VELOCITY_IMPULSE,
    MJ_LINEAR_DAMPING_RATIO,
    MJ_LINEAR_HERTZ,
    MJ_LINEAR_SPRING_IMPULSE,
    MJ_LINEAR_VELOCITY,
    MJ_LINEAR_VELOCITY_IMPULSE,
    MJ_MAX_SPRING_FORCE,
    MJ_MAX_SPRING_TORQUE,
    MJ_MAX_VELOCITY_FORCE,
    MJ_MAX_VELOCITY_TORQUE,
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
    PLJ_DAMPING_RATIO,
    PLJ_HERTZ,
    PLJ_MAX_TORQUE,
    PLJ_PERP_IMPULSE,
    RJ_DAMPING_RATIO,
    RJ_ENABLE,
    RJ_ENABLE_LIMIT,
    RJ_ENABLE_MOTOR,
    RJ_ENABLE_SPRING,
    RJ_HERTZ,
    RJ_LINEAR_IMPULSE,
    RJ_LOWER_ANGLE,
    RJ_LOWER_IMPULSE,
    RJ_MAX_MOTOR_TORQUE,
    RJ_MOTOR_IMPULSE,
    RJ_MOTOR_SPEED,
    RJ_PERP_IMPULSE,
    RJ_SPRING_IMPULSE,
    RJ_TARGET_ANGLE,
    RJ_UPPER_ANGLE,
    RJ_UPPER_IMPULSE,
    SJ_CONE_ANGLE,
    SJ_DAMPING_RATIO,
    SJ_ENABLE,
    SJ_ENABLE_CONE_LIMIT,
    SJ_ENABLE_MOTOR,
    SJ_ENABLE_SPRING,
    SJ_ENABLE_TWIST_LIMIT,
    SJ_HERTZ,
    SJ_LINEAR_IMPULSE,
    SJ_LOWER_TWIST_ANGLE,
    SJ_LOWER_TWIST_IMPULSE,
    SJ_MAX_MOTOR_TORQUE,
    SJ_MOTOR_IMPULSE,
    SJ_MOTOR_VELOCITY,
    SJ_SPRING_IMPULSE,
    SJ_SWING_IMPULSE,
    SJ_TARGET_ROTATION,
    SJ_UPPER_TWIST_ANGLE,
    SJ_UPPER_TWIST_IMPULSE,
    WHJ_ANGULAR_IMPULSE,
    WHJ_ENABLE,
    WHJ_ENABLE_SPIN_MOTOR,
    WHJ_ENABLE_STEERING,
    WHJ_ENABLE_STEERING_LIMIT,
    WHJ_ENABLE_SUSPENSION_LIMIT,
    WHJ_ENABLE_SUSPENSION_SPRING,
    WHJ_LINEAR_IMPULSE,
    WHJ_LOWER_STEERING_IMPULSE,
    WHJ_LOWER_STEERING_LIMIT,
    WHJ_LOWER_SUSPENSION_IMPULSE,
    WHJ_LOWER_SUSPENSION_LIMIT,
    WHJ_MAX_SPIN_TORQUE,
    WHJ_MAX_STEERING_TORQUE,
    WHJ_SPIN_IMPULSE,
    WHJ_SPIN_SPEED,
    WHJ_STEERING_DAMPING_RATIO,
    WHJ_STEERING_HERTZ,
    WHJ_STEERING_SPRING_IMPULSE,
    WHJ_SUSPENSION_DAMPING_RATIO,
    WHJ_SUSPENSION_HERTZ,
    WHJ_SUSPENSION_SPRING_IMPULSE,
    WHJ_TARGET_STEERING_ANGLE,
    WHJ_UPPER_STEERING_IMPULSE,
    WHJ_UPPER_STEERING_LIMIT,
    WHJ_UPPER_SUSPENSION_IMPULSE,
    WHJ_UPPER_SUSPENSION_LIMIT,
    WJ_ANGULAR_DAMPING_RATIO,
    WJ_ANGULAR_HERTZ,
    WJ_ANGULAR_IMPULSE,
    WJ_LINEAR_DAMPING_RATIO,
    WJ_LINEAR_HERTZ,
    WJ_LINEAR_IMPULSE,
} from "./columns";
import type { SolveLayout } from "./contactsolver";
import { OVERFLOW_INDEX, SetType } from "./core";
import type { DistanceJoint } from "./distanceJoint";
import { type JointSim, JointType } from "./joint";
import type { Mat3, Quat, Transform, Vec3 } from "./math";
import type { MotorJoint } from "./motorJoint";
import type { ParallelJoint } from "./parallelJoint";
import type { PrismaticJoint } from "./prismaticJoint";
import type { RevoluteJoint } from "./revoluteJoint";
import type { SphericalJoint } from "./sphericalJoint";
import type { WeldJoint } from "./weldJoint";
import type { WheelJoint } from "./wheelJoint";
import type { WorldState } from "./world";

/** The joint spans the staged solve needs: the colored total (the `PrepareJoints` sweep) and the
 * overflow span (run serially). Colored joints occupy `[0, jointTotal)`, overflow the tail. */
export type JointLayout = {
    jointTotal: number;
    overflowJointStart: number;
    overflowJointCount: number;
};

function writeVec3(f32: Float32Array, o: number, v: Vec3): void {
    f32[o] = v.x;
    f32[o + 1] = v.y;
    f32[o + 2] = v.z;
}

function writeQuat(f32: Float32Array, o: number, q: Quat): void {
    f32[o] = q.v.x;
    f32[o + 1] = q.v.y;
    f32[o + 2] = q.v.z;
    f32[o + 3] = q.s;
}

function writeMat3(f32: Float32Array, o: number, m: Mat3): void {
    f32[o] = m.cx.x;
    f32[o + 1] = m.cx.y;
    f32[o + 2] = m.cx.z;
    f32[o + 3] = m.cy.x;
    f32[o + 4] = m.cy.y;
    f32[o + 5] = m.cy.z;
    f32[o + 6] = m.cz.x;
    f32[o + 7] = m.cz.y;
    f32[o + 8] = m.cz.z;
}

function writeTransform(f32: Float32Array, o: number, t: Transform): void {
    writeVec3(f32, o, t.p);
    writeQuat(f32, o + 3, t.q);
}

/** Write one joint's full record into slot `slot` of the flat joint column. Reads the two bodies' sim
 * data (invMass/invInertia + pose) straight from their solver sets — the same fields the serial
 * prepare reads — and the type config + persistent impulses from the sim payload. */
function writeRecord(
    world: WorldState,
    f32: Float32Array,
    u32: Uint32Array,
    slot: number,
    sim: JointSim,
): void {
    const base = slot * JOINT_STRIDE;

    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];
    const bodySimA = world.solverSets[bodyA.setIndex].bodySims[bodyA.localIndex];
    const bodySimB = world.solverSets[bodyB.setIndex].bodySims[bodyB.localIndex];

    u32[base + J_TYPE] = sim.type;
    u32[base + J_SIM_INDEX_A] =
        bodyA.setIndex === SetType.Awake ? bodyA.localIndex : NULL_INDEX >>> 0;
    u32[base + J_SIM_INDEX_B] =
        bodyB.setIndex === SetType.Awake ? bodyB.localIndex : NULL_INDEX >>> 0;

    f32[base + J_INV_MASS_A] = bodySimA.invMass;
    f32[base + J_INV_MASS_B] = bodySimB.invMass;
    writeMat3(f32, base + J_INV_IA, bodySimA.invInertiaWorld);
    writeMat3(f32, base + J_INV_IB, bodySimB.invInertiaWorld);
    writeQuat(f32, base + J_QA, bodySimA.transform.q);
    writeVec3(f32, base + J_LOCAL_CENTER_A, bodySimA.localCenter);
    writeVec3(f32, base + J_CENTER_A, bodySimA.center);
    writeQuat(f32, base + J_QB, bodySimB.transform.q);
    writeVec3(f32, base + J_LOCAL_CENTER_B, bodySimB.localCenter);
    writeVec3(f32, base + J_CENTER_B, bodySimB.center);
    writeTransform(f32, base + J_LOCAL_FRAME_A, sim.localFrameA);
    writeTransform(f32, base + J_LOCAL_FRAME_B, sim.localFrameB);
    f32[base + J_CONSTRAINT_HERTZ] = sim.constraintHertz;
    f32[base + J_CONSTRAINT_DAMPING] = sim.constraintDampingRatio;

    if (sim.type === JointType.Distance) {
        const j = sim.data as DistanceJoint;
        f32[base + DJ_LENGTH] = j.length;
        f32[base + DJ_HERTZ] = j.hertz;
        f32[base + DJ_DAMPING_RATIO] = j.dampingRatio;
        f32[base + DJ_LOWER_SPRING_FORCE] = j.lowerSpringForce;
        f32[base + DJ_UPPER_SPRING_FORCE] = j.upperSpringForce;
        f32[base + DJ_MIN_LENGTH] = j.minLength;
        f32[base + DJ_MAX_LENGTH] = j.maxLength;
        f32[base + DJ_MAX_MOTOR_FORCE] = j.maxMotorForce;
        f32[base + DJ_MOTOR_SPEED] = j.motorSpeed;
        u32[base + DJ_ENABLE] =
            (j.enableSpring ? DJ_ENABLE_SPRING : 0) |
            (j.enableLimit ? DJ_ENABLE_LIMIT : 0) |
            (j.enableMotor ? DJ_ENABLE_MOTOR : 0);
        f32[base + DJ_IMPULSE] = j.impulse;
        f32[base + DJ_LOWER_IMPULSE] = j.lowerImpulse;
        f32[base + DJ_UPPER_IMPULSE] = j.upperImpulse;
        f32[base + DJ_MOTOR_IMPULSE] = j.motorImpulse;
    } else if (sim.type === JointType.Weld) {
        const j = sim.data as WeldJoint;
        f32[base + WJ_LINEAR_HERTZ] = j.linearHertz;
        f32[base + WJ_LINEAR_DAMPING_RATIO] = j.linearDampingRatio;
        f32[base + WJ_ANGULAR_HERTZ] = j.angularHertz;
        f32[base + WJ_ANGULAR_DAMPING_RATIO] = j.angularDampingRatio;
        writeVec3(f32, base + WJ_LINEAR_IMPULSE, j.linearImpulse);
        writeVec3(f32, base + WJ_ANGULAR_IMPULSE, j.angularImpulse);
    } else if (sim.type === JointType.Revolute) {
        const j = sim.data as RevoluteJoint;
        f32[base + RJ_HERTZ] = j.hertz;
        f32[base + RJ_DAMPING_RATIO] = j.dampingRatio;
        f32[base + RJ_MAX_MOTOR_TORQUE] = j.maxMotorTorque;
        f32[base + RJ_MOTOR_SPEED] = j.motorSpeed;
        f32[base + RJ_TARGET_ANGLE] = j.targetAngle;
        f32[base + RJ_LOWER_ANGLE] = j.lowerAngle;
        f32[base + RJ_UPPER_ANGLE] = j.upperAngle;
        u32[base + RJ_ENABLE] =
            (j.enableSpring ? RJ_ENABLE_SPRING : 0) |
            (j.enableMotor ? RJ_ENABLE_MOTOR : 0) |
            (j.enableLimit ? RJ_ENABLE_LIMIT : 0);
        writeVec3(f32, base + RJ_LINEAR_IMPULSE, j.linearImpulse);
        f32[base + RJ_PERP_IMPULSE] = j.perpImpulse.x;
        f32[base + RJ_PERP_IMPULSE + 1] = j.perpImpulse.y;
        f32[base + RJ_SPRING_IMPULSE] = j.springImpulse;
        f32[base + RJ_MOTOR_IMPULSE] = j.motorImpulse;
        f32[base + RJ_LOWER_IMPULSE] = j.lowerImpulse;
        f32[base + RJ_UPPER_IMPULSE] = j.upperImpulse;
    } else if (sim.type === JointType.Spherical) {
        const j = sim.data as SphericalJoint;
        f32[base + SJ_HERTZ] = j.hertz;
        f32[base + SJ_DAMPING_RATIO] = j.dampingRatio;
        f32[base + SJ_MAX_MOTOR_TORQUE] = j.maxMotorTorque;
        writeVec3(f32, base + SJ_MOTOR_VELOCITY, j.motorVelocity);
        f32[base + SJ_LOWER_TWIST_ANGLE] = j.lowerTwistAngle;
        f32[base + SJ_UPPER_TWIST_ANGLE] = j.upperTwistAngle;
        f32[base + SJ_CONE_ANGLE] = j.coneAngle;
        writeQuat(f32, base + SJ_TARGET_ROTATION, j.targetRotation);
        u32[base + SJ_ENABLE] =
            (j.enableSpring ? SJ_ENABLE_SPRING : 0) |
            (j.enableMotor ? SJ_ENABLE_MOTOR : 0) |
            (j.enableConeLimit ? SJ_ENABLE_CONE_LIMIT : 0) |
            (j.enableTwistLimit ? SJ_ENABLE_TWIST_LIMIT : 0);
        writeVec3(f32, base + SJ_LINEAR_IMPULSE, j.linearImpulse);
        writeVec3(f32, base + SJ_SPRING_IMPULSE, j.springImpulse);
        writeVec3(f32, base + SJ_MOTOR_IMPULSE, j.motorImpulse);
        f32[base + SJ_LOWER_TWIST_IMPULSE] = j.lowerTwistImpulse;
        f32[base + SJ_UPPER_TWIST_IMPULSE] = j.upperTwistImpulse;
        f32[base + SJ_SWING_IMPULSE] = j.swingImpulse;
    } else if (sim.type === JointType.Prismatic) {
        const j = sim.data as PrismaticJoint;
        f32[base + PJ_HERTZ] = j.hertz;
        f32[base + PJ_DAMPING_RATIO] = j.dampingRatio;
        f32[base + PJ_MAX_MOTOR_FORCE] = j.maxMotorForce;
        f32[base + PJ_MOTOR_SPEED] = j.motorSpeed;
        f32[base + PJ_TARGET_TRANSLATION] = j.targetTranslation;
        f32[base + PJ_LOWER_TRANSLATION] = j.lowerTranslation;
        f32[base + PJ_UPPER_TRANSLATION] = j.upperTranslation;
        u32[base + PJ_ENABLE] =
            (j.enableSpring ? PJ_ENABLE_SPRING : 0) |
            (j.enableMotor ? PJ_ENABLE_MOTOR : 0) |
            (j.enableLimit ? PJ_ENABLE_LIMIT : 0);
        f32[base + PJ_PERP_IMPULSE] = j.perpImpulse.x;
        f32[base + PJ_PERP_IMPULSE + 1] = j.perpImpulse.y;
        writeVec3(f32, base + PJ_ANGULAR_IMPULSE, j.angularImpulse);
        f32[base + PJ_SPRING_IMPULSE] = j.springImpulse;
        f32[base + PJ_MOTOR_IMPULSE] = j.motorImpulse;
        f32[base + PJ_LOWER_IMPULSE] = j.lowerImpulse;
        f32[base + PJ_UPPER_IMPULSE] = j.upperImpulse;
    } else if (sim.type === JointType.Wheel) {
        const j = sim.data as WheelJoint;
        f32[base + WHJ_MAX_SPIN_TORQUE] = j.maxSpinTorque;
        f32[base + WHJ_SPIN_SPEED] = j.spinSpeed;
        f32[base + WHJ_LOWER_SUSPENSION_LIMIT] = j.lowerSuspensionLimit;
        f32[base + WHJ_UPPER_SUSPENSION_LIMIT] = j.upperSuspensionLimit;
        f32[base + WHJ_SUSPENSION_HERTZ] = j.suspensionHertz;
        f32[base + WHJ_SUSPENSION_DAMPING_RATIO] = j.suspensionDampingRatio;
        f32[base + WHJ_LOWER_STEERING_LIMIT] = j.lowerSteeringLimit;
        f32[base + WHJ_UPPER_STEERING_LIMIT] = j.upperSteeringLimit;
        f32[base + WHJ_TARGET_STEERING_ANGLE] = j.targetSteeringAngle;
        f32[base + WHJ_MAX_STEERING_TORQUE] = j.maxSteeringTorque;
        f32[base + WHJ_STEERING_HERTZ] = j.steeringHertz;
        f32[base + WHJ_STEERING_DAMPING_RATIO] = j.steeringDampingRatio;
        u32[base + WHJ_ENABLE] =
            (j.enableSpinMotor ? WHJ_ENABLE_SPIN_MOTOR : 0) |
            (j.enableSuspensionSpring ? WHJ_ENABLE_SUSPENSION_SPRING : 0) |
            (j.enableSuspensionLimit ? WHJ_ENABLE_SUSPENSION_LIMIT : 0) |
            (j.enableSteering ? WHJ_ENABLE_STEERING : 0) |
            (j.enableSteeringLimit ? WHJ_ENABLE_STEERING_LIMIT : 0);
        f32[base + WHJ_LINEAR_IMPULSE] = j.linearImpulse.x;
        f32[base + WHJ_LINEAR_IMPULSE + 1] = j.linearImpulse.y;
        f32[base + WHJ_ANGULAR_IMPULSE] = j.angularImpulse.x;
        f32[base + WHJ_ANGULAR_IMPULSE + 1] = j.angularImpulse.y;
        f32[base + WHJ_SPIN_IMPULSE] = j.spinImpulse;
        f32[base + WHJ_SUSPENSION_SPRING_IMPULSE] = j.suspensionSpringImpulse;
        f32[base + WHJ_LOWER_SUSPENSION_IMPULSE] = j.lowerSuspensionImpulse;
        f32[base + WHJ_UPPER_SUSPENSION_IMPULSE] = j.upperSuspensionImpulse;
        f32[base + WHJ_STEERING_SPRING_IMPULSE] = j.steeringSpringImpulse;
        f32[base + WHJ_LOWER_STEERING_IMPULSE] = j.lowerSteeringImpulse;
        f32[base + WHJ_UPPER_STEERING_IMPULSE] = j.upperSteeringImpulse;
    } else if (sim.type === JointType.Motor) {
        const j = sim.data as MotorJoint;
        writeVec3(f32, base + MJ_LINEAR_VELOCITY, j.linearVelocity);
        writeVec3(f32, base + MJ_ANGULAR_VELOCITY, j.angularVelocity);
        f32[base + MJ_MAX_VELOCITY_FORCE] = j.maxVelocityForce;
        f32[base + MJ_MAX_VELOCITY_TORQUE] = j.maxVelocityTorque;
        f32[base + MJ_LINEAR_HERTZ] = j.linearHertz;
        f32[base + MJ_LINEAR_DAMPING_RATIO] = j.linearDampingRatio;
        f32[base + MJ_ANGULAR_HERTZ] = j.angularHertz;
        f32[base + MJ_ANGULAR_DAMPING_RATIO] = j.angularDampingRatio;
        f32[base + MJ_MAX_SPRING_FORCE] = j.maxSpringForce;
        f32[base + MJ_MAX_SPRING_TORQUE] = j.maxSpringTorque;
        writeVec3(f32, base + MJ_LINEAR_VELOCITY_IMPULSE, j.linearVelocityImpulse);
        writeVec3(f32, base + MJ_ANGULAR_VELOCITY_IMPULSE, j.angularVelocityImpulse);
        writeVec3(f32, base + MJ_LINEAR_SPRING_IMPULSE, j.linearSpringImpulse);
        writeVec3(f32, base + MJ_ANGULAR_SPRING_IMPULSE, j.angularSpringImpulse);
    } else if (sim.type === JointType.Parallel) {
        const j = sim.data as ParallelJoint;
        f32[base + PLJ_HERTZ] = j.hertz;
        f32[base + PLJ_DAMPING_RATIO] = j.dampingRatio;
        f32[base + PLJ_MAX_TORQUE] = j.maxTorque;
        f32[base + PLJ_PERP_IMPULSE] = j.perpImpulse.x;
        f32[base + PLJ_PERP_IMPULSE + 1] = j.perpImpulse.y;
    } else if (sim.type === JointType.Filter) {
        // A filter joint is a collision filter with no solve (b3PrepareJoint/Solve break on it); the
        // header is enough — the kernel dispatch no-ops it. This is what lets a jointed scene route
        // wholly through the kernel: no awake joint type reaches the throw below.
    } else {
        throw new Error(`tumble: joints-in-kernel does not support joint type ${sim.type}`);
    }
}

/**
 * Marshal every awake joint into the flat joint column and write each active color's joint span into
 * the color-span column: colored joints first (per-color concatenated, the order `computeLayout`
 * fixed), then the overflow color's joints. @returns the spans the staged solve reads.
 */
export function marshalJoints(world: WorldState, layout: SolveLayout, cols: Columns): JointLayout {
    const f32 = cols.joint;
    const u32 = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
    const span = cols.colorSpan;

    let slot = 0;
    for (let i = 0; i < layout.colors.length; ++i) {
        const joints = layout.colors[i].color.jointSims;
        const o = i * 6;
        span[o + 4] = slot;
        span[o + 5] = joints.length;
        for (const sim of joints) {
            writeRecord(world, f32, u32, slot, sim);
            slot += 1;
        }
    }
    const jointTotal = slot;

    const overflowJoints = world.constraintGraph.colors[OVERFLOW_INDEX].jointSims;
    const overflowJointStart = slot;
    for (const sim of overflowJoints) {
        writeRecord(world, f32, u32, slot, sim);
        slot += 1;
    }

    return { jointTotal, overflowJointStart, overflowJointCount: overflowJoints.length };
}

/** Total awake joint count (colored + overflow) — the joint column size to reserve. */
export function countJoints(world: WorldState, layout: SolveLayout): number {
    let n = world.constraintGraph.colors[OVERFLOW_INDEX].jointSims.length;
    for (const span of layout.colors) {
        n += span.color.jointSims.length;
    }
    return n;
}

/** Read the solved impulses back out of the joint column into each `JointSim.data` (the persistent
 * warm-start state), mirroring the marshal order. */
export function readbackJointImpulses(world: WorldState, layout: SolveLayout, cols: Columns): void {
    const f32 = cols.joint;
    let slot = 0;
    const read = (sim: JointSim) => {
        const base = slot * JOINT_STRIDE;
        if (sim.type === JointType.Distance) {
            const j = sim.data as DistanceJoint;
            j.impulse = f32[base + DJ_IMPULSE];
            j.lowerImpulse = f32[base + DJ_LOWER_IMPULSE];
            j.upperImpulse = f32[base + DJ_UPPER_IMPULSE];
            j.motorImpulse = f32[base + DJ_MOTOR_IMPULSE];
        } else if (sim.type === JointType.Weld) {
            const j = sim.data as WeldJoint;
            j.linearImpulse = {
                x: f32[base + WJ_LINEAR_IMPULSE],
                y: f32[base + WJ_LINEAR_IMPULSE + 1],
                z: f32[base + WJ_LINEAR_IMPULSE + 2],
            };
            j.angularImpulse = {
                x: f32[base + WJ_ANGULAR_IMPULSE],
                y: f32[base + WJ_ANGULAR_IMPULSE + 1],
                z: f32[base + WJ_ANGULAR_IMPULSE + 2],
            };
        } else if (sim.type === JointType.Revolute) {
            const j = sim.data as RevoluteJoint;
            j.linearImpulse = {
                x: f32[base + RJ_LINEAR_IMPULSE],
                y: f32[base + RJ_LINEAR_IMPULSE + 1],
                z: f32[base + RJ_LINEAR_IMPULSE + 2],
            };
            j.perpImpulse = { x: f32[base + RJ_PERP_IMPULSE], y: f32[base + RJ_PERP_IMPULSE + 1] };
            j.springImpulse = f32[base + RJ_SPRING_IMPULSE];
            j.motorImpulse = f32[base + RJ_MOTOR_IMPULSE];
            j.lowerImpulse = f32[base + RJ_LOWER_IMPULSE];
            j.upperImpulse = f32[base + RJ_UPPER_IMPULSE];
        } else if (sim.type === JointType.Spherical) {
            const j = sim.data as SphericalJoint;
            j.linearImpulse = {
                x: f32[base + SJ_LINEAR_IMPULSE],
                y: f32[base + SJ_LINEAR_IMPULSE + 1],
                z: f32[base + SJ_LINEAR_IMPULSE + 2],
            };
            j.springImpulse = {
                x: f32[base + SJ_SPRING_IMPULSE],
                y: f32[base + SJ_SPRING_IMPULSE + 1],
                z: f32[base + SJ_SPRING_IMPULSE + 2],
            };
            j.motorImpulse = {
                x: f32[base + SJ_MOTOR_IMPULSE],
                y: f32[base + SJ_MOTOR_IMPULSE + 1],
                z: f32[base + SJ_MOTOR_IMPULSE + 2],
            };
            j.lowerTwistImpulse = f32[base + SJ_LOWER_TWIST_IMPULSE];
            j.upperTwistImpulse = f32[base + SJ_UPPER_TWIST_IMPULSE];
            j.swingImpulse = f32[base + SJ_SWING_IMPULSE];
        } else if (sim.type === JointType.Prismatic) {
            const j = sim.data as PrismaticJoint;
            j.perpImpulse = { x: f32[base + PJ_PERP_IMPULSE], y: f32[base + PJ_PERP_IMPULSE + 1] };
            j.angularImpulse = {
                x: f32[base + PJ_ANGULAR_IMPULSE],
                y: f32[base + PJ_ANGULAR_IMPULSE + 1],
                z: f32[base + PJ_ANGULAR_IMPULSE + 2],
            };
            j.springImpulse = f32[base + PJ_SPRING_IMPULSE];
            j.motorImpulse = f32[base + PJ_MOTOR_IMPULSE];
            j.lowerImpulse = f32[base + PJ_LOWER_IMPULSE];
            j.upperImpulse = f32[base + PJ_UPPER_IMPULSE];
        } else if (sim.type === JointType.Wheel) {
            const j = sim.data as WheelJoint;
            j.linearImpulse = {
                x: f32[base + WHJ_LINEAR_IMPULSE],
                y: f32[base + WHJ_LINEAR_IMPULSE + 1],
            };
            j.angularImpulse = {
                x: f32[base + WHJ_ANGULAR_IMPULSE],
                y: f32[base + WHJ_ANGULAR_IMPULSE + 1],
            };
            j.spinImpulse = f32[base + WHJ_SPIN_IMPULSE];
            j.suspensionSpringImpulse = f32[base + WHJ_SUSPENSION_SPRING_IMPULSE];
            j.lowerSuspensionImpulse = f32[base + WHJ_LOWER_SUSPENSION_IMPULSE];
            j.upperSuspensionImpulse = f32[base + WHJ_UPPER_SUSPENSION_IMPULSE];
            j.steeringSpringImpulse = f32[base + WHJ_STEERING_SPRING_IMPULSE];
            j.lowerSteeringImpulse = f32[base + WHJ_LOWER_STEERING_IMPULSE];
            j.upperSteeringImpulse = f32[base + WHJ_UPPER_STEERING_IMPULSE];
        } else if (sim.type === JointType.Motor) {
            const j = sim.data as MotorJoint;
            j.linearVelocityImpulse = {
                x: f32[base + MJ_LINEAR_VELOCITY_IMPULSE],
                y: f32[base + MJ_LINEAR_VELOCITY_IMPULSE + 1],
                z: f32[base + MJ_LINEAR_VELOCITY_IMPULSE + 2],
            };
            j.angularVelocityImpulse = {
                x: f32[base + MJ_ANGULAR_VELOCITY_IMPULSE],
                y: f32[base + MJ_ANGULAR_VELOCITY_IMPULSE + 1],
                z: f32[base + MJ_ANGULAR_VELOCITY_IMPULSE + 2],
            };
            j.linearSpringImpulse = {
                x: f32[base + MJ_LINEAR_SPRING_IMPULSE],
                y: f32[base + MJ_LINEAR_SPRING_IMPULSE + 1],
                z: f32[base + MJ_LINEAR_SPRING_IMPULSE + 2],
            };
            j.angularSpringImpulse = {
                x: f32[base + MJ_ANGULAR_SPRING_IMPULSE],
                y: f32[base + MJ_ANGULAR_SPRING_IMPULSE + 1],
                z: f32[base + MJ_ANGULAR_SPRING_IMPULSE + 2],
            };
        } else if (sim.type === JointType.Parallel) {
            const j = sim.data as ParallelJoint;
            j.perpImpulse = {
                x: f32[base + PLJ_PERP_IMPULSE],
                y: f32[base + PLJ_PERP_IMPULSE + 1],
            };
        }
        slot += 1;
    };
    for (const span of layout.colors) {
        for (const sim of span.color.jointSims) read(sim);
    }
    for (const sim of world.constraintGraph.colors[OVERFLOW_INDEX].jointSims) read(sim);
}
