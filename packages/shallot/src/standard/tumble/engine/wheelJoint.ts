// The wheel joint — Box3D's wheel_joint.c (Erin Catto, MIT). A car-suspension joint: the wheel slides
// along body A's local x-axis (suspension, with optional spring + limits), spins freely about body B's
// local z (optional spin motor), and either steers about body A's x (soft spring + limits) or is held
// collinear. Closed by a point-to-line linear constraint. Every op is fround-wrapped, mirroring the C
// scalar expression tree exactly. See the README.

import { NULL_INDEX } from "./array";
import { BodyFlags, getBodyState, getBodyTransformQuick, identityBodyState } from "./body";
import type { StepContext } from "./contactsolver";
import { SetType } from "./core";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";
import {
    atan2,
    clampf,
    FLT_MIN,
    f32,
    type Mat2,
    type Mat3,
    mat2,
    mat3,
    maxf,
    quat,
    type Transform,
    type Vec2,
    type Vec3,
    vec3,
} from "./math";
import { makeSoft, type Softness } from "./softness";
import type { WorldState } from "./world";

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
    indexA: number;
    indexB: number;
    frameA: Transform;
    frameB: Transform;
    deltaCenter: Vec3;
    spinMass: number;
    suspensionMass: number;
    steeringMass: number;
    suspensionSoftness: Softness;
    steeringSoftness: Softness;
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

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});

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
        indexA: 0,
        indexB: 0,
        frameA: identityTransform(),
        frameB: identityTransform(),
        deltaCenter: { x: 0, y: 0, z: 0 },
        spinMass: 0,
        suspensionMass: 0,
        steeringMass: 0,
        suspensionSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        steeringSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        enableSpinMotor: def.enableSpinMotor,
        enableSuspensionSpring: def.enableSuspensionSpring,
        enableSuspensionLimit: def.enableSuspensionLimit,
        enableSteering: def.enableSteering,
        enableSteeringLimit: def.enableSteeringLimit,
    };
    pair.sim.data = data;
    return pair;
}

// The steering constraint axis (twist about body A's x). Recomputed identically wherever it is needed;
// its inputs are pure f32 dots so every call is bit-identical (b3SolveWheelJoint's steeringAxis block).
function steeringAxisOf(matrixA: Mat3, matrixB: Mat3): Vec3 {
    const cs = vec3.dot(matrixB.cz, matrixA.cz);
    const ss = f32(-vec3.dot(matrixB.cz, matrixA.cy));
    let den = f32(f32(cs * cs) + f32(ss * ss));
    den = den > 0 ? f32(1 / den) : 0;
    return vec3.scale(
        den,
        vec3.cross(matrixB.cz, vec3.sub(vec3.scale(-cs, matrixA.cy), vec3.scale(ss, matrixA.cz))),
    );
}

export function prepareWheelJoint(sim: JointSim, context: StepContext): void {
    const world = context.world;
    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];

    const setA = world.solverSets[bodyA.setIndex];
    const setB = world.solverSets[bodyB.setIndex];
    const bodySimA = setA.bodySims[bodyA.localIndex];
    const bodySimB = setB.bodySims[bodyB.localIndex];

    sim.invMassA = bodySimA.invMass;
    sim.invMassB = bodySimB.invMass;
    sim.invIA = bodySimA.invInertiaWorld;
    sim.invIB = bodySimB.invInertiaWorld;

    const invInertiaSum = mat3.add(sim.invIA, sim.invIB);
    sim.fixedRotation = mat3.det(invInertiaSum) < f32(1000 * FLT_MIN);

    const joint = sim.data as WheelJoint;
    joint.indexA = bodyA.setIndex === SetType.Awake ? bodyA.localIndex : NULL_INDEX;
    joint.indexB = bodyB.setIndex === SetType.Awake ? bodyB.localIndex : NULL_INDEX;

    joint.frameA = {
        q: quat.mul(bodySimA.transform.q, sim.localFrameA.q),
        p: quat.rotate(bodySimA.transform.q, vec3.sub(sim.localFrameA.p, bodySimA.localCenter)),
    };
    joint.frameB = {
        q: quat.mul(bodySimB.transform.q, sim.localFrameB.q),
        p: quat.rotate(bodySimB.transform.q, vec3.sub(sim.localFrameB.p, bodySimB.localCenter)),
    };
    joint.deltaCenter = vec3.sub(bodySimB.center, bodySimA.center);

    const rA = joint.frameA.p;
    const rB = joint.frameB.p;

    const matrixA = mat3.fromQuat(joint.frameA.q);
    const matrixB = mat3.fromQuat(joint.frameB.q);

    {
        const suspensionAxis = matrixA.cx;
        const rAn = vec3.cross(rA, suspensionAxis);
        const rBn = vec3.cross(rB, suspensionAxis);
        const k = f32(
            f32(
                f32(f32(sim.invMassA + sim.invMassB) + vec3.dot(rAn, mat3.mulV(sim.invIA, rAn))) +
                    vec3.dot(rBn, mat3.mulV(sim.invIB, rBn)),
            ),
        );
        joint.suspensionMass = k > 0 ? f32(1 / k) : 0;
    }

    joint.suspensionSoftness = makeSoft(
        joint.suspensionHertz,
        joint.suspensionDampingRatio,
        context.h,
    );
    joint.steeringSoftness = makeSoft(joint.steeringHertz, joint.steeringDampingRatio, context.h);

    {
        const spinAxis = matrixB.cz;
        const k = vec3.dot(spinAxis, mat3.mulV(invInertiaSum, spinAxis));
        joint.spinMass = k > 0 ? f32(1 / k) : 0;
    }

    {
        const steeringAxis = steeringAxisOf(matrixA, matrixB);
        const k = vec3.dot(steeringAxis, mat3.mulV(invInertiaSum, steeringAxis));
        joint.steeringMass = k > 0 ? f32(1 / k) : 0;
    }

    if (context.enableWarmStarting === false) {
        joint.linearImpulse = { x: 0, y: 0 };
        joint.angularImpulse = { x: 0, y: 0 };
        joint.spinImpulse = 0;
        joint.suspensionSpringImpulse = 0;
        joint.lowerSuspensionImpulse = 0;
        joint.upperSuspensionImpulse = 0;
        joint.steeringSpringImpulse = 0;
        joint.lowerSteeringImpulse = 0;
        joint.upperSteeringImpulse = 0;
    }
}

export function warmStartWheelJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as WheelJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    const d = vec3.add(
        vec3.add(vec3.sub(stateB.deltaPosition, stateA.deltaPosition), joint.deltaCenter),
        vec3.sub(rB, rA),
    );

    const quatA = quat.mul(stateA.deltaRotation, joint.frameA.q);
    let quatB = quat.mul(stateB.deltaRotation, joint.frameB.q);
    if (quat.dot(quatA, quatB) < 0) {
        quatB = quat.negate(quatB);
    }

    const matrixA = mat3.fromQuat(quatA);
    const matrixB = mat3.fromQuat(quatB);

    const sAx = vec3.cross(vec3.add(d, rA), matrixA.cx);
    const sBx = vec3.cross(rB, matrixA.cx);
    const sAy = vec3.cross(vec3.add(d, rA), matrixA.cy);
    const sBy = vec3.cross(rB, matrixA.cy);
    const sAz = vec3.cross(vec3.add(d, rA), matrixA.cz);
    const sBz = vec3.cross(rB, matrixA.cz);

    const suspensionImpulse = f32(
        f32(joint.suspensionSpringImpulse + joint.lowerSuspensionImpulse) -
            joint.upperSuspensionImpulse,
    );

    const linearImpulseY = joint.linearImpulse.x;
    const linearImpulseZ = joint.linearImpulse.y;
    const angularImpulseX = joint.angularImpulse.x;
    const angularImpulseY = joint.angularImpulse.y;

    const P = vec3.blend3(
        suspensionImpulse,
        matrixA.cx,
        linearImpulseY,
        matrixA.cy,
        linearImpulseZ,
        matrixA.cz,
    );
    const La = vec3.blend3(suspensionImpulse, sAx, linearImpulseY, sAy, linearImpulseZ, sAz);
    const Lb = vec3.blend3(suspensionImpulse, sBx, linearImpulseY, sBy, linearImpulseZ, sBz);
    let angImp = vec3.scale(joint.spinImpulse, matrixA.cz);

    const spinAxis = matrixB.cz;

    if (joint.enableSteering) {
        const steeringAxis = steeringAxisOf(matrixA, matrixB);
        const perpAxis = vec3.cross(spinAxis, matrixA.cx);
        const steeringImpulse = f32(
            f32(joint.steeringSpringImpulse + joint.lowerSteeringImpulse) -
                joint.upperSteeringImpulse,
        );
        angImp = vec3.blend3(
            angularImpulseX,
            perpAxis,
            joint.spinImpulse,
            spinAxis,
            steeringImpulse,
            steeringAxis,
        );
    } else {
        const relQ = quat.invMul(quatA, quatB);
        const perpAxisX = vec3.scale(
            f32(0.5),
            quat.rotate(
                quatA,
                vec3.add(vec3.scale(relQ.s, vec3.axisX()), vec3.cross(relQ.v, vec3.axisX())),
            ),
        );
        const perpAxisY = vec3.scale(
            f32(0.5),
            quat.rotate(
                quatA,
                vec3.add(vec3.scale(relQ.s, vec3.axisY()), vec3.cross(relQ.v, vec3.axisY())),
            ),
        );
        angImp = vec3.add(
            angImp,
            vec3.blend3(
                angularImpulseX,
                perpAxisX,
                angularImpulseY,
                perpAxisY,
                joint.spinImpulse,
                spinAxis,
            ),
        );
    }

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.linearVelocity = vec3.mulSub(stateA.linearVelocity, mA, P);
        stateA.angularVelocity = vec3.sub(
            stateA.angularVelocity,
            mat3.mulV(iA, vec3.add(La, angImp)),
        );
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.linearVelocity = vec3.mulAdd(stateB.linearVelocity, mB, P);
        stateB.angularVelocity = vec3.add(
            stateB.angularVelocity,
            mat3.mulV(iB, vec3.add(Lb, angImp)),
        );
    }
}

export function solveWheelJoint(sim: JointSim, context: StepContext, useBias: boolean): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as WheelJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const fixedRotation = sim.fixedRotation;

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    const quatA = quat.mul(stateA.deltaRotation, joint.frameA.q);
    let quatB = quat.mul(stateB.deltaRotation, joint.frameB.q);
    if (quat.dot(quatA, quatB) < 0) {
        quatB = quat.negate(quatB);
    }

    const relQ = quat.invMul(quatA, quatB);
    const matrixA = mat3.fromQuat(quatA);
    const matrixB = mat3.fromQuat(quatB);

    const d = vec3.add(
        vec3.add(vec3.sub(stateB.deltaPosition, stateA.deltaPosition), joint.deltaCenter),
        vec3.sub(rB, rA),
    );
    const sAx = vec3.cross(vec3.add(d, rA), matrixA.cx);
    const sBx = vec3.cross(rB, matrixA.cx);
    const sAy = vec3.cross(vec3.add(d, rA), matrixA.cy);
    const sBy = vec3.cross(rB, matrixA.cy);
    const sAz = vec3.cross(vec3.add(d, rA), matrixA.cz);
    const sBz = vec3.cross(rB, matrixA.cz);

    const translation = vec3.dot(matrixA.cx, d);

    const cs = vec3.dot(matrixB.cz, matrixA.cz);
    const ss = f32(-vec3.dot(matrixB.cz, matrixA.cy));
    const steeringAxis = steeringAxisOf(matrixA, matrixB);

    // motor constraint
    if (joint.enableSpinMotor && fixedRotation === false) {
        const spinAxis = matrixB.cz;
        const cdot = f32(vec3.dot(vec3.sub(wB, wA), spinAxis) - joint.spinSpeed);
        let impulse = f32(-joint.spinMass * cdot);
        const oldImpulse = joint.spinImpulse;
        const maxImpulse = f32(context.h * joint.maxSpinTorque);
        joint.spinImpulse = clampf(f32(joint.spinImpulse + impulse), -maxImpulse, maxImpulse);
        impulse = f32(joint.spinImpulse - oldImpulse);

        wA = vec3.sub(wA, mat3.mulV(iA, vec3.scale(impulse, spinAxis)));
        wB = vec3.add(wB, mat3.mulV(iB, vec3.scale(impulse, spinAxis)));
    }

    // suspension
    if (joint.enableSuspensionSpring) {
        // A real spring, applied even during relax.
        const c = translation;
        const bias = f32(joint.suspensionSoftness.biasRate * c);
        const massScale = joint.suspensionSoftness.massScale;
        const impulseScale = joint.suspensionSoftness.impulseScale;

        const cdot = f32(
            f32(vec3.dot(matrixA.cx, vec3.sub(vB, vA)) + vec3.dot(sBx, wB)) - vec3.dot(sAx, wA),
        );
        const impulse = f32(
            f32(f32(f32(-massScale * joint.suspensionMass) * f32(cdot + bias))) -
                f32(impulseScale * joint.suspensionSpringImpulse),
        );
        joint.suspensionSpringImpulse = f32(joint.suspensionSpringImpulse + impulse);

        const linearImpulse = vec3.scale(impulse, matrixA.cx);
        const angularImpulseA = vec3.scale(impulse, sAx);
        const angularImpulseB = vec3.scale(impulse, sBx);

        vA = vec3.mulSub(vA, mA, linearImpulse);
        wA = vec3.sub(wA, mat3.mulV(iA, angularImpulseA));
        vB = vec3.mulAdd(vB, mB, linearImpulse);
        wB = vec3.add(wB, mat3.mulV(iB, angularImpulseB));
    }

    // steering
    if (joint.enableSteering && fixedRotation === false) {
        const steeringAngle = atan2(ss, cs);

        {
            // A real spring, applied even during relax.
            const c = f32(steeringAngle - joint.targetSteeringAngle);
            const bias = f32(joint.steeringSoftness.biasRate * c);
            const massScale = joint.steeringSoftness.massScale;
            const impulseScale = joint.steeringSoftness.impulseScale;

            const cdot = vec3.dot(steeringAxis, vec3.sub(wB, wA));
            const oldImpulse = joint.steeringSpringImpulse;
            let impulse = f32(
                f32(f32(f32(-massScale * joint.steeringMass) * f32(cdot + bias))) -
                    f32(impulseScale * oldImpulse),
            );
            const maxImpulse = f32(context.h * joint.maxSteeringTorque);
            joint.steeringSpringImpulse = clampf(
                f32(oldImpulse + impulse),
                -maxImpulse,
                maxImpulse,
            );
            impulse = f32(joint.steeringSpringImpulse - oldImpulse);

            wA = vec3.sub(wA, mat3.mulV(iA, vec3.scale(impulse, steeringAxis)));
            wB = vec3.add(wB, mat3.mulV(iB, vec3.scale(impulse, steeringAxis)));
        }

        if (joint.enableSteeringLimit) {
            // Lower limit
            {
                const c = f32(steeringAngle - joint.lowerSteeringLimit);
                let bias = 0;
                let massScale = f32(1);
                let impulseScale = 0;
                if (c > 0) {
                    bias = f32(c * context.invH);
                } else if (useBias) {
                    bias = f32(sim.constraintSoftness.biasRate * c);
                    massScale = sim.constraintSoftness.massScale;
                    impulseScale = sim.constraintSoftness.impulseScale;
                }

                const cdot = vec3.dot(steeringAxis, vec3.sub(wB, wA));
                const oldImpulse = joint.lowerSteeringImpulse;
                let impulse = f32(
                    f32(f32(f32(-massScale * joint.steeringMass) * f32(cdot + bias))) -
                        f32(impulseScale * oldImpulse),
                );
                joint.lowerSteeringImpulse = maxf(f32(oldImpulse + impulse), 0);
                impulse = f32(joint.lowerSteeringImpulse - oldImpulse);

                wA = vec3.sub(wA, mat3.mulV(iA, vec3.scale(impulse, steeringAxis)));
                wB = vec3.add(wB, mat3.mulV(iB, vec3.scale(impulse, steeringAxis)));
            }

            // Upper limit (signs flipped to keep c positive when satisfied)
            {
                const c = f32(joint.upperSteeringLimit - steeringAngle);
                let bias = 0;
                let massScale = f32(1);
                let impulseScale = 0;
                if (c > 0) {
                    bias = f32(c * context.invH);
                } else if (useBias) {
                    bias = f32(sim.constraintSoftness.biasRate * c);
                    massScale = sim.constraintSoftness.massScale;
                    impulseScale = sim.constraintSoftness.impulseScale;
                }

                // sign flipped on cdot
                const cdot = vec3.dot(steeringAxis, vec3.sub(wA, wB));
                const oldImpulse = joint.upperSteeringImpulse;
                let impulse = f32(
                    f32(f32(f32(-massScale * joint.steeringMass) * f32(cdot + bias))) -
                        f32(impulseScale * oldImpulse),
                );
                joint.upperSteeringImpulse = maxf(f32(oldImpulse + impulse), 0);
                impulse = f32(joint.upperSteeringImpulse - oldImpulse);

                // sign flipped on applied impulse
                wA = vec3.add(wA, mat3.mulV(iA, vec3.scale(impulse, steeringAxis)));
                wB = vec3.sub(wB, mat3.mulV(iB, vec3.scale(impulse, steeringAxis)));
            }
        }
    }

    if (joint.enableSuspensionLimit) {
        // Lower limit
        {
            const c = f32(translation - joint.lowerSuspensionLimit);
            let bias = 0;
            let massScale = f32(1);
            let impulseScale = 0;
            if (c > 0) {
                bias = f32(c * context.invH);
            } else if (useBias) {
                bias = f32(sim.constraintSoftness.biasRate * c);
                massScale = sim.constraintSoftness.massScale;
                impulseScale = sim.constraintSoftness.impulseScale;
            }

            const cdot = f32(
                f32(vec3.dot(matrixA.cx, vec3.sub(vB, vA)) + vec3.dot(sBx, wB)) - vec3.dot(sAx, wA),
            );
            let impulse = f32(
                f32(f32(f32(-massScale * joint.suspensionMass) * f32(cdot + bias))) -
                    f32(impulseScale * joint.lowerSuspensionImpulse),
            );
            const oldImpulse = joint.lowerSuspensionImpulse;
            joint.lowerSuspensionImpulse = maxf(f32(oldImpulse + impulse), 0);
            impulse = f32(joint.lowerSuspensionImpulse - oldImpulse);

            const linearImpulse = vec3.scale(impulse, matrixA.cx);
            const angularImpulseA = vec3.scale(impulse, sAx);
            const angularImpulseB = vec3.scale(impulse, sBx);

            vA = vec3.mulSub(vA, mA, linearImpulse);
            wA = vec3.sub(wA, mat3.mulV(iA, angularImpulseA));
            vB = vec3.mulAdd(vB, mB, linearImpulse);
            wB = vec3.add(wB, mat3.mulV(iB, angularImpulseB));
        }

        // Upper limit (signs flipped to keep c positive when satisfied)
        {
            const c = f32(joint.upperSuspensionLimit - translation);
            let bias = 0;
            let massScale = f32(1);
            let impulseScale = 0;
            if (c > 0) {
                bias = f32(c * context.invH);
            } else if (useBias) {
                bias = f32(sim.constraintSoftness.biasRate * c);
                massScale = sim.constraintSoftness.massScale;
                impulseScale = sim.constraintSoftness.impulseScale;
            }

            // sign flipped on cdot
            const cdot = f32(
                f32(vec3.dot(matrixA.cx, vec3.sub(vA, vB)) + vec3.dot(sAx, wA)) - vec3.dot(sBx, wB),
            );
            let impulse = f32(
                f32(f32(f32(-massScale * joint.suspensionMass) * f32(cdot + bias))) -
                    f32(impulseScale * joint.upperSuspensionImpulse),
            );
            const oldImpulse = joint.upperSuspensionImpulse;
            joint.upperSuspensionImpulse = maxf(f32(oldImpulse + impulse), 0);
            impulse = f32(joint.upperSuspensionImpulse - oldImpulse);

            const linearImpulse = vec3.scale(impulse, matrixA.cx);
            const angularImpulseA = vec3.scale(impulse, sAx);
            const angularImpulseB = vec3.scale(impulse, sBx);

            // sign flipped on applied impulse
            vA = vec3.mulAdd(vA, mA, linearImpulse);
            wA = vec3.add(wA, mat3.mulV(iA, angularImpulseA));
            vB = vec3.mulSub(vB, mB, linearImpulse);
            wB = vec3.sub(wB, mat3.mulV(iB, angularImpulseB));
        }
    }

    // Collinearity constraint
    if (fixedRotation === false) {
        if (joint.enableSteering === true) {
            let bias = 0;
            let massScale = f32(1);
            let impulseScale = 0;
            if (useBias) {
                const c = vec3.dot(matrixA.cx, matrixB.cz);
                bias = f32(sim.constraintSoftness.biasRate * c);
                massScale = sim.constraintSoftness.massScale;
                impulseScale = sim.constraintSoftness.impulseScale;
            }

            const u = vec3.cross(matrixB.cz, matrixA.cx);
            const cdot = vec3.dot(vec3.sub(wB, wA), u);

            const invInertiaSum = mat3.add(iA, iB);
            const k = vec3.dot(u, mat3.mulV(invInertiaSum, u));
            const perpMass = k > 0 ? f32(1 / k) : 0;

            const deltaImpulse = f32(
                f32(f32(f32(-massScale * perpMass) * f32(cdot + bias))) -
                    f32(impulseScale * joint.angularImpulse.x),
            );
            joint.angularImpulse.x = f32(joint.angularImpulse.x + deltaImpulse);

            wA = vec3.mulSub(wA, deltaImpulse, mat3.mulV(iA, u));
            wB = vec3.mulAdd(wB, deltaImpulse, mat3.mulV(iB, u));
        } else {
            let bias: Vec2 = { x: 0, y: 0 };
            let massScale = f32(1);
            let impulseScale = 0;
            if (useBias) {
                const c: Vec2 = { x: relQ.v.x, y: relQ.v.y };
                bias = {
                    x: f32(sim.constraintSoftness.biasRate * c.x),
                    y: f32(sim.constraintSoftness.biasRate * c.y),
                };
                massScale = sim.constraintSoftness.massScale;
                impulseScale = sim.constraintSoftness.impulseScale;
            }

            // Collinearity constraint as 2-by-2
            const perpAxisX = vec3.scale(
                f32(0.5),
                quat.rotate(
                    quatA,
                    vec3.add(vec3.scale(relQ.s, vec3.axisX()), vec3.cross(relQ.v, vec3.axisX())),
                ),
            );
            const perpAxisY = vec3.scale(
                f32(0.5),
                quat.rotate(
                    quatA,
                    vec3.add(vec3.scale(relQ.s, vec3.axisY()), vec3.cross(relQ.v, vec3.axisY())),
                ),
            );

            const invInertiaSum = mat3.add(iA, iB);
            const kxx = vec3.dot(perpAxisX, mat3.mulV(invInertiaSum, perpAxisX));
            const kyy = vec3.dot(perpAxisY, mat3.mulV(invInertiaSum, perpAxisY));
            const kxy = vec3.dot(perpAxisX, mat3.mulV(invInertiaSum, perpAxisY));

            const k: Mat2 = { cx: { x: kxx, y: kxy }, cy: { x: kxy, y: kyy } };

            const wRel = vec3.sub(wB, wA);
            const cdot: Vec2 = { x: vec3.dot(wRel, perpAxisX), y: vec3.dot(wRel, perpAxisY) };
            const oldImpulse = joint.angularImpulse;
            const cdotPlusBias: Vec2 = { x: f32(cdot.x + bias.x), y: f32(cdot.y + bias.y) };
            const sol = mat2.solve(k, cdotPlusBias);
            const deltaImpulse: Vec2 = {
                x: f32(f32(-massScale * sol.x) - f32(impulseScale * oldImpulse.x)),
                y: f32(f32(-massScale * sol.y) - f32(impulseScale * oldImpulse.y)),
            };
            joint.angularImpulse = {
                x: f32(oldImpulse.x + deltaImpulse.x),
                y: f32(oldImpulse.y + deltaImpulse.y),
            };

            const angularImpulse = vec3.blend2(
                deltaImpulse.x,
                perpAxisX,
                deltaImpulse.y,
                perpAxisY,
            );
            wA = vec3.sub(wA, mat3.mulV(iA, angularImpulse));
            wB = vec3.add(wB, mat3.mulV(iB, angularImpulse));
        }
    }

    // Solve point-to-line constraint
    {
        const perpY = matrixA.cy;
        const perpZ = matrixA.cz;

        let bias: Vec2 = { x: 0, y: 0 };
        let massScale = f32(1);
        let impulseScale = 0;
        if (useBias) {
            const c: Vec2 = { x: vec3.dot(perpY, d), y: vec3.dot(perpZ, d) };
            bias = {
                x: f32(sim.constraintSoftness.biasRate * c.x),
                y: f32(sim.constraintSoftness.biasRate * c.y),
            };
            massScale = sim.constraintSoftness.massScale;
            impulseScale = sim.constraintSoftness.impulseScale;
        }

        const vRel = vec3.sub(
            vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vA),
            vec3.cross(wA, vec3.add(rA, d)),
        );
        const cdot: Vec2 = { x: vec3.dot(perpY, vRel), y: vec3.dot(perpZ, vRel) };

        const kyy = f32(
            f32(
                f32(f32(mA + mB) + vec3.dot(sAy, mat3.mulV(iA, sAy))) +
                    vec3.dot(sBy, mat3.mulV(iB, sBy)),
            ),
        );
        const kyz = f32(vec3.dot(sAy, mat3.mulV(iA, sAz)) + vec3.dot(sBy, mat3.mulV(iB, sBz)));
        const kzz = f32(
            f32(
                f32(f32(mA + mB) + vec3.dot(sAz, mat3.mulV(iA, sAz))) +
                    vec3.dot(sBz, mat3.mulV(iB, sBz)),
            ),
        );

        const k: Mat2 = { cx: { x: kyy, y: kyz }, cy: { x: kyz, y: kzz } };

        const oldImpulse = joint.linearImpulse;
        const cdotPlusBias: Vec2 = { x: f32(cdot.x + bias.x), y: f32(cdot.y + bias.y) };
        const sol = mat2.solve(k, cdotPlusBias);
        const deltaImpulse: Vec2 = {
            x: f32(f32(-massScale * sol.x) - f32(impulseScale * oldImpulse.x)),
            y: f32(f32(-massScale * sol.y) - f32(impulseScale * oldImpulse.y)),
        };
        joint.linearImpulse = {
            x: f32(oldImpulse.x + deltaImpulse.x),
            y: f32(oldImpulse.y + deltaImpulse.y),
        };

        const linearImpulse = vec3.blend2(deltaImpulse.x, perpY, deltaImpulse.y, perpZ);

        vA = vec3.mulSub(vA, mA, linearImpulse);
        wA = vec3.sub(wA, mat3.mulV(iA, vec3.blend2(deltaImpulse.x, sAy, deltaImpulse.y, sAz)));
        vB = vec3.mulAdd(vB, mB, linearImpulse);
        wB = vec3.add(wB, mat3.mulV(iB, vec3.blend2(deltaImpulse.x, sBy, deltaImpulse.y, sBz)));
    }

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.linearVelocity = vA;
        stateA.angularVelocity = wA;
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.linearVelocity = vB;
        stateB.angularVelocity = wB;
    }
}

// --- Force / torque accessors -----------------------------------------------------------------

/** The reaction force this joint applies (b3GetWheelJointForce). */
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
