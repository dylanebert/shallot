// The motor joint — Box3D's motor_joint.c (Erin Catto, MIT). Drives two bodies toward target relative
// linear + angular velocities (each capped by a max force/torque) and optionally holds a soft spring at
// the reference pose. Four independent sub-solves: angular spring, angular velocity, linear spring,
// linear velocity. Solve takes no useBias. Every op is fround-wrapped, mirroring the C scalar
// expression tree exactly. See the README.

import { NULL_INDEX } from "./array";
import { BodyFlags, identityBodyState } from "./body";
import type { StepContext } from "./contactsolver";
import { SetType } from "./core";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";
import {
    FLT_MIN,
    f32,
    type Mat3,
    mat3,
    type Quat,
    quat,
    type Transform,
    type Vec3,
    vec3,
} from "./math";
import { makeSoft, type Softness } from "./softness";
import type { WorldState } from "./world";

/** Motor joint payload (b3MotorJoint). Impulses persist across steps for warm starting. */
export type MotorJoint = {
    linearVelocity: Vec3;
    angularVelocity: Vec3;
    maxVelocityForce: number;
    maxVelocityTorque: number;
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
    maxSpringForce: number;
    maxSpringTorque: number;
    linearVelocityImpulse: Vec3;
    angularVelocityImpulse: Vec3;
    linearSpringImpulse: Vec3;
    angularSpringImpulse: Vec3;
    indexA: number;
    indexB: number;
    frameA: Transform;
    frameB: Transform;
    deltaCenter: Vec3;
    linearSpring: Softness;
    angularSpring: Softness;
    angularMass: Mat3;
};

/** Motor joint definition (b3MotorJointDef), body handles resolved to a base JointDef. */
export type MotorJointDef = {
    base: JointDef;
    linearVelocity: Vec3;
    maxVelocityForce: number;
    angularVelocity: Vec3;
    maxVelocityTorque: number;
    linearHertz: number;
    linearDampingRatio: number;
    maxSpringForce: number;
    angularHertz: number;
    angularDampingRatio: number;
    maxSpringTorque: number;
};

/** @returns the ported motor joint definition defaults (b3DefaultMotorJointDef). */
export function defaultMotorJointDef(base: JointDef): MotorJointDef {
    return {
        base,
        linearVelocity: { x: 0, y: 0, z: 0 },
        maxVelocityForce: 0,
        angularVelocity: { x: 0, y: 0, z: 0 },
        maxVelocityTorque: 0,
        linearHertz: 0,
        linearDampingRatio: 0,
        maxSpringForce: 0,
        angularHertz: 0,
        angularDampingRatio: 0,
        maxSpringTorque: 0,
    };
}

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});
const zeroVec3 = (): Vec3 => ({ x: 0, y: 0, z: 0 });
const zeroSoftness = (): Softness => ({ biasRate: 0, massScale: 0, impulseScale: 0 });
const zeroMat3 = (): Mat3 => ({
    cx: { x: 0, y: 0, z: 0 },
    cy: { x: 0, y: 0, z: 0 },
    cz: { x: 0, y: 0, z: 0 },
});

/** Create a motor joint (b3CreateMotorJoint). @returns the joint handle + sim. */
export function createMotorJoint(
    world: WorldState,
    def: MotorJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Motor);
    const data: MotorJoint = {
        linearVelocity: { ...def.linearVelocity },
        angularVelocity: { ...def.angularVelocity },
        maxVelocityForce: def.maxVelocityForce,
        maxVelocityTorque: def.maxVelocityTorque,
        linearHertz: def.linearHertz,
        linearDampingRatio: def.linearDampingRatio,
        angularHertz: def.angularHertz,
        angularDampingRatio: def.angularDampingRatio,
        maxSpringForce: def.maxSpringForce,
        maxSpringTorque: def.maxSpringTorque,
        linearVelocityImpulse: zeroVec3(),
        angularVelocityImpulse: zeroVec3(),
        linearSpringImpulse: zeroVec3(),
        angularSpringImpulse: zeroVec3(),
        indexA: 0,
        indexB: 0,
        frameA: identityTransform(),
        frameB: identityTransform(),
        deltaCenter: zeroVec3(),
        linearSpring: zeroSoftness(),
        angularSpring: zeroSoftness(),
        angularMass: zeroMat3(),
    };
    pair.sim.data = data;
    return pair;
}

export function prepareMotorJoint(sim: JointSim, context: StepContext): void {
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

    const joint = sim.data as MotorJoint;
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

    joint.linearSpring = makeSoft(joint.linearHertz, joint.linearDampingRatio, context.h);
    joint.angularSpring = makeSoft(joint.angularHertz, joint.angularDampingRatio, context.h);
    joint.angularMass = mat3.invert(invInertiaSum);

    if (context.enableWarmStarting === false) {
        joint.linearVelocityImpulse = zeroVec3();
        joint.angularVelocityImpulse = zeroVec3();
        joint.linearSpringImpulse = zeroVec3();
        joint.angularSpringImpulse = zeroVec3();
    }
}

export function warmStartMotorJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const joint = sim.data as MotorJoint;
    const dummy = identityBodyState();
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    const linearImpulse = vec3.add(joint.linearVelocityImpulse, joint.linearSpringImpulse);
    const angularImpulse = vec3.add(joint.angularVelocityImpulse, joint.angularSpringImpulse);

    // The C writes velocities unconditionally here (no dynamic-flag guard); the dummy state absorbs
    // static-body writes.
    stateA.linearVelocity = vec3.mulSub(stateA.linearVelocity, mA, linearImpulse);
    stateA.angularVelocity = vec3.sub(
        stateA.angularVelocity,
        mat3.mulV(iA, vec3.add(vec3.cross(rA, linearImpulse), angularImpulse)),
    );
    stateB.linearVelocity = vec3.mulAdd(stateB.linearVelocity, mB, linearImpulse);
    stateB.angularVelocity = vec3.add(
        stateB.angularVelocity,
        mat3.mulV(iB, vec3.add(vec3.cross(rB, linearImpulse), angularImpulse)),
    );
}

export function solveMotorJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as MotorJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const quatA = quat.mul(stateA.deltaRotation, joint.frameA.q);
    let quatB = quat.mul(stateB.deltaRotation, joint.frameB.q);
    if (quat.dot(quatA, quatB) < 0) {
        quatB = quat.negate(quatB);
    }
    const relQ = quat.invMul(quatA, quatB);

    // angular spring
    if (joint.maxSpringTorque > 0 && joint.angularHertz > 0) {
        const targetQuat: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
        const deltaRotation = quat.deltaToRotation(relQ, targetQuat);
        const c = vec3.neg(quat.rotate(quatA, deltaRotation));

        const bias = vec3.scale(joint.angularSpring.biasRate, c);
        const massScale = joint.angularSpring.massScale;
        const impulseScale = joint.angularSpring.impulseScale;

        const cdot = vec3.sub(wB, wA);
        const maxImpulse = f32(context.h * joint.maxSpringTorque);
        const oldImpulse = joint.angularSpringImpulse;
        let impulse = vec3.mulSub(
            vec3.scale(-massScale, mat3.mulV(joint.angularMass, vec3.add(cdot, bias))),
            impulseScale,
            oldImpulse,
        );
        joint.angularSpringImpulse = vec3.add(oldImpulse, impulse);
        if (vec3.lengthSq(joint.angularSpringImpulse) > f32(maxImpulse * maxImpulse)) {
            joint.angularSpringImpulse = vec3.scale(
                maxImpulse,
                vec3.normalize(joint.angularSpringImpulse),
            );
        }
        impulse = vec3.sub(joint.angularSpringImpulse, oldImpulse);

        wA = vec3.sub(wA, mat3.mulV(iA, impulse));
        wB = vec3.add(wB, mat3.mulV(iB, impulse));
    }

    // angular velocity
    if (joint.maxVelocityTorque > 0) {
        const cdot = vec3.sub(vec3.sub(wB, wA), joint.angularVelocity);
        let impulse = vec3.neg(mat3.mulV(joint.angularMass, cdot));

        const maxImpulse = f32(context.h * joint.maxVelocityTorque);
        const oldImpulse = joint.angularVelocityImpulse;
        joint.angularVelocityImpulse = vec3.add(oldImpulse, impulse);
        if (vec3.lengthSq(joint.angularVelocityImpulse) > f32(maxImpulse * maxImpulse)) {
            joint.angularVelocityImpulse = vec3.scale(
                maxImpulse,
                vec3.normalize(joint.angularVelocityImpulse),
            );
        }
        impulse = vec3.sub(joint.angularVelocityImpulse, oldImpulse);

        wA = vec3.sub(wA, mat3.mulV(iA, impulse));
        wB = vec3.add(wB, mat3.mulV(iB, impulse));
    }

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    // linear spring
    if (joint.maxSpringForce > 0 && joint.linearHertz > 0) {
        const dcA = stateA.deltaPosition;
        const dcB = stateB.deltaPosition;
        const c = vec3.add(vec3.add(vec3.sub(dcB, dcA), vec3.sub(rB, rA)), joint.deltaCenter);

        const bias = vec3.scale(joint.linearSpring.biasRate, c);
        const massScale = joint.linearSpring.massScale;
        const impulseScale = joint.linearSpring.impulseScale;

        const cdot = vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vec3.add(vA, vec3.cross(wA, rA)));

        const k = linearK(mA, mB, sim.invIA, sim.invIB, rA, rB);
        const b = mat3.solve(k, vec3.add(cdot, bias));

        const oldImpulse = joint.linearSpringImpulse;
        let impulse = vec3.mulSub(vec3.scale(-massScale, b), impulseScale, oldImpulse);
        const maxImpulse = f32(context.h * joint.maxSpringForce);
        joint.linearSpringImpulse = vec3.add(joint.linearSpringImpulse, impulse);
        if (vec3.lengthSq(joint.linearSpringImpulse) > f32(maxImpulse * maxImpulse)) {
            joint.linearSpringImpulse = vec3.scale(
                maxImpulse,
                vec3.normalize(joint.linearSpringImpulse),
            );
        }
        impulse = vec3.sub(joint.linearSpringImpulse, oldImpulse);

        vA = vec3.mulSub(vA, mA, impulse);
        wA = vec3.sub(wA, mat3.mulV(iA, vec3.cross(rA, impulse)));
        vB = vec3.mulAdd(vB, mB, impulse);
        wB = vec3.add(wB, mat3.mulV(iB, vec3.cross(rB, impulse)));
    }

    // linear velocity
    if (joint.maxVelocityForce > 0) {
        let cdot = vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vec3.add(vA, vec3.cross(wA, rA)));
        cdot = vec3.sub(cdot, joint.linearVelocity);

        const k = linearK(mA, mB, sim.invIA, sim.invIB, rA, rB);
        const b = mat3.solve(k, cdot);
        let impulse = vec3.neg(b);

        const oldImpulse = joint.linearVelocityImpulse;
        const maxImpulse = f32(context.h * joint.maxVelocityForce);
        joint.linearVelocityImpulse = vec3.add(joint.linearVelocityImpulse, impulse);
        if (vec3.lengthSq(joint.linearVelocityImpulse) > f32(maxImpulse * maxImpulse)) {
            joint.linearVelocityImpulse = vec3.scale(
                maxImpulse,
                vec3.normalize(joint.linearVelocityImpulse),
            );
        }
        impulse = vec3.sub(joint.linearVelocityImpulse, oldImpulse);

        vA = vec3.mulSub(vA, mA, impulse);
        wA = vec3.sub(wA, mat3.mulV(iA, vec3.cross(rA, impulse)));
        vB = vec3.mulAdd(vB, mB, impulse);
        wB = vec3.add(wB, mat3.mulV(iB, vec3.cross(rB, impulse)));
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

// K = (1/mA + 1/mB) * I - skew(rA) * invIA * skew(rA) - skew(rB) * invIB * skew(rB), the point-to-point
// effective-mass matrix shared by the linear spring and linear velocity sub-solves.
function linearK(mA: number, mB: number, invIA: Mat3, invIB: Mat3, rA: Vec3, rB: Vec3): Mat3 {
    const sA = mat3.skew(rA);
    const sB = mat3.skew(rB);
    const kA = mat3.mul(sA, mat3.mul(invIA, sA));
    const kB = mat3.mul(sB, mat3.mul(invIB, sB));
    const k = mat3.neg(mat3.add(kA, kB));
    const mm = f32(mA + mB);
    k.cx.x = f32(k.cx.x + mm);
    k.cy.y = f32(k.cy.y + mm);
    k.cz.z = f32(k.cz.z + mm);
    return k;
}

// --- Force / torque accessors -----------------------------------------------------------------

/** The reaction force this joint applies (b3GetMotorJointForce). */
export function getMotorJointForce(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as MotorJoint;
    return vec3.scale(world.invH, vec3.add(joint.linearVelocityImpulse, joint.linearSpringImpulse));
}

/** The reaction torque this joint applies (b3GetMotorJointTorque). */
export function getMotorJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as MotorJoint;
    return vec3.scale(
        world.invH,
        vec3.add(joint.angularVelocityImpulse, joint.angularSpringImpulse),
    );
}
