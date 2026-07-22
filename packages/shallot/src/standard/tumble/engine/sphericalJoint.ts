// The spherical (ball-and-socket) joint — Box3D's spherical_joint.c (Erin Catto, MIT). A point-to-point
// linear constraint with optional angular spring, angular-velocity motor, a swing (cone) limit about
// body A's local z, and a twist limit about the shared axis. Every op is fround-wrapped, mirroring the
// C scalar expression tree exactly. See the README.

import { NULL_INDEX } from "./array";
import { BodyFlags, getBodyTransformQuick, identityBodyState } from "./body";
import type { StepContext } from "./contactsolver";
import { SetType } from "./core";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";
import {
    clampf,
    FLT_MIN,
    f32,
    type Mat3,
    mat3,
    maxf,
    minf,
    PI,
    type Quat,
    quat,
    type Transform,
    type Vec3,
    vec3,
} from "./math";
import { makeSoft, type Softness } from "./softness";
import type { WorldState } from "./world";

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
    indexA: number;
    indexB: number;
    frameA: Transform;
    frameB: Transform;
    deltaCenter: Vec3;
    swingAxis: Vec3;
    twistJacobian: Vec3;
    rotationMass: Mat3;
    swingMass: number;
    twistMass: number;
    springSoftness: Softness;
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

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});
const zeroVec3 = (): Vec3 => ({ x: 0, y: 0, z: 0 });
const zeroMat3 = (): Mat3 => ({
    cx: { x: 0, y: 0, z: 0 },
    cy: { x: 0, y: 0, z: 0 },
    cz: { x: 0, y: 0, z: 0 },
});

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
        indexA: 0,
        indexB: 0,
        frameA: identityTransform(),
        frameB: identityTransform(),
        deltaCenter: zeroVec3(),
        swingAxis: zeroVec3(),
        twistJacobian: zeroVec3(),
        rotationMass: zeroMat3(),
        swingMass: 0,
        twistMass: 0,
        springSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        enableSpring: def.enableSpring,
        enableMotor: def.enableMotor,
        enableConeLimit: def.enableConeLimit,
        enableTwistLimit: def.enableTwistLimit,
    };
    pair.sim.data = data;
    return pair;
}

export function prepareSphericalJoint(sim: JointSim, context: StepContext): void {
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

    const joint = sim.data as SphericalJoint;
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

    // Cone axis is body A's z-axis; twist axis is body B's z-axis.
    const coneAxis = quat.rotate(joint.frameA.q, vec3.axisZ());
    const twistAxis = quat.rotate(joint.frameB.q, vec3.axisZ());

    if (joint.enableConeLimit) {
        const swingAxis = vec3.normalize(vec3.cross(coneAxis, twistAxis));
        const k = vec3.dot(swingAxis, mat3.mulV(invInertiaSum, swingAxis));
        joint.swingMass = k > 0 ? f32(1 / k) : 0;
        joint.swingAxis = swingAxis;
    }

    if (joint.enableTwistLimit) {
        const relQ = quat.invMul(joint.frameA.q, joint.frameB.q);
        const num = f32(f32(relQ.v.x * relQ.v.x) + f32(relQ.v.y * relQ.v.y));
        const den = f32(f32(relQ.v.z * relQ.v.z) + f32(relQ.s * relQ.s));
        const tanThetaOver2 = f32(Math.sqrt(f32(num / den)));

        const swingAxis = vec3.normalize(vec3.cross(coneAxis, twistAxis));
        const perpAxis = vec3.cross(swingAxis, coneAxis);
        const twistJacobian = vec3.mulAdd(coneAxis, tanThetaOver2, perpAxis);
        const k = vec3.dot(twistJacobian, mat3.mulV(invInertiaSum, twistJacobian));
        joint.twistMass = k > 0 ? f32(1 / k) : 0;
        joint.twistJacobian = twistJacobian;
    }

    if (sim.fixedRotation === false) {
        joint.rotationMass = mat3.invert(invInertiaSum);
    } else {
        joint.rotationMass = zeroMat3();
    }

    joint.springSoftness = makeSoft(joint.hertz, joint.dampingRatio, context.h);

    if (context.enableWarmStarting === false) {
        joint.linearImpulse = zeroVec3();
        joint.motorImpulse = zeroVec3();
        joint.springImpulse = zeroVec3();
        joint.swingImpulse = 0;
        joint.lowerTwistImpulse = 0;
        joint.upperTwistImpulse = 0;
    }
}

export function warmStartSphericalJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as SphericalJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    let angularImpulse = vec3.add(joint.springImpulse, joint.motorImpulse);
    angularImpulse = vec3.mulSub(angularImpulse, joint.swingImpulse, joint.swingAxis);
    angularImpulse = vec3.mulAdd(
        angularImpulse,
        f32(joint.lowerTwistImpulse - joint.upperTwistImpulse),
        joint.twistJacobian,
    );

    vA = vec3.mulSub(vA, mA, joint.linearImpulse);
    wA = vec3.sub(wA, mat3.mulV(iA, vec3.add(vec3.cross(rA, joint.linearImpulse), angularImpulse)));
    vB = vec3.mulAdd(vB, mB, joint.linearImpulse);
    wB = vec3.add(wB, mat3.mulV(iB, vec3.add(vec3.cross(rB, joint.linearImpulse), angularImpulse)));

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.linearVelocity = vA;
        stateA.angularVelocity = wA;
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.linearVelocity = vB;
        stateB.angularVelocity = wB;
    }
}

export function solveSphericalJoint(sim: JointSim, context: StepContext, useBias: boolean): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as SphericalJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const fixedRotation = sim.fixedRotation;
    const quatA = quat.mul(stateA.deltaRotation, joint.frameA.q);
    const quatB = quat.mul(stateB.deltaRotation, joint.frameB.q);

    const relQ = quat.invMul(quatA, quatB);

    // Solve spring
    if (joint.enableSpring && fixedRotation === false) {
        const deltaRotation = quat.deltaToRotation(relQ, joint.targetRotation);
        const c = vec3.neg(quat.rotate(quatA, deltaRotation));

        const bias = vec3.scale(joint.springSoftness.biasRate, c);
        const massScale = joint.springSoftness.massScale;
        const impulseScale = joint.springSoftness.impulseScale;
        const cdot = vec3.sub(wB, wA);

        const impulse = vec3.mulSub(
            vec3.scale(-massScale, mat3.mulV(joint.rotationMass, vec3.add(cdot, bias))),
            impulseScale,
            joint.springImpulse,
        );
        joint.springImpulse = vec3.add(joint.springImpulse, impulse);

        wA = vec3.sub(wA, mat3.mulV(iA, impulse));
        wB = vec3.add(wB, mat3.mulV(iB, impulse));
    }

    if (joint.enableMotor && fixedRotation === false) {
        const cdot = vec3.sub(wB, wA);

        let lambda = vec3.neg(mat3.mulV(joint.rotationMass, vec3.sub(cdot, joint.motorVelocity)));
        let newImpulse = vec3.add(joint.motorImpulse, lambda);
        const length = f32(Math.sqrt(vec3.lengthSq(newImpulse)));
        const maxImpulse = f32(joint.maxMotorTorque * context.h);
        if (length > maxImpulse) {
            newImpulse = vec3.scale(f32(maxImpulse / length), newImpulse);
        }

        lambda = vec3.sub(newImpulse, joint.motorImpulse);
        joint.motorImpulse = newImpulse;

        wA = vec3.sub(wA, mat3.mulV(iA, lambda));
        wB = vec3.add(wB, mat3.mulV(iB, lambda));
    }

    if (joint.enableTwistLimit && fixedRotation === false) {
        const twistAngle = quat.getTwistAngle(relQ);
        const twistJacobian = joint.twistJacobian;

        // Lower limit
        {
            const c = f32(twistAngle - joint.lowerTwistAngle);
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

            const cdot = vec3.dot(vec3.sub(wB, wA), twistJacobian);
            const oldImpulse = joint.lowerTwistImpulse;
            let deltaImpulse = f32(
                f32(f32(f32(-massScale * joint.twistMass) * f32(cdot + bias))) -
                    f32(impulseScale * oldImpulse),
            );
            joint.lowerTwistImpulse = maxf(f32(oldImpulse + deltaImpulse), 0);
            deltaImpulse = f32(joint.lowerTwistImpulse - oldImpulse);

            wA = vec3.mulSub(wA, deltaImpulse, mat3.mulV(iA, twistJacobian));
            wB = vec3.mulAdd(wB, deltaImpulse, mat3.mulV(iB, twistJacobian));
        }

        // Upper limit
        {
            const c = f32(joint.upperTwistAngle - twistAngle);
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

            // sign flipped on Cdot
            const cdot = vec3.dot(vec3.sub(wA, wB), twistJacobian);
            const oldImpulse = joint.upperTwistImpulse;
            let deltaImpulse = f32(
                f32(f32(f32(-massScale * joint.twistMass) * f32(cdot + bias))) -
                    f32(impulseScale * oldImpulse),
            );
            joint.upperTwistImpulse = maxf(f32(oldImpulse + deltaImpulse), 0);
            deltaImpulse = f32(joint.upperTwistImpulse - oldImpulse);

            // sign flipped on applied impulse
            wA = vec3.mulAdd(wA, deltaImpulse, mat3.mulV(iA, twistJacobian));
            wB = vec3.mulSub(wB, deltaImpulse, mat3.mulV(iB, twistJacobian));
        }
    }

    if (joint.enableConeLimit && fixedRotation === false) {
        const swingAngle = quat.getSwingAngle(relQ);
        const swingAxis = joint.swingAxis;

        const c = f32(joint.coneAngle - swingAngle);
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

        // sign flipped on Cdot
        const cdot = vec3.dot(vec3.sub(wA, wB), swingAxis);
        const oldImpulse = joint.swingImpulse;
        let deltaImpulse = f32(
            f32(f32(f32(-massScale * joint.swingMass) * f32(cdot + bias))) -
                f32(impulseScale * oldImpulse),
        );
        joint.swingImpulse = maxf(f32(oldImpulse + deltaImpulse), 0);
        deltaImpulse = f32(joint.swingImpulse - oldImpulse);

        // sign flipped on applied impulse
        wA = vec3.mulAdd(wA, deltaImpulse, mat3.mulV(iA, swingAxis));
        wB = vec3.mulSub(wB, deltaImpulse, mat3.mulV(iB, swingAxis));
    }

    // Solve point-to-point constraint
    {
        const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
        const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

        const cdot = vec3.sub(vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vA), vec3.cross(wA, rA));

        let bias: Vec3 = { x: 0, y: 0, z: 0 };
        let massScale = f32(1);
        let impulseScale = 0;
        if (useBias) {
            const dcA = stateA.deltaPosition;
            const dcB = stateB.deltaPosition;
            let separation = vec3.add(vec3.sub(dcB, dcA), vec3.sub(rB, rA));
            separation = vec3.add(separation, joint.deltaCenter);
            bias = vec3.scale(sim.constraintSoftness.biasRate, separation);
            massScale = sim.constraintSoftness.massScale;
            impulseScale = sim.constraintSoftness.impulseScale;
        }

        // K = [(1/mA + 1/mB) * E - skew(rA) * invIA * skew(rA) - skew(rB) * invIB * skew(rB)]
        const sA = mat3.skew(rA);
        const sB = mat3.skew(rB);
        const kA = mat3.mul(sA, mat3.mul(sim.invIA, sA));
        const kB = mat3.mul(sB, mat3.mul(sim.invIB, sB));
        const k = mat3.neg(mat3.add(kA, kB));
        const mm = f32(mA + mB);
        k.cx.x = f32(k.cx.x + mm);
        k.cy.y = f32(k.cy.y + mm);
        k.cz.z = f32(k.cz.z + mm);

        const b = mat3.solve(k, vec3.add(cdot, bias));

        const impulse = vec3.sub(
            vec3.scale(-massScale, b),
            vec3.scale(impulseScale, joint.linearImpulse),
        );
        joint.linearImpulse = vec3.add(joint.linearImpulse, impulse);

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

// --- Force / torque accessors -----------------------------------------------------------------

/** The reaction force this joint applies (b3GetSphericalJointForce). */
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
