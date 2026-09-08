// The revolute (hinge) joint — Box3D's revolute_joint.c (Erin Catto, MIT). A point-to-point linear
// constraint plus a collinearity (perpendicularity) constraint about the hinge axis (body A's local
// z), with optional angular spring, motor, and twist limits. This is the arithmetic-critical solver
// path: every op is fround-wrapped, mirroring the C scalar expression tree exactly (no FMA, no SIMD
// min/max). See the README.

import { NULL_INDEX } from "./array";
import { BodyFlags, getBodyTransformQuick, identityBodyState } from "./body";
import type { StepContext } from "./contactsolver";
import { SetType } from "./core";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";
import {
    clampf,
    FLT_MIN,
    f32,
    type Mat2,
    mat2,
    mat3,
    maxf,
    minf,
    PI,
    quat,
    type Transform,
    type Vec2,
    type Vec3,
    vec2,
    vec3,
} from "./math";
import { makeSoft, type Softness } from "./softness";
import type { WorldState } from "./world";

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
    indexA: number;
    indexB: number;
    frameA: Transform;
    frameB: Transform;
    rotationAxisZ: Vec3;
    perpAxisX: Vec3;
    perpAxisY: Vec3;
    deltaCenter: Vec3;
    axialMass: number;
    springSoftness: Softness;
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

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
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
        indexA: 0,
        indexB: 0,
        frameA: identityTransform(),
        frameB: identityTransform(),
        rotationAxisZ: { x: 0, y: 0, z: 0 },
        perpAxisX: { x: 0, y: 0, z: 0 },
        perpAxisY: { x: 0, y: 0, z: 0 },
        deltaCenter: { x: 0, y: 0, z: 0 },
        axialMass: 0,
        springSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        enableSpring: def.enableSpring,
        enableLimit: def.enableLimit,
        enableMotor: def.enableMotor,
    };
    sim.data = data;
    return pair;
}

export function prepareRevoluteJoint(sim: JointSim, context: StepContext): void {
    const world = context.world;
    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];

    const setA = world.solverSets[bodyA.setIndex];
    const setB = world.solverSets[bodyB.setIndex];
    const localIndexA = bodyA.localIndex;
    const localIndexB = bodyB.localIndex;
    const bodySimA = setA.bodySims[localIndexA];
    const bodySimB = setB.bodySims[localIndexB];

    sim.invMassA = bodySimA.invMass;
    sim.invMassB = bodySimB.invMass;
    sim.invIA = bodySimA.invInertiaWorld;
    sim.invIB = bodySimB.invInertiaWorld;

    const invInertiaSum = mat3.add(sim.invIA, sim.invIB);
    sim.fixedRotation = mat3.det(invInertiaSum) < f32(1000 * FLT_MIN);

    const joint = sim.data as RevoluteJoint;
    joint.indexA = bodyA.setIndex === SetType.Awake ? localIndexA : NULL_INDEX;
    joint.indexB = bodyB.setIndex === SetType.Awake ? localIndexB : NULL_INDEX;

    // World-space joint frames relative to each body's center of mass.
    joint.frameA = {
        q: quat.mul(bodySimA.transform.q, sim.localFrameA.q),
        p: quat.rotate(bodySimA.transform.q, vec3.sub(sim.localFrameA.p, bodySimA.localCenter)),
    };
    joint.frameB = {
        q: quat.mul(bodySimB.transform.q, sim.localFrameB.q),
        p: quat.rotate(bodySimB.transform.q, vec3.sub(sim.localFrameB.p, bodySimB.localCenter)),
    };
    joint.deltaCenter = vec3.sub(bodySimB.center, bodySimA.center);

    {
        const rotationAxisZ = quat.rotate(joint.frameA.q, vec3.axisZ());
        const k = vec3.dot(rotationAxisZ, mat3.mulV(invInertiaSum, rotationAxisZ));
        joint.axialMass = k > 0 ? f32(1 / k) : 0;
        joint.rotationAxisZ = rotationAxisZ;
    }

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

    joint.springSoftness = makeSoft(joint.hertz, joint.dampingRatio, context.h);

    if (context.enableWarmStarting === false) {
        joint.linearImpulse = { x: 0, y: 0, z: 0 };
        joint.perpImpulse = { x: 0, y: 0 };
        joint.motorImpulse = 0;
        joint.springImpulse = 0;
        joint.lowerImpulse = 0;
        joint.upperImpulse = 0;
    }
}

export function warmStartRevoluteJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as RevoluteJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    const axialImpulse = f32(
        f32(f32(joint.springImpulse + joint.motorImpulse) + joint.lowerImpulse) -
            joint.upperImpulse,
    );
    let angularImpulse = vec3.add(
        vec3.scale(joint.perpImpulse.x, joint.perpAxisX),
        vec3.scale(joint.perpImpulse.y, joint.perpAxisY),
    );
    angularImpulse = vec3.mulAdd(angularImpulse, axialImpulse, joint.rotationAxisZ);

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

export function solveRevoluteJoint(sim: JointSim, context: StepContext, useBias: boolean): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as RevoluteJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const fixedRotation = sim.fixedRotation;
    const quatA = quat.mul(stateA.deltaRotation, joint.frameA.q);
    let quatB = quat.mul(stateB.deltaRotation, joint.frameB.q);

    if (quat.dot(quatA, quatB) < 0) {
        quatB = quat.negate(quatB);
    }

    const relQ = quat.invMul(quatA, quatB);

    // Solve spring
    if (joint.enableSpring && fixedRotation === false) {
        const targetAngle = joint.targetAngle;
        const angle = quat.getTwistAngle(relQ);
        const c = f32(angle - targetAngle);

        const bias = f32(joint.springSoftness.biasRate * c);
        const massScale = joint.springSoftness.massScale;
        const impulseScale = joint.springSoftness.impulseScale;
        const cdot = vec3.dot(vec3.sub(wB, wA), joint.rotationAxisZ);

        const deltaImpulse = f32(
            f32(f32(f32(-massScale * joint.axialMass) * f32(cdot + bias))) -
                f32(impulseScale * joint.springImpulse),
        );
        joint.springImpulse = f32(joint.springImpulse + deltaImpulse);

        wA = vec3.mulSub(wA, deltaImpulse, mat3.mulV(iA, joint.rotationAxisZ));
        wB = vec3.mulAdd(wB, deltaImpulse, mat3.mulV(iB, joint.rotationAxisZ));
    }

    if (joint.enableMotor && fixedRotation === false) {
        const cdot = f32(vec3.dot(vec3.sub(wB, wA), joint.rotationAxisZ) - joint.motorSpeed);

        let deltaImpulse = f32(-joint.axialMass * cdot);
        let newImpulse = f32(joint.motorImpulse + deltaImpulse);
        const maxImpulse = f32(joint.maxMotorTorque * context.h);
        newImpulse = clampf(newImpulse, -maxImpulse, maxImpulse);
        deltaImpulse = f32(newImpulse - joint.motorImpulse);
        joint.motorImpulse = newImpulse;

        wA = vec3.mulSub(wA, deltaImpulse, mat3.mulV(iA, joint.rotationAxisZ));
        wB = vec3.mulAdd(wB, deltaImpulse, mat3.mulV(iB, joint.rotationAxisZ));
    }

    if (joint.enableLimit && fixedRotation === false) {
        const angle = quat.getTwistAngle(relQ);
        const axis = joint.rotationAxisZ;

        // Lower limit
        {
            const c = f32(angle - joint.lowerAngle);
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

            const cdot = vec3.dot(vec3.sub(wB, wA), axis);
            const oldImpulse = joint.lowerImpulse;
            let deltaImpulse = f32(
                f32(f32(f32(-massScale * joint.axialMass) * f32(cdot + bias))) -
                    f32(impulseScale * oldImpulse),
            );
            joint.lowerImpulse = maxf(f32(oldImpulse + deltaImpulse), 0);
            deltaImpulse = f32(joint.lowerImpulse - oldImpulse);

            wA = vec3.mulSub(wA, deltaImpulse, mat3.mulV(iA, axis));
            wB = vec3.mulAdd(wB, deltaImpulse, mat3.mulV(iB, axis));
        }

        // Upper limit
        {
            const c = f32(joint.upperAngle - angle);
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
            const cdot = vec3.dot(vec3.sub(wA, wB), axis);
            const oldImpulse = joint.upperImpulse;
            let deltaImpulse = f32(
                f32(f32(f32(-massScale * joint.axialMass) * f32(cdot + bias))) -
                    f32(impulseScale * oldImpulse),
            );
            joint.upperImpulse = maxf(f32(oldImpulse + deltaImpulse), 0);
            deltaImpulse = f32(joint.upperImpulse - oldImpulse);

            // sign flipped on applied impulse
            wA = vec3.mulAdd(wA, deltaImpulse, mat3.mulV(iA, axis));
            wB = vec3.mulSub(wB, deltaImpulse, mat3.mulV(iB, axis));
        }
    }

    // Collinearity constraint
    if (fixedRotation === false) {
        let bias: Vec2 = { x: 0, y: 0 };
        let massScale = f32(1);
        let impulseScale = 0;

        if (useBias) {
            const c: Vec2 = { x: relQ.v.x, y: relQ.v.y };
            bias = vec2.scale(sim.constraintSoftness.biasRate, c);
            massScale = sim.constraintSoftness.massScale;
            impulseScale = sim.constraintSoftness.impulseScale;
        }

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
        joint.perpAxisX = perpAxisX;
        joint.perpAxisY = perpAxisY;

        const invInertiaSum = mat3.add(iA, iB);
        const kxx = vec3.dot(perpAxisX, mat3.mulV(invInertiaSum, perpAxisX));
        const kyy = vec3.dot(perpAxisY, mat3.mulV(invInertiaSum, perpAxisY));
        const kxy = vec3.dot(perpAxisX, mat3.mulV(invInertiaSum, perpAxisY));

        const k: Mat2 = { cx: { x: kxx, y: kxy }, cy: { x: kxy, y: kyy } };

        const wRel = vec3.sub(wB, wA);
        const cdot: Vec2 = { x: vec3.dot(wRel, perpAxisX), y: vec3.dot(wRel, perpAxisY) };
        const oldImpulse = joint.perpImpulse;
        const sol = mat2.solve(k, vec2.add(cdot, bias));
        const deltaImpulse = vec2.sub(
            vec2.scale(-massScale, sol),
            vec2.scale(impulseScale, oldImpulse),
        );
        joint.perpImpulse = vec2.add(joint.perpImpulse, deltaImpulse);

        const angularImpulse = vec3.add(
            vec3.scale(deltaImpulse.x, perpAxisX),
            vec3.scale(deltaImpulse.y, perpAxisY),
        );
        wA = vec3.sub(wA, mat3.mulV(iA, angularImpulse));
        wB = vec3.add(wB, mat3.mulV(iB, angularImpulse));
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
            const separation = vec3.add(
                vec3.add(vec3.sub(dcB, dcA), vec3.sub(rB, rA)),
                joint.deltaCenter,
            );
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

/** The reaction force this joint applies (b3GetRevoluteJointForce). */
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
