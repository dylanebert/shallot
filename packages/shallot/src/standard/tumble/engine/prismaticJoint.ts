// The prismatic (slider) joint — Box3D's prismatic_joint.c (Erin Catto, MIT). Constrains two bodies
// to slide along body A's local x-axis: a 2-DOF point-to-line linear constraint, a full rotation
// constraint (no relative rotation), plus optional axial spring, motor, and translation limits. Every
// op is fround-wrapped, mirroring the C scalar expression tree exactly. See the README.

import { NULL_INDEX } from "./array";
import {
    BodyFlags,
    getBodySim,
    getBodyState,
    getBodyTransformQuick,
    identityBodyState,
} from "./body";
import type { StepContext } from "./contactsolver";
import { SetType } from "./core";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";
import {
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
    vec2,
    vec3,
} from "./math";
import { makeSoft, type Softness } from "./softness";
import type { WorldState } from "./world";

/** Prismatic joint payload (b3PrismaticJoint). Impulses persist across steps for warm starting. */
export type PrismaticJoint = {
    perpImpulse: Vec2;
    angularImpulse: Vec3;
    springImpulse: number;
    motorImpulse: number;
    lowerImpulse: number;
    upperImpulse: number;
    hertz: number;
    dampingRatio: number;
    maxMotorForce: number;
    motorSpeed: number;
    targetTranslation: number;
    lowerTranslation: number;
    upperTranslation: number;
    indexA: number;
    indexB: number;
    frameA: Transform;
    frameB: Transform;
    jointAxis: Vec3;
    perpAxisY: Vec3;
    perpAxisZ: Vec3;
    deltaCenter: Vec3;
    rotationMass: Mat3;
    springSoftness: Softness;
    enableSpring: boolean;
    enableLimit: boolean;
    enableMotor: boolean;
};

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

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});
const zeroMat3 = (): Mat3 => ({
    cx: { x: 0, y: 0, z: 0 },
    cy: { x: 0, y: 0, z: 0 },
    cz: { x: 0, y: 0, z: 0 },
});

/** Create a prismatic joint (b3CreatePrismaticJoint). @returns the joint handle + sim. */
export function createPrismaticJoint(
    world: WorldState,
    def: PrismaticJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Prismatic);
    const data: PrismaticJoint = {
        perpImpulse: { x: 0, y: 0 },
        angularImpulse: { x: 0, y: 0, z: 0 },
        springImpulse: 0,
        motorImpulse: 0,
        lowerImpulse: 0,
        upperImpulse: 0,
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        maxMotorForce: def.maxMotorForce,
        motorSpeed: def.motorSpeed,
        targetTranslation: def.targetTranslation,
        lowerTranslation: def.lowerTranslation,
        upperTranslation: def.upperTranslation,
        indexA: 0,
        indexB: 0,
        frameA: identityTransform(),
        frameB: identityTransform(),
        jointAxis: { x: 0, y: 0, z: 0 },
        perpAxisY: { x: 0, y: 0, z: 0 },
        perpAxisZ: { x: 0, y: 0, z: 0 },
        deltaCenter: { x: 0, y: 0, z: 0 },
        rotationMass: zeroMat3(),
        springSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        enableSpring: def.enableSpring,
        enableLimit: def.enableLimit,
        enableMotor: def.enableMotor,
    };
    pair.sim.data = data;
    return pair;
}

export function preparePrismaticJoint(sim: JointSim, context: StepContext): void {
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

    const joint = sim.data as PrismaticJoint;
    joint.indexA = bodyA.setIndex === SetType.Awake ? bodyA.localIndex : NULL_INDEX;
    joint.indexB = bodyB.setIndex === SetType.Awake ? bodyB.localIndex : NULL_INDEX;

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
    joint.rotationMass = mat3.invert(invInertiaSum);

    // Initial joint axes in world space.
    const matrixA = mat3.fromQuat(joint.frameA.q);
    joint.jointAxis = matrixA.cx;
    joint.perpAxisY = matrixA.cy;
    joint.perpAxisZ = matrixA.cz;

    joint.springSoftness = makeSoft(joint.hertz, joint.dampingRatio, context.h);

    if (context.enableWarmStarting === false) {
        joint.perpImpulse = { x: 0, y: 0 };
        joint.angularImpulse = { x: 0, y: 0, z: 0 };
        joint.motorImpulse = 0;
        joint.springImpulse = 0;
        joint.lowerImpulse = 0;
        joint.upperImpulse = 0;
    }
}

export function warmStartPrismaticJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as PrismaticJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);
    const d = vec3.add(
        vec3.add(vec3.sub(stateB.deltaPosition, stateA.deltaPosition), joint.deltaCenter),
        vec3.sub(rB, rA),
    );
    const jointAxis = quat.rotate(stateA.deltaRotation, joint.jointAxis);
    const sAx = vec3.cross(vec3.add(rA, d), jointAxis);
    const sBx = vec3.cross(rB, jointAxis);

    const perpY = quat.rotate(stateA.deltaRotation, joint.perpAxisY);
    const perpZ = quat.rotate(stateA.deltaRotation, joint.perpAxisZ);
    const sAy = vec3.cross(vec3.add(rA, d), perpY);
    const sBy = vec3.cross(rB, perpY);
    const sAz = vec3.cross(vec3.add(rA, d), perpZ);
    const sBz = vec3.cross(rB, perpZ);

    const axialImpulse = f32(
        f32(f32(joint.springImpulse + joint.motorImpulse) + joint.lowerImpulse) -
            joint.upperImpulse,
    );
    const perpImpulse = joint.perpImpulse;

    const P = vec3.blend3(axialImpulse, jointAxis, perpImpulse.x, perpY, perpImpulse.y, perpZ);
    const La = vec3.add(
        vec3.blend3(axialImpulse, sAx, perpImpulse.x, sAy, perpImpulse.y, sAz),
        joint.angularImpulse,
    );
    const Lb = vec3.add(
        vec3.blend3(axialImpulse, sBx, perpImpulse.x, sBy, perpImpulse.y, sBz),
        joint.angularImpulse,
    );

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;
    vA = vec3.mulSub(vA, mA, P);
    wA = vec3.sub(wA, mat3.mulV(iA, La));
    vB = vec3.mulAdd(vB, mB, P);
    wB = vec3.add(wB, mat3.mulV(iB, Lb));

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.linearVelocity = vA;
        stateA.angularVelocity = wA;
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.linearVelocity = vB;
        stateB.angularVelocity = wB;
    }
}

export function solvePrismaticJoint(sim: JointSim, context: StepContext, useBias: boolean): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as PrismaticJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const fixedRotation = sim.fixedRotation;
    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    const dcA = stateA.deltaPosition;
    const dcB = stateB.deltaPosition;
    const d = vec3.add(vec3.add(vec3.sub(dcB, dcA), joint.deltaCenter), vec3.sub(rB, rA));

    const jointAxis = quat.rotate(stateA.deltaRotation, joint.jointAxis);
    const sAx = vec3.cross(vec3.add(rA, d), jointAxis);
    const sBx = vec3.cross(rB, jointAxis);
    const jointTranslation = vec3.dot(d, jointAxis);
    const targetTranslation = joint.targetTranslation;

    // The axial effective mass must be fresh to avoid divergence when the joint is stressed.
    const ka = f32(
        f32(
            f32(f32(mA + mB) + vec3.dot(sAx, mat3.mulV(iA, sAx))) +
                vec3.dot(sBx, mat3.mulV(iB, sBx)),
        ),
    );
    const axialMass = ka > 0 ? f32(1 / ka) : 0;

    // Solve spring
    if (joint.enableSpring && fixedRotation === false) {
        const c = f32(jointTranslation - targetTranslation);

        const bias = f32(joint.springSoftness.biasRate * c);
        const massScale = joint.springSoftness.massScale;
        const impulseScale = joint.springSoftness.impulseScale;

        const vRel = vec3.sub(
            vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vA),
            vec3.cross(wA, vec3.add(rA, d)),
        );
        const cdot = vec3.dot(vRel, jointAxis);
        const deltaImpulse = f32(
            f32(f32(f32(-massScale * axialMass) * f32(cdot + bias))) -
                f32(impulseScale * joint.springImpulse),
        );
        joint.springImpulse = f32(joint.springImpulse + deltaImpulse);

        const P = vec3.scale(deltaImpulse, jointAxis);
        const La = vec3.scale(deltaImpulse, sAx);
        const Lb = vec3.scale(deltaImpulse, sBx);

        vA = vec3.mulSub(vA, mA, P);
        wA = vec3.sub(wA, mat3.mulV(iA, La));
        vB = vec3.mulAdd(vB, mB, P);
        wB = vec3.add(wB, mat3.mulV(iB, Lb));
    }

    if (joint.enableMotor && fixedRotation === false) {
        const vRel = vec3.sub(
            vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vA),
            vec3.cross(wA, vec3.add(rA, d)),
        );
        const cdot = f32(vec3.dot(vRel, jointAxis) - joint.motorSpeed);

        let deltaImpulse = f32(-axialMass * cdot);
        let newImpulse = f32(joint.motorImpulse + deltaImpulse);
        const maxImpulse = f32(joint.maxMotorForce * context.h);
        newImpulse = clampf(newImpulse, -maxImpulse, maxImpulse);
        deltaImpulse = f32(newImpulse - joint.motorImpulse);
        joint.motorImpulse = newImpulse;

        const P = vec3.scale(deltaImpulse, jointAxis);
        const La = vec3.scale(deltaImpulse, sAx);
        const Lb = vec3.scale(deltaImpulse, sBx);

        vA = vec3.mulSub(vA, mA, P);
        wA = vec3.sub(wA, mat3.mulV(iA, La));
        vB = vec3.mulAdd(vB, mB, P);
        wB = vec3.add(wB, mat3.mulV(iB, Lb));
    }

    if (joint.enableLimit && fixedRotation === false) {
        const speculativeDistance = f32(
            f32(0.25) * f32(joint.upperTranslation - joint.lowerTranslation),
        );

        // Lower limit
        {
            const c = f32(jointTranslation - joint.lowerTranslation);
            if (c < speculativeDistance) {
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

                const vRel = vec3.sub(
                    vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vA),
                    vec3.cross(wA, vec3.add(rA, d)),
                );
                const cdot = vec3.dot(vRel, jointAxis);
                const oldImpulse = joint.lowerImpulse;
                let deltaImpulse = f32(
                    f32(f32(f32(-massScale * axialMass) * f32(cdot + bias))) -
                        f32(impulseScale * oldImpulse),
                );
                joint.lowerImpulse = maxf(f32(oldImpulse + deltaImpulse), 0);
                deltaImpulse = f32(joint.lowerImpulse - oldImpulse);

                const P = vec3.scale(deltaImpulse, jointAxis);
                const La = vec3.scale(deltaImpulse, sAx);
                const Lb = vec3.scale(deltaImpulse, sBx);

                vA = vec3.mulSub(vA, mA, P);
                wA = vec3.sub(wA, mat3.mulV(iA, La));
                vB = vec3.mulAdd(vB, mB, P);
                wB = vec3.add(wB, mat3.mulV(iB, Lb));
            } else {
                joint.lowerImpulse = 0;
            }
        }

        // Upper limit
        {
            const c = f32(joint.upperTranslation - jointTranslation);
            if (c < speculativeDistance) {
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
                const vRel = vec3.sub(
                    vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vA),
                    vec3.cross(wA, vec3.add(rA, d)),
                );
                const cdot = f32(-vec3.dot(vRel, jointAxis));
                const oldImpulse = joint.upperImpulse;
                const deltaImpulse = f32(
                    f32(f32(f32(-massScale * axialMass) * f32(cdot + bias))) -
                        f32(impulseScale * oldImpulse),
                );
                joint.upperImpulse = maxf(f32(oldImpulse + deltaImpulse), 0);

                // sign flipped on applied impulse
                const negDeltaImpulse = f32(oldImpulse - joint.upperImpulse);
                const P = vec3.scale(negDeltaImpulse, jointAxis);
                const La = vec3.scale(negDeltaImpulse, sAx);
                const Lb = vec3.scale(negDeltaImpulse, sBx);

                vA = vec3.mulSub(vA, mA, P);
                wA = vec3.sub(wA, mat3.mulV(iA, La));
                vB = vec3.mulAdd(vB, mB, P);
                wB = vec3.add(wB, mat3.mulV(iB, Lb));
            } else {
                joint.upperImpulse = 0;
            }
        }
    }

    // Rotation constraint
    if (fixedRotation === false) {
        let bias: Vec3 = { x: 0, y: 0, z: 0 };
        let massScale = f32(1);
        let impulseScale = 0;

        if (useBias) {
            const quatA = quat.mul(stateA.deltaRotation, joint.frameA.q);
            const quatB = quat.mul(stateB.deltaRotation, joint.frameB.q);

            const relQ = quat.invMul(quatA, quatB);
            const targetQuat = quat.identity();
            const deltaRotation = quat.deltaToRotation(relQ, targetQuat);
            const c = vec3.neg(quat.rotate(quatA, deltaRotation));

            bias = vec3.scale(sim.constraintSoftness.biasRate, c);
            massScale = sim.constraintSoftness.massScale;
            impulseScale = sim.constraintSoftness.impulseScale;
        }

        const cdot = vec3.sub(wB, wA);
        const impulse = vec3.sub(
            vec3.scale(-massScale, mat3.mulV(joint.rotationMass, vec3.add(cdot, bias))),
            vec3.scale(impulseScale, joint.angularImpulse),
        );
        joint.angularImpulse = vec3.add(joint.angularImpulse, impulse);

        wA = vec3.sub(wA, mat3.mulV(iA, impulse));
        wB = vec3.add(wB, mat3.mulV(iB, impulse));
    }

    // Solve point-to-line constraint
    {
        const perpY = quat.rotate(stateA.deltaRotation, joint.perpAxisY);
        const perpZ = quat.rotate(stateA.deltaRotation, joint.perpAxisZ);

        let bias: Vec2 = { x: 0, y: 0 };
        let massScale = f32(1);
        let impulseScale = 0;
        if (useBias) {
            const c: Vec2 = { x: vec3.dot(perpY, d), y: vec3.dot(perpZ, d) };
            bias = vec2.scale(sim.constraintSoftness.biasRate, c);
            massScale = sim.constraintSoftness.massScale;
            impulseScale = sim.constraintSoftness.impulseScale;
        }

        const vRel = vec3.sub(
            vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vA),
            vec3.cross(wA, vec3.add(rA, d)),
        );
        const cdot: Vec2 = { x: vec3.dot(perpY, vRel), y: vec3.dot(perpZ, vRel) };

        const sAy = vec3.cross(vec3.add(rA, d), perpY);
        const sBy = vec3.cross(rB, perpY);
        const sAz = vec3.cross(vec3.add(rA, d), perpZ);
        const sBz = vec3.cross(rB, perpZ);

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

        const K: Mat2 = { cx: { x: kyy, y: kyz }, cy: { x: kyz, y: kzz } };

        const oldImpulse = joint.perpImpulse;
        const sol = mat2.solve(K, vec2.add(cdot, bias));
        const deltaImpulse = vec2.sub(
            vec2.scale(-massScale, sol),
            vec2.scale(impulseScale, oldImpulse),
        );
        joint.perpImpulse = vec2.add(oldImpulse, deltaImpulse);

        const P = vec3.blend2(deltaImpulse.x, perpY, deltaImpulse.y, perpZ);

        vA = vec3.mulSub(vA, mA, P);
        wA = vec3.sub(wA, mat3.mulV(iA, vec3.blend2(deltaImpulse.x, sAy, deltaImpulse.y, sAz)));
        vB = vec3.mulAdd(vB, mB, P);
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

/** The reaction force this joint applies (b3GetPrismaticJointForce). */
export function getPrismaticJointForce(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as PrismaticJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);

    // impulse in joint space
    const impulse: Vec3 = {
        x: joint.perpImpulse.x,
        y: joint.perpImpulse.y,
        z: f32(
            f32(f32(joint.motorImpulse + joint.lowerImpulse) + joint.upperImpulse) +
                joint.springImpulse,
        ),
    };

    let force = vec3.scale(world.invH, impulse);
    force = quat.rotate(sim.localFrameA.q, force);
    force = quat.rotate(transformA.q, force);
    return force;
}

/** The reaction torque this joint applies (b3GetPrismaticJointTorque). */
export function getPrismaticJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as PrismaticJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);

    let torque = vec3.scale(world.invH, joint.angularImpulse);
    torque = quat.rotate(sim.localFrameA.q, torque);
    torque = quat.rotate(transformA.q, torque);
    return torque;
}

/** The current translation along the joint axis (b3PrismaticJoint_GetTranslation). */
export function prismaticJointTranslation(world: WorldState, sim: JointSim): number {
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);

    let jointAxis = quat.rotate(sim.localFrameA.q, vec3.axisX());
    jointAxis = quat.rotate(transformA.q, jointAxis);

    const anchorA = quat.rotate(transformA.q, sim.localFrameA.p);
    const anchorB = quat.rotate(transformB.q, sim.localFrameB.p);
    const d = vec3.add(vec3.sub(transformB.p, transformA.p), vec3.sub(anchorB, anchorA));
    return vec3.dot(d, jointAxis);
}

/** The current translation speed along the joint axis (b3PrismaticJoint_GetSpeed). */
export function prismaticJointSpeed(world: WorldState, sim: JointSim): number {
    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];
    const bodySimA = getBodySim(world, bodyA);
    const bodySimB = getBodySim(world, bodyB);
    const stateA = getBodyState(world, bodyA);
    const stateB = getBodyState(world, bodyB);

    const qA = bodySimA.transform.q;
    const qB = bodySimB.transform.q;

    const axisA = quat.rotate(qA, quat.rotate(sim.localFrameA.q, vec3.axisX()));
    const rA = quat.rotate(qA, vec3.sub(sim.localFrameA.p, bodySimA.localCenter));
    const rB = quat.rotate(qB, vec3.sub(sim.localFrameB.p, bodySimB.localCenter));

    // Difference the centers directly; positions are f32 in the single-precision build.
    const d = vec3.add(vec3.sub(bodySimB.center, bodySimA.center), vec3.sub(rB, rA));

    const zero: Vec3 = { x: 0, y: 0, z: 0 };
    const vA = stateA ? stateA.linearVelocity : zero;
    const vB = stateB ? stateB.linearVelocity : zero;
    const wA = stateA ? stateA.angularVelocity : zero;
    const wB = stateB ? stateB.angularVelocity : zero;

    const vRel = vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vec3.add(vA, vec3.cross(wA, rA)));

    // The axis moves with body A, so account for its rotation.
    return f32(vec3.dot(d, vec3.cross(wA, axisA)) + vec3.dot(axisA, vRel));
}
