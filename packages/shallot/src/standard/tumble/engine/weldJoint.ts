// The weld joint — Box3D's weld_joint.c (Erin Catto, MIT). Rigidly fixes two bodies: an angular
// constraint (relative rotation held at the reference) plus a point-to-point linear constraint, each
// optionally softened by a spring (linearHertz / angularHertz). Every op is fround-wrapped, mirroring
// the C scalar expression tree exactly. See the README.

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

/** Weld joint payload (b3WeldJoint). Impulses persist across steps for warm starting. */
export type WeldJoint = {
    linearImpulse: Vec3;
    angularImpulse: Vec3;
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
    indexA: number;
    indexB: number;
    frameA: Transform;
    frameB: Transform;
    deltaCenter: Vec3;
    angularMass: Mat3;
    linearSpring: Softness;
    angularSpring: Softness;
};

/** Weld joint definition (b3WeldJointDef), body handles resolved to a base JointDef. */
export type WeldJointDef = {
    base: JointDef;
    linearHertz: number;
    linearDampingRatio: number;
    angularHertz: number;
    angularDampingRatio: number;
};

/** @returns the ported weld joint definition defaults (b3DefaultWeldJointDef). */
export function defaultWeldJointDef(base: JointDef): WeldJointDef {
    return {
        base,
        linearHertz: 0,
        linearDampingRatio: 0,
        angularHertz: 0,
        angularDampingRatio: 0,
    };
}

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});

const zeroSoftness = (): Softness => ({ biasRate: 0, massScale: 0, impulseScale: 0 });

/** Create a weld joint (b3CreateWeldJoint). @returns the joint handle + sim. */
export function createWeldJoint(
    world: WorldState,
    def: WeldJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Weld);
    const data: WeldJoint = {
        linearImpulse: { x: 0, y: 0, z: 0 },
        angularImpulse: { x: 0, y: 0, z: 0 },
        linearHertz: def.linearHertz,
        linearDampingRatio: def.linearDampingRatio,
        angularHertz: def.angularHertz,
        angularDampingRatio: def.angularDampingRatio,
        indexA: 0,
        indexB: 0,
        frameA: identityTransform(),
        frameB: identityTransform(),
        deltaCenter: { x: 0, y: 0, z: 0 },
        angularMass: {
            cx: { x: 0, y: 0, z: 0 },
            cy: { x: 0, y: 0, z: 0 },
            cz: { x: 0, y: 0, z: 0 },
        },
        linearSpring: zeroSoftness(),
        angularSpring: zeroSoftness(),
    };
    pair.sim.data = data;
    return pair;
}

export function prepareWeldJoint(sim: JointSim, context: StepContext): void {
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

    const joint = sim.data as WeldJoint;
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
    joint.angularMass = mat3.invert(invInertiaSum);

    joint.linearSpring =
        joint.linearHertz === 0
            ? sim.constraintSoftness
            : makeSoft(joint.linearHertz, joint.linearDampingRatio, context.h);
    joint.angularSpring =
        joint.angularHertz === 0
            ? sim.constraintSoftness
            : makeSoft(joint.angularHertz, joint.angularDampingRatio, context.h);

    if (context.enableWarmStarting === false) {
        joint.linearImpulse = { x: 0, y: 0, z: 0 };
        joint.angularImpulse = { x: 0, y: 0, z: 0 };
    }
}

export function warmStartWeldJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as WeldJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
    const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

    vA = vec3.mulSub(vA, mA, joint.linearImpulse);
    wA = vec3.sub(
        wA,
        mat3.mulV(iA, vec3.add(vec3.cross(rA, joint.linearImpulse), joint.angularImpulse)),
    );

    vB = vec3.mulAdd(vB, mB, joint.linearImpulse);
    wB = vec3.add(
        wB,
        mat3.mulV(iB, vec3.add(vec3.cross(rB, joint.linearImpulse), joint.angularImpulse)),
    );

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.linearVelocity = vA;
        stateA.angularVelocity = wA;
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.linearVelocity = vB;
        stateB.angularVelocity = wB;
    }
}

export function solveWeldJoint(sim: JointSim, context: StepContext, useBias: boolean): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as WeldJoint;
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

    // angular constraint
    if (fixedRotation === false) {
        let bias: Vec3 = { x: 0, y: 0, z: 0 };
        let massScale = f32(1);
        let impulseScale = 0;
        if (useBias || joint.angularHertz > 0) {
            const targetQuat: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
            const deltaRotation = quat.deltaToRotation(relQ, targetQuat);
            const c = vec3.neg(quat.rotate(quatA, deltaRotation));

            bias = vec3.scale(joint.angularSpring.biasRate, c);
            massScale = joint.angularSpring.massScale;
            impulseScale = joint.angularSpring.impulseScale;
        }

        const cdot = vec3.sub(wB, wA);
        const impulse = vec3.mulSub(
            vec3.scale(-massScale, mat3.mulV(joint.angularMass, vec3.add(cdot, bias))),
            impulseScale,
            joint.angularImpulse,
        );
        joint.angularImpulse = vec3.add(joint.angularImpulse, impulse);

        wA = vec3.sub(wA, mat3.mulV(iA, impulse));
        wB = vec3.add(wB, mat3.mulV(iB, impulse));
    }

    // linear constraint
    {
        const rA = quat.rotate(stateA.deltaRotation, joint.frameA.p);
        const rB = quat.rotate(stateB.deltaRotation, joint.frameB.p);

        const cdot = vec3.sub(vec3.add(vB, vec3.cross(wB, rB)), vec3.add(vA, vec3.cross(wA, rA)));

        let bias: Vec3 = { x: 0, y: 0, z: 0 };
        let massScale = f32(1);
        let impulseScale = 0;
        if (useBias || joint.linearHertz > 0) {
            const dcA = stateA.deltaPosition;
            const dcB = stateB.deltaPosition;
            const separation = vec3.add(
                vec3.add(vec3.sub(dcB, dcA), vec3.sub(rB, rA)),
                joint.deltaCenter,
            );
            bias = vec3.scale(joint.linearSpring.biasRate, separation);
            massScale = joint.linearSpring.massScale;
            impulseScale = joint.linearSpring.impulseScale;
        }

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

        const impulse = vec3.mulSub(vec3.scale(-massScale, b), impulseScale, joint.linearImpulse);
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

/** The reaction force this joint applies (b3GetWeldJointForce). */
export function getWeldJointForce(world: WorldState, sim: JointSim): Vec3 {
    return vec3.scale(world.invH, (sim.data as WeldJoint).linearImpulse);
}

/** The reaction torque this joint applies (b3GetWeldJointTorque). */
export function getWeldJointTorque(world: WorldState, sim: JointSim): Vec3 {
    return vec3.scale(world.invH, (sim.data as WeldJoint).angularImpulse);
}
