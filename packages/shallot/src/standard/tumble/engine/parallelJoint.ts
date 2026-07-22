// The parallel joint — Box3D's parallel_joint.c (Erin Catto, MIT). Holds two bodies' rotation frames
// collinear about their local z axes (a 2-DOF angular constraint), driven by a soft spring capped at
// maxTorque; no linear constraint. Solve takes no useBias (a pure velocity/soft constraint). Every op
// is fround-wrapped, mirroring the C scalar expression tree exactly. See the README.

import { NULL_INDEX } from "./array";
import { BodyFlags, identityBodyState } from "./body";
import type { StepContext } from "./contactsolver";
import { SetType } from "./core";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";
import {
    FLT_MAX,
    FLT_MIN,
    f32,
    type Mat2,
    mat2,
    mat3,
    type Quat,
    quat,
    type Vec2,
    type Vec3,
    vec2,
    vec3,
} from "./math";
import { makeSoft, type Softness } from "./softness";
import type { WorldState } from "./world";

/** Parallel joint payload (b3ParallelJoint). Impulse persists across steps for warm starting. */
export type ParallelJoint = {
    perpImpulse: Vec2;
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
    indexA: number;
    indexB: number;
    quatA: Quat;
    quatB: Quat;
    perpAxisX: Vec3;
    perpAxisY: Vec3;
    softness: Softness;
};

/** Parallel joint definition (b3ParallelJointDef), body handles resolved to a base JointDef. */
export type ParallelJointDef = {
    base: JointDef;
    hertz: number;
    dampingRatio: number;
    maxTorque: number;
};

/** @returns the ported parallel joint definition defaults (b3DefaultParallelJointDef). */
export function defaultParallelJointDef(base: JointDef): ParallelJointDef {
    return { base, hertz: 1, dampingRatio: 1, maxTorque: FLT_MAX };
}

const identityQuat = (): Quat => ({ v: { x: 0, y: 0, z: 0 }, s: 1 });

/** Create a parallel joint (b3CreateParallelJoint). @returns the joint handle + sim. */
export function createParallelJoint(
    world: WorldState,
    def: ParallelJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Parallel);
    const data: ParallelJoint = {
        perpImpulse: { x: 0, y: 0 },
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        maxTorque: def.maxTorque,
        indexA: 0,
        indexB: 0,
        quatA: identityQuat(),
        quatB: identityQuat(),
        perpAxisX: { x: 0, y: 0, z: 0 },
        perpAxisY: { x: 0, y: 0, z: 0 },
        softness: { biasRate: 0, massScale: 0, impulseScale: 0 },
    };
    pair.sim.data = data;
    return pair;
}

// The two perpendicular collinearity axes in world space, from the relative rotation (relQ) of the
// two joint frames. relQ = inv(quatA) * quatB; the axes are half the rotated imaginary parts.
function perpAxes(qA: Quat, relQ: Quat): { x: Vec3; y: Vec3 } {
    return {
        x: vec3.scale(
            f32(0.5),
            quat.rotate(
                qA,
                vec3.add(vec3.scale(relQ.s, vec3.axisX()), vec3.cross(relQ.v, vec3.axisX())),
            ),
        ),
        y: vec3.scale(
            f32(0.5),
            quat.rotate(
                qA,
                vec3.add(vec3.scale(relQ.s, vec3.axisY()), vec3.cross(relQ.v, vec3.axisY())),
            ),
        ),
    };
}

export function prepareParallelJoint(sim: JointSim, context: StepContext): void {
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

    const joint = sim.data as ParallelJoint;
    joint.indexA = bodyA.setIndex === SetType.Awake ? bodyA.localIndex : NULL_INDEX;
    joint.indexB = bodyB.setIndex === SetType.Awake ? bodyB.localIndex : NULL_INDEX;

    joint.quatA = quat.mul(bodySimA.transform.q, sim.localFrameA.q);
    joint.quatB = quat.mul(bodySimB.transform.q, sim.localFrameB.q);

    const relQ = quat.invMul(joint.quatA, joint.quatB);
    const axes = perpAxes(joint.quatA, relQ);
    joint.perpAxisX = axes.x;
    joint.perpAxisY = axes.y;

    joint.softness = makeSoft(joint.hertz, joint.dampingRatio, context.h);

    if (context.enableWarmStarting === false) {
        joint.perpImpulse = { x: 0, y: 0 };
    }
}

export function warmStartParallelJoint(sim: JointSim, context: StepContext): void {
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as ParallelJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let wA = stateA.angularVelocity;
    let wB = stateB.angularVelocity;

    const angularImpulse = vec3.blend2(
        joint.perpImpulse.x,
        joint.perpAxisX,
        joint.perpImpulse.y,
        joint.perpAxisY,
    );

    wA = vec3.sub(wA, mat3.mulV(iA, angularImpulse));
    wB = vec3.add(wB, mat3.mulV(iB, angularImpulse));

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.angularVelocity = wA;
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.angularVelocity = wB;
    }
}

export function solveParallelJoint(sim: JointSim, context: StepContext): void {
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as ParallelJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let wA = stateA.angularVelocity;
    let wB = stateB.angularVelocity;

    const fixedRotation = sim.fixedRotation;
    const quatA = quat.mul(stateA.deltaRotation, joint.quatA);
    let quatB = quat.mul(stateB.deltaRotation, joint.quatB);

    if (quat.dot(quatA, quatB) < 0) {
        quatB = quat.negate(quatB);
    }

    const relQ = quat.invMul(quatA, quatB);

    if (fixedRotation === false && joint.maxTorque > 0) {
        const c: Vec2 = { x: relQ.v.x, y: relQ.v.y };
        const bias = vec2.scale(joint.softness.biasRate, c);
        const massScale = joint.softness.massScale;
        const impulseScale = joint.softness.impulseScale;

        const axes = perpAxes(quatA, relQ);
        const perpAxisX = axes.x;
        const perpAxisY = axes.y;
        joint.perpAxisX = perpAxisX;
        joint.perpAxisY = perpAxisY;

        const invInertiaSum = mat3.add(iA, iB);
        const kxx = vec3.dot(perpAxisX, mat3.mulV(invInertiaSum, perpAxisX));
        const kyy = vec3.dot(perpAxisY, mat3.mulV(invInertiaSum, perpAxisY));
        const kxy = vec3.dot(perpAxisX, mat3.mulV(invInertiaSum, perpAxisY));
        const k: Mat2 = { cx: { x: kxx, y: kxy }, cy: { x: kxy, y: kyy } };

        const wRel = vec3.sub(wB, wA);
        const cdot: Vec2 = { x: vec3.dot(wRel, perpAxisX), y: vec3.dot(wRel, perpAxisY) };

        const maxImpulse = f32(context.h * joint.maxTorque);
        const oldImpulse = joint.perpImpulse;
        const sol = mat2.solve(k, vec2.add(cdot, bias));
        let deltaImpulse = vec2.sub(
            vec2.scale(-massScale, sol),
            vec2.scale(impulseScale, oldImpulse),
        );
        joint.perpImpulse = vec2.add(oldImpulse, deltaImpulse);
        if (vec2.lengthSquared(joint.perpImpulse) > f32(maxImpulse * maxImpulse)) {
            const s = f32(maxImpulse / vec2.length(joint.perpImpulse));
            joint.perpImpulse = vec2.scale(s, joint.perpImpulse);
        }

        deltaImpulse = vec2.sub(joint.perpImpulse, oldImpulse);

        const angularImpulse = vec3.blend2(deltaImpulse.x, perpAxisX, deltaImpulse.y, perpAxisY);
        wA = vec3.sub(wA, mat3.mulV(iA, angularImpulse));
        wB = vec3.add(wB, mat3.mulV(iB, angularImpulse));
    }

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.angularVelocity = wA;
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.angularVelocity = wB;
    }
}

// --- Force / torque accessors -----------------------------------------------------------------

/** The reaction torque this joint applies (b3GetParallelJointTorque). */
export function getParallelJointTorque(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as ParallelJoint;
    const relQ = quat.invMul(joint.quatA, joint.quatB);
    const axes = perpAxes(joint.quatA, relQ);
    joint.perpAxisX = axes.x;
    joint.perpAxisY = axes.y;

    const angularImpulse = vec3.blend2(
        joint.perpImpulse.x,
        joint.perpAxisX,
        joint.perpImpulse.y,
        joint.perpAxisY,
    );
    return vec3.scale(world.invH, angularImpulse);
}
