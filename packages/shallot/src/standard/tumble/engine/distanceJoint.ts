// The distance joint — Box3D's distance_joint.c (Erin Catto, MIT). A 1-D constraint along the axis
// between two anchor points: rigid at a fixed length, or (when the spring is enabled) a soft spring
// with optional min/max limits and a motor. Every op is fround-wrapped, mirroring the C scalar
// expression tree exactly. See the README.

import { NULL_INDEX } from "./array";
import { BodyFlags, getBodyTransformQuick, identityBodyState } from "./body";
import type { StepContext } from "./contactsolver";
import { HUGE, LINEAR_SLOP, SetType } from "./core";
import { createJoint, type Joint, type JointDef, type JointSim, JointType } from "./joint";
import { clampf, FLT_MAX, f32, mat3, maxf, quat, type Vec3, vec3 } from "./math";
import { makeSoft, type Softness } from "./softness";
import type { WorldState } from "./world";

/** Distance joint payload (b3DistanceJoint). Impulses persist across steps for warm starting. */
export type DistanceJoint = {
    length: number;
    hertz: number;
    dampingRatio: number;
    lowerSpringForce: number;
    upperSpringForce: number;
    minLength: number;
    maxLength: number;
    maxMotorForce: number;
    motorSpeed: number;
    impulse: number;
    lowerImpulse: number;
    upperImpulse: number;
    motorImpulse: number;
    indexA: number;
    indexB: number;
    anchorA: Vec3;
    anchorB: Vec3;
    deltaCenter: Vec3;
    distanceSoftness: Softness;
    axialMass: number;
    enableSpring: boolean;
    enableLimit: boolean;
    enableMotor: boolean;
};

/** Distance joint definition (b3DistanceJointDef), body handles resolved to a base JointDef. */
export type DistanceJointDef = {
    base: JointDef;
    length: number;
    enableSpring: boolean;
    hertz: number;
    dampingRatio: number;
    lowerSpringForce: number;
    upperSpringForce: number;
    enableLimit: boolean;
    minLength: number;
    maxLength: number;
    enableMotor: boolean;
    maxMotorForce: number;
    motorSpeed: number;
};

/** @returns the ported distance joint definition defaults (b3DefaultDistanceJointDef). */
export function defaultDistanceJointDef(base: JointDef): DistanceJointDef {
    return {
        base,
        length: 1,
        enableSpring: false,
        hertz: 0,
        dampingRatio: 0,
        lowerSpringForce: -FLT_MAX,
        upperSpringForce: FLT_MAX,
        enableLimit: false,
        minLength: 0,
        maxLength: HUGE,
        enableMotor: false,
        maxMotorForce: 0,
        motorSpeed: 0,
    };
}

/** Create a distance joint (b3CreateDistanceJoint). @returns the joint handle + sim. */
export function createDistanceJoint(
    world: WorldState,
    def: DistanceJointDef,
): { joint: Joint; sim: JointSim } {
    const pair = createJoint(world, def.base, JointType.Distance);
    const data: DistanceJoint = {
        length: maxf(def.length, LINEAR_SLOP),
        hertz: def.hertz,
        dampingRatio: def.dampingRatio,
        lowerSpringForce: def.lowerSpringForce,
        upperSpringForce: def.upperSpringForce,
        minLength: maxf(def.minLength, LINEAR_SLOP),
        maxLength: maxf(def.minLength, def.maxLength),
        maxMotorForce: def.maxMotorForce,
        motorSpeed: def.motorSpeed,
        impulse: 0,
        lowerImpulse: 0,
        upperImpulse: 0,
        motorImpulse: 0,
        indexA: 0,
        indexB: 0,
        anchorA: { x: 0, y: 0, z: 0 },
        anchorB: { x: 0, y: 0, z: 0 },
        deltaCenter: { x: 0, y: 0, z: 0 },
        distanceSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        axialMass: 0,
        enableSpring: def.enableSpring,
        enableLimit: def.enableLimit,
        enableMotor: def.enableMotor,
    };
    pair.sim.data = data;
    return pair;
}

export function prepareDistanceJoint(sim: JointSim, context: StepContext): void {
    const world = context.world;
    const bodyA = world.bodies[sim.bodyIdA];
    const bodyB = world.bodies[sim.bodyIdB];

    const setA = world.solverSets[bodyA.setIndex];
    const setB = world.solverSets[bodyB.setIndex];
    const bodySimA = setA.bodySims[bodyA.localIndex];
    const bodySimB = setB.bodySims[bodyB.localIndex];

    const mA = bodySimA.invMass;
    const iA = bodySimA.invInertiaWorld;
    const mB = bodySimB.invMass;
    const iB = bodySimB.invInertiaWorld;

    sim.invMassA = mA;
    sim.invMassB = mB;
    sim.invIA = iA;
    sim.invIB = iB;

    const joint = sim.data as DistanceJoint;
    joint.indexA = bodyA.setIndex === SetType.Awake ? bodyA.localIndex : NULL_INDEX;
    joint.indexB = bodyB.setIndex === SetType.Awake ? bodyB.localIndex : NULL_INDEX;

    joint.anchorA = quat.rotate(
        bodySimA.transform.q,
        vec3.sub(sim.localFrameA.p, bodySimA.localCenter),
    );
    joint.anchorB = quat.rotate(
        bodySimB.transform.q,
        vec3.sub(sim.localFrameB.p, bodySimB.localCenter),
    );
    joint.deltaCenter = vec3.sub(bodySimB.center, bodySimA.center);

    const rA = joint.anchorA;
    const rB = joint.anchorB;
    const separation = vec3.add(vec3.sub(rB, rA), joint.deltaCenter);
    const axis = vec3.normalize(separation);

    const crA = vec3.cross(rA, axis);
    const crB = vec3.cross(rB, axis);
    const k = f32(
        f32(f32(mA + mB) + vec3.dot(crA, mat3.mulV(iA, crA))) + vec3.dot(crB, mat3.mulV(iB, crB)),
    );
    joint.axialMass = k > 0 ? f32(1 / k) : 0;

    joint.distanceSoftness = makeSoft(joint.hertz, joint.dampingRatio, context.h);

    if (context.enableWarmStarting === false) {
        joint.impulse = 0;
        joint.lowerImpulse = 0;
        joint.upperImpulse = 0;
        joint.motorImpulse = 0;
    }
}

export function warmStartDistanceJoint(sim: JointSim, context: StepContext): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as DistanceJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    const rA = quat.rotate(stateA.deltaRotation, joint.anchorA);
    const rB = quat.rotate(stateB.deltaRotation, joint.anchorB);

    const ds = vec3.add(vec3.sub(stateB.deltaPosition, stateA.deltaPosition), vec3.sub(rB, rA));
    const separation = vec3.add(joint.deltaCenter, ds);
    const axis = vec3.normalize(separation);

    const axialImpulse = f32(
        f32(f32(joint.impulse + joint.lowerImpulse) - joint.upperImpulse) + joint.motorImpulse,
    );
    const p = vec3.scale(axialImpulse, axis);

    if ((stateA.flags & BodyFlags.dynamicFlag) !== 0) {
        stateA.linearVelocity = vec3.mulSub(stateA.linearVelocity, mA, p);
        stateA.angularVelocity = vec3.sub(stateA.angularVelocity, mat3.mulV(iA, vec3.cross(rA, p)));
    }
    if ((stateB.flags & BodyFlags.dynamicFlag) !== 0) {
        stateB.linearVelocity = vec3.mulAdd(stateB.linearVelocity, mB, p);
        stateB.angularVelocity = vec3.add(stateB.angularVelocity, mat3.mulV(iB, vec3.cross(rB, p)));
    }
}

export function solveDistanceJoint(sim: JointSim, context: StepContext, useBias: boolean): void {
    const mA = sim.invMassA;
    const mB = sim.invMassB;
    const iA = sim.invIA;
    const iB = sim.invIB;

    const dummy = identityBodyState();
    const joint = sim.data as DistanceJoint;
    const stateA = joint.indexA === NULL_INDEX ? dummy : context.states[joint.indexA];
    const stateB = joint.indexB === NULL_INDEX ? dummy : context.states[joint.indexB];

    let vA = stateA.linearVelocity;
    let wA = stateA.angularVelocity;
    let vB = stateB.linearVelocity;
    let wB = stateB.angularVelocity;

    const rA = quat.rotate(stateA.deltaRotation, joint.anchorA);
    const rB = quat.rotate(stateB.deltaRotation, joint.anchorB);

    const ds = vec3.add(vec3.sub(stateB.deltaPosition, stateA.deltaPosition), vec3.sub(rB, rA));
    const separation = vec3.add(joint.deltaCenter, ds);

    const length = vec3.length(separation);
    const axis = vec3.normalize(separation);

    // relative velocity of the two anchors projected onto the axis.
    const relVel = (): number => {
        const vr = vec3.add(vec3.sub(vB, vA), vec3.sub(vec3.cross(wB, rB), vec3.cross(wA, rA)));
        return vec3.dot(axis, vr);
    };

    const applyP = (impulse: number): void => {
        const p = vec3.scale(impulse, axis);
        vA = vec3.mulSub(vA, mA, p);
        wA = vec3.sub(wA, mat3.mulV(iA, vec3.cross(rA, p)));
        vB = vec3.mulAdd(vB, mB, p);
        wB = vec3.add(wB, mat3.mulV(iB, vec3.cross(rB, p)));
    };

    if (joint.enableSpring && (joint.minLength < joint.maxLength || joint.enableLimit === false)) {
        // spring
        if (joint.hertz > 0) {
            const cdot = relVel();
            const c = f32(length - joint.length);
            const bias = f32(joint.distanceSoftness.biasRate * c);

            const m = f32(joint.distanceSoftness.massScale * joint.axialMass);
            const oldImpulse = joint.impulse;
            let impulse = f32(
                f32(f32(-m) * f32(cdot + bias)) -
                    f32(joint.distanceSoftness.impulseScale * oldImpulse),
            );
            const h = context.h;
            joint.impulse = clampf(
                f32(joint.impulse + impulse),
                f32(joint.lowerSpringForce * h),
                f32(joint.upperSpringForce * h),
            );
            impulse = f32(joint.impulse - oldImpulse);
            applyP(impulse);
        }

        if (joint.enableLimit) {
            // lower limit
            {
                const cdot = relVel();
                const c = f32(length - joint.minLength);
                let bias = 0;
                let massCoeff = f32(1);
                let impulseCoeff = 0;
                if (c > 0) {
                    bias = f32(c * context.invH);
                } else if (useBias) {
                    bias = f32(sim.constraintSoftness.biasRate * c);
                    massCoeff = sim.constraintSoftness.massScale;
                    impulseCoeff = sim.constraintSoftness.impulseScale;
                }

                let impulse = f32(
                    f32(f32(f32(-massCoeff * joint.axialMass) * f32(cdot + bias))) -
                        f32(impulseCoeff * joint.lowerImpulse),
                );
                const newImpulse = maxf(0, f32(joint.lowerImpulse + impulse));
                impulse = f32(newImpulse - joint.lowerImpulse);
                joint.lowerImpulse = newImpulse;
                applyP(impulse);
            }

            // upper limit (impulse sign flipped)
            {
                const vr = vec3.add(
                    vec3.sub(vA, vB),
                    vec3.sub(vec3.cross(wA, rA), vec3.cross(wB, rB)),
                );
                const cdot = vec3.dot(axis, vr);
                const c = f32(joint.maxLength - length);
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

                let impulse = f32(
                    f32(f32(f32(-massScale * joint.axialMass) * f32(cdot + bias))) -
                        f32(impulseScale * joint.upperImpulse),
                );
                const newImpulse = maxf(0, f32(joint.upperImpulse + impulse));
                impulse = f32(newImpulse - joint.upperImpulse);
                joint.upperImpulse = newImpulse;

                const p = vec3.scale(-impulse, axis);
                vA = vec3.mulSub(vA, mA, p);
                wA = vec3.sub(wA, mat3.mulV(iA, vec3.cross(rA, p)));
                vB = vec3.mulAdd(vB, mB, p);
                wB = vec3.add(wB, mat3.mulV(iB, vec3.cross(rB, p)));
            }
        }

        if (joint.enableMotor) {
            const cdot = relVel();
            let impulse = f32(joint.axialMass * f32(joint.motorSpeed - cdot));
            const oldImpulse = joint.motorImpulse;
            const maxImpulse = f32(context.h * joint.maxMotorForce);
            joint.motorImpulse = clampf(f32(joint.motorImpulse + impulse), -maxImpulse, maxImpulse);
            impulse = f32(joint.motorImpulse - oldImpulse);
            applyP(impulse);
        }
    } else {
        // rigid constraint
        const cdot = relVel();
        const c = f32(length - joint.length);

        let bias = 0;
        let massScale = f32(1);
        let impulseScale = 0;
        if (useBias) {
            bias = f32(sim.constraintSoftness.biasRate * c);
            massScale = sim.constraintSoftness.massScale;
            impulseScale = sim.constraintSoftness.impulseScale;
        }

        const impulse = f32(
            f32(f32(f32(-massScale * joint.axialMass) * f32(cdot + bias))) -
                f32(impulseScale * joint.impulse),
        );
        joint.impulse = f32(joint.impulse + impulse);
        applyP(impulse);
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

// --- Force accessor ---------------------------------------------------------------------------

/** The reaction force this joint applies (b3GetDistanceJointForce). */
export function getDistanceJointForce(world: WorldState, sim: JointSim): Vec3 {
    const joint = sim.data as DistanceJoint;
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);

    const pA = vec3.add(quat.rotate(transformA.q, sim.localFrameA.p), transformA.p);
    const pB = vec3.add(quat.rotate(transformB.q, sim.localFrameB.p), transformB.p);
    const d = vec3.sub(pB, pA);
    const axis = vec3.normalize(d);
    const force = f32(
        f32(
            f32(f32(joint.impulse + joint.lowerImpulse) - joint.upperImpulse) + joint.motorImpulse,
        ) * world.invH,
    );
    return vec3.scale(force, axis);
}

/** The current distance between the two anchor points (b3DistanceJoint_GetCurrentLength). */
export function distanceJointCurrentLength(world: WorldState, sim: JointSim): number {
    const transformA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
    const transformB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);
    const pA = vec3.add(quat.rotate(transformA.q, sim.localFrameA.p), transformA.p);
    const pB = vec3.add(quat.rotate(transformB.q, sim.localFrameB.p), transformB.p);
    const d = vec3.sub(pB, pA);
    return vec3.length(d);
}
