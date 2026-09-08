// Joints — the common machinery from Box3D's joint.c (Erin Catto, MIT). A joint constrains two
// bodies. The organizational handle (b3Joint) lives in world.joints and threads two doubly-linked
// edges through the attached bodies; the simulation payload (b3JointSim) lives in a solver set's
// jointSims column (the overflow graph color while awake). Per-type math lives in one file each
// (revoluteJoint.ts, …); this hub holds create/destroy, the prepare/warm-start/solve dispatch, the
// serial overflow loops, and the reaction/force/torque accessors.
//
// The port targets the force-overflow build: b3AssignJointColor always returns the overflow color,
// so joints join contacts in colors[OVERFLOW_INDEX].jointSims. Every arithmetic op is fround-wrapped
// in the per-type files; see the README.

import { NULL_INDEX } from "./array";
import { type Body, getBodyTransformQuick, wakeBody } from "./body";
import { bufferMove } from "./broadphase";
import { destroyContact } from "./contact";
import type { StepContext } from "./contactsolver";
import { OVERFLOW_INDEX, SetType } from "./core";
import {
    type DistanceJoint,
    getDistanceJointForce,
    prepareDistanceJoint,
    solveDistanceJoint,
    warmStartDistanceJoint,
} from "./distanceJoint";
import { createJointInGraph, removeJointFromGraph } from "./graph";
import { allocId, freeId } from "./ids";
import { linkJoint, unlinkJoint } from "./island";
import {
    absf,
    FLT_MAX,
    f32,
    type Mat3,
    maxf,
    maxInt,
    minf,
    quat,
    type Transform,
    transformWorldPoint,
    type Vec3,
    vec3,
} from "./math";
import {
    getMotorJointForce,
    getMotorJointTorque,
    type MotorJoint,
    prepareMotorJoint,
    solveMotorJoint,
    warmStartMotorJoint,
} from "./motorJoint";
import {
    getParallelJointTorque,
    type ParallelJoint,
    prepareParallelJoint,
    solveParallelJoint,
    warmStartParallelJoint,
} from "./parallelJoint";
import {
    getPrismaticJointForce,
    getPrismaticJointTorque,
    type PrismaticJoint,
    preparePrismaticJoint,
    solvePrismaticJoint,
    warmStartPrismaticJoint,
} from "./prismaticJoint";
import {
    getRevoluteJointForce,
    getRevoluteJointTorque,
    prepareRevoluteJoint,
    type RevoluteJoint,
    solveRevoluteJoint,
    warmStartRevoluteJoint,
} from "./revoluteJoint";
import { makeSoft, type Softness } from "./softness";
import { wakeSolverSet } from "./solverset";
import {
    getSphericalJointForce,
    getSphericalJointTorque,
    prepareSphericalJoint,
    type SphericalJoint,
    solveSphericalJoint,
    warmStartSphericalJoint,
} from "./sphericalJoint";
import { BodyType } from "./types";
import {
    getWeldJointForce,
    getWeldJointTorque,
    prepareWeldJoint,
    solveWeldJoint,
    type WeldJoint,
    warmStartWeldJoint,
} from "./weldJoint";
import {
    getWheelJointForce,
    getWheelJointTorque,
    prepareWheelJoint,
    solveWheelJoint,
    type WheelJoint,
    warmStartWheelJoint,
} from "./wheelJoint";
import type { WorldState } from "./world";

/** Joint kind (b3JointType). Numeric values mirror the C enum order (parallel = 0 … wheel = 8). */
export const JointType = {
    Parallel: 0,
    Distance: 1,
    Filter: 2,
    Motor: 3,
    Prismatic: 4,
    Revolute: 5,
    Spherical: 6,
    Weld: 7,
    Wheel: 8,
} as const;
export type JointType = (typeof JointType)[keyof typeof JointType];

/** The per-type payload union carried by a JointSim (filter carries none). */
export type JointPayload =
    | RevoluteJoint
    | WeldJoint
    | ParallelJoint
    | MotorJoint
    | DistanceJoint
    | PrismaticJoint
    | SphericalJoint
    | WheelJoint;

/** One end of a joint in a body's doubly-linked joint list (b3JointEdge). */
export type JointEdge = { bodyId: number; prevKey: number; nextKey: number };

/** The organizational joint handle stored in world.joints (b3Joint). */
export type Joint = {
    userData: unknown;
    setIndex: number;
    colorIndex: number;
    localIndex: number;
    edges: [JointEdge, JointEdge];
    jointId: number;
    islandId: number;
    islandIndex: number;
    drawScale: number;
    type: JointType;
    generation: number;
    collideConnected: boolean;
};

/** The joint simulation payload stored in a solver set / graph color (b3JointSim). */
export type JointSim = {
    jointId: number;
    bodyIdA: number;
    bodyIdB: number;
    type: JointType;
    localFrameA: Transform;
    localFrameB: Transform;
    invMassA: number;
    invMassB: number;
    invIA: Mat3;
    invIB: Mat3;
    constraintHertz: number;
    constraintDampingRatio: number;
    constraintSoftness: Softness;
    forceThreshold: number;
    torqueThreshold: number;
    fixedRotation: boolean;
    data: JointPayload;
};

/** The resolved base joint definition (b3JointDef, body handles already resolved to ids). */
export type JointDef = {
    bodyIdA: number;
    bodyIdB: number;
    localFrameA: Transform;
    localFrameB: Transform;
    forceThreshold: number;
    torqueThreshold: number;
    constraintHertz: number;
    constraintDampingRatio: number;
    drawScale: number;
    collideConnected: boolean;
    userData: unknown;
};

const identityTransform = (): Transform => ({
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
});

/** @returns the ported base joint definition defaults (b3DefaultJointDef). */
export function defaultJointDef(): JointDef {
    return {
        bodyIdA: NULL_INDEX,
        bodyIdB: NULL_INDEX,
        localFrameA: identityTransform(),
        localFrameB: identityTransform(),
        forceThreshold: FLT_MAX,
        torqueThreshold: FLT_MAX,
        constraintHertz: 60,
        constraintDampingRatio: 2,
        drawScale: 1,
        collideConnected: false,
        userData: null,
    };
}

/** @returns a fresh zeroed joint handle slot. */
export function emptyJoint(): Joint {
    return {
        userData: null,
        setIndex: NULL_INDEX,
        colorIndex: NULL_INDEX,
        localIndex: NULL_INDEX,
        edges: [
            { bodyId: NULL_INDEX, prevKey: NULL_INDEX, nextKey: NULL_INDEX },
            { bodyId: NULL_INDEX, prevKey: NULL_INDEX, nextKey: NULL_INDEX },
        ],
        jointId: NULL_INDEX,
        islandId: NULL_INDEX,
        islandIndex: NULL_INDEX,
        drawScale: 0,
        type: JointType.Parallel,
        generation: 0,
        collideConnected: false,
    };
}

/** @returns a fresh zeroed joint sim (mirrors the C memset before the type payload is filled). */
export function emptyJointSim(): JointSim {
    return {
        jointId: NULL_INDEX,
        bodyIdA: NULL_INDEX,
        bodyIdB: NULL_INDEX,
        type: JointType.Parallel,
        localFrameA: identityTransform(),
        localFrameB: identityTransform(),
        invMassA: 0,
        invMassB: 0,
        invIA: { cx: { x: 0, y: 0, z: 0 }, cy: { x: 0, y: 0, z: 0 }, cz: { x: 0, y: 0, z: 0 } },
        invIB: { cx: { x: 0, y: 0, z: 0 }, cy: { x: 0, y: 0, z: 0 }, cz: { x: 0, y: 0, z: 0 } },
        constraintHertz: 0,
        constraintDampingRatio: 0,
        constraintSoftness: { biasRate: 0, massScale: 0, impulseScale: 0 },
        forceThreshold: 0,
        torqueThreshold: 0,
        fixedRotation: false,
        // Placeholder; the per-type create overwrites this before the joint is used.
        data: null as unknown as JointPayload,
    };
}

/** @returns the joint's live sim, from the graph color when awake else its solver set (b3GetJointSim). */
export function getJointSim(world: WorldState, joint: Joint): JointSim {
    if (joint.setIndex === SetType.Awake) {
        return world.constraintGraph.colors[joint.colorIndex].jointSims[joint.localIndex];
    }
    return world.solverSets[joint.setIndex].jointSims[joint.localIndex];
}

/** Create the common joint handle + sim and place it in the right solver set (b3CreateJoint). */
export function createJoint(
    world: WorldState,
    def: JointDef,
    type: JointType,
): { joint: Joint; sim: JointSim } {
    const bodyA = world.bodies[def.bodyIdA];
    const bodyB = world.bodies[def.bodyIdB];
    const bodyIdA = bodyA.id;
    const bodyIdB = bodyB.id;
    const maxSetIndex = maxInt(bodyA.setIndex, bodyB.setIndex);

    const jointId = allocId(world.jointIdPool);
    if (jointId === world.joints.length) {
        world.joints.push(emptyJoint());
    }

    const joint = world.joints[jointId];
    joint.jointId = jointId;
    joint.userData = def.userData;
    joint.generation += 1;
    joint.setIndex = NULL_INDEX;
    joint.colorIndex = NULL_INDEX;
    joint.localIndex = NULL_INDEX;
    joint.islandId = NULL_INDEX;
    joint.islandIndex = NULL_INDEX;
    joint.drawScale = def.drawScale;
    joint.type = type;
    joint.collideConnected = def.collideConnected;

    // Doubly linked list on bodyA
    joint.edges[0] = { bodyId: bodyIdA, prevKey: NULL_INDEX, nextKey: bodyA.headJointKey };
    const keyA = (jointId << 1) | 0;
    if (bodyA.headJointKey !== NULL_INDEX) {
        const jointA = world.joints[bodyA.headJointKey >> 1];
        jointA.edges[bodyA.headJointKey & 1].prevKey = keyA;
    }
    bodyA.headJointKey = keyA;
    bodyA.jointCount += 1;

    // Doubly linked list on bodyB
    joint.edges[1] = { bodyId: bodyIdB, prevKey: NULL_INDEX, nextKey: bodyB.headJointKey };
    const keyB = (jointId << 1) | 1;
    if (bodyB.headJointKey !== NULL_INDEX) {
        const jointB = world.joints[bodyB.headJointKey >> 1];
        jointB.edges[bodyB.headJointKey & 1].prevKey = keyB;
    }
    bodyB.headJointKey = keyB;
    bodyB.jointCount += 1;

    let sim: JointSim;

    if (bodyA.setIndex === SetType.Disabled || bodyB.setIndex === SetType.Disabled) {
        // if either body is disabled, create in disabled set
        const set = world.solverSets[SetType.Disabled];
        joint.setIndex = SetType.Disabled;
        joint.localIndex = set.jointSims.length;
        sim = emptyJointSim();
        set.jointSims.push(sim);
    } else if (bodyA.type !== BodyType.Dynamic && bodyB.type !== BodyType.Dynamic) {
        // joint is not attached to a dynamic body
        const set = world.solverSets[SetType.Static];
        joint.setIndex = SetType.Static;
        joint.localIndex = set.jointSims.length;
        sim = emptyJointSim();
        set.jointSims.push(sim);
    } else if (bodyA.setIndex === SetType.Awake || bodyB.setIndex === SetType.Awake) {
        // if either body is sleeping, wake it
        if (maxSetIndex >= SetType.FirstSleeping) {
            wakeSolverSet(world, maxSetIndex);
        }
        joint.setIndex = SetType.Awake;
        sim = createJointInGraph(world, joint);
    } else {
        // joint connected between sleeping and/or static bodies
        if (
            bodyA.setIndex !== bodyB.setIndex &&
            bodyA.setIndex >= SetType.FirstSleeping &&
            bodyB.setIndex >= SetType.FirstSleeping
        ) {
            // Merging two sleeping sets needs b3MergeSolverSets — no ported path / fixture reaches it.
            throw new Error(
                "tumble: joint between two separate sleeping sets (needs mergeSolverSets)",
            );
        }
        const setIndex = maxSetIndex;
        const set = world.solverSets[setIndex];
        joint.setIndex = setIndex;
        joint.localIndex = set.jointSims.length;
        sim = emptyJointSim();
        set.jointSims.push(sim);
    }

    sim.jointId = jointId;
    sim.bodyIdA = bodyIdA;
    sim.bodyIdB = bodyIdB;
    sim.localFrameA = def.localFrameA;
    sim.localFrameB = def.localFrameB;
    sim.type = type;
    sim.constraintHertz = def.constraintHertz;
    sim.constraintDampingRatio = def.constraintDampingRatio;
    sim.constraintSoftness = { biasRate: 0, massScale: 1, impulseScale: 0 };
    sim.forceThreshold = def.forceThreshold;
    sim.torqueThreshold = def.torqueThreshold;

    if (joint.setIndex > SetType.Disabled) {
        linkJoint(world, joint);
    }

    return { joint, sim };
}

/**
 * Create a filter joint (b3CreateFilterJoint): connects two bodies solely to suppress collision
 * between them (collideConnected defaults false). It carries no constraint — prepare/warm-start/solve
 * are no-ops — so its sim keeps the null payload placeholder.
 */
export function createFilterJoint(
    world: WorldState,
    def: JointDef,
): { joint: Joint; sim: JointSim } {
    return createJoint(world, def, JointType.Filter);
}

/** Destroy a joint (b3DestroyJointInternal): unlink the body edges, the island, and the solver set. */
export function destroyJointInternal(world: WorldState, joint: Joint, wakeBodies: boolean): void {
    const jointId = joint.jointId;
    const edgeA = joint.edges[0];
    const edgeB = joint.edges[1];

    const idA = edgeA.bodyId;
    const idB = edgeB.bodyId;
    const bodyA = world.bodies[idA];
    const bodyB = world.bodies[idB];

    // Remove from body A
    if (edgeA.prevKey !== NULL_INDEX) {
        const prevJoint = world.joints[edgeA.prevKey >> 1];
        prevJoint.edges[edgeA.prevKey & 1].nextKey = edgeA.nextKey;
    }
    if (edgeA.nextKey !== NULL_INDEX) {
        const nextJoint = world.joints[edgeA.nextKey >> 1];
        nextJoint.edges[edgeA.nextKey & 1].prevKey = edgeA.prevKey;
    }
    const edgeKeyA = (jointId << 1) | 0;
    if (bodyA.headJointKey === edgeKeyA) {
        bodyA.headJointKey = edgeA.nextKey;
    }
    bodyA.jointCount -= 1;

    // Remove from body B
    if (edgeB.prevKey !== NULL_INDEX) {
        const prevJoint = world.joints[edgeB.prevKey >> 1];
        prevJoint.edges[edgeB.prevKey & 1].nextKey = edgeB.nextKey;
    }
    if (edgeB.nextKey !== NULL_INDEX) {
        const nextJoint = world.joints[edgeB.nextKey >> 1];
        nextJoint.edges[edgeB.nextKey & 1].prevKey = edgeB.prevKey;
    }
    const edgeKeyB = (jointId << 1) | 1;
    if (bodyB.headJointKey === edgeKeyB) {
        bodyB.headJointKey = edgeB.nextKey;
    }
    bodyB.jointCount -= 1;

    if (joint.islandId !== NULL_INDEX) {
        unlinkJoint(world, joint);
    }

    // Remove joint from the solver set that owns it
    const setIndex = joint.setIndex;
    const localIndex = joint.localIndex;
    if (setIndex === SetType.Awake) {
        removeJointFromGraph(
            world,
            joint.edges[0].bodyId,
            joint.edges[1].bodyId,
            joint.colorIndex,
            localIndex,
        );
    } else {
        const set = world.solverSets[setIndex];
        const movedIndex = set.jointSims.length - 1;
        const last = set.jointSims[movedIndex];
        set.jointSims[localIndex] = last;
        set.jointSims.pop();
        if (localIndex !== movedIndex) {
            const movedJoint = world.joints[last.jointId];
            movedJoint.localIndex = localIndex;
        }
    }

    // Free joint and id (preserve joint generation)
    joint.setIndex = NULL_INDEX;
    joint.localIndex = NULL_INDEX;
    joint.colorIndex = NULL_INDEX;
    joint.jointId = NULL_INDEX;
    freeId(world.jointIdPool, jointId);

    if (wakeBodies) {
        wakeBody(world, bodyA);
        wakeBody(world, bodyB);
    }
}

// --- Dispatch ---------------------------------------------------------------------------------

/** Prepare one joint: clamp the constraint hertz and dispatch to the type (b3PrepareJoint). */
export function prepareJoint(sim: JointSim, context: StepContext): void {
    const hertz = minf(sim.constraintHertz, f32(0.25 * context.invH));
    sim.constraintSoftness = makeSoft(hertz, sim.constraintDampingRatio, context.h);

    switch (sim.type) {
        case JointType.Parallel:
            prepareParallelJoint(sim, context);
            break;
        case JointType.Distance:
            prepareDistanceJoint(sim, context);
            break;
        case JointType.Motor:
            prepareMotorJoint(sim, context);
            break;
        case JointType.Prismatic:
            preparePrismaticJoint(sim, context);
            break;
        case JointType.Revolute:
            prepareRevoluteJoint(sim, context);
            break;
        case JointType.Spherical:
            prepareSphericalJoint(sim, context);
            break;
        case JointType.Weld:
            prepareWeldJoint(sim, context);
            break;
        case JointType.Wheel:
            prepareWheelJoint(sim, context);
            break;
        case JointType.Filter:
            break;
    }
}

/** Warm-start one joint (b3WarmStartJoint). */
export function warmStartJoint(sim: JointSim, context: StepContext): void {
    switch (sim.type) {
        case JointType.Parallel:
            warmStartParallelJoint(sim, context);
            break;
        case JointType.Distance:
            warmStartDistanceJoint(sim, context);
            break;
        case JointType.Motor:
            warmStartMotorJoint(sim, context);
            break;
        case JointType.Prismatic:
            warmStartPrismaticJoint(sim, context);
            break;
        case JointType.Revolute:
            warmStartRevoluteJoint(sim, context);
            break;
        case JointType.Spherical:
            warmStartSphericalJoint(sim, context);
            break;
        case JointType.Weld:
            warmStartWeldJoint(sim, context);
            break;
        case JointType.Wheel:
            warmStartWheelJoint(sim, context);
            break;
        case JointType.Filter:
            break;
    }
}

/** Solve one joint (b3SolveJoint). */
export function solveJoint(sim: JointSim, context: StepContext, useBias: boolean): void {
    switch (sim.type) {
        case JointType.Parallel:
            // Parallel is a pure soft/velocity constraint; it ignores useBias (b3SolveParallelJoint).
            solveParallelJoint(sim, context);
            break;
        case JointType.Distance:
            solveDistanceJoint(sim, context, useBias);
            break;
        case JointType.Motor:
            // Motor is a pure velocity/spring drive; it ignores useBias (b3SolveMotorJoint).
            solveMotorJoint(sim, context);
            break;
        case JointType.Prismatic:
            solvePrismaticJoint(sim, context, useBias);
            break;
        case JointType.Revolute:
            solveRevoluteJoint(sim, context, useBias);
            break;
        case JointType.Spherical:
            solveSphericalJoint(sim, context, useBias);
            break;
        case JointType.Weld:
            solveWeldJoint(sim, context, useBias);
            break;
        case JointType.Wheel:
            solveWheelJoint(sim, context, useBias);
            break;
        case JointType.Filter:
            break;
    }
}

// --- Per-color + serial overflow loops --------------------------------------------------------

/** Prepare every joint in a graph color (the b3_stagePrepareJoints block / b3PrepareJoints_Overflow). */
export function prepareColorJoints(joints: JointSim[], context: StepContext): void {
    for (let i = 0; i < joints.length; ++i) {
        prepareJoint(joints[i], context);
    }
}

/** Warm-start every joint in a graph color (b3WarmStartJointsTask / b3WarmStartJoints_Overflow). */
export function warmStartColorJoints(joints: JointSim[], context: StepContext): void {
    for (let i = 0; i < joints.length; ++i) {
        warmStartJoint(joints[i], context);
    }
}

/**
 * Flag a joint for a joint event if its reaction force/torque is over threshold (b3SolveJointsTask's
 * event check), unless it is already flagged. A zero threshold reports every awake joint. The flag set is
 * hash-invisible (events are behavioral), so on the pool — where the joints solve in-kernel and
 * {@link solveColorJoints} never runs — this reconstructs the flags from the read-back impulses instead
 * (`readbackJointImpulses` then this pass, in solver.ts). Serial and pooled paths agree for a zero
 * threshold and any settled load; they can differ only for a nonzero threshold a transient spike crosses
 * mid-solve, since the pooled pass sees only the final substep's impulses.
 */
export function flagJointEvent(sim: JointSim, context: StepContext): void {
    if (
        (sim.forceThreshold < FLT_MAX || sim.torqueThreshold < FLT_MAX) &&
        context.jointEventFlags.has(sim.jointId) === false
    ) {
        const reaction = getJointReaction(context.world, sim, context.invH);
        if (reaction.force >= sim.forceThreshold || reaction.torque >= sim.torqueThreshold) {
            context.jointEventFlags.add(sim.jointId);
        }
    }
}

/** Solve every joint in a graph color (b3SolveJointsTask / b3SolveJoints_Overflow). */
export function solveColorJoints(joints: JointSim[], context: StepContext, useBias: boolean): void {
    for (let i = 0; i < joints.length; ++i) {
        const sim = joints[i];
        solveJoint(sim, context, useBias);
        // Flag over-threshold joints for an event on the biased pass only (matching b3SolveJointsTask).
        if (useBias) flagJointEvent(sim, context);
    }
}

/** Prepare every joint in the overflow color (b3PrepareJoints_Overflow). */
export function prepareOverflowJoints(context: StepContext): void {
    prepareColorJoints(context.world.constraintGraph.colors[OVERFLOW_INDEX].jointSims, context);
}

/** Warm-start every joint in the overflow color (b3WarmStartJoints_Overflow). */
export function warmStartOverflowJoints(context: StepContext): void {
    warmStartColorJoints(context.world.constraintGraph.colors[OVERFLOW_INDEX].jointSims, context);
}

/** Solve every joint in the overflow color (b3SolveJoints_Overflow). */
export function solveOverflowJoints(context: StepContext, useBias: boolean): void {
    solveColorJoints(
        context.world.constraintGraph.colors[OVERFLOW_INDEX].jointSims,
        context,
        useBias,
    );
}

// --- Reaction / force / torque ----------------------------------------------------------------

/** The constraint force this joint applies (b3GetJointConstraintForce). */
export function getJointConstraintForce(world: WorldState, sim: JointSim): Vec3 {
    switch (sim.type) {
        case JointType.Distance:
            return getDistanceJointForce(world, sim);
        case JointType.Motor:
            return getMotorJointForce(world, sim);
        case JointType.Prismatic:
            return getPrismaticJointForce(world, sim);
        case JointType.Revolute:
            return getRevoluteJointForce(world, sim);
        case JointType.Spherical:
            return getSphericalJointForce(world, sim);
        case JointType.Weld:
            return getWeldJointForce(world, sim);
        case JointType.Wheel:
            return getWheelJointForce(world, sim);
        case JointType.Parallel:
        case JointType.Filter:
            return { x: 0, y: 0, z: 0 };
    }
}

/** The constraint torque this joint applies (b3GetJointConstraintTorque). */
export function getJointConstraintTorque(world: WorldState, sim: JointSim): Vec3 {
    switch (sim.type) {
        case JointType.Parallel:
            return getParallelJointTorque(world, sim);
        case JointType.Motor:
            return getMotorJointTorque(world, sim);
        case JointType.Prismatic:
            return getPrismaticJointTorque(world, sim);
        case JointType.Revolute:
            return getRevoluteJointTorque(world, sim);
        case JointType.Spherical:
            return getSphericalJointTorque(world, sim);
        case JointType.Weld:
            return getWeldJointTorque(world, sim);
        case JointType.Wheel:
            return getWheelJointTorque(world, sim);
        case JointType.Distance:
        case JointType.Filter:
            return { x: 0, y: 0, z: 0 };
    }
}

/**
 * The scalar reaction force and torque this joint applied last step (b3GetJointReaction), read from
 * the raw impulse accumulators. Feeds the joint-event threshold test; hash-invisible, but ported
 * op-for-op under the same fround discipline as the rest of the solver.
 */
function getJointReaction(
    world: WorldState,
    sim: JointSim,
    invTimeStep: number,
): { force: number; torque: number } {
    let linearImpulse = 0;
    let angularImpulse = 0;

    switch (sim.type) {
        case JointType.Parallel: {
            const joint = sim.data as ParallelJoint;
            angularImpulse = vec3.length({ x: joint.perpImpulse.x, y: joint.perpImpulse.y, z: 0 });
            break;
        }
        case JointType.Distance: {
            const joint = sim.data as DistanceJoint;
            linearImpulse = absf(
                f32(
                    f32(f32(joint.impulse + joint.lowerImpulse) - joint.upperImpulse) +
                        joint.motorImpulse,
                ),
            );
            break;
        }
        case JointType.Motor: {
            const joint = sim.data as MotorJoint;
            linearImpulse = vec3.length(
                vec3.add(joint.linearVelocityImpulse, joint.linearSpringImpulse),
            );
            angularImpulse = vec3.length(
                vec3.add(joint.angularVelocityImpulse, joint.angularSpringImpulse),
            );
            break;
        }
        case JointType.Prismatic: {
            const joint = sim.data as PrismaticJoint;
            linearImpulse = vec3.length({
                x: f32(f32(joint.motorImpulse + joint.lowerImpulse) - joint.upperImpulse),
                y: joint.perpImpulse.x,
                z: joint.perpImpulse.y,
            });
            angularImpulse = vec3.length(joint.angularImpulse);
            break;
        }
        case JointType.Revolute: {
            const joint = sim.data as RevoluteJoint;
            linearImpulse = vec3.length(joint.linearImpulse);
            angularImpulse = vec3.length({
                x: joint.perpImpulse.x,
                y: joint.perpImpulse.y,
                z: f32(f32(joint.motorImpulse + joint.lowerImpulse) - joint.upperImpulse),
            });
            break;
        }
        case JointType.Spherical: {
            const joint = sim.data as SphericalJoint;
            linearImpulse = vec3.length(joint.linearImpulse);

            const xfA = getBodyTransformQuick(world, world.bodies[sim.bodyIdA]);
            const xfB = getBodyTransformQuick(world, world.bodies[sim.bodyIdB]);
            const qA = quat.mul(xfA.q, sim.localFrameA.q);
            const qB = quat.mul(xfB.q, sim.localFrameB.q);

            // Cone axis is body A's z-axis, twist axis body B's; swing is their cross.
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
            angularImpulse = vec3.length(impulse);
            break;
        }
        case JointType.Weld: {
            const joint = sim.data as WeldJoint;
            linearImpulse = vec3.length(joint.linearImpulse);
            angularImpulse = vec3.length(joint.angularImpulse);
            break;
        }
        case JointType.Wheel: {
            const joint = sim.data as WheelJoint;
            const perp = joint.linearImpulse;
            const axial = f32(
                f32(joint.suspensionSpringImpulse + joint.lowerSuspensionImpulse) -
                    joint.upperSuspensionImpulse,
            );
            linearImpulse = f32(
                Math.sqrt(
                    f32(f32(f32(perp.x * perp.x) + f32(perp.y * perp.y)) + f32(axial * axial)),
                ),
            );
            angularImpulse = absf(joint.spinImpulse);
            break;
        }
        case JointType.Filter:
            break;
    }

    return { force: f32(linearImpulse * invTimeStep), torque: f32(angularImpulse * invTimeStep) };
}

// --- Base runtime API (b3Joint_*) -------------------------------------------------------------

/** Toggle whether the two connected bodies collide, updating the broad-phase (b3Joint_SetCollideConnected). */
export function setJointCollideConnected(
    world: WorldState,
    joint: Joint,
    shouldCollide: boolean,
): void {
    if (joint.collideConnected === shouldCollide) {
        return;
    }
    joint.collideConnected = shouldCollide;

    const bodyA = world.bodies[joint.edges[0].bodyId];
    const bodyB = world.bodies[joint.edges[1].bodyId];

    if (shouldCollide) {
        // Tell the broad-phase to look for new pairs on the body with fewest shapes.
        let shapeId = bodyA.shapeCount < bodyB.shapeCount ? bodyA.headShapeId : bodyB.headShapeId;
        while (shapeId !== NULL_INDEX) {
            const shape = world.shapes[shapeId];
            if (shape.proxyKey !== NULL_INDEX) {
                bufferMove(world.broadPhase, shape.proxyKey);
            }
            shapeId = shape.nextShapeId;
        }
    } else {
        destroyContactsBetweenBodies(world, bodyA, bodyB);
    }
}

/** Destroy any contacts between two bodies (b3DestroyContactsBetweenBodies) — walk the shorter list. */
function destroyContactsBetweenBodies(world: WorldState, bodyA: Body, bodyB: Body): void {
    let contactKey: number;
    let otherBodyId: number;
    if (bodyA.contactCount < bodyB.contactCount) {
        contactKey = bodyA.headContactKey;
        otherBodyId = bodyB.id;
    } else {
        contactKey = bodyB.headContactKey;
        otherBodyId = bodyA.id;
    }

    // No need to wake bodies when a joint removes collision between them.
    while (contactKey !== NULL_INDEX) {
        const contactId = contactKey >> 1;
        const edgeIndex = contactKey & 1;
        const contact = world.contacts[contactId];
        contactKey = contact.edges[edgeIndex].nextKey;
        const otherEdgeIndex = edgeIndex ^ 1;
        if (contact.edges[otherEdgeIndex].bodyId === otherBodyId) {
            // Careful: this removes the contact from the list we are walking.
            destroyContact(world, contact, false);
        }
    }
}

/** Wake both bodies attached to a joint (b3Joint_WakeBodies). */
export function wakeJointBodies(world: WorldState, joint: Joint): void {
    world.locked = true;
    wakeBody(world, world.bodies[joint.edges[0].bodyId]);
    wakeBody(world, world.bodies[joint.edges[1].bodyId]);
    world.locked = false;
}

/** The linear separation error at the joint anchors (b3Joint_GetLinearSeparation). */
export function getJointLinearSeparation(world: WorldState, joint: Joint): number {
    const sim = getJointSim(world, joint);
    const xfA = getBodyTransformQuick(world, world.bodies[joint.edges[0].bodyId]);
    const xfB = getBodyTransformQuick(world, world.bodies[joint.edges[1].bodyId]);
    const pA = transformWorldPoint(xfA, sim.localFrameA.p);
    const pB = transformWorldPoint(xfB, sim.localFrameB.p);
    const dp = vec3.sub(pB, pA);

    switch (joint.type) {
        case JointType.Parallel:
        case JointType.Motor:
        case JointType.Filter:
            return 0;
        case JointType.Distance: {
            const dj = sim.data as DistanceJoint;
            const length = vec3.length(dp);
            if (dj.enableSpring) {
                if (dj.enableLimit) {
                    if (length < dj.minLength) return f32(dj.minLength - length);
                    if (length > dj.maxLength) return f32(length - dj.maxLength);
                    return 0;
                }
                return 0;
            }
            return absf(f32(length - dj.length));
        }
        case JointType.Revolute:
        case JointType.Spherical:
            return vec3.length(dp);
        case JointType.Weld: {
            const wj = sim.data as WeldJoint;
            return wj.linearHertz === 0 ? vec3.length(dp) : 0;
        }
        case JointType.Prismatic: {
            const pj = sim.data as PrismaticJoint;
            return axisSeparation(
                xfA,
                dp,
                pj.enableLimit,
                pj.lowerTranslation,
                pj.upperTranslation,
            );
        }
        case JointType.Wheel: {
            const whj = sim.data as WheelJoint;
            return axisSeparation(
                xfA,
                dp,
                whj.enableSuspensionLimit,
                whj.lowerSuspensionLimit,
                whj.upperSuspensionLimit,
            );
        }
    }
}

/** Shared perpendicular + axial-limit separation for prismatic/wheel (their b3Joint_GetLinearSeparation arms). */
function axisSeparation(
    xfA: Transform,
    dp: Vec3,
    enableLimit: boolean,
    lower: number,
    upper: number,
): number {
    const axisA = quat.rotate(xfA.q, vec3.axisX());
    const perpA = vec3.perp(axisA);
    const perpendicular = absf(vec3.dot(perpA, dp));
    let limit = 0;
    if (enableLimit) {
        const translation = vec3.dot(axisA, dp);
        if (translation < lower) limit = f32(lower - translation);
        if (upper < translation) limit = f32(translation - upper);
    }
    return f32(Math.sqrt(f32(f32(perpendicular * perpendicular) + f32(limit * limit))));
}

/** The angular separation error at the joint (b3Joint_GetAngularSeparation). */
export function getJointAngularSeparation(world: WorldState, joint: Joint): number {
    const sim = getJointSim(world, joint);
    const xfA = getBodyTransformQuick(world, world.bodies[joint.edges[0].bodyId]);
    const xfB = getBodyTransformQuick(world, world.bodies[joint.edges[1].bodyId]);
    const relQ = quat.invMul(xfA.q, xfB.q);

    switch (joint.type) {
        case JointType.Distance:
        case JointType.Motor:
        case JointType.Filter:
            return 0;
        case JointType.Parallel:
            // Remove the hinge angle before measuring.
            relQ.v.z = 0;
            return quat.getAngle(relQ);
        case JointType.Prismatic:
            return quat.getAngle(relQ);
        case JointType.Revolute: {
            const rj = sim.data as RevoluteJoint;
            if (rj.enableLimit) {
                const angle = quat.getTwistAngle(relQ);
                if (angle < rj.lowerAngle) return quat.getAngle(relQ);
                if (rj.upperAngle < angle) return quat.getAngle(relQ);
            }
            // Remove the hinge angle.
            relQ.v.z = 0;
            return quat.getAngle(relQ);
        }
        case JointType.Spherical: {
            const sj = sim.data as SphericalJoint;
            let sum = 0;
            if (sj.enableConeLimit) {
                const swingAngle = quat.getSwingAngle(relQ);
                sum = f32(sum + maxf(0, f32(swingAngle - sj.coneAngle)));
            }
            if (sj.enableTwistLimit) {
                const twistAngle = quat.getTwistAngle(relQ);
                sum = f32(sum + maxf(0, f32(sj.lowerTwistAngle - twistAngle)));
                sum = f32(sum + maxf(0, f32(twistAngle - sj.upperTwistAngle)));
            }
            return sum;
        }
        case JointType.Weld: {
            const wj = sim.data as WeldJoint;
            return wj.angularHertz === 0 ? quat.getAngle(relQ) : 0;
        }
        case JointType.Wheel:
            // Unimplemented in the C reference (b3Joint_GetAngularSeparation asserts for wheel).
            throw new Error("tumble: wheel joint angular separation is unimplemented");
    }
}
