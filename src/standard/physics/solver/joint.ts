import { NULL_INDEX } from "../common/array";
import type { EntityId } from "../common/ids";
import { FLT_MAX, type Transform, type Vec3 } from "../common/math";
import { readJointEventUserData, readJointReaction } from "../kernel/jointcolumns";
import { JointField, jointCapacity, jointField, jointViews } from "../kernel/jointrecords";
import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";

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

/** Kernel joint identity index. */
export type Joint = number;

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

export function jointIsValid(world: WorldState, id: EntityId): boolean {
    if (!world.inUse) return false;
    const joint = id.index1 - 1;
    return (
        joint >= 0 &&
        joint < jointCapacity(world) &&
        jointField(world, joint, JointField.setIndex) !== NULL_INDEX &&
        jointField(world, joint, JointField.generation) === id.generation
    );
}

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

export function writeJointDefinition(world: WorldState, def: JointDef): number {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    const ptr = k.jointDefinitionPtr(world.worldId);
    const { words, floats } = jointViews(k);
    const o = ptr >>> 2;
    words[o] = def.bodyIdA;
    words[o + 1] = def.bodyIdB;
    floats[o + 2] = def.drawScale;
    words[o + 3] = Number(def.collideConnected);
    floats[o + 4] = def.localFrameA.p.x;
    floats[o + 5] = def.localFrameA.p.y;
    floats[o + 6] = def.localFrameA.p.z;
    floats[o + 7] = def.localFrameA.q.v.x;
    floats[o + 8] = def.localFrameA.q.v.y;
    floats[o + 9] = def.localFrameA.q.v.z;
    floats[o + 10] = def.localFrameA.q.s;
    floats[o + 11] = def.localFrameB.p.x;
    floats[o + 12] = def.localFrameB.p.y;
    floats[o + 13] = def.localFrameB.p.z;
    floats[o + 14] = def.localFrameB.q.v.x;
    floats[o + 15] = def.localFrameB.q.v.y;
    floats[o + 16] = def.localFrameB.q.v.z;
    floats[o + 17] = def.localFrameB.q.s;
    floats[o + 18] = def.forceThreshold;
    floats[o + 19] = def.torqueThreshold;
    floats[o + 20] = def.constraintHertz;
    floats[o + 21] = def.constraintDampingRatio;
    return ptr;
}

/** Create a joint identity and sim in Box3D's solver-set and island order. */
export function createJointRecord(world: WorldState, def: JointDef, type: JointType): Joint {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    const joint = k.jointCreate(
        def.bodyIdA,
        def.bodyIdB,
        type,
        def.drawScale,
        +def.collideConnected,
        def.localFrameA.p.x,
        def.localFrameA.p.y,
        def.localFrameA.p.z,
        def.localFrameA.q.v.x,
        def.localFrameA.q.v.y,
        def.localFrameA.q.v.z,
        def.localFrameA.q.s,
        def.localFrameB.p.x,
        def.localFrameB.p.y,
        def.localFrameB.p.z,
        def.localFrameB.q.v.x,
        def.localFrameB.q.v.y,
        def.localFrameB.q.v.z,
        def.localFrameB.q.s,
        def.forceThreshold,
        def.torqueThreshold,
        def.constraintHertz,
        def.constraintDampingRatio,
    );
    readJointEventUserData(world);
    world.jointUserData[joint] = def.userData;
    return joint;
}

export function finishJointCreation(
    world: WorldState,
    def: JointDef,
    joint: Joint,
): { joint: Joint } {
    readJointEventUserData(world);
    world.jointUserData[joint] = def.userData;
    return { joint };
}
export function createJoint(world: WorldState, def: JointDef, type: JointType): { joint: Joint } {
    return { joint: createJointRecord(world, def, type) };
}
/** A filter joint suppresses collision and carries no solver constraint. */
export function createFilterJoint(world: WorldState, def: JointDef): { joint: Joint } {
    return createJoint(world, def, JointType.Filter);
}
/** Unlink and free a joint, optionally waking both attached solver sets. */
export function destroyJointInternal(world: WorldState, joint: Joint, wakeBodies: boolean): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    readJointEventUserData(world);
    k.jointDestroy(joint, +wakeBodies);
    world.jointUserData[joint] = null;
}
/** The constraint force this joint applies (b3GetJointConstraintForce). */
export function getJointConstraintForce(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, false);
}
/** The constraint torque this joint applies (b3GetJointConstraintTorque). */
export function getJointConstraintTorque(world: WorldState, sim: Joint): Vec3 {
    return readJointReaction(world, sim, true);
}
/** Toggle whether the two connected bodies collide, updating the broad-phase (b3Joint_SetCollideConnected). */
export function setJointCollideConnected(
    world: WorldState,
    joint: Joint,
    shouldCollide: boolean,
): void {
    kernel(world.ecsState).jointSetCollideConnected(world.worldId, joint, shouldCollide);
}
/** Wake both bodies attached to a joint (b3Joint_WakeBodies). */
export function wakeJointBodies(world: WorldState, joint: Joint): void {
    world.locked = true;
    kernel(world.ecsState).jointWakeBodies(world.worldId, joint);
    world.locked = false;
}
/** The linear separation error at the joint anchors (b3Joint_GetLinearSeparation). */
export function getJointLinearSeparation(world: WorldState, joint: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, joint, 5);
}
/** The angular separation error at the joint (b3Joint_GetAngularSeparation). */
export function getJointAngularSeparation(world: WorldState, joint: Joint): number {
    return kernel(world.ecsState).jointRead(world.worldId, joint, 4);
}
