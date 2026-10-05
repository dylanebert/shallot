import { GRAPH_COLOR_COUNT, OVERFLOW_INDEX, SetType } from "../common/constants";
import type { Quat, Transform, Vec3 } from "../common/math";
import type { SolveLayout } from "../solver/contactsolver";
import type { Joint } from "../solver/joint";
import type { WorldState } from "../world/world";
import {
    J_BODY_INDEX_A,
    J_BODY_INDEX_B,
    J_EVENT,
    J_JOINT_ID,
    J_SIM_INDEX_A,
    J_SIM_INDEX_B,
} from "./columns";
import { kernel } from "./kernel";
import { bodyColumnIndex } from "./stagedbodies";

function jointKernel(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
export function jointArrayKey(joint: Joint): number {
    return joint.setIndex === SetType.Awake ? joint.colorIndex : GRAPH_COLOR_COUNT + joint.setIndex;
}
export function releaseJointArray(world: WorldState, key: number): void {
    jointKernel(world).jointArrayRelease(key);
}
export function jointArrayCount(world: WorldState, key: number): number {
    return jointKernel(world).jointArrayCount(key);
}
export function appendJointRecord(world: WorldState, key: number): number {
    return jointKernel(world).jointArrayAppend(key);
}
export function removeJointRecord(world: WorldState, key: number, index: number): void {
    const moved = jointKernel(world).jointArrayRemove(key, index) >>> 0;
    if (moved !== 0xffffffff) world.joints[moved].localIndex = index;
}
export function moveJointRecord(world: WorldState, joint: Joint, target: number): number {
    const k = jointKernel(world);
    const destination = k.jointArrayCount(target);
    const index = joint.localIndex;
    const moved = k.jointArrayMove(jointArrayKey(joint), index, target) >>> 0;
    if (moved !== 0xffffffff) world.joints[moved].localIndex = index;
    return destination;
}
export function jointAt(world: WorldState, key: number, index: number): Joint {
    return world.joints[jointKernel(world).jointReadWord(key, index, J_JOINT_ID)];
}
export function readJointFloat(world: WorldState, joint: Joint, field: number): number {
    return jointKernel(world).jointReadFloat(jointArrayKey(joint), joint.localIndex, field);
}
export function writeJointFloat(
    world: WorldState,
    joint: Joint,
    field: number,
    value: number,
): void {
    jointKernel(world).jointWriteFloat(jointArrayKey(joint), joint.localIndex, field, value);
}
export function readJointWord(world: WorldState, joint: Joint, field: number): number {
    return jointKernel(world).jointReadWord(jointArrayKey(joint), joint.localIndex, field);
}
export function writeJointWord(
    world: WorldState,
    joint: Joint,
    field: number,
    value: number,
): void {
    jointKernel(world).jointWriteWord(jointArrayKey(joint), joint.localIndex, field, value);
}
export function readJointFlag(
    world: WorldState,
    joint: Joint,
    field: number,
    mask: number,
): boolean {
    return (readJointWord(world, joint, field) & mask) !== 0;
}
export function writeJointFlag(
    world: WorldState,
    joint: Joint,
    field: number,
    mask: number,
    value: boolean,
): void {
    const bits = readJointWord(world, joint, field);
    writeJointWord(world, joint, field, value ? bits | mask : bits & ~mask);
}
export function readJointVec3(world: WorldState, joint: Joint, field: number): Vec3 {
    return {
        x: readJointFloat(world, joint, field),
        y: readJointFloat(world, joint, field + 1),
        z: readJointFloat(world, joint, field + 2),
    };
}
export function writeJointVec3(world: WorldState, joint: Joint, field: number, v: Vec3): void {
    writeJointFloat(world, joint, field, v.x);
    writeJointFloat(world, joint, field + 1, v.y);
    writeJointFloat(world, joint, field + 2, v.z);
}
export function writeJointVec2(
    world: WorldState,
    joint: Joint,
    field: number,
    v: { x: number; y: number },
): void {
    writeJointFloat(world, joint, field, v.x);
    writeJointFloat(world, joint, field + 1, v.y);
}
export function readJointVec2(
    world: WorldState,
    joint: Joint,
    field: number,
): { x: number; y: number } {
    return { x: readJointFloat(world, joint, field), y: readJointFloat(world, joint, field + 1) };
}
export function readJointQuat(world: WorldState, joint: Joint, field: number): Quat {
    return { v: readJointVec3(world, joint, field), s: readJointFloat(world, joint, field + 3) };
}
export function writeJointQuat(world: WorldState, joint: Joint, field: number, q: Quat): void {
    writeJointVec3(world, joint, field, q.v);
    writeJointFloat(world, joint, field + 3, q.s);
}
export function readJointTransform(world: WorldState, joint: Joint, field: number): Transform {
    return { p: readJointVec3(world, joint, field), q: readJointQuat(world, joint, field + 3) };
}
export function writeJointTransform(
    world: WorldState,
    joint: Joint,
    field: number,
    t: Transform,
): void {
    writeJointVec3(world, joint, field, t.p);
    writeJointQuat(world, joint, field + 3, t.q);
}

/** Only body-column addresses remain staged until body records move into the kernel. */
export function stageJointBodies(
    world: WorldState,
    layout: SolveLayout,
    spans: Uint32Array,
): number {
    const k = jointKernel(world);
    let total = 0;
    for (let c = 0; c <= layout.colors.length; ++c) {
        const key = c === layout.colors.length ? OVERFLOW_INDEX : layout.colors[c].colorIndex;
        const count = k.jointArrayCount(key);
        if (c < layout.colors.length) {
            spans[c * 6 + 4] = key;
            spans[c * 6 + 5] = count;
            total += count;
        }
        for (let i = 0; i < count; ++i) {
            const joint = world.joints[k.jointReadWord(key, i, J_JOINT_ID)];
            const a = world.bodies[joint.edges[0].bodyId];
            const b = world.bodies[joint.edges[1].bodyId];
            k.jointWriteWord(
                key,
                i,
                J_SIM_INDEX_A,
                a.setIndex === SetType.Awake ? a.localIndex : 0xffffffff,
            );
            k.jointWriteWord(
                key,
                i,
                J_SIM_INDEX_B,
                b.setIndex === SetType.Awake ? b.localIndex : 0xffffffff,
            );
            k.jointWriteWord(key, i, J_BODY_INDEX_A, bodyColumnIndex(world, a));
            k.jointWriteWord(key, i, J_BODY_INDEX_B, bodyColumnIndex(world, b));
        }
    }
    return total;
}
export function collectJointEvents(
    world: WorldState,
    layout: SolveLayout,
    events: Set<number>,
): void {
    const k = jointKernel(world);
    for (let c = 0; c <= layout.colors.length; ++c) {
        const key = c === layout.colors.length ? OVERFLOW_INDEX : layout.colors[c].colorIndex;
        const count = k.jointArrayCount(key);
        for (let i = 0; i < count; ++i) {
            if (k.jointReadFloat(key, i, J_EVENT) !== 0)
                events.add(k.jointReadWord(key, i, J_JOINT_ID));
        }
    }
}
