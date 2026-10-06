import { GRAPH_COLOR_COUNT, SetType } from "../common/constants";
import type { Quat, Transform, Vec3 } from "../common/math";
import type { Joint } from "../solver/joint";
import type { WorldState } from "../world/world";
import { J_JOINT_ID } from "./columns";
import { JointField, jointField, jointViews } from "./jointrecords";
import { kernel } from "./kernel";

function jointKernel(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
export function readJointReaction(world: WorldState, joint: Joint, torque: boolean): Vec3 {
    const k = jointKernel(world);
    k.jointReaction(world.worldId, joint, world.invH, Number(torque));
    const f = jointViews(k).floats,
        n = k.shapeQueryOutputPtr() >>> 2;
    return { x: f[n], y: f[n + 1], z: f[n + 2] };
}
export function jointArrayKey(world: WorldState, joint: Joint): number {
    return jointField(world, joint, JointField.setIndex) === SetType.Awake
        ? jointField(world, joint, JointField.colorIndex)
        : GRAPH_COLOR_COUNT + jointField(world, joint, JointField.setIndex);
}
export function jointArrayCount(world: WorldState, key: number): number {
    return jointKernel(world).jointArrayCount(key);
}
export function jointAt(world: WorldState, key: number, index: number): Joint {
    return jointKernel(world).jointReadWord(key, index, J_JOINT_ID);
}
export function readJointFloat(world: WorldState, joint: Joint, field: number): number {
    const k = jointKernel(world);
    return jointViews(k).floats[(k.jointSimPtr(joint) >>> 2) + field];
}
export function writeJointFloat(
    world: WorldState,
    joint: Joint,
    field: number,
    value: number,
): void {
    const k = jointKernel(world);
    jointViews(k).floats[(k.jointSimPtr(joint) >>> 2) + field] = value;
}
export function readJointWord(world: WorldState, joint: Joint, field: number): number {
    const k = jointKernel(world);
    return jointViews(k).words[(k.jointSimPtr(joint) >>> 2) + field];
}
export function readJointFlag(
    world: WorldState,
    joint: Joint,
    field: number,
    mask: number,
): boolean {
    return (readJointWord(world, joint, field) & mask) !== 0;
}
export function readJointVec3(
    world: WorldState,
    joint: Joint,
    field: number,
    out: Vec3 = { x: 0, y: 0, z: 0 },
): Vec3 {
    out.x = readJointFloat(world, joint, field);
    out.y = readJointFloat(world, joint, field + 1);
    out.z = readJointFloat(world, joint, field + 2);
    return out;
}
export function writeJointVec3(world: WorldState, joint: Joint, field: number, v: Vec3): void {
    jointKernel(world).jointWriteVec3(joint, field, v.x, v.y, v.z);
}
export function readJointVec2(
    world: WorldState,
    joint: Joint,
    field: number,
): { x: number; y: number } {
    return { x: readJointFloat(world, joint, field), y: readJointFloat(world, joint, field + 1) };
}
export function readJointQuat(
    world: WorldState,
    joint: Joint,
    field: number,
    out: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 },
): Quat {
    readJointVec3(world, joint, field, out.v);
    out.s = readJointFloat(world, joint, field + 3);
    return out;
}
export function writeJointQuat(world: WorldState, joint: Joint, field: number, q: Quat): void {
    jointKernel(world).jointWriteQuat(joint, field, q.v.x, q.v.y, q.v.z, q.s);
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

const eventViews = new WeakMap<WorldState, Uint32Array>();
export function collectJointEvents(world: WorldState): void {
    const k = jointKernel(world);
    k.jointCollectEvents();
    readJointEventUserData(world);
}
export function readJointEventUserData(world: WorldState): void {
    const k = jointKernel(world);
    const count = k.eventCount(world.worldId, 5);
    if (count === 0) return;
    let words = eventViews.get(world);
    if (words?.buffer !== k.memory.buffer) {
        words = new Uint32Array(k.memory.buffer);
        eventViews.set(world, words);
    }
    const start = k.eventBufferPtr(world.worldId, 5) >>> 2;
    for (let i = 0; i < count; ++i) {
        const id = words[start + i * 3 + 2];
        world.jointEventUserData[i] = world.jointUserData[id];
    }
}
