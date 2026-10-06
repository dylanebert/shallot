import type { WorldState } from "../world/world";
import { type Kernel, kernel } from "./kernel";

export const JointField = {
    setIndex: 0,
    colorIndex: 1,
    localIndex: 2,
    bodyIdA: 3,
    prevKeyA: 4,
    nextKeyA: 5,
    bodyIdB: 6,
    prevKeyB: 7,
    nextKeyB: 8,
    jointId: 9,
    islandId: 10,
    islandIndex: 11,
    type: 13,
    generation: 14,
    collideConnected: 15,
} as const;
export type JointField = (typeof JointField)[keyof typeof JointField];

function jointKernel(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
export const JOINT_RECORD_WORDS = 15;
export type JointViews = { words: Int32Array; floats: Float32Array };
const views = new WeakMap<Kernel, JointViews>();
// Reading floats through memory avoids boxed WASM return values at the JavaScript boundary.
export function jointViews(k: Kernel): JointViews {
    let view = views.get(k);
    if (view?.words.buffer !== k.memory.buffer) {
        view = {
            words: new Int32Array(k.memory.buffer),
            floats: new Float32Array(k.memory.buffer),
        };
        views.set(k, view);
    }
    return view;
}
export function jointField(world: WorldState, id: number, field: number): number {
    const k = jointKernel(world);
    const view = jointViews(k);
    const base = (k.jointRecordPtr() >>> 2) + id * JOINT_RECORD_WORDS;
    if (field === JointField.generation) return view.words[base + 14] & 0xffff;
    if (field === JointField.collideConnected) return (view.words[base + 14] >>> 16) & 0xff;
    return view.words[base + field];
}
export function setJointField(world: WorldState, id: number, field: number, value: number): void {
    const k = jointKernel(world);
    const view = jointViews(k);
    const base = (k.jointRecordPtr() >>> 2) + id * JOINT_RECORD_WORDS;
    if (field === JointField.generation)
        view.words[base + 14] = (view.words[base + 14] & ~0xffff) | (value & 0xffff);
    else if (field === JointField.collideConnected)
        view.words[base + 14] = (view.words[base + 14] & ~0xff0000) | (+!!value << 16);
    else view.words[base + field] = value;
}
export function jointDrawScale(world: WorldState, id: number): number {
    const k = jointKernel(world);
    return jointViews(k).floats[(k.jointRecordPtr() >>> 2) + id * JOINT_RECORD_WORDS + 12];
}
export function setJointDrawScale(world: WorldState, id: number, value: number): void {
    const k = jointKernel(world);
    jointViews(k).floats[(k.jointRecordPtr() >>> 2) + id * JOINT_RECORD_WORDS + 12] = value;
}
export function jointCapacity(world: WorldState): number {
    return jointKernel(world).jointRecordCapacity();
}
export function jointCount(world: WorldState): number {
    return jointKernel(world).jointRecordCount();
}
