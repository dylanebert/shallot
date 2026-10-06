import type { WorldState } from "../world/world";
import { type BodySimRef, bodySimSlot } from "./bodycolumns";
import { kernel } from "./kernel";

function active(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
export function createSolverSet(world: WorldState): number {
    return active(world).solverSetCreate();
}
export function solverSetCount(world: WorldState): number {
    return active(world).solverSetCount();
}
export function solverSetIndex(world: WorldState, set: number): number {
    return active(world).solverSetIndex(set);
}
export function releaseSolverSet(world: WorldState, set: number): void {
    active(world).solverSetDestroy(set);
}
export function setBodyCount(world: WorldState, set: number): number {
    return active(world).solverSetBodyCount(set);
}
export function setBodyPush(world: WorldState, set: number, sim: BodySimRef): number {
    const i = active(world).solverSetBodyAppend(set);
    world.bodyStore.refreshViews();
    world.bodyStore.writeSim(bodySimSlot(set, i), sim);
    return i;
}
export function setBodyPop(world: WorldState, set: number): void {
    active(world).solverSetBodyPop(set);
}
export function setBodyRemove(world: WorldState, set: number, index: number): number {
    const last = setBodyCount(world, set) - 1;
    if (index !== last) world.bodyStore.writeSim(bodySimSlot(set, index), bodySimSlot(set, last));
    setBodyPop(world, set);
    return index === last ? -1 : last;
}
export function setArrayCount(world: WorldState, set: number, kind: number): number {
    return active(world).solverSetArrayCount(set, kind);
}
export function setArrayGet(world: WorldState, set: number, kind: number, index: number): number {
    return active(world).solverSetArrayGet(set, kind, index);
}
export function setArrayPush(world: WorldState, set: number, kind: number, value: number): number {
    return active(world).solverSetArrayPush(set, kind, value);
}
export function setArrayRemove(
    world: WorldState,
    set: number,
    kind: number,
    index: number,
): number {
    return active(world).solverSetArrayRemove(set, kind, index);
}
export function setArrayWrite(
    world: WorldState,
    set: number,
    kind: number,
    index: number,
    value: number,
): void {
    active(world).solverSetArrayWrite(set, kind, index, value);
}
export function setArrayPop(world: WorldState, set: number, kind: number): void {
    active(world).solverSetArrayPop(set, kind);
}
/** Observation only; stepping reads indices at their local slot. */
export function setArraySnapshot(world: WorldState, set: number, kind: number): number[] {
    return Array.from({ length: setArrayCount(world, set, kind) }, (_, i) =>
        setArrayGet(world, set, kind, i),
    );
}
