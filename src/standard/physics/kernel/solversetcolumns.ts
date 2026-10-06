import type { WorldState } from "../world/world";
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
    world.bodyStore.forgetSet(set);
}
export function setBodyCount(world: WorldState, set: number): number {
    return active(world).solverSetBodyCount(set);
}
export function setArrayCount(world: WorldState, set: number, kind: number): number {
    return active(world).solverSetArrayCount(set, kind);
}
export function setArrayGet(world: WorldState, set: number, kind: number, index: number): number {
    return active(world).solverSetArrayGet(set, kind, index);
}
/** Observation only; stepping reads indices at their local slot. */
export function setArraySnapshot(world: WorldState, set: number, kind: number): number[] {
    return Array.from({ length: setArrayCount(world, set, kind) }, (_, i) =>
        setArrayGet(world, set, kind, i),
    );
}
