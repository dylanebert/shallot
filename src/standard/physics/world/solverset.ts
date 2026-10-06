import { kernel } from "../kernel/kernel";
import { releaseSolverSet } from "../kernel/solversetcolumns";
import type { Joint } from "../solver/joint";
import type { WorldState } from "./world";

export type SolverSet = number;
export function destroySolverSet(world: WorldState, set: number): void {
    releaseSolverSet(world, set);
}
export function wakeSolverSet(world: WorldState, set: number): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.solverSetWake(set);
}
export function transferBody(
    world: WorldState,
    target: SolverSet,
    _source: SolverSet,
    body: number,
): void {
    kernel(world.ecsState).bodyTransfer(world.worldId, body, target, true);
}
export function transferJoint(
    world: WorldState,
    target: SolverSet,
    _source: SolverSet,
    joint: Joint,
): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.jointTransfer(joint, target);
}
export function trySleepIsland(world: WorldState, id: number): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.solverSetTrySleepIsland(id);
}
