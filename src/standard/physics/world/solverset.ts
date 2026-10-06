import { reclassifyBodyContacts } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import { kernel } from "../kernel/kernel";
import { syncBodyQuery } from "../kernel/shapecolumns";
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
    source: SolverSet,
    body: number,
): void {
    if (target === source) return;
    world.bodyStore.refreshViews();
    const moved = kernel(world.ecsState).bodyTransfer(world.worldId, body, target, true) | 0;
    if (moved !== NULL_INDEX && source === SetType.Awake) syncBodyQuery(world, moved);
    syncBodyQuery(world, body);
    reclassifyBodyContacts(world, body);
}
export function transferJoint(
    world: WorldState,
    target: SolverSet,
    source: SolverSet,
    joint: Joint,
): void {
    if (target === source) return;
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.jointTransfer(joint, target);
}
export function trySleepIsland(world: WorldState, id: number): void {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    k.solverSetTrySleepIsland(id);
}
