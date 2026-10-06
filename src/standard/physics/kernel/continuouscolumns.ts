import type { WorldState } from "../world/world";
import { kernel, ParKind, runPool, workers } from "./kernel";

export function prepareContinuous(world: WorldState, count: number): void {
    const k = kernel(world.ecsState);
    world.shapeStore.refreshViews();
    world.broadPhase.store.refreshIfStale();
    const trees = world.broadPhase.trees;
    k.continuousRoots(trees[0].root, trees[1].root, trees[2].root, world.enableSleep);
    world.bodyStore.refreshContinuous(count);
}
/** Publish task sensor hits serially after the matching finalize or bullet sweep. */
export function consumeContinuous(world: WorldState, count: number, bullets: boolean): void {
    kernel(world.ecsState).sensorConsumeContinuous(world.worldId, count, bullets);
}
export function solveBullets(world: WorldState, count: number): void {
    const k = kernel(world.ecsState);
    const pool = workers(world.ecsState);
    const fork = k.parBuild(ParKind.Bullets, count, (pool?.size ?? 0) + 1, 0);
    if (fork && pool) runPool(world.ecsState, pool, k.runMt);
    else k.runMt();
    world.shapeStore.refreshViews();
    world.broadPhase.store.refreshIfStale();
    consumeContinuous(world, count, true);
    k.treeEnlargePass(count, 1);
}
