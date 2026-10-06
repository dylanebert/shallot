import { kernel, runPool, threads, workers } from "../kernel/kernel";
import type { WorldState } from "../world/world";

/** Run the kernel's pair pass alone for lifecycle observations. */
export function updateBroadPhasePairs(world: WorldState): void {
    const k = kernel(world.ecsState);
    const pool = workers(world.ecsState);
    k.pairsBegin(world.worldId, threads(world.ecsState));
    while (k.stepAdvance() !== 0) {
        if (pool === null) throw new Error("physics: kernel yielded pairs without a pool");
        runPool(world.ecsState, pool, k.runMt);
    }
    world.broadPhase.store.refreshIfStale();
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
}
