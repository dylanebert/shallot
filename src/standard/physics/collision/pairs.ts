import { advanceStep, kernel, threads } from "../kernel/kernel";
import type { WorldState } from "../world/world";

/** Run the kernel's pair pass alone for lifecycle observations. */
export function updateBroadPhasePairs(world: WorldState): void {
    const k = kernel(world.ecsState);
    k.pairsBegin(world.worldId, threads(world.ecsState));
    if (advanceStep(world.ecsState) !== 0)
        throw new Error("physics: pair pass invoked a user callback");
    world.broadPhase.store.refreshIfStale();
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
}
