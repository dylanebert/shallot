import { kernel, kernelState, rethrowQueryError } from "../kernel/kernel";
import type { WorldState } from "../world/world";

/** Run user material callbacks at the kernel's post-collide serial point. */
export function mixContacts(world: WorldState): void {
    const state = kernelState(world.ecsState);
    state.materialWorld = world;
    try {
        kernel(world.ecsState).mixContacts(world.worldId);
        rethrowQueryError(world.ecsState);
    } finally {
        state.materialWorld = null;
    }
}
