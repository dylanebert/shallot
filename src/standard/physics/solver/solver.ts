import { mixContacts } from "../collision/collide";
import { kernel, kernelState, rethrowQueryError, runPool, workers } from "../kernel/kernel";
import type { WorldState } from "../world/world";

/** Drive the kernel's yielded tasks and user callbacks until b3World_Step completes. */
export function solve(world: WorldState): void {
    const k = kernel(world.ecsState);
    const pool = workers(world.ecsState);
    const state = kernelState(world.ecsState);
    state.collisionWorld = world;
    state.materialWorld = world;
    try {
        let code: number;
        while ((code = k.stepAdvance()) !== 0) {
            if (code === 1) {
                if (pool === null) throw new Error("physics: kernel yielded a task without a pool");
                runPool(world.ecsState, pool, k.runMt, true);
            } else if (code === 2) mixContacts(world);
            else throw new Error(`physics: unknown step yield ${code}`);
        }
        rethrowQueryError(world.ecsState);
    } finally {
        state.collisionWorld = null;
        state.materialWorld = null;
    }
}
