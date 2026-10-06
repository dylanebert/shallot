import { BodyFlags } from "../world/body";
import { recordSensorHit } from "../world/sensor";
import type { WorldState } from "../world/world";
import type { BodySimRef } from "./bodycolumns";
import { CONTINUOUS_STRIDE as STRIDE } from "./bodycolumns";
import { S2_BODY_ID, S2_FLAGS, SIM2_STRIDE } from "./columns";
import { kernel, ParKind, runPool, workers } from "./kernel";

export function prepareContinuous(world: WorldState, sims: BodySimRef[]): void {
    const k = kernel(world.ecsState);
    world.shapeStore.refreshViews();
    world.broadPhase.store.refreshIfStale();
    const trees = world.broadPhase.trees;
    k.continuousRoots(trees[0].root, trees[1].root, trees[2].root);
    world.bodyStore.refreshContinuous(sims.length);
    const out = world.bodyStore.continuousF;
    // `sims` is the awake set, whose sim `i` sits at column index `i`, as `finalizeBodies` reads it.
    const sim2U = world.bodyStore.sim2U;
    for (let i = 0; i < sims.length; i++) {
        const body = world.bodies[sim2U[i * SIM2_STRIDE + S2_BODY_ID]];
        out[i * STRIDE] =
            world.enableSleep && body.flags & BodyFlags.enableSleep ? body.sleepThreshold : -1;
    }
}
/** Publish task sensor hits serially after the matching finalize or bullet sweep. */
export function consumeContinuous(world: WorldState, count: number, bullets: boolean): void {
    world.bodyStore.refreshContinuous();
    const out = world.bodyStore.continuousU;
    const sim2 = world.bodyStore.sim2U;
    const mask = BodyFlags.isFast | BodyFlags.isBullet;
    const wanted = BodyFlags.isFast | (bullets ? BodyFlags.isBullet : 0);
    for (let i = 0; i < count; ++i) {
        if ((sim2[i * SIM2_STRIDE + S2_FLAGS] & mask) !== wanted) continue;
        const row = i * STRIDE;
        for (let n = 0; n < out[row + 1]; n++)
            recordSensorHit(world, out[row + 2 + n * 2], out[row + 3 + n * 2]);
    }
}
export function solveBullets(world: WorldState, sims: BodySimRef[]): void {
    const k = kernel(world.ecsState);
    const pool = workers(world.ecsState);
    const fork = k.parBuild(ParKind.Bullets, sims.length, (pool?.size ?? 0) + 1, 0);
    if (fork && pool) runPool(world.ecsState, pool, k.runMt);
    else k.runMt();
    world.shapeStore.refreshViews();
    world.broadPhase.store.refreshIfStale();
    consumeContinuous(world, sims.length, true);
    k.treeEnlargePass(sims.length, 1);
}
