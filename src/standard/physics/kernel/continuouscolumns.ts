import * as bp from "../collision/broadphase";
import { NULL_INDEX } from "../common/array";
import { BodyFlags, type BodySim } from "../world/body";
import { recordSensorHit } from "../world/sensor";
import type { WorldState } from "../world/world";
import { CONTINUOUS_STRIDE as STRIDE } from "./bodycolumns";
import { kernel, ParKind, runPool, workers } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

export function prepareContinuous(world: WorldState, sims: BodySim[]): void {
    const k = kernel(world.ecsState);
    world.shapeStore.refreshViews();
    world.broadPhase.store.refreshIfStale();
    const trees = world.broadPhase.trees;
    k.continuousRoots(trees[0].root, trees[1].root, trees[2].root);
    world.bodyStore.refreshContinuous(sims.length);
    const out = world.bodyStore.continuousF;
    for (let i = 0; i < sims.length; i++) {
        const body = world.bodies[sims[i].bodyId];
        out[i * STRIDE] =
            world.enableSleep && body.flags & BodyFlags.enableSleep ? body.sleepThreshold : -1;
    }
}
export function consumeContinuous(world: WorldState, sim: BodySim, index: number): void {
    world.bodyStore.refreshContinuous();
    const out = world.bodyStore.continuousU;
    const row = index * STRIDE;
    for (let n = 0; n < out[row + 1]; n++)
        recordSensorHit(world, out[row + 2 + n * 2], out[row + 3 + n * 2]);
    const f = world.shapeStore.shapeF;
    const u = world.shapeStore.shapeU;
    let id = world.bodies[sim.bodyId].headShapeId;
    while (id !== NULL_INDEX) {
        const shape = world.shapes[id];
        const o = id * SHAPE_STRIDE;
        const box = shape.aabb;
        box.lowerBound.x = f[o + 34];
        box.lowerBound.y = f[o + 35];
        box.lowerBound.z = f[o + 36];
        box.upperBound.x = f[o + 37];
        box.upperBound.y = f[o + 38];
        box.upperBound.z = f[o + 39];
        if (u[o + 15]) {
            const fat = world.shapeStore.fatF;
            const b = id * 6;
            const box = shape.fatAABB;
            box.lowerBound.x = fat[b];
            box.lowerBound.y = fat[b + 1];
            box.lowerBound.z = fat[b + 2];
            box.upperBound.x = fat[b + 3];
            box.upperBound.y = fat[b + 4];
            box.upperBound.z = fat[b + 5];
            shape.enlargedAABB = true;
        }
        id = shape.nextShapeId;
    }
}
export function enlargeFastProxies(world: WorldState, sim: BodySim): void {
    let id = world.bodies[sim.bodyId].headShapeId;
    while (id !== NULL_INDEX) {
        const shape = world.shapes[id];
        if (shape.enlargedAABB) {
            bp.enlargeProxy(world.broadPhase, shape.proxyKey, shape.fatAABB);
            shape.enlargedAABB = false;
        }
        id = shape.nextShapeId;
    }
}
export function bufferFastBulletMoves(world: WorldState, sim: BodySim): void {
    let id = world.bodies[sim.bodyId].headShapeId;
    while (id !== NULL_INDEX) {
        const shape = world.shapes[id];
        bp.bufferMove(world.broadPhase, shape.proxyKey);
        id = shape.nextShapeId;
    }
}
export function solveBullets(world: WorldState, sims: BodySim[]): void {
    const k = kernel(world.ecsState);
    const pool = workers(world.ecsState);
    const fork = k.parBuild(ParKind.Bullets, sims.length, (pool?.size ?? 0) + 1, 0, 0);
    if (fork && pool) runPool(world.ecsState, pool, k.runMt);
    else k.runMt();
    world.shapeStore.refreshViews();
    world.broadPhase.store.refreshIfStale();
    bp.beginEnlargePass(world.broadPhase);
    for (let i = 0; i < sims.length; i++) {
        const sim = sims[i];
        if (
            (sim.flags & (BodyFlags.isFast | BodyFlags.isBullet)) !==
            (BodyFlags.isFast | BodyFlags.isBullet)
        )
            continue;
        consumeContinuous(world, sim, i);
        if (!(sim.flags & BodyFlags.enlargeBounds)) continue;
        sim.flags &= ~BodyFlags.enlargeBounds;
        let id = world.bodies[sim.bodyId].headShapeId;
        while (id !== NULL_INDEX) {
            const shape = world.shapes[id];
            if (shape.enlargedAABB) {
                shape.enlargedAABB = false;
                bp.queueEnlargement(world.broadPhase, shape.proxyKey, shape.fatAABB);
            }
            id = shape.nextShapeId;
        }
    }
    bp.finishEnlargePass(world.broadPhase);
}
