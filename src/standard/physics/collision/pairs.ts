import { BodyType } from "../common/types";
import { bodyType } from "../kernel/filtercolumns";
import { kernel, ParKind, runPool, workers } from "../kernel/kernel";
import type { Body } from "../world/body";
import type { WorldState } from "../world/world";
import { bodiesFiltered } from "./bodyfilter";
import { clearMoves, moveCount } from "./broadphase";
import { createContact } from "./contact";
import { ensureResident } from "./table";

export function shouldBodiesCollide(world: WorldState, a: Body, b: Body): boolean {
    return (
        (bodyType(world, a.id) === BodyType.Dynamic ||
            bodyType(world, b.id) === BodyType.Dynamic) &&
        !bodiesFiltered(world, a.id, b.id)
    );
}

let memory: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
function heap(buffer: ArrayBufferLike): Uint32Array {
    if (memory.buffer !== buffer) memory = new Uint32Array(buffer);
    return memory;
}

/** Join the kernel pair task, then create contacts from its per-proxy LIFO lists in place. */
export function updateBroadPhasePairs(world: WorldState): void {
    const broad = world.broadPhase;
    broad.store.refreshIfStale();
    const count = moveCount(broad);
    if (count === 0) return;
    ensureResident(broad.pairSet);
    const k = kernel(world.ecsState);
    for (;;) {
        k.reservePairs();
        const pool = workers(world.ecsState);
        const fork = k.parBuild(ParKind.Pairs, count, (pool?.size ?? 0) + 1, k.broadSetCap(), 0);
        if (fork && pool) runPool(world.ecsState, pool, k.runMt);
        else k.runMt();
        if (k.pairsOverflow() === 0) break;
    }
    k.rebuildTrees();
    broad.store.refreshIfStale();
    const heads = k.pairsCandEndPtr() >>> 2;
    const pairs = k.pairsCandPtr() >>> 2;
    for (let i = 0; i < count; ++i) {
        let entry = heap(k.memory.buffer)[heads + i];
        while (entry !== 0xffffffff) {
            const u = heap(k.memory.buffer);
            const o = pairs + entry * 4;
            const child = u[o];
            const a = u[o + 1];
            const b = u[o + 2];
            entry = u[o + 3];
            createContact(world, world.shapes[a], world.shapes[b], child);
        }
    }
    clearMoves(broad);
}
