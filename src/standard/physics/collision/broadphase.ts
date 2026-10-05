import type { World } from "../../../engine";
// Broad-phase — a port of Box3D's src/broad_phase.c (Erin Catto, MIT), the container over three
// dynamic trees (static / kinematic / dynamic) plus the move buffer that records which proxies
// changed this step, in deterministic insertion order.

import type { AABB } from "../common/math";
import { type BroadStore, createBroadStore } from "../kernel/broadcolumns";
import { kernel } from "../kernel/kernel";
import type { DynamicTree } from "../kernel/treecolumns";
import * as tree from "../kernel/treecolumns";
import { createSet, type HashSet } from "./table";

// b3BodyType. Static must be 0 so the proxy-key pack/unpack (2-bit type) round-trips.
export const BodyType = {
    Static: 0,
    Kinematic: 1,
    Dynamic: 2,
} as const;
export type BodyTypeValue = (typeof BodyType)[keyof typeof BodyType];

// Store the proxy type in the lower 2 bits of the key; the remaining bits hold the proxy id.
export const proxyType = (key: number): BodyTypeValue => (key & 3) as BodyTypeValue;
export const proxyId = (key: number): number => key >> 2;
export const proxyKey = (id: number, type: BodyTypeValue): number => (id << 2) | type;

export type BroadPhase = {
    // Logical initialization belongs to the World, including when its snapshot predates a claim.
    initialization: BroadStore["initialization"];
    trees: DynamicTree[];
    // Hash set of active shape pairs (b3ShapePairKey), so a pair isn't turned into a second
    // contact. Written by contact create/destroy; read by pair finding (solver stage).
    pairSet: HashSet;
    // Re-derives tree and move views after a kernel memory grow.
    store: BroadStore;
    captureCheckpoint(): unknown;
    restoreCheckpoint(state: unknown): void;
};

export function moveCount(bp: BroadPhase): number {
    bp.store.refreshIfStale();
    return bp.store.moveState[0] ?? 0;
}
export function moveKey(bp: BroadPhase, index: number): number {
    bp.store.refreshIfStale();
    return bp.store.moveData[index];
}
export function clearMoves(bp: BroadPhase): void {
    bp.store.refreshIfStale();
    kernel(bp.store.ecsState).broadClearMoves();
}
export function isMoved(bp: BroadPhase, type: BodyTypeValue, id: number): boolean {
    bp.store.refreshIfStale();
    const bits = bp.store.movedBits[type];
    return ((bits[id >>> 5] ?? 0) & (1 << (id & 31))) !== 0;
}

const maxInt = (a: number, b: number): number => (a > b ? a : b);

export function createBroadPhase(
    world: World | undefined,
    capacity: {
        staticShapeCount: number;
        dynamicShapeCount: number;
        contactCount?: number;
    },
    worldId = 0,
): BroadPhase {
    const staticCapacity = maxInt(16, capacity.staticShapeCount);
    const dynamicCapacity = maxInt(16, capacity.dynamicShapeCount);

    // `store.world` is wired once the world is fully constructed.
    const store = createBroadStore(world, worldId);

    const trees: DynamicTree[] = [];
    trees[BodyType.Static] = tree.createTree(staticCapacity, store, BodyType.Static);
    trees[BodyType.Kinematic] = tree.createTree(16, store, BodyType.Kinematic);
    trees[BodyType.Dynamic] = tree.createTree(dynamicCapacity, store, BodyType.Dynamic);
    store.trees = trees;

    const pairSet = createSet(2 * (capacity.contactCount ?? 0), store);

    return {
        initialization: store.initialization,
        trees,
        pairSet,
        store,
        captureCheckpoint() {
            return {
                trees: this.trees.map((tree) => tree.captureCheckpoint()),
            };
        },
        restoreCheckpoint(state: unknown): void {
            const saved = state as { trees: unknown[] };
            for (let i = 0; i < this.trees.length; i++)
                this.trees[i].restoreCheckpoint(saved.trees[i]);
        },
    };
}

// This is what triggers new contact pairs to be created. Must be called in deterministic order.
export function bufferMove(bp: BroadPhase, queryProxy: number): void {
    bp.store.refreshIfStale();
    kernel(bp.store.ecsState).broadBufferMove(queryProxy);
}

export function createProxy(
    bp: BroadPhase,
    type: BodyTypeValue,
    box: AABB,
    categoryHi: number,
    categoryLo: number,
    shapeIndex: number,
    forcePairCreation: boolean,
): number {
    // The resident tree views may have been detached by a `memory.grow` since the last broad-phase op
    // (a sibling region reserve, or a shape/body create). Re-derive if so — O(1) when still fresh.
    bp.store.refreshIfStale();
    tree.reserveProxy(bp.trees[type]);
    return kernel(bp.store.ecsState).broadCreateProxy(
        type,
        box.lowerBound.x,
        box.lowerBound.y,
        box.lowerBound.z,
        box.upperBound.x,
        box.upperBound.y,
        box.upperBound.z,
        categoryHi,
        categoryLo,
        shapeIndex,
        Number(forcePairCreation),
    );
}

export function destroyProxy(bp: BroadPhase, key: number): void {
    bp.store.refreshIfStale();
    kernel(bp.store.ecsState).broadDestroyProxy(key);
}

export function moveProxy(bp: BroadPhase, key: number, box: AABB): void {
    bp.store.refreshIfStale();
    kernel(bp.store.ecsState).broadMoveProxy(
        key,
        box.lowerBound.x,
        box.lowerBound.y,
        box.lowerBound.z,
        box.upperBound.x,
        box.upperBound.y,
        box.upperBound.z,
    );
}

export function enlargeProxy(bp: BroadPhase, key: number, box: AABB): void {
    const type = proxyType(key);
    if (type === BodyType.Static) throw new Error("broadphase: cannot enlarge a static proxy");
    bp.store.refreshIfStale();
    kernel(bp.store.ecsState).broadEnlargeProxy(
        key,
        box.lowerBound.x,
        box.lowerBound.y,
        box.lowerBound.z,
        box.upperBound.x,
        box.upperBound.y,
        box.upperBound.z,
    );
}

export function testOverlap(bp: BroadPhase, keyA: number, keyB: number): boolean {
    bp.store.refreshIfStale();
    return kernel(bp.store.ecsState).broadTestOverlap(keyA, keyB) !== 0;
}

/** Clear a proxy's moved flag (b3ClearBit on movedProxies). */
export function clearMoved(bp: BroadPhase, type: BodyTypeValue, id: number): void {
    bp.store.refreshIfStale();
    kernel(bp.store.ecsState).broadClearMoved(type, id);
}
