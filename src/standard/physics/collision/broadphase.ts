import type { World } from "../../../engine";
// Broad-phase — a port of Box3D's src/broad_phase.c (Erin Catto, MIT), the container over three
// dynamic trees (static / kinematic / dynamic) plus the move buffer that records which proxies
// changed this step, in deterministic insertion order.

import { type BitSet, clearBit } from "../common/bitset";
import type { AABB } from "../common/math";
import { aabb } from "../common/math";
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
    // Resident move membership and insertion order, shared by user edits, refits and pair queries.
    movedProxies: BitSet[];
    moveArray: ResidentMoves;
    // Hash set of active shape pairs (b3ShapePairKey), so a pair isn't turned into a second
    // contact. Written by contact create/destroy; read by pair finding (solver stage).
    pairSet: HashSet;
    // The resident broad-phase region's view manager: the trees + pairSet node/slot arrays live in the
    // kernel's linear memory, and this rewrites their views after any grow (broadcolumns.ts).
    store: BroadStore;
};

class ResidentMoves {
    readonly store: BroadStore;
    constructor(store: BroadStore) {
        this.store = store;
    }
    get count(): number {
        this.store.refreshIfStale();
        return this.store.moveState[0] ?? 0;
    }
    get(index: number): number {
        this.store.refreshIfStale();
        return this.store.moveData[index];
    }
    clear(): void {
        if (this.count !== 0) kernel(this.store.ecsState).broadClearMoves();
    }
}
class ResidentBits implements BitSet {
    readonly store: BroadStore;
    readonly index: number;
    constructor(store: BroadStore, index: number) {
        this.store = store;
        this.index = index;
    }
    get bits(): Uint32Array {
        this.store.refreshIfStale();
        return this.store.movedBits[this.index];
    }
    get blockCount(): number {
        return this.bits.length;
    }
    get blockCapacity(): number {
        return this.bits.length;
    }
}

const maxInt = (a: number, b: number): number => (a > b ? a : b);

export function createBroadPhase(
    world: World | undefined,
    capacity: {
        staticShapeCount: number;
        dynamicShapeCount: number;
        contactCount?: number;
    },
): BroadPhase {
    const staticCapacity = maxInt(16, capacity.staticShapeCount);
    const dynamicCapacity = maxInt(16, capacity.dynamicShapeCount);

    // The trees + pairSet node/slot pools are kernel-resident (broadcolumns.ts); the store owns their
    // views and reservations. Register the trees + set on it after creating them so a grow can rewrite
    // every view in place. `store.world` is wired once the world is fully constructed (makeWorldState).
    const store = createBroadStore(world);

    const trees: DynamicTree[] = [];
    trees[BodyType.Static] = tree.createTree(staticCapacity, store, BodyType.Static);
    trees[BodyType.Kinematic] = tree.createTree(16, store, BodyType.Kinematic);
    trees[BodyType.Dynamic] = tree.createTree(dynamicCapacity, store, BodyType.Dynamic);
    store.trees = trees;

    const movedProxies: BitSet[] = [];
    movedProxies[BodyType.Static] = new ResidentBits(store, BodyType.Static);
    movedProxies[BodyType.Kinematic] = new ResidentBits(store, BodyType.Kinematic);
    movedProxies[BodyType.Dynamic] = new ResidentBits(store, BodyType.Dynamic);

    const moveArray = new ResidentMoves(store);

    const pairSet = createSet(2 * (capacity.contactCount ?? 0), store);
    store.set = pairSet;

    return {
        initialization: store.initialization,
        trees,
        movedProxies,
        moveArray,
        pairSet,
        store,
    };
}

// This is what triggers new contact pairs to be created. Must be called in deterministic order.
export function bufferMove(bp: BroadPhase, queryProxy: number): void {
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
    const id = tree.createProxy(
        bp.trees[type],
        box,
        categoryHi,
        categoryLo,
        shapeIndex,
        type !== BodyType.Static || forcePairCreation,
    );
    return proxyKey(id, type);
}

export function destroyProxy(bp: BroadPhase, key: number): void {
    bp.store.refreshIfStale();
    tree.destroyProxy(bp.trees[proxyType(key)], proxyId(key));
}

export function moveProxy(bp: BroadPhase, key: number, box: AABB): void {
    kernel(bp.store.ecsState).treeMoveProxy(
        proxyType(key),
        proxyId(key),
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
    tree.enlargeProxy(bp.trees[type], proxyId(key), box);
}

// Scratch the two proxy AABBs are read into (getAABBInto — the tree holds no live AABB to alias).
const overlapA: AABB = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };
const overlapB: AABB = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };

export function testOverlap(bp: BroadPhase, keyA: number, keyB: number): boolean {
    bp.store.refreshIfStale();
    tree.getAABBInto(bp.trees[proxyType(keyA)], proxyId(keyA), overlapA);
    tree.getAABBInto(bp.trees[proxyType(keyB)], proxyId(keyB), overlapB);
    return aabb.overlaps(overlapA, overlapB);
}

/** Clear a proxy's moved flag (b3ClearBit on movedProxies). */
export function clearMoved(bp: BroadPhase, type: BodyTypeValue, id: number): void {
    clearBit(bp.movedProxies[type], id);
}
