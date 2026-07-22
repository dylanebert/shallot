// Broad-phase — a port of Box3D's src/broad_phase.c (Erin Catto, MIT), the container over three
// dynamic trees (static / kinematic / dynamic) plus the move buffer that records which proxies
// changed this step, in deterministic insertion order.
//
// Stage-5 scope: the trees, proxy lifecycle, and the move buffer (bit-set-mirrors-array invariant).
// The pair-finding half — b3PairQueryCallback / b3FindPairsTask / b3UpdateBroadPhasePairs and the
// pairSet — is world-coupled (shapes, bodies, contacts, filters, sensors, joints) and ports at the
// lifecycle/solver stages (7/8) where those types exist, alongside contact creation. It composes
// from the tree Query already implemented here; building it now would need stub world types.

import { GrowVec } from "./array";
import { type BitSet, clearBit, createBitSet, getBit, setBitGrow } from "./bitset";
import { type BroadStore, createBroadStore } from "./broadcolumns";
import type { AABB } from "./math";
import { aabb } from "./math";
import { createSet, type HashSet } from "./table";
import type { DynamicTree } from "./tree";
import * as tree from "./tree";

// b3BodyType. Local for now; hoists to lifecycle at stage 7. Static must be 0 so the proxy-key
// pack/unpack (2-bit type) round-trips.
export const BodyType = {
    Static: 0,
    Kinematic: 1,
    Dynamic: 2,
} as const;
export type BodyTypeValue = (typeof BodyType)[keyof typeof BodyType];
const BODY_TYPE_COUNT = 3;

// Store the proxy type in the lower 2 bits of the key; the remaining bits hold the proxy id.
export const proxyType = (key: number): BodyTypeValue => (key & 3) as BodyTypeValue;
export const proxyId = (key: number): number => key >> 2;
export const proxyKey = (id: number, type: BodyTypeValue): number => (id << 2) | type;

export type BroadPhase = {
    trees: DynamicTree[];
    // Per body-type bit sets indexed by proxyId, marking proxies moved this step. Paired with
    // moveArray, which preserves deterministic insertion order for pair queries. TS-only (copied into a
    // kernel slab per step for the in-kernel query, 3d).
    movedProxies: BitSet[];
    moveArray: GrowVec<Int32Array>;
    // Hash set of active shape pairs (b3ShapePairKey), so a pair isn't turned into a second
    // contact. Written by contact create/destroy; read by pair finding (solver stage).
    pairSet: HashSet;
    // The resident broad-phase region's view manager: the trees + pairSet node/slot arrays live in the
    // kernel's linear memory, and this rewrites their views after any grow (broadcolumns.ts).
    store: BroadStore;
};

const maxInt = (a: number, b: number): number => (a > b ? a : b);

export function createBroadPhase(capacity: {
    staticShapeCount: number;
    dynamicShapeCount: number;
    contactCount?: number;
}): BroadPhase {
    const staticCapacity = maxInt(16, capacity.staticShapeCount);
    const dynamicCapacity = maxInt(16, capacity.dynamicShapeCount);

    // The trees + pairSet node/slot pools are kernel-resident (broadcolumns.ts); the store owns their
    // views and reservations. Register the trees + set on it after creating them so a grow can rewrite
    // every view in place. `store.world` is wired once the world is fully constructed (makeWorldState).
    const store = createBroadStore();

    const trees: DynamicTree[] = [];
    trees[BodyType.Static] = tree.createTree(staticCapacity, store, BodyType.Static);
    trees[BodyType.Kinematic] = tree.createTree(16, store, BodyType.Kinematic);
    trees[BodyType.Dynamic] = tree.createTree(dynamicCapacity, store, BodyType.Dynamic);
    store.trees = trees;

    const movedProxies: BitSet[] = [];
    movedProxies[BodyType.Static] = createBitSet(staticCapacity);
    movedProxies[BodyType.Kinematic] = createBitSet(16);
    movedProxies[BodyType.Dynamic] = createBitSet(dynamicCapacity);

    const moveArray = new GrowVec((n: number) => new Int32Array(n), capacity.dynamicShapeCount);

    const pairSet = createSet(2 * (capacity.contactCount ?? 0), store);
    store.set = pairSet;

    return { trees, movedProxies, moveArray, pairSet, store };
}

// This is what triggers new contact pairs to be created. Must be called in deterministic order.
export function bufferMove(bp: BroadPhase, queryProxy: number): void {
    const type = proxyType(queryProxy);
    const id = proxyId(queryProxy);
    const set = bp.movedProxies[type];
    if (getBit(set, id) === false) {
        setBitGrow(set, id);
        bp.moveArray.push(queryProxy);
    }
}

function unBufferMove(bp: BroadPhase, proxyKeyValue: number): void {
    const type = proxyType(proxyKeyValue);
    const id = proxyId(proxyKeyValue);
    const set = bp.movedProxies[type];

    if (getBit(set, id)) {
        clearBit(set, id);
        // Purge from move buffer. Linear search.
        const count = bp.moveArray.count;
        for (let i = 0; i < count; ++i) {
            if (bp.moveArray.get(i) === proxyKeyValue) {
                bp.moveArray.removeSwap(i);
                break;
            }
        }
    }
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
    const id = tree.createProxy(bp.trees[type], box, categoryHi, categoryLo, shapeIndex);
    const key = proxyKey(id, type);
    if (type !== BodyType.Static || forcePairCreation) {
        bufferMove(bp, key);
    }
    return key;
}

export function destroyProxy(bp: BroadPhase, key: number): void {
    bp.store.refreshIfStale();
    unBufferMove(bp, key);
    tree.destroyProxy(bp.trees[proxyType(key)], proxyId(key));
}

export function moveProxy(bp: BroadPhase, key: number, box: AABB): void {
    bp.store.refreshIfStale();
    tree.moveProxy(bp.trees[proxyType(key)], proxyId(key), box);
    bufferMove(bp, key);
}

export function enlargeProxy(bp: BroadPhase, key: number, box: AABB): void {
    const type = proxyType(key);
    if (type === BodyType.Static) throw new Error("broadphase: cannot enlarge a static proxy");
    bp.store.refreshIfStale();
    tree.enlargeProxy(bp.trees[type], proxyId(key), box);
    bufferMove(bp, key);
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

export function getShapeIndex(bp: BroadPhase, key: number): number {
    bp.store.refreshIfStale();
    return tree.getUserData(bp.trees[proxyType(key)], proxyId(key));
}

/** @returns whether a proxy is flagged as moved this step (b3GetBit on movedProxies). */
export function getMoved(bp: BroadPhase, type: BodyTypeValue, id: number): boolean {
    return getBit(bp.movedProxies[type], id);
}

/** Clear a proxy's moved flag (b3ClearBit on movedProxies). */
export function clearMoved(bp: BroadPhase, type: BodyTypeValue, id: number): void {
    clearBit(bp.movedProxies[type], id);
}

export function validate(bp: BroadPhase): void {
    tree.validate(bp.trees[BodyType.Dynamic]);
    tree.validate(bp.trees[BodyType.Kinematic]);
}

export function validateNoEnlarged(bp: BroadPhase): void {
    for (let j = 0; j < BODY_TYPE_COUNT; ++j) {
        tree.validateNoEnlarged(bp.trees[j]);
    }
}
