// Broad-phase pair finding — Box3D's b3UpdateBroadPhasePairs + b3PairQueryCallback (broad_phase.c,
// Erin Catto, MIT). Each moved proxy queries the trees for overlapping proxies; new, un-filtered,
// distinct-body pairs become contacts. Contact creation order is deterministic: proxies in
// move-buffer order, candidates within a proxy in LIFO discovery order (matching the C move-result
// list), so the assigned contact ids — and therefore the solver order — match the reference.
//
// The query DFS, moved-proxy dedup, pair-set-membership rejection, and the two tree rebuilds run
// in-kernel over the resident broad-phase region (kernel/src/pairwork.rs), which returns a candidate
// slab (steady-state ≈ empty). TS copies the move buffer + dynamic moved-bitset in, then applies the
// surviving filters (self-body / sensor / shouldShapesCollide / joint walk) over the returned
// candidates and creates the contacts. A found compound leaf stays on the TS path: the kernel emits a
// placeholder, and TS maps the query bounds into the compound's local frame and recurses its inner
// tree, each overlapping child a candidate with its child index. fround per the README.

import { intVec, NULL_INDEX } from "./array";
import { type Body, getBodyTransformQuick } from "./body";
import * as bp from "./broadphase";
import { type CompoundData, queryCompound } from "./compound";
import { createContact } from "./contact";
import { kernel } from "./kernel";
import { type AABB, aabb, vec3, xf } from "./math";
import { containsKey, ensureResident } from "./table";
import * as tree from "./tree";
import { BodyType, type FilterBits } from "./types";
import type { WorldState } from "./world";

/** @returns whether two shapes' filters allow a collision (b3ShouldShapesCollide). */
export function shouldShapesCollide(a: FilterBits, b: FilterBits): boolean {
    if (a.groupIndex === b.groupIndex && a.groupIndex !== 0) {
        return a.groupIndex > 0;
    }
    return (
        ((a.maskHi & b.categoryHi) | (a.maskLo & b.categoryLo)) !== 0 &&
        ((a.categoryHi & b.maskHi) | (a.categoryLo & b.maskLo)) !== 0
    );
}

// Whether two bodies may collide (b3ShouldBodiesCollide). At least one must be dynamic, and no joint
// connecting them may have collideConnected disabled. Walks the shorter of the two bodies' joint lists.
export function shouldBodiesCollide(world: WorldState, bodyA: Body, bodyB: Body): boolean {
    if (bodyA.type !== BodyType.Dynamic && bodyB.type !== BodyType.Dynamic) {
        return false;
    }

    let jointKey: number;
    let otherBodyId: number;
    if (bodyA.jointCount < bodyB.jointCount) {
        jointKey = bodyA.headJointKey;
        otherBodyId = bodyB.id;
    } else {
        jointKey = bodyB.headJointKey;
        otherBodyId = bodyA.id;
    }

    while (jointKey !== NULL_INDEX) {
        const jointId = jointKey >> 1;
        const edgeIndex = jointKey & 1;
        const otherEdgeIndex = edgeIndex ^ 1;

        const joint = world.joints[jointId];
        if (
            joint.collideConnected === false &&
            joint.edges[otherEdgeIndex].bodyId === otherBodyId
        ) {
            return false;
        }

        jointKey = joint.edges[edgeIndex].nextKey;
    }

    return true;
}

// The survivor slab: one flat entry per pair that passed every filter, appended in discovery order,
// with the moved proxy that found it delimiting a range. `survEnd[i]` is proxy i's exclusive end; its
// start is `survEnd[i - 1]` (0 for the first). Persistent scratch — pair finding runs once per step and
// is not re-entrant, so the buffers are reused rather than rebuilt.
const candShapeA = intVec();
const candShapeB = intVec();
const candChild = intVec();
const survEnd = intVec();

// Scratch the moved proxy's fat AABB is read into during compound expansion (getAABBInto — zero-alloc;
// the tree holds no live AABB object to alias).
const fatScratch: AABB = { lowerBound: vec3.zero(), upperBound: vec3.zero() };

// u32 slots per kernel candidate entry (flag, shapeA, shapeB) — mirrors CAND_STRIDE in pairwork.rs.
const CAND_STRIDE = 3;
// The candidate-slab capacity handed to the kernel; grows monotonically on overflow (a cold-step event
// only — steady state emits ≈0 entries). Persisted across steps to avoid re-growing.
let candCap = 256;

/** @returns whether shapes `a`/`b` pass every non-membership filter (self-body, sensor, category, joint). */
function filtersPass(world: WorldState, shapeA: number, shapeB: number): boolean {
    const sa = world.shapes[shapeA];
    const sb = world.shapes[shapeB];
    if (sa.bodyId === sb.bodyId) return false;
    if (sa.sensorIndex !== NULL_INDEX || sb.sensorIndex !== NULL_INDEX) return false;
    if (shouldShapesCollide(sa.filter, sb.filter) === false) return false;
    return shouldBodiesCollide(world, world.bodies[sa.bodyId], world.bodies[sb.bodyId]);
}

/**
 * Find new collision pairs, create contacts, rebuild the trees, and reset the move buffer. The query DFS,
 * moved-proxy dedup, pair-set-membership rejection, and the two tree rebuilds run in the kernel over the
 * resident broad-phase region (pairwork.rs); TS copies the move buffer + dynamic moved-bitset into the
 * kernel slab, applies the surviving filters (self-body / sensor / shouldShapesCollide / joint walk)
 * over the returned candidates — expanding any compound leaf against its inner tree here — and creates
 * the contacts in the exact enumeration order.
 */
export function updateBroadPhasePairs(world: WorldState): void {
    const broadPhase = world.broadPhase;
    const moveArray = broadPhase.moveArray;
    const moveCount = moveArray.count;
    if (moveCount === 0) {
        return;
    }

    // Re-derive the resident tree + pairSet views if a `memory.grow` since the last broad-phase op
    // detached them (a prior step's solve reserve, or a between-step create).
    broadPhase.store.refreshIfStale();
    // Reserve + zero the pair-set on this world's first use — the in-kernel `queryPairs` reads it for the
    // membership dedup, and the singleton region carries a prior world's stale hashes until it is zeroed
    // (in the pre-3d path a TS `containsKey`/`addKey` did this; the kernel query never reserves).
    ensureResident(broadPhase.pairSet);

    const k = kernel();
    const trees = broadPhase.trees;
    const movedDyn = broadPhase.movedProxies[BodyType.Dynamic];
    const movedWords = movedDyn.blockCount;
    // Rebuild-leaf scratch is sized to the larger of the two rebuilt trees (dynamic + kinematic).
    const maxProxy = Math.max(
        trees[BodyType.Dynamic].proxyCount,
        trees[BodyType.Kinematic].proxyCount,
        1,
    );

    // Copy the per-step inputs into the kernel slab and run the query, growing + re-running if the
    // candidate slab overflowed (a cold-step event; the query mutates neither the trees nor the pair-set,
    // so a re-run is side-effect-free). `reservePairs` may grow memory, so the slab views are re-derived
    // from the headers after each reserve.
    let entryCount = 0;
    for (;;) {
        k.reservePairs(moveCount, movedWords, candCap, maxProxy);
        const buf = k.memory.buffer;
        const state = new Uint32Array(buf, k.pairsStatePtr(), 3 * 4);
        for (let t = 0; t < 3; ++t) {
            const tr = trees[t];
            state[t * 4] = tr.root >>> 0;
            state[t * 4 + 1] = tr.nodeCount;
            state[t * 4 + 2] = tr.freeList >>> 0;
            state[t * 4 + 3] = tr.proxyCount;
        }
        const move = new Uint32Array(buf, k.pairsMovePtr(), moveCount);
        for (let i = 0; i < moveCount; ++i) move[i] = moveArray.get(i);
        if (movedWords > 0) {
            new Uint32Array(buf, k.pairsMovedPtr(), movedWords).set(
                movedDyn.bits.subarray(0, movedWords),
            );
        }

        // Pass the pair-set's logical capacity — the resident region is grow-only across worlds, so
        // `broadSetCap()` can exceed this world's table.
        entryCount = k.queryPairs(broadPhase.pairSet.capacity);
        if (entryCount <= candCap) break;
        candCap = entryCount + (entryCount >> 1);
    }

    // Copy the candidate slab out of linear memory before any pair-set grow (compound membership or a
    // later `createContact`) can relocate it — the slab lives at the solver base, above the pair-set, so
    // a `reserveBroad` grow shifts it out from under a view captured here.
    const buf = k.memory.buffer;
    const candEnd = new Uint32Array(buf, k.pairsCandEndPtr(), moveCount).slice();
    const cand = new Uint32Array(buf, k.pairsCandPtr(), entryCount * CAND_STRIDE).slice();

    // A `reservePairs` grow above detaches the tree views the compound expansion below reads.
    broadPhase.store.refreshIfStale();

    candShapeA.clear();
    candShapeB.clear();
    candChild.clear();
    survEnd.clear();

    // Phase 1 (TS half) — apply the surviving filters over the kernel's candidates, expanding compound
    // placeholders against their inner trees, in discovery order. All membership tests run against the
    // step-start pair-set (no contact is created until phase 3), matching the C's ordering.
    let entryStart = 0;
    for (let i = 0; i < moveCount; ++i) {
        const end = candEnd[i];
        const queryKey = moveArray.get(i);
        for (let e = entryStart; e < end; ++e) {
            const o = e * CAND_STRIDE;
            const flag = cand[o];
            const shapeA = cand[o + 1];
            const shapeB = cand[o + 2];
            if (flag === 0) {
                if (filtersPass(world, shapeA, shapeB)) {
                    candShapeA.push(shapeA);
                    candShapeB.push(shapeB);
                    candChild.push(0);
                }
            } else {
                // Compound placeholder: `shapeA` is the compound shape, `shapeB` the query shape. Map the
                // moved proxy's fat AABB into the compound's frame, walk its inner tree, and emit each
                // overlapping child that passes membership + filters. Dedup already ran in the kernel.
                const compoundShape = world.shapes[shapeA];
                const fatAABB = tree.getAABBInto(
                    trees[bp.proxyType(queryKey)],
                    bp.proxyId(queryKey),
                    fatScratch,
                );
                const compoundTransform = getBodyTransformQuick(
                    world,
                    world.bodies[compoundShape.bodyId],
                );
                const localAABB = aabb.transform(xf.invert(compoundTransform), fatAABB);
                queryCompound(
                    compoundShape.compound as CompoundData,
                    localAABB,
                    (childIndex: number): boolean => {
                        if (containsKey(broadPhase.pairSet, shapeA, shapeB, childIndex))
                            return true;
                        if (filtersPass(world, shapeA, shapeB)) {
                            candShapeA.push(shapeA);
                            candShapeB.push(shapeB);
                            candChild.push(childIndex);
                        }
                        return true;
                    },
                );
            }
        }
        survEnd.push(candShapeA.count);
        entryStart = end;
    }

    // Phase 2 — rebuild the dynamic + kinematic trees in the kernel, then fold each new root/count/free
    // list back into the TS tree structs (the pool bytes are resident; these scalars are TS-side).
    k.rebuildTrees();
    broadPhase.store.refreshIfStale();
    const rebuildOut = new Uint32Array(k.memory.buffer, k.pairsRebuildOutPtr(), 2 * 3);
    for (const [slot, type] of [
        [0, BodyType.Dynamic],
        [1, BodyType.Kinematic],
    ] as const) {
        const tr = trees[type];
        tr.root = rebuildOut[slot * 3] | 0;
        tr.nodeCount = rebuildOut[slot * 3 + 1];
        tr.freeList = rebuildOut[slot * 3 + 2] | 0;
    }

    // Phase 3 — create contacts in deterministic order (proxies in order; candidates LIFO, so each
    // proxy's range walks backward).
    let start = 0;
    for (let i = 0; i < moveCount; ++i) {
        const end = survEnd.get(i);
        for (let kk = end - 1; kk >= start; --kk) {
            createContact(
                world,
                world.shapes[candShapeA.get(kk)],
                world.shapes[candShapeB.get(kk)],
                candChild.get(kk),
            );
        }
        start = end;
    }

    // Phase 4 — reset the move buffer: clear only the bits that were set this step.
    for (let i = 0; i < moveArray.count; ++i) {
        const key = moveArray.get(i);
        bp.clearMoved(broadPhase, bp.proxyType(key), bp.proxyId(key));
    }
    moveArray.clear();
}
