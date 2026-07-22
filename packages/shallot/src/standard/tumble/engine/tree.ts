// Dynamic AABB tree — a from-scratch port of Box3D's src/dynamic_tree.c (Erin Catto, MIT).
//
// The tree stores proxies (leaf AABBs) and answers overlap / ray / box / closest queries. It is
// the broadphase's acceleration structure and is bit-exact against the C reference: every float
// operation is fround-disciplined and every scalar branch mirrors the DISABLE_SIMD build (never an
// intrinsic's NaN/±0 semantics). See the README.
//
// Faithful to the compiled config: B3_TREE_HEURISTIC == 0 (median split), so the SAH partition is
// not ported. b3DynamicTree_Save/Load (FILE* I/O, dev tooling) and b3DynamicTree_GetByteCount (a C
// blob-layout artifact) are dropped — no runtime meaning in TS.
//
// Layout: one flat node pool mirroring C's 48-byte `b3TreeNode` — 12 four-byte slots per node over a
// single ArrayBuffer, read through dual Float32Array (aabb) / Int32Array (everything else) views.
// A visit loads one contiguous record instead of dereferencing a 4-object graph. Slots per node:
//   [0..5] aabb lowerBound.xyz / upperBound.xyz (f32)
//   [6..7] categoryBits hi / lo (the C's u64, held as two u32 halves; the mask test runs per visit)
//   [8..9] child1 / child2 (internal) — slot 8 aliases userData (leaf, a plain shape index), under
//          the isLeaf read guard, mirroring the C's children/userData union
//   [10]   parent (allocated) / next (free) — the C's parent/next union
//   [11]   height (high 16 bits) | flags (low 16 bits), matching C's uint16 height + uint16 flags

import { ALL_BITS_HI, ALL_BITS_LO } from "./core";
import {
    type AABB,
    aabb,
    FLT_MAX,
    f32,
    maxf,
    minf,
    testBoundsRayOverlap,
    type Vec3,
    vec3,
} from "./math";

const NULL_INDEX = -1;

// b3TreeNodeFlags (low 16 bits of slot 11)
const ALLOCATED = 0x0001;
const ENLARGED = 0x0002;
const LEAF = 0x0004;

const STACK_SIZE = 1024; // B3_TREE_STACK_SIZE

// Four-byte slots per node (mirrors sizeof(b3TreeNode) / 4 == 48 / 4).
const STRIDE = 12;
// Rebuild build-stack fields per item (b3RebuildItem: nodeIndex, childCount, startIndex,
// splitIndex, endIndex).
const ITEM_STRIDE = 5;

/**
 * The kernel-resident node pool's grow hook. When set, the tree's `nf`/`ni` are views over the kernel's
 * broad-phase region (broadcolumns.ts) rather than a private ArrayBuffer, and `growTree` reserves the
 * region + rewrites the views. Absent (standalone trees, e.g. tests) → the tree owns a private buffer.
 */
export type TreeBacking = { growTree(treeIndex: number, nodeCapacity: number): void };

export type DynamicTree = {
    // Node pool: dual views over one ArrayBuffer. Reassigned on grow, so never cache across an
    // allocateNode call (the buffer is reallocated — the view-staleness trap). When `store` is set the
    // buffer is the kernel's shared linear memory, and `store.growTree` reassigns these views.
    nf: Float32Array;
    ni: Int32Array;
    root: number;
    nodeCount: number;
    nodeCapacity: number;
    proxyCount: number;
    freeList: number;
    // Kernel-resident backing (broadcolumns.ts), or null for a standalone private-buffer tree. Set →
    // `nf`/`ni` view the kernel region; a grow routes through `store.growTree` + rebuilds the free list.
    store: TreeBacking | null;
    // Which of the three broad-phase tree pools this is (0 static / 1 kinematic / 2 dynamic); only
    // meaningful when `store` is set.
    treeIndex: number;
    // The node capacity the first resident grow jumps to (`2 * proxyCapacity - 1`); the region is
    // reserved lazily, so `nodeCapacity` starts 0 and the first `allocateNode` sizes it to this.
    initNodeCapacity: number;
    // rebuild scratch (median-split path), the C's own flat shapes
    leafIndices: Int32Array;
    leafCenters: Float32Array; // 3 f32 per leaf
    rebuildCapacity: number;
    gatherStack: Int32Array; // rebuild DFS gather stack (C's `int stack[]`)
    buildStack: Int32Array; // rebuild buildTree item stack (ITEM_STRIDE ints per item)
};

export type TreeStats = { nodeVisits: number; leafVisits: number };

export type RayCastInput = { origin: Vec3; translation: Vec3; maxFraction: number };
export type BoxCastInput = { box: AABB; translation: Vec3; maxFraction: number };

// Query callbacks close over their context (idiomatic TS); the C void* context is dropped.
export type QueryCallback = (proxyId: number, userData: number) => boolean;
export type RayCastCallback = (input: RayCastInput, proxyId: number, userData: number) => number;
export type BoxCastCallback = (input: BoxCastInput, proxyId: number, userData: number) => number;
export type QueryClosestCallback = (
    minDistanceSqr: number,
    proxyId: number,
    userData: number,
) => number;

const maxInt = (a: number, b: number): number => (a > b ? a : b);

// --- flat-slot accessors --------------------------------------------------------------------
// Height in the high 16 bits, flags in the low 16 — so `flags |= bit` and the LEAF/ALLOCATED masks
// touch slot 11 directly (they live in the low half) while height reads/writes shift.

const heightOf = (ni: Int32Array, i: number): number => ni[i * STRIDE + 11] >>> 16;
function setHeight(ni: Int32Array, i: number, h: number): void {
    const n = i * STRIDE + 11;
    ni[n] = (ni[n] & 0xffff) | (h << 16);
}
const isLeaf = (ni: Int32Array, i: number): boolean => (ni[i * STRIDE + 11] & LEAF) !== 0;
const isAllocated = (ni: Int32Array, i: number): boolean => (ni[i * STRIDE + 11] & ALLOCATED) !== 0;

// aabb.perimeter (b3Perimeter) on node i's flat aabb — exact fround expression tree.
function nodePerimeter(nf: Float32Array, i: number): number {
    const n = i * STRIDE;
    const wx = f32(nf[n + 3] - nf[n]);
    const wy = f32(nf[n + 4] - nf[n + 1]);
    const wz = f32(nf[n + 5] - nf[n + 2]);
    return f32(2 * f32(f32(f32(wx * wz) + f32(wy * wx)) + f32(wz * wy)));
}

// Perimeter of aabb.union(node i, node j) without materializing the union (findBestSibling / rotate
// cost this for candidates it doesn't keep). union lower = min lowers, upper = max uppers.
function unionPerimeter(nf: Float32Array, i: number, j: number): number {
    const a = i * STRIDE;
    const b = j * STRIDE;
    const lx = minf(nf[a], nf[b]);
    const ly = minf(nf[a + 1], nf[b + 1]);
    const lz = minf(nf[a + 2], nf[b + 2]);
    const ux = maxf(nf[a + 3], nf[b + 3]);
    const uy = maxf(nf[a + 4], nf[b + 4]);
    const uz = maxf(nf[a + 5], nf[b + 5]);
    const wx = f32(ux - lx);
    const wy = f32(uy - ly);
    const wz = f32(uz - lz);
    return f32(2 * f32(f32(f32(wx * wz) + f32(wy * wx)) + f32(wz * wy)));
}

// aabb.union(node i, node j) written into node k's aabb slots (b3AABB_Union).
function unionInto(nf: Float32Array, i: number, j: number, k: number): void {
    const a = i * STRIDE;
    const b = j * STRIDE;
    const d = k * STRIDE;
    nf[d] = minf(nf[a], nf[b]);
    nf[d + 1] = minf(nf[a + 1], nf[b + 1]);
    nf[d + 2] = minf(nf[a + 2], nf[b + 2]);
    nf[d + 3] = maxf(nf[a + 3], nf[b + 3]);
    nf[d + 4] = maxf(nf[a + 4], nf[b + 4]);
    nf[d + 5] = maxf(nf[a + 5], nf[b + 5]);
}

// aabb.center (b3AABB_Center) of node i into `out`: 0.5 * (upper + lower).
function centerInto(nf: Float32Array, i: number, out: Vec3): void {
    const n = i * STRIDE;
    out.x = f32(0.5 * f32(nf[n + 3] + nf[n]));
    out.y = f32(0.5 * f32(nf[n + 4] + nf[n + 1]));
    out.z = f32(0.5 * f32(nf[n + 5] + nf[n + 2]));
}

// Write an AABB object's six floats into node i's aabb slots. Values are already f32-valued, so the
// Float32Array store-round is identity.
function writeAABB(nf: Float32Array, i: number, box: AABB): void {
    const n = i * STRIDE;
    nf[n] = box.lowerBound.x;
    nf[n + 1] = box.lowerBound.y;
    nf[n + 2] = box.lowerBound.z;
    nf[n + 3] = box.upperBound.x;
    nf[n + 4] = box.upperBound.y;
    nf[n + 5] = box.upperBound.z;
}

function readAABB(nf: Float32Array, i: number): AABB {
    const n = i * STRIDE;
    return {
        lowerBound: { x: nf[n], y: nf[n + 1], z: nf[n + 2] },
        upperBound: { x: nf[n + 3], y: nf[n + 4], z: nf[n + 5] },
    };
}

// aabb.enlarge (b3EnlargeAABB): grow node i's aabb in place to contain `box`; returns whether it changed.
function nodeEnlarge(nf: Float32Array, i: number, box: AABB): boolean {
    const n = i * STRIDE;
    let changed = false;
    if (box.lowerBound.x < nf[n]) {
        nf[n] = box.lowerBound.x;
        changed = true;
    }
    if (box.lowerBound.y < nf[n + 1]) {
        nf[n + 1] = box.lowerBound.y;
        changed = true;
    }
    if (box.lowerBound.z < nf[n + 2]) {
        nf[n + 2] = box.lowerBound.z;
        changed = true;
    }
    if (nf[n + 3] < box.upperBound.x) {
        nf[n + 3] = box.upperBound.x;
        changed = true;
    }
    if (nf[n + 4] < box.upperBound.y) {
        nf[n + 4] = box.upperBound.y;
        changed = true;
    }
    if (nf[n + 5] < box.upperBound.z) {
        nf[n + 5] = box.upperBound.z;
        changed = true;
    }
    return changed;
}

export function createTree(
    proxyCapacity: number,
    store: TreeBacking | null = null,
    treeIndex = 0,
): DynamicTree {
    const capacity = maxInt(proxyCapacity, 16);
    // maximum node count for a full binary tree is 2 * leafCount - 1
    const nodeCapacity = 2 * capacity - 1;

    // Resident tree: the node pool lives in the kernel's broad-phase region, reserved lazily on the
    // first allocateNode (which sizes it to initNodeCapacity). It starts with empty views + a null free
    // list, so the first alloc takes the grow path exactly as a full pool would.
    if (store !== null) {
        return {
            nf: new Float32Array(0),
            ni: new Int32Array(0),
            root: NULL_INDEX,
            nodeCount: 0,
            nodeCapacity: 0,
            proxyCount: 0,
            freeList: NULL_INDEX,
            store,
            treeIndex,
            initNodeCapacity: nodeCapacity,
            leafIndices: new Int32Array(0),
            leafCenters: new Float32Array(0),
            rebuildCapacity: 0,
            gatherStack: new Int32Array(STACK_SIZE),
            buildStack: new Int32Array(STACK_SIZE * ITEM_STRIDE),
        };
    }

    const buf = new ArrayBuffer(nodeCapacity * STRIDE * 4);
    const nf = new Float32Array(buf);
    const ni = new Int32Array(buf);
    // Fresh buffer is zeroed: only the free-list `next` chain needs setting (slot 10). aabb 0,
    // category 0, children/parent 0, height|flags 0 are the free-node defaults; resetToDefault
    // fills real values on allocate.
    for (let i = 0; i < nodeCapacity - 1; ++i) ni[i * STRIDE + 10] = i + 1;
    ni[(nodeCapacity - 1) * STRIDE + 10] = NULL_INDEX;

    return {
        nf,
        ni,
        root: NULL_INDEX,
        nodeCount: 0,
        nodeCapacity,
        proxyCount: 0,
        freeList: 0,
        store: null,
        treeIndex: 0,
        initNodeCapacity: nodeCapacity,
        leafIndices: new Int32Array(0),
        leafCenters: new Float32Array(0),
        rebuildCapacity: 0,
        gatherStack: new Int32Array(STACK_SIZE),
        buildStack: new Int32Array(STACK_SIZE * ITEM_STRIDE),
    };
}

// *node = b3_defaultTreeNode: fresh allocated leaf-less node.
function resetToDefault(nf: Float32Array, ni: Int32Array, i: number): void {
    const n = i * STRIDE;
    nf[n] = 0;
    nf[n + 1] = 0;
    nf[n + 2] = 0;
    nf[n + 3] = 0;
    nf[n + 4] = 0;
    nf[n + 5] = 0;
    ni[n + 6] = ALL_BITS_HI;
    ni[n + 7] = ALL_BITS_LO;
    ni[n + 8] = NULL_INDEX; // child1 / userData union
    ni[n + 9] = NULL_INDEX; // child2
    ni[n + 10] = NULL_INDEX; // parent / next union
    ni[n + 11] = ALLOCATED; // height 0, flags ALLOCATED
}

function allocateNode(tree: DynamicTree): number {
    if (tree.freeList === NULL_INDEX) {
        // The free list is empty. Grow the pool (to initNodeCapacity on the first resident reserve,
        // else 1.5x) and rebuild the free list from nodeCount. At this point nodeCount == the old
        // capacity (the pool is full), so the new free run is [nodeCount, nodeCapacity) either way.
        const oldCapacity = tree.nodeCapacity;
        tree.nodeCapacity =
            oldCapacity === 0 ? tree.initNodeCapacity : oldCapacity + (oldCapacity >> 1);

        if (tree.store !== null) {
            // Resident: reserve the kernel region (grow-only) and re-derive the views. The region is
            // base-anchored within the broad region, so a grow preserves this pool's existing bytes in
            // place (through a byte memmove — NaN category words survive); no copy is needed here.
            tree.store.growTree(tree.treeIndex, tree.nodeCapacity);
        } else {
            const buf = new ArrayBuffer(tree.nodeCapacity * STRIDE * 4);
            const ni2 = new Int32Array(buf);
            // Copy through the Int32 view: an exact 32-bit word copy preserves float slot bit patterns
            // too. Copying through the Float32 view would canonicalize NaN-valued category words
            // (0xffffffff reads as NaN) and corrupt them.
            ni2.set(tree.ni);
            tree.nf = new Float32Array(buf);
            tree.ni = ni2;
        }

        const ni = tree.ni;
        for (let i = tree.nodeCount; i < tree.nodeCapacity - 1; ++i) {
            ni[i * STRIDE + 10] = i + 1;
        }
        ni[(tree.nodeCapacity - 1) * STRIDE + 10] = NULL_INDEX;
        tree.freeList = tree.nodeCount;
    }

    const nodeIndex = tree.freeList;
    tree.freeList = tree.ni[nodeIndex * STRIDE + 10];
    resetToDefault(tree.nf, tree.ni, nodeIndex);
    tree.nodeCount += 1;
    return nodeIndex;
}

function freeNode(tree: DynamicTree, nodeId: number): void {
    const n = nodeId * STRIDE;
    tree.ni[n + 10] = tree.freeList; // next
    tree.ni[n + 11] &= ~0xffff; // flags = 0 (height untouched)
    tree.freeList = nodeId;
    tree.nodeCount -= 1;
}

// Scratch for findBestSibling's fallback node-distance branch (both children contain D).
const sibCenterD: Vec3 = { x: 0, y: 0, z: 0 };
const sibCenter1: Vec3 = { x: 0, y: 0, z: 0 };
const sibCenter2: Vec3 = { x: 0, y: 0, z: 0 };

// Greedy SAH sibling selection. See dynamic_tree.c for the case analysis. `leaf` is the node being
// inserted (its aabb is boxD).
function findBestSibling(tree: DynamicTree, leaf: number): number {
    const nf = tree.nf;
    const ni = tree.ni;
    centerInto(nf, leaf, sibCenterD);
    const areaD = nodePerimeter(nf, leaf);

    const rootIndex = tree.root;

    let areaBase = nodePerimeter(nf, rootIndex);
    let directCost = unionPerimeter(nf, rootIndex, leaf);
    let inheritedCost = 0;

    let bestSibling = rootIndex;
    let bestCost = directCost;

    let index = rootIndex;
    while (isLeaf(ni, index) === false) {
        const child1 = ni[index * STRIDE + 8];
        const child2 = ni[index * STRIDE + 9];

        const cost = f32(directCost + inheritedCost);
        if (cost < bestCost) {
            bestSibling = index;
            bestCost = cost;
        }

        inheritedCost = f32(inheritedCost + f32(directCost - areaBase));

        const leaf1 = isLeaf(ni, child1);
        const leaf2 = isLeaf(ni, child2);

        let lowerCost1 = FLT_MAX;
        const directCost1 = unionPerimeter(nf, child1, leaf);
        let area1 = 0;
        if (leaf1) {
            const cost1 = f32(directCost1 + inheritedCost);
            if (cost1 < bestCost) {
                bestSibling = child1;
                bestCost = cost1;
            }
        } else {
            area1 = nodePerimeter(nf, child1);
            lowerCost1 = f32(f32(inheritedCost + directCost1) + minf(f32(areaD - area1), 0));
        }

        let lowerCost2 = FLT_MAX;
        const directCost2 = unionPerimeter(nf, child2, leaf);
        let area2 = 0;
        if (leaf2) {
            const cost2 = f32(directCost2 + inheritedCost);
            if (cost2 < bestCost) {
                bestSibling = child2;
                bestCost = cost2;
            }
        } else {
            area2 = nodePerimeter(nf, child2);
            lowerCost2 = f32(f32(inheritedCost + directCost2) + minf(f32(areaD - area2), 0));
        }

        if (leaf1 && leaf2) break;

        if (bestCost <= lowerCost1 && bestCost <= lowerCost2) break;

        if (lowerCost1 === lowerCost2 && leaf1 === false) {
            // Both children fully contain D. Fall back to node distance.
            centerInto(nf, child1, sibCenter1);
            centerInto(nf, child2, sibCenter2);
            const d1 = vec3.sub(sibCenter1, sibCenterD);
            const d2 = vec3.sub(sibCenter2, sibCenterD);
            lowerCost1 = vec3.lengthSq(d1);
            lowerCost2 = vec3.lengthSq(d2);
        }

        if (lowerCost1 < lowerCost2 && leaf1 === false) {
            index = child1;
            areaBase = area1;
            directCost = directCost1;
        } else {
            index = child2;
            areaBase = area2;
            directCost = directCost2;
        }
    }

    return bestSibling;
}

const RotateType = {
    None: 0,
    BF: 1,
    BG: 2,
    CD: 3,
    CE: 4,
} as const;

// OR two nodes' category halves into a destination node (b3 category propagation).
function orCategory(ni: Int32Array, dst: number, a: number, b: number): void {
    const d = dst * STRIDE;
    const na = a * STRIDE;
    const nb = b * STRIDE;
    ni[d + 6] = (ni[na + 6] | ni[nb + 6]) >>> 0;
    ni[d + 7] = (ni[na + 7] | ni[nb + 7]) >>> 0;
}

// Propagate the ENLARGED flag: dst.flags |= (a.flags | b.flags) & ENLARGED. ENLARGED lives in the
// low 16 bits, so the mask extracts it cleanly across both slot-11 words.
function orEnlarged(ni: Int32Array, dst: number, a: number, b: number): void {
    ni[dst * STRIDE + 11] |= (ni[a * STRIDE + 11] | ni[b * STRIDE + 11]) & ENLARGED;
}

// Perform a rotation if node A is imbalanced. Ported case-for-case from dynamic_tree.c.
function rotateNodes(tree: DynamicTree, iA: number): void {
    const nf = tree.nf;
    const ni = tree.ni;
    if (isLeaf(ni, iA)) return;

    const iB = ni[iA * STRIDE + 8];
    const iC = ni[iA * STRIDE + 9];

    const isLeafB = isLeaf(ni, iB);
    const isLeafC = isLeaf(ni, iC);

    if (isLeafB && !isLeafC) {
        const iF = ni[iC * STRIDE + 8];
        const iG = ni[iC * STRIDE + 9];

        const costBase = nodePerimeter(nf, iC);
        const costBF = unionPerimeter(nf, iB, iG);
        const costBG = unionPerimeter(nf, iB, iF);

        if (costBase < costBF && costBase < costBG) return;

        if (costBF < costBG) {
            // Swap B and F
            ni[iA * STRIDE + 8] = iF;
            ni[iC * STRIDE + 8] = iB;
            ni[iB * STRIDE + 10] = iC;
            ni[iF * STRIDE + 10] = iA;
            unionInto(nf, iB, iG, iC);
            setHeight(ni, iC, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iG)));
            setHeight(ni, iA, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iF)));
            orCategory(ni, iC, iB, iG);
            orCategory(ni, iA, iC, iF);
            orEnlarged(ni, iC, iB, iG);
            orEnlarged(ni, iA, iC, iF);
        } else {
            // Swap B and G
            ni[iA * STRIDE + 8] = iG;
            ni[iC * STRIDE + 9] = iB;
            ni[iB * STRIDE + 10] = iC;
            ni[iG * STRIDE + 10] = iA;
            unionInto(nf, iB, iF, iC);
            setHeight(ni, iC, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iF)));
            setHeight(ni, iA, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iG)));
            orCategory(ni, iC, iB, iF);
            orCategory(ni, iA, iC, iG);
            orEnlarged(ni, iC, iB, iF);
            orEnlarged(ni, iA, iC, iG);
        }
    } else if (isLeafC && !isLeafB) {
        const iD = ni[iB * STRIDE + 8];
        const iE = ni[iB * STRIDE + 9];

        const costBase = nodePerimeter(nf, iB);
        const costCD = unionPerimeter(nf, iC, iE);
        const costCE = unionPerimeter(nf, iC, iD);

        if (costBase < costCD && costBase < costCE) return;

        if (costCD < costCE) {
            // Swap C and D
            ni[iA * STRIDE + 9] = iD;
            ni[iB * STRIDE + 8] = iC;
            ni[iC * STRIDE + 10] = iB;
            ni[iD * STRIDE + 10] = iA;
            unionInto(nf, iC, iE, iB);
            setHeight(ni, iB, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iE)));
            setHeight(ni, iA, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iD)));
            orCategory(ni, iB, iC, iE);
            orCategory(ni, iA, iB, iD);
            orEnlarged(ni, iB, iC, iE);
            orEnlarged(ni, iA, iB, iD);
        } else {
            // Swap C and E
            ni[iA * STRIDE + 9] = iE;
            ni[iB * STRIDE + 9] = iC;
            ni[iC * STRIDE + 10] = iB;
            ni[iE * STRIDE + 10] = iA;
            unionInto(nf, iC, iD, iB);
            setHeight(ni, iB, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iD)));
            setHeight(ni, iA, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iE)));
            orCategory(ni, iB, iC, iD);
            orCategory(ni, iA, iB, iE);
            orEnlarged(ni, iB, iC, iD);
            orEnlarged(ni, iA, iB, iE);
        }
    } else if (!isLeafB && !isLeafC) {
        const iD = ni[iB * STRIDE + 8];
        const iE = ni[iB * STRIDE + 9];
        const iF = ni[iC * STRIDE + 8];
        const iG = ni[iC * STRIDE + 9];

        const areaB = nodePerimeter(nf, iB);
        const areaC = nodePerimeter(nf, iC);
        const costBase = f32(areaB + areaC);
        let bestRotation: number = RotateType.None;
        let bestCost = costBase;

        const costBF = f32(areaB + unionPerimeter(nf, iB, iG));
        if (costBF < bestCost) {
            bestRotation = RotateType.BF;
            bestCost = costBF;
        }

        const costBG = f32(areaB + unionPerimeter(nf, iB, iF));
        if (costBG < bestCost) {
            bestRotation = RotateType.BG;
            bestCost = costBG;
        }

        const costCD = f32(areaC + unionPerimeter(nf, iC, iE));
        if (costCD < bestCost) {
            bestRotation = RotateType.CD;
            bestCost = costCD;
        }

        const costCE = f32(areaC + unionPerimeter(nf, iC, iD));
        if (costCE < bestCost) {
            bestRotation = RotateType.CE;
        }

        switch (bestRotation) {
            case RotateType.None:
                break;
            case RotateType.BF:
                ni[iA * STRIDE + 8] = iF;
                ni[iC * STRIDE + 8] = iB;
                ni[iB * STRIDE + 10] = iC;
                ni[iF * STRIDE + 10] = iA;
                unionInto(nf, iB, iG, iC);
                setHeight(ni, iC, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iG)));
                setHeight(ni, iA, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iF)));
                orCategory(ni, iC, iB, iG);
                orCategory(ni, iA, iC, iF);
                orEnlarged(ni, iC, iB, iG);
                orEnlarged(ni, iA, iC, iF);
                break;
            case RotateType.BG:
                ni[iA * STRIDE + 8] = iG;
                ni[iC * STRIDE + 9] = iB;
                ni[iB * STRIDE + 10] = iC;
                ni[iG * STRIDE + 10] = iA;
                unionInto(nf, iB, iF, iC);
                setHeight(ni, iC, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iF)));
                setHeight(ni, iA, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iG)));
                orCategory(ni, iC, iB, iF);
                orCategory(ni, iA, iC, iG);
                orEnlarged(ni, iC, iB, iF);
                orEnlarged(ni, iA, iC, iG);
                break;
            case RotateType.CD:
                ni[iA * STRIDE + 9] = iD;
                ni[iB * STRIDE + 8] = iC;
                ni[iC * STRIDE + 10] = iB;
                ni[iD * STRIDE + 10] = iA;
                unionInto(nf, iC, iE, iB);
                setHeight(ni, iB, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iE)));
                setHeight(ni, iA, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iD)));
                orCategory(ni, iB, iC, iE);
                orCategory(ni, iA, iB, iD);
                orEnlarged(ni, iB, iC, iE);
                orEnlarged(ni, iA, iB, iD);
                break;
            case RotateType.CE:
                ni[iA * STRIDE + 9] = iE;
                ni[iB * STRIDE + 9] = iC;
                ni[iC * STRIDE + 10] = iB;
                ni[iE * STRIDE + 10] = iA;
                unionInto(nf, iC, iD, iB);
                setHeight(ni, iB, 1 + maxInt(heightOf(ni, iC), heightOf(ni, iD)));
                setHeight(ni, iA, 1 + maxInt(heightOf(ni, iB), heightOf(ni, iE)));
                orCategory(ni, iB, iC, iD);
                orCategory(ni, iA, iB, iE);
                orEnlarged(ni, iB, iC, iD);
                orEnlarged(ni, iA, iB, iE);
                break;
        }
    }
}

function insertLeaf(tree: DynamicTree, leaf: number, shouldRotate: boolean): void {
    if (tree.root === NULL_INDEX) {
        tree.root = leaf;
        tree.ni[leaf * STRIDE + 10] = NULL_INDEX; // parent
        return;
    }

    // Stage 1: find the best sibling for this node
    const sibling = findBestSibling(tree, leaf);

    // Stage 2: create a new parent for the leaf and sibling
    const oldParent = tree.ni[sibling * STRIDE + 10];
    const newParent = allocateNode(tree);
    // allocateNode may have grown (reallocated) the pool — re-fetch the views.
    const nf = tree.nf;
    const ni = tree.ni;

    ni[newParent * STRIDE + 10] = oldParent; // parent
    unionInto(nf, leaf, sibling, newParent);
    orCategory(ni, newParent, leaf, sibling);
    setHeight(ni, newParent, heightOf(ni, sibling) + 1);
    // Internal node: slot 8 holds child1 (aliases the C's UINT64_MAX userData, never read).

    if (oldParent !== NULL_INDEX) {
        if (ni[oldParent * STRIDE + 8] === sibling) {
            ni[oldParent * STRIDE + 8] = newParent;
        } else {
            ni[oldParent * STRIDE + 9] = newParent;
        }
        ni[newParent * STRIDE + 8] = sibling;
        ni[newParent * STRIDE + 9] = leaf;
        ni[sibling * STRIDE + 10] = newParent;
        ni[leaf * STRIDE + 10] = newParent;
    } else {
        ni[newParent * STRIDE + 8] = sibling;
        ni[newParent * STRIDE + 9] = leaf;
        ni[sibling * STRIDE + 10] = newParent;
        ni[leaf * STRIDE + 10] = newParent;
        tree.root = newParent;
    }

    // Stage 3: walk back up fixing heights and AABBs
    let index = ni[leaf * STRIDE + 10];
    while (index !== NULL_INDEX) {
        const child1 = ni[index * STRIDE + 8];
        const child2 = ni[index * STRIDE + 9];

        unionInto(nf, child1, child2, index);
        orCategory(ni, index, child1, child2);
        setHeight(ni, index, 1 + maxInt(heightOf(ni, child1), heightOf(ni, child2)));
        orEnlarged(ni, index, child1, child2);

        if (shouldRotate) rotateNodes(tree, index);

        index = ni[index * STRIDE + 10];
    }
}

function removeLeaf(tree: DynamicTree, leaf: number): void {
    if (leaf === tree.root) {
        tree.root = NULL_INDEX;
        return;
    }

    const ni = tree.ni;
    const nf = tree.nf;
    const parent = ni[leaf * STRIDE + 10];
    const grandParent = ni[parent * STRIDE + 10];
    let sibling: number;
    if (ni[parent * STRIDE + 8] === leaf) {
        sibling = ni[parent * STRIDE + 9];
    } else {
        sibling = ni[parent * STRIDE + 8];
    }

    if (grandParent !== NULL_INDEX) {
        if (ni[grandParent * STRIDE + 8] === parent) {
            ni[grandParent * STRIDE + 8] = sibling;
        } else {
            ni[grandParent * STRIDE + 9] = sibling;
        }
        ni[sibling * STRIDE + 10] = grandParent;
        freeNode(tree, parent);

        let index = grandParent;
        while (index !== NULL_INDEX) {
            const child1 = ni[index * STRIDE + 8];
            const child2 = ni[index * STRIDE + 9];
            unionInto(nf, child1, child2, index);
            orCategory(ni, index, child1, child2);
            setHeight(ni, index, 1 + maxInt(heightOf(ni, child1), heightOf(ni, child2)));
            index = ni[index * STRIDE + 10];
        }
    } else {
        tree.root = sibling;
        ni[sibling * STRIDE + 10] = NULL_INDEX;
        freeNode(tree, parent);
    }
}

export function createProxy(
    tree: DynamicTree,
    box: AABB,
    categoryHi: number,
    categoryLo: number,
    userData: number,
): number {
    const proxyId = allocateNode(tree);
    const nf = tree.nf;
    const ni = tree.ni;
    const n = proxyId * STRIDE;
    writeAABB(nf, proxyId, box);
    ni[n + 8] = userData; // leaf: slot 8 is userData
    ni[n + 6] = categoryHi;
    ni[n + 7] = categoryLo;
    setHeight(ni, proxyId, 0);
    ni[n + 11] = (ni[n + 11] & ~0xffff) | (ALLOCATED | LEAF);

    insertLeaf(tree, proxyId, true);
    tree.proxyCount += 1;
    return proxyId;
}

export function destroyProxy(tree: DynamicTree, proxyId: number): void {
    removeLeaf(tree, proxyId);
    freeNode(tree, proxyId);
    tree.proxyCount -= 1;
}

export function getProxyCount(tree: DynamicTree): number {
    return tree.proxyCount;
}

export function moveProxy(tree: DynamicTree, proxyId: number, box: AABB): void {
    removeLeaf(tree, proxyId);
    writeAABB(tree.nf, proxyId, box);
    insertLeaf(tree, proxyId, false);
}

export function enlargeProxy(tree: DynamicTree, proxyId: number, box: AABB): void {
    const nf = tree.nf;
    const ni = tree.ni;
    writeAABB(nf, proxyId, box);

    let parentIndex = ni[proxyId * STRIDE + 10];
    while (parentIndex !== NULL_INDEX) {
        const changed = nodeEnlarge(nf, parentIndex, box);
        ni[parentIndex * STRIDE + 11] |= ENLARGED;
        const next = ni[parentIndex * STRIDE + 10];
        parentIndex = next;
        if (changed === false) break;
    }

    while (parentIndex !== NULL_INDEX) {
        if (ni[parentIndex * STRIDE + 11] & ENLARGED) break;
        ni[parentIndex * STRIDE + 11] |= ENLARGED;
        parentIndex = ni[parentIndex * STRIDE + 10];
    }
}

export function setCategoryBits(
    tree: DynamicTree,
    proxyId: number,
    categoryHi: number,
    categoryLo: number,
): void {
    const ni = tree.ni;
    ni[proxyId * STRIDE + 6] = categoryHi;
    ni[proxyId * STRIDE + 7] = categoryLo;

    let nodeIndex = ni[proxyId * STRIDE + 10];
    while (nodeIndex !== NULL_INDEX) {
        const child1 = ni[nodeIndex * STRIDE + 8];
        const child2 = ni[nodeIndex * STRIDE + 9];
        orCategory(ni, nodeIndex, child1, child2);
        nodeIndex = ni[nodeIndex * STRIDE + 10];
    }
}

export function getCategoryHi(tree: DynamicTree, proxyId: number): number {
    return tree.ni[proxyId * STRIDE + 6] >>> 0;
}

export function getCategoryLo(tree: DynamicTree, proxyId: number): number {
    return tree.ni[proxyId * STRIDE + 7] >>> 0;
}

export function getHeight(tree: DynamicTree): number {
    if (tree.root === NULL_INDEX) return 0;
    return heightOf(tree.ni, tree.root);
}

export function getAreaRatio(tree: DynamicTree): number {
    if (tree.root === NULL_INDEX) return 0;
    const nf = tree.nf;
    const ni = tree.ni;
    const rootArea = nodePerimeter(nf, tree.root);

    let totalArea = 0;
    for (let i = 0; i < tree.nodeCapacity; ++i) {
        if (isAllocated(ni, i) === false || isLeaf(ni, i) || i === tree.root) continue;
        totalArea = f32(totalArea + nodePerimeter(nf, i));
    }
    return f32(totalArea / rootArea);
}

export function getRootBounds(tree: DynamicTree): AABB {
    if (tree.root !== NULL_INDEX) return readAABB(tree.nf, tree.root);
    return { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };
}

/** Materialize proxy `proxyId`'s aabb (b3DynamicTree_GetAABB). Cold callers; hot paths use getAABBInto. */
export function getAABB(tree: DynamicTree, proxyId: number): AABB {
    return readAABB(tree.nf, proxyId);
}

/** Read proxy `proxyId`'s aabb into `out` (zero-alloc; hot broad-phase reads). Returns `out`. */
export function getAABBInto(tree: DynamicTree, proxyId: number, out: AABB): AABB {
    const nf = tree.nf;
    const n = proxyId * STRIDE;
    out.lowerBound.x = nf[n];
    out.lowerBound.y = nf[n + 1];
    out.lowerBound.z = nf[n + 2];
    out.upperBound.x = nf[n + 3];
    out.upperBound.y = nf[n + 4];
    out.upperBound.z = nf[n + 5];
    return out;
}

export function getUserData(tree: DynamicTree, proxyId: number): number {
    return tree.ni[proxyId * STRIDE + 8];
}

function bitMatch(
    categoryHi: number,
    categoryLo: number,
    maskHi: number,
    maskLo: number,
    requireAllBits: boolean,
): boolean {
    const hi = (categoryHi & maskHi) >>> 0;
    const lo = (categoryLo & maskLo) >>> 0;
    return requireAllBits ? hi === maskHi && lo === maskLo : hi !== 0 || lo !== 0;
}

// Query scratch, one context per nesting depth: the C's `int stack[B3_TREE_STACK_SIZE]` is a stack
// local, so a query costs no allocation. A query callback may itself query (a compound leaf recurses
// into the compound's inner tree), so the contexts pool by depth rather than being one singleton.
type QueryContext = { stack: Int32Array; stats: TreeStats };
const queryContexts: QueryContext[] = [];
const NO_VISITS: Readonly<TreeStats> = { nodeVisits: 0, leafVisits: 0 };
let queryDepth = 0;

/**
 * The returned stats are borrowed from the depth's scratch context — read them before the next query
 * at the same depth, or copy them out.
 */
export function query(
    tree: DynamicTree,
    box: AABB,
    maskHi: number,
    maskLo: number,
    requireAllBits: boolean,
    callback: QueryCallback,
): Readonly<TreeStats> {
    if (tree.nodeCount === 0) return NO_VISITS;

    const depth = queryDepth;
    let context = queryContexts[depth];
    if (context === undefined) {
        context = { stack: new Int32Array(STACK_SIZE), stats: { nodeVisits: 0, leafVisits: 0 } };
        queryContexts[depth] = context;
    }

    const result = context.stats;
    result.nodeVisits = 0;
    result.leafVisits = 0;

    // Hoist the query box + node pool views; the traversal loop touches only flat slots (no per-visit
    // allocation, no object-graph deref).
    const nf = tree.nf;
    const ni = tree.ni;
    const blx = box.lowerBound.x;
    const bly = box.lowerBound.y;
    const blz = box.lowerBound.z;
    const bhx = box.upperBound.x;
    const bhy = box.upperBound.y;
    const bhz = box.upperBound.z;

    const stack = context.stack;
    let stackCount = 0;
    stack[stackCount++] = tree.root;
    queryDepth = depth + 1;

    try {
        while (stackCount > 0) {
            const nodeId = stack[--stackCount];
            const n = nodeId * STRIDE;
            result.nodeVisits += 1;

            const hi = (ni[n + 6] & maskHi) >>> 0;
            const lo = (ni[n + 7] & maskLo) >>> 0;
            const match = requireAllBits ? hi === maskHi && lo === maskLo : hi !== 0 || lo !== 0;
            // Conjunction form, not C's disjoint-form b3AABB_Overlaps: equivalent for all
            // finite operands, and tree AABBs are always finite. On a NaN AABB (an upstream
            // contract violation) C would visit the node; this skips it.
            const overlaps =
                nf[n + 3] >= blx &&
                nf[n] <= bhx &&
                nf[n + 4] >= bly &&
                nf[n + 1] <= bhy &&
                nf[n + 5] >= blz &&
                nf[n + 2] <= bhz;

            if (match && overlaps) {
                if ((ni[n + 11] & LEAF) !== 0) {
                    const proceed = callback(nodeId, ni[n + 8]);
                    result.leafVisits += 1;
                    if (proceed === false) return result;
                } else if (stackCount < STACK_SIZE - 1) {
                    stack[stackCount++] = ni[n + 8];
                    stack[stackCount++] = ni[n + 9];
                }
            }
        }
    } finally {
        queryDepth = depth;
    }
    return result;
}

// Scratch for the cold cast/closest node reads (per-call, not per-visit — cast traversal is far
// colder than query; not gold-plated).
const castLo: Vec3 = { x: 0, y: 0, z: 0 };
const castHi: Vec3 = { x: 0, y: 0, z: 0 };
const castCenter1: Vec3 = { x: 0, y: 0, z: 0 };
const castCenter2: Vec3 = { x: 0, y: 0, z: 0 };
const castNodeAABB: AABB = { lowerBound: castLo, upperBound: castHi };

function loadBounds(nf: Float32Array, i: number, lo: Vec3, hi: Vec3): void {
    const n = i * STRIDE;
    lo.x = nf[n];
    lo.y = nf[n + 1];
    lo.z = nf[n + 2];
    hi.x = nf[n + 3];
    hi.y = nf[n + 4];
    hi.z = nf[n + 5];
}

function distanceToNodeSqr(nf: Float32Array, point: Vec3, i: number): number {
    loadBounds(nf, i, castLo, castHi);
    const r = vec3.sub(point, vec3.clamp(point, castLo, castHi));
    return vec3.dot(r, r);
}

export function queryClosest(
    tree: DynamicTree,
    point: Vec3,
    maskHi: number,
    maskLo: number,
    requireAllBits: boolean,
    callback: QueryClosestCallback,
    minDistanceSqr: number,
): { stats: TreeStats; minDistanceSqr: number } {
    const result: TreeStats = { nodeVisits: 0, leafVisits: 0 };
    if (tree.nodeCount === 0) return { stats: result, minDistanceSqr };

    const nf = tree.nf;
    const ni = tree.ni;
    let minSqr = minDistanceSqr;
    const stack: { nodeIndex: number; distanceToNodeSqr: number }[] = [
        {
            nodeIndex: tree.root,
            distanceToNodeSqr: distanceToNodeSqr(nf, point, tree.root),
        },
    ];

    while (stack.length > 0) {
        const item = stack.pop() as { nodeIndex: number; distanceToNodeSqr: number };
        const idx = item.nodeIndex;
        result.nodeVisits += 1;

        if (bitMatch(ni[idx * STRIDE + 6], ni[idx * STRIDE + 7], maskHi, maskLo, requireAllBits)) {
            if (item.distanceToNodeSqr < minSqr) {
                if (isLeaf(ni, idx)) {
                    const dd = callback(minSqr, idx, ni[idx * STRIDE + 8]);
                    if (dd < minSqr) minSqr = dd;
                    result.leafVisits += 1;
                } else if (stack.length < STACK_SIZE - 1) {
                    const child1 = ni[idx * STRIDE + 8];
                    const child2 = ni[idx * STRIDE + 9];
                    const item1 = {
                        nodeIndex: child1,
                        distanceToNodeSqr: distanceToNodeSqr(nf, point, child1),
                    };
                    const item2 = {
                        nodeIndex: child2,
                        distanceToNodeSqr: distanceToNodeSqr(nf, point, child2),
                    };
                    // Iterate the closest child first as we pop off the stack.
                    if (item2.distanceToNodeSqr < item1.distanceToNodeSqr) {
                        stack.push(item1);
                        stack.push(item2);
                    } else {
                        stack.push(item2);
                        stack.push(item1);
                    }
                }
            }
        }
    }

    return { stats: result, minDistanceSqr: minSqr };
}

export function rayCast(
    tree: DynamicTree,
    input: RayCastInput,
    maskHi: number,
    maskLo: number,
    requireAllBits: boolean,
    callback: RayCastCallback,
): TreeStats {
    const result: TreeStats = { nodeVisits: 0, leafVisits: 0 };
    if (tree.nodeCount === 0) return result;

    const nf = tree.nf;
    const ni = tree.ni;
    const p1 = input.origin;
    const d = input.translation;
    let maxFraction = input.maxFraction;

    let p2 = vec3.mulAdd(p1, maxFraction, d);
    const segmentAABB: AABB = { lowerBound: vec3.min(p1, p2), upperBound: vec3.max(p1, p2) };

    const stack: number[] = [tree.root];
    const subInput: RayCastInput = { origin: p1, translation: d, maxFraction };

    while (stack.length > 0) {
        const nodeId = stack.pop() as number;
        result.nodeVisits += 1;

        loadBounds(nf, nodeId, castLo, castHi);
        if (
            bitMatch(
                ni[nodeId * STRIDE + 6],
                ni[nodeId * STRIDE + 7],
                maskHi,
                maskLo,
                requireAllBits,
            ) === false ||
            aabb.overlaps(castNodeAABB, segmentAABB) === false
        ) {
            continue;
        }

        if (testBoundsRayOverlap(castLo, castHi, p1, d) === false) {
            continue;
        }

        if (isLeaf(ni, nodeId)) {
            subInput.maxFraction = maxFraction;
            const value = callback(subInput, nodeId, ni[nodeId * STRIDE + 8]);
            result.leafVisits += 1;

            if (value === 0) return result;

            if (value > 0 && value <= maxFraction) {
                maxFraction = value;
                p2 = vec3.mulAdd(p1, maxFraction, d);
                segmentAABB.lowerBound = vec3.min(p1, p2);
                segmentAABB.upperBound = vec3.max(p1, p2);
            }
        } else if (stack.length < STACK_SIZE - 1) {
            const child1 = ni[nodeId * STRIDE + 8];
            const child2 = ni[nodeId * STRIDE + 9];
            centerInto(nf, child1, castCenter1);
            centerInto(nf, child2, castCenter2);
            if (vec3.distanceSq(castCenter1, p1) < vec3.distanceSq(castCenter2, p1)) {
                stack.push(child2);
                stack.push(child1);
            } else {
                stack.push(child1);
                stack.push(child2);
            }
        }
    }

    return result;
}

export function boxCast(
    tree: DynamicTree,
    input: BoxCastInput,
    maskHi: number,
    maskLo: number,
    requireAllBits: boolean,
    callback: BoxCastCallback,
): TreeStats {
    const stats: TreeStats = { nodeVisits: 0, leafVisits: 0 };
    if (tree.nodeCount === 0) return stats;

    const nf = tree.nf;
    const ni = tree.ni;
    const originAABB = input.box;
    const p1 = aabb.center(originAABB);
    const extension = aabb.extents(originAABB);
    let maxFraction = input.maxFraction;

    let t = vec3.scale(maxFraction, input.translation);
    const totalAABB: AABB = {
        lowerBound: vec3.min(originAABB.lowerBound, vec3.add(originAABB.lowerBound, t)),
        upperBound: vec3.max(originAABB.upperBound, vec3.add(originAABB.upperBound, t)),
    };

    const subInput: BoxCastInput = { box: input.box, translation: input.translation, maxFraction };
    const stack: number[] = [tree.root];

    while (stack.length > 0) {
        const nodeId = stack.pop() as number;
        stats.nodeVisits += 1;

        loadBounds(nf, nodeId, castLo, castHi);
        if (
            bitMatch(
                ni[nodeId * STRIDE + 6],
                ni[nodeId * STRIDE + 7],
                maskHi,
                maskLo,
                requireAllBits,
            ) === false ||
            aabb.overlaps(castNodeAABB, totalAABB) === false
        ) {
            continue;
        }

        // radius extension is added to the node in this case
        const lower = vec3.sub(castLo, extension);
        const upper = vec3.add(castHi, extension);
        if (testBoundsRayOverlap(lower, upper, p1, input.translation) === false) {
            continue;
        }

        if (isLeaf(ni, nodeId)) {
            subInput.maxFraction = maxFraction;
            const value = callback(subInput, nodeId, ni[nodeId * STRIDE + 8]);
            stats.leafVisits += 1;

            if (value === 0) return stats;

            if (value > 0 && value < maxFraction) {
                maxFraction = value;
                t = vec3.scale(maxFraction, input.translation);
                totalAABB.lowerBound = vec3.min(
                    originAABB.lowerBound,
                    vec3.add(originAABB.lowerBound, t),
                );
                totalAABB.upperBound = vec3.max(
                    originAABB.upperBound,
                    vec3.add(originAABB.upperBound, t),
                );
            }
        } else if (stack.length < STACK_SIZE - 1) {
            const child1 = ni[nodeId * STRIDE + 8];
            const child2 = ni[nodeId * STRIDE + 9];
            centerInto(nf, child1, castCenter1);
            centerInto(nf, child2, castCenter2);
            if (vec3.distanceSq(castCenter1, p1) < vec3.distanceSq(castCenter2, p1)) {
                stack.push(child2);
                stack.push(child1);
            } else {
                stack.push(child1);
                stack.push(child2);
            }
        }
    }

    return stats;
}

// --- rebuild (median split, B3_TREE_HEURISTIC == 0) -----------------------------------------

// Median split of leaf centers along the longest axis, Hoare partition. Returns the left count.
// `indices` / `centers` are the tree's flat scratch, offset by `start` (centers stride 3).
function partitionMid(
    indices: Int32Array,
    centers: Float32Array,
    start: number,
    count: number,
): number {
    if (count <= 2) return (count / 2) | 0;

    let lx = centers[start * 3];
    let ly = centers[start * 3 + 1];
    let lz = centers[start * 3 + 2];
    let ux = lx;
    let uy = ly;
    let uz = lz;
    for (let i = 1; i < count; ++i) {
        const c = (start + i) * 3;
        lx = minf(lx, centers[c]);
        ly = minf(ly, centers[c + 1]);
        lz = minf(lz, centers[c + 2]);
        ux = maxf(ux, centers[c]);
        uy = maxf(uy, centers[c + 1]);
        uz = maxf(uz, centers[c + 2]);
    }

    const dx = f32(ux - lx);
    const dy = f32(uy - ly);
    const dz = f32(uz - lz);
    const cx = f32(0.5 * f32(lx + ux));
    const cy = f32(0.5 * f32(ly + uy));
    const cz = f32(0.5 * f32(lz + uz));

    let i1 = 0;
    let i2 = count;

    const swap = (a: number, b: number): void => {
        const ia = start + a;
        const ib = start + b;
        const ti = indices[ia];
        indices[ia] = indices[ib];
        indices[ib] = ti;
        for (let k = 0; k < 3; ++k) {
            const tc = centers[ia * 3 + k];
            centers[ia * 3 + k] = centers[ib * 3 + k];
            centers[ib * 3 + k] = tc;
        }
    };

    if (dx >= dy && dx >= dz) {
        const pivot = cx;
        while (i1 < i2) {
            while (i1 < i2 && centers[(start + i1) * 3] < pivot) i1 += 1;
            while (i1 < i2 && centers[(start + i2 - 1) * 3] >= pivot) i2 -= 1;
            if (i1 < i2) {
                swap(i1, i2 - 1);
                i1 += 1;
                i2 -= 1;
            }
        }
    } else if (dy >= dz) {
        const pivot = cy;
        while (i1 < i2) {
            while (i1 < i2 && centers[(start + i1) * 3 + 1] < pivot) i1 += 1;
            while (i1 < i2 && centers[(start + i2 - 1) * 3 + 1] >= pivot) i2 -= 1;
            if (i1 < i2) {
                swap(i1, i2 - 1);
                i1 += 1;
                i2 -= 1;
            }
        }
    } else {
        const pivot = cz;
        while (i1 < i2) {
            while (i1 < i2 && centers[(start + i1) * 3 + 2] < pivot) i1 += 1;
            while (i1 < i2 && centers[(start + i2 - 1) * 3 + 2] >= pivot) i2 -= 1;
            if (i1 < i2) {
                swap(i1, i2 - 1);
                i1 += 1;
                i2 -= 1;
            }
        }
    }

    if (i1 > 0 && i1 < count) return i1;
    return (count / 2) | 0;
}

function buildTree(tree: DynamicTree, leafCount: number): number {
    const leafIndices = tree.leafIndices;
    const leafCenters = tree.leafCenters;

    if (leafCount === 1) {
        tree.ni[leafIndices[0] * STRIDE + 10] = NULL_INDEX; // parent
        return leafIndices[0];
    }

    // Flat item stack: ITEM_STRIDE ints per item — [0]=nodeIndex, [1]=childCount, [2]=startIndex,
    // [3]=splitIndex, [4]=endIndex (b3RebuildItem).
    const bs = tree.buildStack;
    let top = 0;

    bs[0] = allocateNode(tree);
    bs[1] = -1;
    bs[2] = 0;
    bs[4] = leafCount;
    bs[3] = partitionMid(leafIndices, leafCenters, 0, leafCount);

    for (;;) {
        const base = top * ITEM_STRIDE;
        bs[base + 1] += 1;

        if (bs[base + 1] === 2) {
            if (top === 0) break;

            const parentBase = (top - 1) * ITEM_STRIDE;
            const parentNode = bs[parentBase];
            const nodeIndex = bs[base];
            if (bs[parentBase + 1] === 0) {
                tree.ni[parentNode * STRIDE + 8] = nodeIndex;
            } else {
                tree.ni[parentNode * STRIDE + 9] = nodeIndex;
            }

            const ni = tree.ni;
            const nf = tree.nf;
            ni[nodeIndex * STRIDE + 10] = parentNode; // parent

            const child1 = ni[nodeIndex * STRIDE + 8];
            const child2 = ni[nodeIndex * STRIDE + 9];
            unionInto(nf, child1, child2, nodeIndex);
            setHeight(ni, nodeIndex, 1 + maxInt(heightOf(ni, child1), heightOf(ni, child2)));
            orCategory(ni, nodeIndex, child1, child2);

            top -= 1;
        } else {
            let startIndex: number;
            let endIndex: number;
            if (bs[base + 1] === 0) {
                startIndex = bs[base + 2];
                endIndex = bs[base + 3];
            } else {
                startIndex = bs[base + 3];
                endIndex = bs[base + 4];
            }

            const count = endIndex - startIndex;

            if (count === 1) {
                const childIndex = leafIndices[startIndex];
                const nodeIndex = bs[base];
                if (bs[base + 1] === 0) {
                    tree.ni[nodeIndex * STRIDE + 8] = childIndex;
                } else {
                    tree.ni[nodeIndex * STRIDE + 9] = childIndex;
                }
                tree.ni[childIndex * STRIDE + 10] = nodeIndex; // parent
            } else {
                top += 1;
                const split = partitionMid(leafIndices, leafCenters, startIndex, count);
                const alloc = allocateNode(tree);
                // allocateNode may have grown the pool; the flat scratch (bs / leaf*) is independent
                // of it, and subsequent slot reads re-derive tree.ni/nf fresh, so only capture indices.
                const nb = top * ITEM_STRIDE;
                bs[nb] = alloc;
                bs[nb + 1] = -1;
                bs[nb + 2] = startIndex;
                bs[nb + 4] = endIndex;
                bs[nb + 3] = split + startIndex;
            }
        }
    }

    const rootIndex = bs[0];
    const ni = tree.ni;
    const nf = tree.nf;
    const child1 = ni[rootIndex * STRIDE + 8];
    const child2 = ni[rootIndex * STRIDE + 9];
    unionInto(nf, child1, child2, rootIndex);
    setHeight(ni, rootIndex, 1 + maxInt(heightOf(ni, child1), heightOf(ni, child2)));
    orCategory(ni, rootIndex, child1, child2);

    return rootIndex;
}

export function rebuild(tree: DynamicTree, fullBuild: boolean): number {
    const proxyCount = tree.proxyCount;
    if (proxyCount === 0) return 0;

    if (proxyCount > tree.rebuildCapacity) {
        const newCapacity = proxyCount + ((proxyCount / 2) | 0);
        tree.leafIndices = new Int32Array(newCapacity);
        tree.leafCenters = new Float32Array(newCapacity * 3);
        tree.rebuildCapacity = newCapacity;
    }

    let leafCount = 0;
    const gatherStack = tree.gatherStack;
    let gatherCount = 0;
    const ni = tree.ni;
    const nf = tree.nf;
    const leafIndices = tree.leafIndices;
    const leafCenters = tree.leafCenters;

    let nodeIndex = tree.root;

    // Gather grown proxies + un-grown internal nodes as rebuild leaves; free the grown internals.
    for (;;) {
        const n = nodeIndex * STRIDE;
        if (
            isLeaf(ni, nodeIndex) === true ||
            ((ni[n + 11] & ENLARGED) === 0 && fullBuild === false)
        ) {
            leafIndices[leafCount] = nodeIndex;
            leafCenters[leafCount * 3] = f32(0.5 * f32(nf[n + 3] + nf[n]));
            leafCenters[leafCount * 3 + 1] = f32(0.5 * f32(nf[n + 4] + nf[n + 1]));
            leafCenters[leafCount * 3 + 2] = f32(0.5 * f32(nf[n + 5] + nf[n + 2]));
            leafCount += 1;
            ni[n + 10] = NULL_INDEX; // parent
        } else {
            const doomedNodeIndex = nodeIndex;
            nodeIndex = ni[n + 8]; // child1
            if (gatherCount < STACK_SIZE) gatherStack[gatherCount++] = ni[n + 9]; // child2
            freeNode(tree, doomedNodeIndex);
            continue;
        }

        if (gatherCount === 0) break;
        nodeIndex = gatherStack[--gatherCount];
    }

    tree.root = buildTree(tree, leafCount);
    // C calls b3DynamicTree_Validate here, but only under B3_ENABLE_VALIDATION (off in the fixture
    // build this port targets). Validation is a caller-invoked debug tool, not part of the hot path.
    return leafCount;
}

// --- validation (B3_ENABLE_VALIDATION) ------------------------------------------------------

function computeHeightRecurse(tree: DynamicTree, nodeId: number): number {
    const ni = tree.ni;
    if (isLeaf(ni, nodeId)) return 0;
    const height1 = computeHeightRecurse(tree, ni[nodeId * STRIDE + 8]);
    const height2 = computeHeightRecurse(tree, ni[nodeId * STRIDE + 9]);
    return 1 + maxInt(height1, height2);
}

function validateStructure(tree: DynamicTree, index: number): void {
    if (index === NULL_INDEX) return;
    const ni = tree.ni;
    const n = index * STRIDE;
    if (index === tree.root && ni[n + 10] !== NULL_INDEX) {
        throw new Error("tree: root has a parent");
    }
    const flags = ni[n + 11] & 0xffff;
    if (flags !== 0 && (flags & ALLOCATED) === 0) {
        throw new Error(`tree: node ${index} has flags without allocated bit`);
    }
    if (isLeaf(ni, index)) {
        if (heightOf(ni, index) !== 0) throw new Error(`tree: leaf ${index} height != 0`);
        return;
    }
    const child1 = ni[n + 8];
    const child2 = ni[n + 9];
    if (ni[child1 * STRIDE + 10] !== index || ni[child2 * STRIDE + 10] !== index) {
        throw new Error(`tree: node ${index} child parent mismatch`);
    }
    if ((ni[child1 * STRIDE + 11] | ni[child2 * STRIDE + 11]) & ENLARGED) {
        if ((ni[n + 11] & ENLARGED) === 0) {
            throw new Error(`tree: node ${index} missing enlarged propagation`);
        }
    }
    validateStructure(tree, child1);
    validateStructure(tree, child2);
}

function validateMetrics(tree: DynamicTree, index: number): void {
    if (index === NULL_INDEX) return;
    const ni = tree.ni;
    if (isLeaf(ni, index)) {
        if (heightOf(ni, index) !== 0) throw new Error(`tree: leaf ${index} height != 0`);
        return;
    }
    const child1 = ni[index * STRIDE + 8];
    const child2 = ni[index * STRIDE + 9];
    const height = 1 + maxInt(heightOf(ni, child1), heightOf(ni, child2));
    if (heightOf(ni, index) !== height) throw new Error(`tree: node ${index} height mismatch`);
    const box = readAABB(tree.nf, index);
    if (
        !aabb.contains(box, readAABB(tree.nf, child1)) ||
        !aabb.contains(box, readAABB(tree.nf, child2))
    ) {
        throw new Error(`tree: node ${index} does not contain a child`);
    }
    const categoryHi = (ni[child1 * STRIDE + 6] | ni[child2 * STRIDE + 6]) >>> 0;
    const categoryLo = (ni[child1 * STRIDE + 7] | ni[child2 * STRIDE + 7]) >>> 0;
    if (
        ni[index * STRIDE + 6] >>> 0 !== categoryHi ||
        ni[index * STRIDE + 7] >>> 0 !== categoryLo
    ) {
        throw new Error(`tree: node ${index} category bits mismatch`);
    }
    validateMetrics(tree, child1);
    validateMetrics(tree, child2);
}

/** Assert tree invariants (structure, heights, containment, free-list count). Throws on violation. */
export function validate(tree: DynamicTree): void {
    if (tree.root === NULL_INDEX) return;

    validateStructure(tree, tree.root);
    validateMetrics(tree, tree.root);

    let freeCount = 0;
    let freeIndex = tree.freeList;
    while (freeIndex !== NULL_INDEX) {
        freeIndex = tree.ni[freeIndex * STRIDE + 10];
        freeCount += 1;
    }

    const height = getHeight(tree);
    const computedHeight = computeHeightRecurse(tree, tree.root);
    if (height !== computedHeight) throw new Error("tree: stored height != computed height");
    if (tree.nodeCount + freeCount !== tree.nodeCapacity) {
        throw new Error("tree: node count + free count != capacity");
    }
}

/** Assert no allocated node still carries the enlarged flag (post-rebuild invariant). */
export function validateNoEnlarged(tree: DynamicTree): void {
    const ni = tree.ni;
    for (let i = 0; i < tree.nodeCapacity; ++i) {
        const flags = ni[i * STRIDE + 11];
        if (flags & ALLOCATED && flags & ENLARGED) {
            throw new Error(`tree: allocated node ${i} still enlarged`);
        }
    }
}

/**
 * A snapshot of node `i`'s fields, mirroring the pre-flat `TreeNode` shape. Debug/read accessor for
 * tests — mints per call; the engine reads slots directly.
 */
export type NodeView = {
    aabb: AABB;
    categoryHi: number;
    categoryLo: number;
    child1: number;
    child2: number;
    userData: number;
    parent: number;
    next: number;
    height: number;
    flags: number;
};

export function readNode(tree: DynamicTree, i: number): NodeView {
    const ni = tree.ni;
    const n = i * STRIDE;
    return {
        aabb: readAABB(tree.nf, i),
        categoryHi: ni[n + 6] >>> 0,
        categoryLo: ni[n + 7] >>> 0,
        child1: ni[n + 8],
        child2: ni[n + 9],
        userData: ni[n + 8],
        parent: ni[n + 10],
        next: ni[n + 10],
        height: ni[n + 11] >>> 16,
        flags: ni[n + 11] & 0xffff,
    };
}

export { ALLOCATED, ENLARGED, LEAF, NULL_INDEX };
