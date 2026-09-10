import { describe, expect, test } from "bun:test";
import { hi32, lo32 } from "./bits";
import { ALL_BITS_HI, ALL_BITS_LO } from "./core";
import { type AABB, FLT_MAX, type Vec3, vec3 } from "./math";
import {
    type BoxCastInput,
    boxCast,
    createProxy,
    createTree,
    type DynamicTree,
    destroyProxy,
    enlargeProxy,
    getAABB,
    getCategoryHi,
    getCategoryLo,
    getHeight,
    getProxyCount,
    getRootBounds,
    moveProxy,
    NULL_INDEX,
    query,
    queryClosest,
    type RayCastInput,
    rayCast,
    readNode,
    rebuild,
    setCategoryBits,
    validate,
    validateNoEnlarged,
} from "./tree";
import gold from "./tree.gold.json";

const u64 = (hi: number, lo: number): bigint => (BigInt(hi) << 32n) | BigInt(lo);

const dv = new DataView(new ArrayBuffer(4));
function fromBits(hex: string): number {
    dv.setUint32(0, Number.parseInt(hex, 16));
    return dv.getFloat32(0);
}
function bits(f: number): string {
    dv.setFloat32(0, f);
    return dv.getUint32(0).toString(16).padStart(8, "0");
}
function bitEqual(got: number, want: string, label: string) {
    const w = fromBits(want);
    if (!Object.is(got, w)) {
        throw new Error(`${label}: got 0x${bits(got)} (${got}), want ${want} (${w})`);
    }
}
const vecFromHex = (a: string[]): Vec3 => ({
    x: fromBits(a[0]),
    y: fromBits(a[1]),
    z: fromBits(a[2]),
});
const aabbFromHex = (o: { lo: string[]; hi: string[] }): AABB => ({
    lowerBound: vecFromHex(o.lo),
    upperBound: vecFromHex(o.hi),
});

function assertAABB(got: AABB, want: { lo: string[]; hi: string[] }, label: string) {
    bitEqual(got.lowerBound.x, want.lo[0], `${label}.lo.x`);
    bitEqual(got.lowerBound.y, want.lo[1], `${label}.lo.y`);
    bitEqual(got.lowerBound.z, want.lo[2], `${label}.lo.z`);
    bitEqual(got.upperBound.x, want.hi[0], `${label}.hi.x`);
    bitEqual(got.upperBound.y, want.hi[1], `${label}.hi.y`);
    bitEqual(got.upperBound.z, want.hi[2], `${label}.hi.z`);
}

// Gold op records are a heterogeneous JSON stream; each handler reads the fields it needs.
type Op = any;

function freeChain(tree: DynamicTree): number[] {
    const chain: number[] = [];
    let i = tree.freeList;
    while (i !== NULL_INDEX) {
        chain.push(i);
        i = readNode(tree, i).next;
    }
    return chain;
}

function assertCheckpoint(tree: DynamicTree, op: Op) {
    const t = op.tree;
    const name = op.name;
    expect(tree.root, `${name}.root`).toBe(t.root);
    expect(tree.nodeCount, `${name}.nodeCount`).toBe(t.nodeCount);
    expect(tree.nodeCapacity, `${name}.nodeCapacity`).toBe(t.nodeCapacity);
    expect(tree.proxyCount, `${name}.proxyCount`).toBe(t.proxyCount);
    expect(tree.freeList, `${name}.freeList`).toBe(t.freeList);
    expect(freeChain(tree), `${name}.freeChain`).toEqual(t.freeChain);

    // Allocated nodes, in ascending index order (matches the emitter).
    const allocated: number[] = [];
    for (let i = 0; i < tree.nodeCapacity; ++i) {
        if (readNode(tree, i).flags & 0x0001) allocated.push(i);
    }
    expect(
        allocated.length,
        `${name}: allocated node count (${allocated.map(String).join(",")})`,
    ).toBe(t.nodes.length);

    for (let k = 0; k < t.nodes.length; ++k) {
        const want = t.nodes[k];
        const i = want.i;
        const node = readNode(tree, i);
        const leaf = (node.flags & 0x0004) !== 0;
        const label = `${name}.node[${i}]`;
        expect(i, `${label}: index`).toBe(allocated[k]);
        expect(leaf, `${label}.leaf`).toBe(want.leaf);
        expect(node.flags, `${label}.flags`).toBe(want.flags);
        expect(node.height, `${label}.height`).toBe(want.height);
        expect(node.parent, `${label}.parent`).toBe(want.parent);
        expect(u64(node.categoryHi, node.categoryLo).toString(), `${label}.category`).toBe(
            want.category,
        );
        assertAABB(node.aabb, want.aabb, label);
        if (leaf) {
            expect(String(node.userData), `${label}.userData`).toBe(want.userData);
        } else {
            expect(node.child1, `${label}.child1`).toBe(want.child1);
            expect(node.child2, `${label}.child2`).toBe(want.child2);
        }
    }
}

function runQuery(tree: DynamicTree, op: Op) {
    const hits: number[] = [];
    const stats = query(
        tree,
        aabbFromHex(op.aabb),
        hi32(BigInt(op.mask)),
        lo32(BigInt(op.mask)),
        op.requireAll,
        (proxyId) => {
            hits.push(proxyId);
            return true;
        },
    );
    expect(hits, "query.hits").toEqual(op.hits);
    expect(stats.nodeVisits, "query.nodeVisits").toBe(op.nodeVisits);
    expect(stats.leafVisits, "query.leafVisits").toBe(op.leafVisits);
}

function runRayCast(tree: DynamicTree, op: Op) {
    const hits: number[] = [];
    const shrink = fromBits(op.shrink);
    const input: RayCastInput = {
        origin: vecFromHex(op.origin),
        translation: vecFromHex(op.translation),
        maxFraction: fromBits(op.maxFraction),
    };
    const stats = rayCast(
        tree,
        input,
        hi32(BigInt(op.mask)),
        lo32(BigInt(op.mask)),
        op.requireAll,
        (sub: RayCastInput, proxyId: number) => {
            hits.push(proxyId);
            return shrink < 0 ? sub.maxFraction : shrink;
        },
    );
    expect(hits, "raycast.hits").toEqual(op.hits);
    expect(stats.nodeVisits, "raycast.nodeVisits").toBe(op.nodeVisits);
    expect(stats.leafVisits, "raycast.leafVisits").toBe(op.leafVisits);
}

function runBoxCast(tree: DynamicTree, op: Op) {
    const hits: number[] = [];
    const input: BoxCastInput = {
        box: aabbFromHex(op.box),
        translation: vecFromHex(op.translation),
        maxFraction: fromBits(op.maxFraction),
    };
    const stats = boxCast(
        tree,
        input,
        hi32(BigInt(op.mask)),
        lo32(BigInt(op.mask)),
        op.requireAll,
        (sub: BoxCastInput, proxyId: number) => {
            hits.push(proxyId);
            return sub.maxFraction;
        },
    );
    expect(hits, "boxcast.hits").toEqual(op.hits);
    expect(stats.nodeVisits, "boxcast.nodeVisits").toBe(op.nodeVisits);
    expect(stats.leafVisits, "boxcast.leafVisits").toBe(op.leafVisits);
}

function runClosest(tree: DynamicTree, op: Op) {
    const hits: number[] = [];
    const point = vecFromHex(op.point);
    const { stats, minDistanceSqr } = queryClosest(
        tree,
        point,
        hi32(BigInt(op.mask)),
        lo32(BigInt(op.mask)),
        op.requireAll,
        (minSqr: number, proxyId: number) => {
            hits.push(proxyId);
            if (op.shrink === 0) return minSqr;
            // Mirror the emitter's b3DistanceToBoxSqr: |point - clamp(point, box)|².
            const box = getAABB(tree, proxyId);
            const r = vec3.sub(point, vec3.clamp(point, box.lowerBound, box.upperBound));
            return vec3.dot(r, r);
        },
        FLT_MAX, // matches the emitter's `float minDistanceSqr = FLT_MAX`
    );
    expect(hits, "closest.hits").toEqual(op.hits);
    bitEqual(minDistanceSqr, op.minDistanceSqr, "closest.minDistanceSqr");
    expect(stats.nodeVisits, "closest.nodeVisits").toBe(op.nodeVisits);
    expect(stats.leafVisits, "closest.leafVisits").toBe(op.leafVisits);
}

describe("dynamic tree — bit-exact op-stream replay vs C", () => {
    test("replay tree.gold.json", () => {
        const tree = createTree(gold.proxyCapacity);
        const handles: number[] = [];

        for (const op of gold.ops as Op[]) {
            switch (op.op) {
                case "create":
                    handles.push(
                        createProxy(
                            tree,
                            aabbFromHex(op.aabb),
                            hi32(BigInt(op.category)),
                            lo32(BigInt(op.category)),
                            Number(op.userData),
                        ),
                    );
                    break;
                case "move":
                    moveProxy(tree, handles[op.handle], aabbFromHex(op.aabb));
                    break;
                case "enlarge":
                    enlargeProxy(tree, handles[op.handle], aabbFromHex(op.aabb));
                    break;
                case "destroy":
                    destroyProxy(tree, handles[op.handle]);
                    break;
                case "rebuild":
                    rebuild(tree, op.full);
                    break;
                case "checkpoint":
                    assertCheckpoint(tree, op);
                    break;
                case "query":
                    runQuery(tree, op);
                    break;
                case "raycast":
                    runRayCast(tree, op);
                    break;
                case "boxcast":
                    runBoxCast(tree, op);
                    break;
                case "closest":
                    runClosest(tree, op);
                    break;
                default:
                    throw new Error(`unknown op ${op.op}`);
            }
        }

        // Structural invariants hold after the full stream.
        validate(tree);
        validateNoEnlarged(tree);
    });
});

describe("dynamic tree — behavior", () => {
    test("empty tree queries return nothing", () => {
        const tree = createTree(16);
        expect(getHeight(tree)).toBe(0);
        const stats = query(
            tree,
            { lowerBound: { x: -1, y: -1, z: -1 }, upperBound: { x: 1, y: 1, z: 1 } },
            ALL_BITS_HI,
            ALL_BITS_LO,
            false,
            () => true,
        );
        expect(stats.nodeVisits).toBe(0);
        expect(stats.leafVisits).toBe(0);
    });

    test("single proxy: root is the leaf, height 0", () => {
        const tree = createTree(16);
        const box: AABB = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 1, y: 1, z: 1 } };
        const id = createProxy(tree, box, ALL_BITS_HI, ALL_BITS_LO, 7);
        expect(tree.root).toBe(id);
        expect(getHeight(tree)).toBe(0);
        expect(tree.proxyCount).toBe(1);
        validate(tree);
    });

    test("destroy-all empties the tree and validates", () => {
        const tree = createTree(16);
        const ids: number[] = [];
        for (let i = 0; i < 10; ++i) {
            ids.push(
                createProxy(
                    tree,
                    {
                        lowerBound: { x: i, y: 0, z: 0 },
                        upperBound: { x: i + 1, y: 1, z: 1 },
                    },
                    ALL_BITS_HI,
                    ALL_BITS_LO,
                    i,
                ),
            );
        }
        validate(tree);
        for (const id of ids) destroyProxy(tree, id);
        expect(tree.root).toBe(NULL_INDEX);
        expect(tree.proxyCount).toBe(0);
    });

    test("setCategoryBits propagates the OR up to the root", () => {
        const tree = createTree(16);
        const box = (x: number): AABB => ({
            lowerBound: { x, y: 0, z: 0 },
            upperBound: { x: x + 1, y: 1, z: 1 },
        });
        const a = createProxy(tree, box(0), 0, 0x1, 0);
        createProxy(tree, box(0.5), 0, 0x2, 1);
        createProxy(tree, box(10), 0, 0x4, 2);

        // Root's category is the OR of every leaf.
        expect(getCategoryLo(tree, tree.root)).toBe(0x7);

        setCategoryBits(tree, a, 0, 0x10);
        expect(getCategoryLo(tree, a)).toBe(0x10);
        // The change ORs back up to the root.
        expect(getCategoryLo(tree, tree.root)).toBe(0x16);
        expect(getCategoryHi(tree, tree.root)).toBe(0);
        expect(getProxyCount(tree)).toBe(3);
        validate(tree);

        // getRootBounds returns a copy, not the live node aabb.
        const bounds = getRootBounds(tree);
        bounds.lowerBound.x = 999;
        expect(getAABB(tree, tree.root).lowerBound.x).not.toBe(999);
    });
});

describe("dynamic tree — filter bits", () => {
    const leafBox: AABB = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 1, y: 1, z: 1 } };
    const queryBox: AABB = {
        lowerBound: { x: -1, y: -1, z: -1 },
        upperBound: { x: 2, y: 2, z: 2 },
    };

    const hits = (
        categoryHi: number,
        categoryLo: number,
        maskHi: number,
        maskLo: number,
        requireAllBits: boolean,
    ): number => {
        const tree = createTree(16);
        createProxy(tree, leafBox, categoryHi, categoryLo, 0);
        let count = 0;
        query(tree, queryBox, maskHi, maskLo, requireAllBits, () => {
            count += 1;
            return true;
        });
        return count;
    };

    test("requireAllBits matches a mask with a half's top bit set", () => {
        // `0xffffffff & 0xffffffff` is -1 as a signed int32, so each half of the AND must be
        // normalized back to unsigned before the equality — or an all-ones mask never matches.
        expect(hits(ALL_BITS_HI, ALL_BITS_LO, ALL_BITS_HI, ALL_BITS_LO, true)).toBe(1);
        expect(hits(0, 0x80000000, 0, 0x80000000, true)).toBe(1); // bit 31 alone
        expect(hits(0x80000000, 0, 0x80000000, 0, true)).toBe(1); // bit 63 alone

        // A category missing one of the mask's bits still fails in either half.
        expect(hits(0, 0x80000000, 0, 0xc0000000, true)).toBe(0);
        expect(hits(0x80000000, 0, 0xc0000000, 0, true)).toBe(0);
    });

    test("any-bit match sees categories above bit 32 without aliasing the halves", () => {
        expect(hits(0x00000100, 0, 0x00000100, 0, false)).toBe(1); // bit 40
        expect(hits(0x80000000, 0, 0x80000000, 0, false)).toBe(1); // bit 63
        expect(hits(0, 0x00000100, 0x00000100, 0, false)).toBe(0);
        expect(hits(0x00000100, 0, 0, 0x00000100, false)).toBe(0);
    });
});

describe("dynamic tree — pool grow", () => {
    test("category words (0xffffffff) survive the node-pool grow copy", () => {
        // createTree floors nodeCapacity at 2*16-1 = 31 nodes; each proxy past the first costs 2
        // nodes (leaf + new parent), so 20 proxies (39 nodes) forces at least one allocateNode grow.
        const tree = createTree(16);
        const initialCapacity = tree.nodeCapacity;
        const proxyCount = 20;
        const ids: number[] = [];
        for (let i = 0; i < proxyCount; ++i) {
            const box: AABB = {
                lowerBound: { x: i, y: 0, z: 0 },
                upperBound: { x: i + 1, y: 1, z: 1 },
            };
            ids.push(createProxy(tree, box, ALL_BITS_HI, ALL_BITS_LO, i));
        }

        expect(tree.nodeCapacity).toBeGreaterThan(initialCapacity);

        const hits: number[] = [];
        query(
            tree,
            { lowerBound: { x: -1, y: -1, z: -1 }, upperBound: { x: proxyCount + 1, y: 2, z: 2 } },
            ALL_BITS_HI,
            ALL_BITS_LO,
            true, // requireAllBits
            (proxyId) => {
                hits.push(proxyId);
                return true;
            },
        );
        expect(hits.sort((a, b) => a - b)).toEqual([...ids].sort((a, b) => a - b));
    });
});

describe("dynamic tree — nested query", () => {
    // The broad phase queries a compound leaf's inner tree from inside an outer query's callback, so
    // the query scratch (traversal stack + stats) pools by depth. A shared singleton would let the
    // inner query clobber the outer one's stack and stats.
    const boxAt = (x: number): AABB => ({
        lowerBound: { x: x - 0.5, y: -0.5, z: -0.5 },
        upperBound: { x: x + 0.5, y: 0.5, z: 0.5 },
    });
    const wide: AABB = {
        lowerBound: { x: -10, y: -1, z: -1 },
        upperBound: { x: 10, y: 1, z: 1 },
    };

    test("an inner query does not disturb the outer traversal or its stats", () => {
        const outer = createTree(16);
        for (let i = 0; i < 4; ++i) createProxy(outer, boxAt(i), ALL_BITS_HI, ALL_BITS_LO, i);
        const inner = createTree(16);
        for (let i = 0; i < 3; ++i) createProxy(inner, boxAt(i), ALL_BITS_HI, ALL_BITS_LO, 100 + i);

        const flat = query(outer, wide, ALL_BITS_HI, ALL_BITS_LO, false, () => true);
        const flatNodeVisits = flat.nodeVisits;
        const flatLeafVisits = flat.leafVisits;

        const outerHits: number[] = [];
        const innerHits: number[] = [];
        const nested = query(outer, wide, ALL_BITS_HI, ALL_BITS_LO, false, (_proxyId, userData) => {
            outerHits.push(userData);
            query(inner, wide, ALL_BITS_HI, ALL_BITS_LO, false, (_id, innerData) => {
                innerHits.push(innerData);
                return true;
            });
            return true;
        });

        expect(outerHits.length).toBe(4);
        expect(innerHits.length).toBe(12); // 3 inner leaves per outer leaf
        expect(nested.nodeVisits).toBe(flatNodeVisits);
        expect(nested.leafVisits).toBe(flatLeafVisits);
    });
});
