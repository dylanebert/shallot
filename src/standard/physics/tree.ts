import type { Resource, World } from "../../engine";
import type { RayCastInput } from "./collision/distance";
import type { AABB, Vec3 } from "./common/records";
import {
    assertKernelEntry,
    type Kernel,
    kernel,
    rethrowQueryError,
    setQueryCallback,
} from "./kernel/kernel";

const treeBrand: unique symbol = Symbol("PhysicsTree");
/** An independently owned dynamic AABB tree, released explicitly or with its World. */
export interface PhysicsTree {
    readonly [treeBrand]: true;
}
export interface PhysicsTreeFilter {
    /** Unsigned 64-bit category mask; defaults to all bits. */
    maskBits?: bigint;
    /** Match every mask bit instead of any mask bit; defaults to false. */
    requireAllBits?: boolean;
}
/** Borrowed until the next query on this tree. */
export interface PhysicsTreeQueryResult {
    nodeVisits: number;
    leafVisits: number;
}
/** Borrowed until the next query on this tree. */
export interface PhysicsTreeClosestResult extends PhysicsTreeQueryResult {
    /** Final squared distance in square meters. */
    distanceSquared: number;
}
/** Borrowed for the duration of a ray callback; copy values to retain them. */
export interface PhysicsTreeRayHit {
    proxyId: number;
    userData: bigint;
    maxFraction: number;
}
/** Borrowed for the duration of a closest callback; copy values to retain them. */
export interface PhysicsTreeClosestHit {
    proxyId: number;
    userData: bigint;
    /** Current search limit in square meters. */
    minDistanceSquared: number;
}
export interface PhysicsTreeStats {
    root: number;
    nodeCount: number;
    nodeCapacity: number;
    proxyCount: number;
    height: number;
    /** Internal-node surface areas, excluding the root, divided by root surface area; zero for an empty tree. */
    areaRatio: number;
}
/** Borrowed until the next node read on this tree. Bounds are in meters; root depth is zero. */
export interface PhysicsTreeNode {
    bounds: AABB;
    parent: number;
    child1: number;
    child2: number;
    height: number;
    leaf: boolean;
    userData: bigint;
}

class TreeState {
    readonly world: World;
    readonly kernel: Kernel;
    readonly pointer: number;
    readonly wordsPointer: number;
    releasePending = false;
    words = new Uint32Array(0);
    floats = new Float32Array(0);
    nodes = new Uint32Array(0);
    bounds = new Float32Array(0);
    users = new Map<number, bigint>();
    active = false;
    overlap: ((proxyId: number, userData: bigint) => boolean) | null = null;
    cast: ((hit: PhysicsTreeRayHit) => number) | null = null;
    closest: ((hit: PhysicsTreeClosestHit) => number) | null = null;
    readonly bits = new DataView(new ArrayBuffer(8));
    readonly result: PhysicsTreeClosestResult = {
        nodeVisits: 0,
        leafVisits: 0,
        distanceSquared: 0,
    };
    readonly stats: PhysicsTreeStats = {
        root: -1,
        nodeCount: 0,
        nodeCapacity: 0,
        proxyCount: 0,
        height: 0,
        areaRatio: 0,
    };
    readonly rayHit: PhysicsTreeRayHit = { proxyId: -1, userData: 0n, maxFraction: 0 };
    readonly closestHit: PhysicsTreeClosestHit = {
        proxyId: -1,
        userData: 0n,
        minDistanceSquared: 0,
    };
    readonly node: PhysicsTreeNode = {
        bounds: { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } },
        parent: -1,
        child1: -1,
        child2: -1,
        height: 0,
        leaf: false,
        userData: 0n,
    };
    readonly callback = (_kind: number, id: number): number => {
        this.refreshWords();
        const user = this.users.get(id)!;
        if (this.overlap) return Number(this.overlap(id, user));
        if (this.cast) {
            this.rayHit.proxyId = id;
            this.rayHit.userData = user;
            this.rayHit.maxFraction = this.floats[6];
            return this.cast(this.rayHit);
        }
        this.closestHit.proxyId = id;
        this.closestHit.userData = user;
        this.closestHit.minDistanceSquared = this.floats[6];
        return this.closest!(this.closestHit);
    };
    constructor(world: World, capacity: number) {
        this.world = world;
        this.kernel = kernel(world);
        this.pointer = this.kernel.standaloneTreeCreate(capacity) >>> 0;
        this.wordsPointer = this.kernel.standaloneTreeWords(this.pointer) >>> 0;
        this.refresh();
    }
    refreshWords(): void {
        const memory = this.kernel.memory.buffer;
        if (this.words.buffer !== memory) {
            this.words = new Uint32Array(memory, this.wordsPointer, 32);
            this.floats = new Float32Array(memory, this.wordsPointer, 32);
        }
    }
    refresh(): void {
        this.refreshWords();
        const memory = this.kernel.memory.buffer;
        const pool = this.kernel.standaloneTreePool(this.pointer) >>> 0;
        const length = this.words[10] * 12;
        if (
            this.nodes.buffer !== memory ||
            this.nodes.byteOffset !== pool ||
            this.nodes.length !== length
        ) {
            this.nodes = new Uint32Array(memory, pool, length);
            this.bounds = new Float32Array(memory, pool, length);
        }
    }
    call(op: number, id = 0, category = 0n, user = 0n, limit = 0, all = false): number {
        this.bits.setBigUint64(0, category, true);
        const low = this.bits.getUint32(0, true),
            high = this.bits.getUint32(4, true);
        this.bits.setBigUint64(0, user, true);
        const result = this.kernel.standaloneTreeCall(
            this.pointer,
            op,
            id,
            high,
            low,
            Number(all),
            this.bits.getUint32(0, true),
            this.bits.getUint32(4, true),
            limit,
        );
        this.refresh();
        return result;
    }
    input(lo: Vec3, hi?: Vec3): void {
        this.refresh();
        this.floats[0] = lo.x;
        this.floats[1] = lo.y;
        this.floats[2] = lo.z;
        this.floats[3] = hi?.x ?? 0;
        this.floats[4] = hi?.y ?? 0;
        this.floats[5] = hi?.z ?? 0;
    }
}
const trees: Resource<Map<PhysicsTree, TreeState>> = {
    create(world) {
        const live = new Map<PhysicsTree, TreeState>();
        world.onDispose(() => {
            for (const state of live.values()) {
                if (state.active) state.releasePending = true;
                else state.kernel.standaloneTreeDestroy(state.pointer);
            }
            live.clear();
        });
        return live;
    },
};
function stateFor(world: World, tree: PhysicsTree): TreeState {
    assertKernelEntry(world);
    const state = world.resource(trees).get(tree);
    if (!state) throw new Error("physics tree is not live in this World");
    if (state.active) throw new Error("physics tree cannot be accessed from its query callback");
    state.refresh();
    return state;
}
function uint(value: number): void {
    if (!Number.isInteger(value) || value < 0 || value > 0x7fffffff)
        throw new Error("physics tree capacity must be a nonnegative 31-bit integer");
}
function bits(value: bigint): void {
    if (value < 0n || value > 0xffffffffffffffffn)
        throw new Error("physics tree bits and user data must be unsigned 64-bit integers");
}
function ordered(lo: number, hi: number): boolean {
    return Number.isFinite(Math.fround(lo)) && Number.isFinite(Math.fround(hi)) && lo <= hi;
}
function bounds(value: AABB): void {
    if (
        !ordered(value.lowerBound.x, value.upperBound.x) ||
        !ordered(value.lowerBound.y, value.upperBound.y) ||
        !ordered(value.lowerBound.z, value.upperBound.z)
    )
        throw new Error("physics tree bounds must be finite and ordered");
}
/** Box3D's dynamic tree. Capacity is the initial leaf capacity; growth preserves proxy IDs. */
export function createPhysicsTree(world: World, capacity = 16): PhysicsTree {
    uint(capacity);
    assertKernelEntry(world);
    const tree: PhysicsTree = { [treeBrand]: true };
    world.resource(trees).set(tree, new TreeState(world, capacity));
    return tree;
}
/** Refuses foreign or already destroyed trees. All proxy IDs become invalid. */
export function destroyPhysicsTree(world: World, tree: PhysicsTree): void {
    const state = stateFor(world, tree);
    kernel(world).standaloneTreeDestroy(state.pointer);
    world.resource(trees).delete(tree);
}
/** Bounds are exact, not fattened. Proxies live until the tree is destroyed. */
export function createPhysicsTreeProxy(
    world: World,
    tree: PhysicsTree,
    box: AABB,
    categoryBits = 1n,
    userData = 0n,
): number {
    const state = stateFor(world, tree);
    bounds(box);
    bits(categoryBits);
    bits(userData);
    state.input(box.lowerBound, box.upperBound);
    const id = state.call(0, 0, categoryBits, userData);
    state.users.set(id, userData);
    return id;
}
/** A full rebuild also rebuilds unchanged subtrees. Proxy IDs and user data are preserved. */
export function rebuildPhysicsTree(world: World, tree: PhysicsTree, fullBuild = true): void {
    stateFor(world, tree).call(3, Number(fullBuild));
}
/** Borrowed until the next statistics read on this tree. */
export function getPhysicsTreeStats(world: World, tree: PhysicsTree): PhysicsTreeStats {
    const s = stateFor(world, tree),
        out = s.stats;
    s.call(10);
    out.root = s.words[8] | 0;
    out.nodeCount = s.words[9];
    out.nodeCapacity = s.words[10];
    out.proxyCount = s.words[11];
    out.height = s.words[12];
    out.areaRatio = s.floats[13];
    return out;
}
/** Checks topology, heights, bounds, categories, counts and the free list; throws on corruption. */
export function validatePhysicsTree(world: World, tree: PhysicsTree): void {
    if (!stateFor(world, tree).call(7)) throw new Error("invalid physics tree");
}
/** Returns null for a free slot; internal nodes have no user data. Child IDs are -1 on leaves. */
export function getPhysicsTreeNode(
    world: World,
    tree: PhysicsTree,
    id: number,
): PhysicsTreeNode | null {
    const s = stateFor(world, tree);
    if (!Number.isInteger(id) || id < 0 || id >= s.words[10])
        throw new Error("physics tree node index out of range");
    const n = id * 12,
        flags = s.nodes[n + 11];
    if (!(flags & 1)) return null;
    const out = s.node;
    out.bounds.lowerBound.x = s.bounds[n];
    out.bounds.lowerBound.y = s.bounds[n + 1];
    out.bounds.lowerBound.z = s.bounds[n + 2];
    out.bounds.upperBound.x = s.bounds[n + 3];
    out.bounds.upperBound.y = s.bounds[n + 4];
    out.bounds.upperBound.z = s.bounds[n + 5];
    out.parent = s.nodes[n + 10] | 0;
    out.height = flags >>> 16;
    out.leaf = !!(flags & 4);
    out.child1 = out.leaf ? -1 : s.nodes[n + 8] | 0;
    out.child2 = out.leaf ? -1 : s.nodes[n + 9] | 0;
    out.userData = out.leaf ? s.users.get(id)! : 0n;
    return out;
}
function query(
    s: TreeState,
    op: number,
    limit: number,
    filter?: PhysicsTreeFilter,
): PhysicsTreeClosestResult {
    const mask = filter?.maskBits ?? 0xffffffffffffffffn;
    s.active = true;
    const previous = setQueryCallback(s.world, s.callback);
    try {
        bits(mask);
        s.call(op, 0, mask, 0n, limit, filter?.requireAllBits ?? false);
        rethrowQueryError(s.world);
        s.result.nodeVisits = s.words[14];
        s.result.leafVisits = s.words[15];
        s.result.distanceSquared = s.floats[16];
        return s.result;
    } finally {
        try {
            setQueryCallback(s.world, previous);
        } finally {
            s.active = false;
            s.overlap = null;
            s.cast = null;
            s.closest = null;
            if (s.releasePending) s.kernel.standaloneTreeDestroy(s.pointer);
        }
    }
}
/** Returning false stops traversal. Query callbacks cannot access this tree, but may query another tree. */
export function queryPhysicsTree(
    world: World,
    tree: PhysicsTree,
    box: AABB,
    callback: (proxyId: number, userData: bigint) => boolean,
    filter?: PhysicsTreeFilter,
): PhysicsTreeQueryResult {
    const s = stateFor(world, tree);
    bounds(box);
    s.input(box.lowerBound, box.upperBound);
    s.overlap = callback;
    return query(s, 4, 0, filter);
}
/** Callback returns -1 to ignore, 0 to stop, or a positive fraction to clip the ray. */
export function rayCastPhysicsTree(
    world: World,
    tree: PhysicsTree,
    input: RayCastInput,
    callback: (hit: PhysicsTreeRayHit) => number,
    filter?: PhysicsTreeFilter,
): PhysicsTreeQueryResult {
    const s = stateFor(world, tree);
    s.input(input.origin, input.translation);
    s.cast = callback;
    return query(s, 5, input.maxFraction, filter);
}
/** Callback returns a squared distance in square meters; smaller values tighten the search. */
export function queryClosestPhysicsTree(
    world: World,
    tree: PhysicsTree,
    point: Vec3,
    maxDistanceSquared: number,
    callback: (hit: PhysicsTreeClosestHit) => number,
    filter?: PhysicsTreeFilter,
): PhysicsTreeClosestResult {
    const s = stateFor(world, tree);
    s.input(point);
    s.closest = callback;
    return query(s, 6, maxDistanceSquared, filter);
}

/** Owned, portable little-endian image of the tree, including proxy IDs and free slots; not Box3D's host-specific file format. */
export function savePhysicsTree(world: World, tree: PhysicsTree): Uint8Array {
    const s = stateFor(world, tree);
    s.call(10);
    const bytes = new Uint8Array(32 + s.nodes.byteLength),
        view = new DataView(bytes.buffer);
    view.setUint32(0, 0x31543342, true);
    for (let i = 0; i < 6; i++) view.setUint32(4 + i * 4, s.words[8 + i], true);
    view.setInt32(28, s.words[17], true);
    for (let i = 0; i < s.nodes.length; i++) view.setUint32(32 + i * 4, s.nodes[i], true);
    return bytes;
}
/** Restores an owned image. Scale must be positive and finite. Malformed or incompatible images throw. */
export function loadPhysicsTree(world: World, image: Uint8Array, scale = 1): PhysicsTree {
    if (!Number.isFinite(Math.fround(scale)) || Math.fround(scale) <= 0 || image.byteLength < 32)
        throw new Error("invalid physics tree image or scale");
    const view = new DataView(image.buffer, image.byteOffset, image.byteLength),
        capacity = view.getUint32(12, true);
    if (
        view.getUint32(0, true) !== 0x31543342 ||
        capacity < 31 ||
        image.byteLength !== 32 + capacity * 48
    )
        throw new Error("invalid physics tree image");
    // Reproduce the exact node capacity, including pools that grew by 50 percent.
    const tree = createPhysicsTree(world, 16),
        s = stateFor(world, tree);
    try {
        s.call(9, capacity);
        s.users.clear();
        for (let i = 0; i < 6; i++) s.words[8 + i] = view.getUint32(4 + i * 4, true);
        s.words[17] = view.getUint32(28, true);
        for (let i = 0; i < s.nodes.length; i++) s.nodes[i] = view.getUint32(32 + i * 4, true);
        for (let i = 0; i < s.words[10]; i++) {
            const n = i * 12;
            if ((s.nodes[n + 11] & 5) !== 5) continue;
            s.bits.setUint32(0, s.nodes[n + 8], true);
            s.bits.setUint32(4, s.nodes[n + 9], true);
            s.users.set(i, s.bits.getBigUint64(0, true));
        }
        if (!s.call(8, 0, 0n, 0n, scale)) throw new Error("invalid physics tree image");
        return tree;
    } catch (error) {
        destroyPhysicsTree(world, tree);
        throw error;
    }
}
