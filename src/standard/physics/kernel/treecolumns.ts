import type { World } from "../../../engine";
import type { AABB } from "../common/records";
import { kernel, rethrowQueryError, setQueryCallback } from "./kernel";
import { guardViews } from "./views";

export type TreeStats = { nodeVisits: number; leafVisits: number };
export type TreeBacking = {
    ecsState: World | undefined;
    growTree(index: number, capacity: number): void;
    refreshIfStale(): void;
    ensureViews(): void;
    reserveTreeWork(depth: number, words: number): number;
};
export type DynamicTree = {
    nf: Float32Array;
    ni: Int32Array;
    state: Int32Array;
    residentState: boolean;
    root: number;
    nodeCount: number;
    nodeCapacity: number;
    proxyCount: number;
    freeList: number;
    store: TreeBacking | null;
    treeIndex: number;
    initNodeCapacity: number;
    handle: number;
    compoundNodes?: Uint32Array;
    captureCheckpoint(): unknown;
    restoreCheckpoint(state: unknown): void;
};
const STRIDE = 12;
let depth = 0;
class TreeMetadata {
    state = new Int32Array([0, 2, -1, 0, 0, 0]);
    captureCheckpoint() {
        const tree = this as unknown as DynamicTree;
        return {
            nodeCapacity: tree.nodeCapacity,
            initNodeCapacity: tree.initNodeCapacity,
            residentState: tree.residentState,
            state: tree.residentState ? null : tree.state.slice(),
        };
    }
    restoreCheckpoint(state: unknown): void {
        const tree = this as unknown as DynamicTree;
        const { state: privateState, ...metadata } = state as ReturnType<
            TreeMetadata["captureCheckpoint"]
        >;
        Object.assign(tree, metadata);
        if (privateState !== null) tree.state = privateState;
    }
    get root() {
        return this.state[0];
    }
    set root(value: number) {
        this.state[0] = value;
    }
    get nodeCount() {
        return this.state[1];
    }
    set nodeCount(value: number) {
        this.state[1] = value;
    }
    get freeList() {
        return this.state[2];
    }
    set freeList(value: number) {
        this.state[2] = value;
    }
    get proxyCount() {
        return this.state[3];
    }
    set proxyCount(value: number) {
        this.state[3] = value;
    }
}
export function createTree(
    capacity: number,
    store: TreeBacking | null = null,
    treeIndex = 0,
): DynamicTree {
    const tree: DynamicTree = Object.assign(new TreeMetadata(), {
        nf: new Float32Array(0),
        ni: new Int32Array(0),
        residentState: false,
        nodeCapacity: 0,
        store,
        treeIndex,
        initNodeCapacity: 2 * Math.max(capacity, 16),
        handle: 0,
    });
    if (store) guardViews(tree, store);
    return tree;
}
function handle(t: DynamicTree): number {
    if (!t.handle) t.handle = kernel(undefined).standaloneTreeCreate(t.initNodeCapacity / 2) >>> 0;
    return t.handle;
}
export function destroyTree(t: DynamicTree): void {
    if (t.handle) {
        kernel(undefined).standaloneTreeDestroy(t.handle);
        t.handle = 0;
    }
}
function publish(t: DynamicTree): void {
    const k = kernel(undefined),
        h = handle(t);
    const state = new Uint32Array(k.memory.buffer, k.standaloneTreeWords(h) >>> 0, 32);
    t.root = 0;
    t.nodeCount = state[9];
    t.nodeCapacity = state[10];
    t.proxyCount = state[11];
    t.freeList = state[17] | 0;
    const pool = new Uint32Array(
        k.memory.buffer,
        k.standaloneTreePool(h) >>> 0,
        Math.max(state[10], state[18]) * 20,
    );
    t.ni = new Int32Array(t.nodeCapacity * 12);
    t.nf = new Float32Array(t.ni.buffer);
    t.compoundNodes = new Uint32Array(t.nodeCount * 8);
    for (let i = 0; i < t.nodeCapacity; i++) {
        const p = i * 20,
            n = i * 8;
        t.ni.set(pool.subarray(p, p + 12), i * 12);
        if (i >= t.nodeCount) continue;
        t.compoundNodes.set(pool.subarray(p, p + 6), n);
        const flags = pool[p + 11];
        if (!(flags & 1))
            t.compoundNodes.set(
                [0x7f800000, 0x7f800000, 0x7f800000, 0xff800000, 0xff800000, 0xff800000],
                n,
            );
        t.compoundNodes[n + 6] = !(flags & 1)
            ? 0xbfffffff
            : flags & 4
              ? 0x80000000 | pool[p + 18]
              : pool[p + 8];
        t.compoundNodes[n + 7] = flags & 4 ? pool[p + 8] : flags >>> 16;
    }
}
function mutation(
    t: DynamicTree,
    op: number,
    id: number,
    box?: AABB,
    hi = 0,
    lo = 0,
    user: number | bigint = 0,
): number {
    const userLow = typeof user === "bigint" ? Number(user & 0xffffffffn) : user;
    const userHigh =
        typeof user === "bigint" ? Number(user >> 32n) : Math.floor(user / 0x100000000);
    if (t.store) {
        t.store.refreshIfStale();
        return kernel(t.store.ecsState).treeMutateResident(
            t.treeIndex,
            op,
            id,
            box?.lowerBound.x ?? 0,
            box?.lowerBound.y ?? 0,
            box?.lowerBound.z ?? 0,
            box?.upperBound.x ?? 0,
            box?.upperBound.y ?? 0,
            box?.upperBound.z ?? 0,
            hi,
            lo,
            userLow,
            userHigh,
        );
    }
    const k = kernel(undefined),
        h = handle(t);
    if (box)
        new Float32Array(k.memory.buffer, k.standaloneTreeWords(h) >>> 0, 6).set([
            box.lowerBound.x,
            box.lowerBound.y,
            box.lowerBound.z,
            box.upperBound.x,
            box.upperBound.y,
            box.upperBound.z,
        ]);
    const result = k.standaloneTreeCall(
        h,
        [0, 12, 13, 11, 3][op],
        id,
        hi,
        lo,
        0,
        userLow,
        userHigh,
        0,
    );
    publish(t);
    return result;
}
/** Rebuilt compound tree columns own their bytes, not a native tree handle. */
export function buildTree(boxes: readonly AABB[], hi: number, lo: number): DynamicTree {
    const t = createTree(boxes.length);
    try {
        const k = kernel(undefined),
            h = handle(t);
        for (let i = 0; i < boxes.length; i++) {
            const b = boxes[i];
            new Float32Array(k.memory.buffer, k.standaloneTreeWords(h) >>> 0, 6).set([
                b.lowerBound.x,
                b.lowerBound.y,
                b.lowerBound.z,
                b.upperBound.x,
                b.upperBound.y,
                b.upperBound.z,
            ]);
            k.standaloneTreeCall(h, 14, 0, hi, lo, 0, i, 0, 0);
        }
        k.standaloneTreeCall(h, 3, 1, 0, 0, 0, 0, 0, 0);
        publish(t);
        // Baked compounds retain the immutable node/proxy prefixes, not the mutable pool capacity.
        t.nodeCapacity = t.nodeCount;
        t.nf = t.nf.subarray(0, t.nodeCount * STRIDE);
        t.ni = t.ni.subarray(0, t.nodeCount * STRIDE);
        return t;
    } finally {
        destroyTree(t);
    }
}
export function createProxy(
    t: DynamicTree,
    box: AABB,
    hi: number,
    lo: number,
    user: number | bigint,
): number {
    if (!t.store || typeof user !== "number") return mutation(t, 0, 0, box, hi, lo, user);
    t.store.refreshIfStale();
    return kernel(t.store.ecsState).treeCreateProxy(
        t.treeIndex,
        box.lowerBound.x,
        box.lowerBound.y,
        box.lowerBound.z,
        box.upperBound.x,
        box.upperBound.y,
        box.upperBound.z,
        hi,
        lo,
        user,
    );
}
export function enlargeProxy(t: DynamicTree, id: number, box: AABB): void {
    mutation(t, 2, id, box);
}
export function destroyProxy(t: DynamicTree, id: number): void {
    mutation(t, 3, id);
}
export function rebuild(t: DynamicTree, full: boolean): number {
    return mutation(t, 4, Number(full));
}
export function query(
    t: DynamicTree,
    box: AABB,
    hi: number,
    lo: number,
    all: boolean,
    callback: (id: number, user: bigint) => boolean,
    context: undefined,
    wide: true,
): TreeStats;
export function query<C = undefined>(
    t: DynamicTree,
    box: AABB,
    hi: number,
    lo: number,
    all: boolean,
    callback: (id: number, user: number, context: C) => boolean,
    context?: C,
): TreeStats;
export function query(
    t: DynamicTree,
    box: AABB,
    hi: number,
    lo: number,
    all: boolean,
    callback:
        | ((id: number, user: number, context: any) => boolean)
        | ((id: number, user: bigint) => boolean),
    context?: any,
    wide = false,
): TreeStats {
    const world = t.store?.ecsState;
    const previous = setQueryCallback(world, (_kind, id, user, high) =>
        Number(
            (callback as (id: number, user: number | bigint, context: unknown) => boolean)(
                id,
                wide ? BigInt(user >>> 0) | (BigInt(high >>> 0) << 32n) : user >>> 0,
                context,
            ),
        ),
    );
    const d = depth++;
    try {
        const k = kernel(world);
        if (!t.store) {
            const h = handle(t),
                wp = k.standaloneTreeWords(h) >>> 0;
            new Float32Array(k.memory.buffer, wp, 6).set([
                box.lowerBound.x,
                box.lowerBound.y,
                box.lowerBound.z,
                box.upperBound.x,
                box.upperBound.y,
                box.upperBound.z,
            ]);
            k.standaloneTreeCall(h, 4, 0, hi, lo, Number(all), 0, 0, 0);
            rethrowQueryError(world);
            const out = new Uint32Array(k.memory.buffer, wp, 32);
            return { nodeVisits: out[14], leafVisits: out[15] };
        }
        t.store.refreshIfStale();
        const state = t.store.reserveTreeWork(d, 6);
        k.treeQuery(
            0,
            t.treeIndex,
            0,
            0,
            box.lowerBound.x,
            box.lowerBound.y,
            box.lowerBound.z,
            box.upperBound.x,
            box.upperBound.y,
            box.upperBound.z,
            hi,
            lo,
            Number(all),
            state,
        );
        rethrowQueryError(world);
        const out = new Uint32Array(k.memory.buffer, state, 6);
        return { nodeVisits: out[4], leafVisits: out[5] };
    } finally {
        depth = d;
        setQueryCallback(world, previous);
    }
}
export function getAABBInto(t: DynamicTree, id: number, out: AABB): AABB {
    const n = id * STRIDE;
    out.lowerBound.x = t.nf[n];
    out.lowerBound.y = t.nf[n + 1];
    out.lowerBound.z = t.nf[n + 2];
    out.upperBound.x = t.nf[n + 3];
    out.upperBound.y = t.nf[n + 4];
    out.upperBound.z = t.nf[n + 5];
    return out;
}
export function getAABB(t: DynamicTree, id: number): AABB {
    return getAABBInto(t, id, {
        lowerBound: { x: 0, y: 0, z: 0 },
        upperBound: { x: 0, y: 0, z: 0 },
    });
}
export function readNode(t: DynamicTree, i: number) {
    const n = i * STRIDE;
    return {
        aabb: getAABB(t, i),
        categoryHi: t.ni[n + 6] >>> 0,
        categoryLo: t.ni[n + 7] >>> 0,
        child1: t.ni[n + 8],
        child2: t.ni[n + 9],
        userData: t.ni[n + 8],
        parent: t.ni[n + 10],
        next: t.ni[n + 10],
        height: t.ni[n + 11] >>> 16,
        flags: t.ni[n + 11] & 0xffff,
    };
}
