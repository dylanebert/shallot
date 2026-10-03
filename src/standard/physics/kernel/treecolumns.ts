import type { World } from "../../../engine";
import type { AABB } from "../common/math";
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
    captureCheckpoint(): unknown;
    restoreCheckpoint(state: unknown): void;
};
export const NULL_INDEX = -1;
const STRIDE = 12;
let depth = 0;
class TreeMetadata {
    state: Int32Array;
    constructor(state: Int32Array) {
        this.state = state;
    }
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
    get root(): number {
        return this.state[0];
    }
    set root(value: number) {
        this.state[0] = value;
    }
    get nodeCount(): number {
        return this.state[1];
    }
    set nodeCount(value: number) {
        this.state[1] = value;
    }
    get freeList(): number {
        return this.state[2];
    }
    set freeList(value: number) {
        this.state[2] = value;
    }
    get proxyCount(): number {
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
    const cap = 2 * Math.max(capacity, 16) - 1;
    const buffer = new ArrayBuffer(store ? 0 : cap * STRIDE * 4);
    const t: DynamicTree = Object.assign(
        new TreeMetadata(new Int32Array([-1, 0, store ? -1 : 0, 0, 0, 0])),
        {
            nf: new Float32Array(buffer),
            ni: new Int32Array(buffer),
            residentState: false,
            nodeCapacity: store ? 0 : cap,
            store,
            treeIndex,
            initNodeCapacity: cap,
        },
    );
    if (store) guardViews(t, store);
    else freeRun(t, 0, cap);
    return t;
}
function freeRun(t: DynamicTree, start: number, end: number): void {
    for (let i = start; i < end; i++) t.ni[i * STRIDE + 10] = i + 1 === end ? -1 : i + 1;
}
function reserve(t: DynamicTree): void {
    if (t.nodeCapacity - t.nodeCount >= 2) return;
    const old = t.nodeCapacity;
    const cap = old === 0 ? t.initNodeCapacity : old + (old >> 1);
    if (t.store) t.store.growTree(t.treeIndex, cap);
    else {
        const ni = new Int32Array(cap * STRIDE);
        ni.set(t.ni);
        t.ni = ni;
        t.nf = new Float32Array(ni.buffer);
    }
    t.nodeCapacity = cap;
    freeRun(t, old, cap);
    if (t.freeList === -1) t.freeList = old;
    else {
        let i = t.freeList;
        while (t.ni[i * STRIDE + 10] !== -1) i = t.ni[i * STRIDE + 10];
        t.ni[i * STRIDE + 10] = old;
    }
}
function withColumns<T>(
    t: DynamicTree,
    mutate: boolean,
    run: (ptr: number, state: number) => T,
): T {
    t.store?.refreshIfStale();
    const k = kernel(t.store?.ecsState);
    const d = depth++;
    try {
        const words = 6 + (t.store ? 0 : t.ni.length) + Math.max(t.proxyCount, 1) * 4;
        const state = t.store
            ? t.store.reserveTreeWork(d, words)
            : k.reserveTreeWork(d, words) >>> 0;
        t.store?.refreshIfStale();
        const s = new Int32Array(k.memory.buffer, state, 6);
        s.set([t.root, t.nodeCount, t.freeList, t.proxyCount, 0, 0]);
        const ptr = t.store ? t.ni.byteOffset : state + 24;
        if (!t.store) new Int32Array(k.memory.buffer, ptr, t.ni.length).set(t.ni);
        const result = run(ptr, state);
        if (mutate) {
            const out = new Int32Array(k.memory.buffer, state, 6);
            [t.root, t.nodeCount, t.freeList, t.proxyCount] = out;
            if (!t.store) t.ni.set(new Int32Array(k.memory.buffer, ptr, t.ni.length));
        }
        rethrowQueryError(t.store?.ecsState);
        return result;
    } finally {
        depth = d;
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
    buffer = true,
): number {
    if (op === 0) {
        t.store?.refreshIfStale();
        reserve(t);
    }
    if (t.store) return residentMutation(t, op, id, box, hi, lo, user, buffer);
    return uploadedMutation(t, op, id, box, hi, lo, user);
}
function residentMutation(
    t: DynamicTree,
    op: number,
    id: number,
    box: AABB | undefined,
    hi: number,
    lo: number,
    user: number | bigint,
    buffer: boolean,
): number {
    t.store!.refreshIfStale();
    return kernel(t.store!.ecsState).treeMutateResident(
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
        typeof user === "number" ? user : Number(user & 0xffffffffn),
        typeof user === "number" ? 0 : Number(user >> 32n),
        Number(buffer),
    );
}
function uploadedMutation(
    t: DynamicTree,
    op: number,
    id: number,
    box: AABB | undefined,
    hi: number,
    lo: number,
    user: number | bigint,
): number {
    return withColumns(t, true, (ptr, state) =>
        kernel(t.store?.ecsState).treeMutate(
            ptr,
            t.nodeCapacity,
            state,
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
            Number(BigInt(user) & 0xffffffffn),
            Number(BigInt(user) >> 32n),
        ),
    );
}
export function createProxy(
    t: DynamicTree,
    box: AABB,
    hi: number,
    lo: number,
    user: number | bigint,
    buffer = t.treeIndex !== 0,
): number {
    if (!t.store || typeof user !== "number") return mutation(t, 0, 0, box, hi, lo, user, buffer);
    t.store.refreshIfStale();
    reserve(t);
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
        Number(buffer),
    );
}
export function moveProxy(t: DynamicTree, id: number, box: AABB): void {
    if (!t.store) {
        mutation(t, 1, id, box);
        return;
    }
    t.store.refreshIfStale();
    kernel(t.store.ecsState).treeMoveProxy(
        t.treeIndex,
        id,
        box.lowerBound.x,
        box.lowerBound.y,
        box.lowerBound.z,
        box.upperBound.x,
        box.upperBound.y,
        box.upperBound.z,
    );
}
export function enlargeProxy(t: DynamicTree, id: number, box: AABB): void {
    if (!t.store) {
        mutation(t, 2, id, box);
        return;
    }
    t.store.refreshIfStale();
    kernel(t.store.ecsState).treeEnlargeProxy(
        t.treeIndex,
        id,
        box.lowerBound.x,
        box.lowerBound.y,
        box.lowerBound.z,
        box.upperBound.x,
        box.upperBound.y,
        box.upperBound.z,
    );
}
export function destroyProxy(t: DynamicTree, id: number): void {
    if (!t.store) {
        mutation(t, 3, id);
        return;
    }
    t.store.refreshIfStale();
    kernel(t.store.ecsState).treeDestroyProxy(t.treeIndex, id);
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
    try {
        return withColumns(t, false, (ptr, state) => {
            const k = kernel(world);
            k.treeQuery(
                ptr,
                t.nodeCapacity,
                t.root,
                t.nodeCount,
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
            const out = new Uint32Array(k.memory.buffer, state, 6);
            return { nodeVisits: out[4], leafVisits: out[5] };
        });
    } finally {
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
