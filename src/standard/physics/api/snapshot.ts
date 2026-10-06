import { kernel } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";
import type { CheckpointStore } from "../kernel/views";
import type { WorldState } from "../world/world";
import type { PhysicsWorld } from "./world";

/** Reusable snapshot of a wasm-backed physics world. */
export interface WorldSnapshot {
    /** Copied logical state. userData values retain identity; their id associations and names are copied. */
    readonly state: unknown;
    /** the detached bytes of this World's persistent kernel regions */
    readonly bytes: Uint8Array;
}

type StoreName = "body" | "shape" | "manifold" | "broadPhase" | "broadStore" | "query";
type StoreMarker = { readonly snapshotStore: StoreName };

const STORE_MARKERS: Record<StoreName, StoreMarker> = {
    body: Object.freeze({ snapshotStore: "body" }),
    shape: Object.freeze({ snapshotStore: "shape" }),
    manifold: Object.freeze({ snapshotStore: "manifold" }),
    broadPhase: Object.freeze({ snapshotStore: "broadPhase" }),
    broadStore: Object.freeze({ snapshotStore: "broadStore" }),
    query: Object.freeze({ snapshotStore: "query" }),
};

function snapshotStores(state: WorldState): Map<object, StoreName> {
    const stores = new Map<object, StoreName>([
        [state.bodyStore, "body"],
        [state.shapeStore, "shape"],
        [state.manifoldStore, "manifold"],
        [state.broadPhase, "broadPhase"],
        [state.broadPhase.store, "broadStore"],
    ]);
    if (state.queryColumns) stores.set(state.queryColumns, "query");
    return stores;
}

function clone<T>(value: T, seen: Map<object, unknown>, stores: Map<object, StoreName>): T {
    if (value === null || typeof value !== "object") return value;
    const store = stores.get(value as object);
    if (store) return STORE_MARKERS[store] as T;
    const prior = seen.get(value as object);
    if (prior !== undefined) return prior as T;
    if (ArrayBuffer.isView(value)) {
        const view = value as unknown as { constructor: new (source: unknown) => unknown };
        const copy = new view.constructor(value) as T;
        seen.set(value as object, copy);
        return copy;
    }
    if (value instanceof ArrayBuffer) return value.slice(0) as T;
    if (value instanceof Map) {
        const out = new Map();
        seen.set(value, out);
        for (const [k, v] of value) out.set(clone(k, seen, stores), clone(v, seen, stores));
        return out as T;
    }
    if (value instanceof Set) {
        const out = new Set();
        seen.set(value, out);
        for (const item of value) out.add(clone(item, seen, stores));
        return out as T;
    }
    if (Array.isArray(value)) {
        const out: unknown[] = [];
        seen.set(value, out);
        for (const item of value) out.push(clone(item, seen, stores));
        return out as T;
    }
    const out = Object.create(Object.getPrototypeOf(value)) as Record<PropertyKey, unknown>;
    seen.set(value as object, out);
    for (const key of Reflect.ownKeys(value)) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor && "value" in descriptor && key !== "userData")
            // Upload registers contain no logical state and may hold detached linear-memory views.
            descriptor.value =
                key === "geometryUploadScratch"
                    ? undefined
                    : key === "bodyUserData" ||
                        key === "shapeUserData" ||
                        key === "jointUserData" ||
                        key === "jointEventUserData"
                      ? descriptor.value.slice()
                      : clone(descriptor.value, seen, stores);
        if (descriptor) Object.defineProperty(out, key, descriptor);
    }
    return out as T;
}

function restoreClone<T>(
    value: T,
    seen: Map<object, unknown>,
    stores: Record<StoreName, object>,
    root?: WorldState,
): T {
    if (value === null || typeof value !== "object") return value;
    for (const name of Object.keys(STORE_MARKERS) as StoreName[]) {
        if ((value as object) === STORE_MARKERS[name]) return stores[name] as T;
    }
    const prior = seen.get(value as object);
    if (prior !== undefined) return prior as T;
    if (ArrayBuffer.isView(value)) {
        const view = value as unknown as { constructor: new (source: unknown) => unknown };
        const copy = new view.constructor(value) as T;
        seen.set(value as object, copy);
        return copy;
    }
    if (value instanceof ArrayBuffer) return value.slice(0) as T;
    if (value instanceof Map) {
        const out = new Map();
        seen.set(value, out);
        for (const [k, v] of value)
            out.set(restoreClone(k, seen, stores), restoreClone(v, seen, stores));
        return out as T;
    }
    if (value instanceof Set) {
        const out = new Set();
        seen.set(value, out);
        for (const item of value) out.add(restoreClone(item, seen, stores));
        return out as T;
    }
    if (Array.isArray(value)) {
        const out: unknown[] = [];
        seen.set(value, out);
        for (const item of value) out.push(restoreClone(item, seen, stores));
        return out as T;
    }
    const out = (root ?? Object.create(Object.getPrototypeOf(value))) as Record<
        PropertyKey,
        unknown
    >;
    seen.set(value as object, out);
    for (const key of Reflect.ownKeys(out)) {
        if (!Reflect.has(value, key)) Reflect.deleteProperty(out, key);
    }
    for (const key of Reflect.ownKeys(value)) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor && "value" in descriptor && key !== "userData")
            descriptor.value =
                key === "bodyUserData" ||
                key === "shapeUserData" ||
                key === "jointUserData" ||
                key === "jointEventUserData"
                    ? descriptor.value.slice()
                    : restoreClone(descriptor.value, seen, stores);
        if (descriptor) Object.defineProperty(out, key, descriptor);
    }
    return out as T;
}

type SnapshotState = {
    world: WorldState;
    checkpoints: Partial<Record<StoreName, unknown>>;
    /** plain data an owner records beside the solver; {@link restore} ignores it */
    bindings?: unknown;
};

/** Capture logical state and kernel regions, retaining opaque userData values, with an owner's plain `bindings`. */
export function snapshot(physicsWorld: PhysicsWorld, bindings?: unknown): WorldSnapshot {
    const state = physicsWorld.state;
    const k = kernel(state.ecsState);
    const length = k.worldSnapshot(state.worldId);
    const pointer = k.worldSnapshotBuffer(length);
    const stores = snapshotStores(state);
    const checkpoints: Partial<Record<StoreName, unknown>> = {};
    for (const [store, name] of stores) {
        checkpoints[name] = clone(
            (store as CheckpointStore).captureCheckpoint(),
            new Map(),
            stores,
        );
    }
    const seen = state.ecsState
        ? new Map<object, unknown>([[state.ecsState, null]])
        : new Map<object, unknown>();
    return {
        // The ECS owner is identity, not solver data; snapshots never clone or retain it.
        state: {
            world: clone(state, seen, stores),
            checkpoints,
            bindings,
        },
        bytes: new Uint8Array(k.memory.buffer, pointer, length).slice(),
    };
}

/** @returns the `bindings` a snapshot was captured with, if any. */
export function snapshotBindings(snapshot: WorldSnapshot): unknown {
    return (snapshot.state as SnapshotState | null)?.bindings;
}

/** Restore into a live compatible World, preserving its identity and every sibling's state. */
export function restore(physicsWorld: PhysicsWorld, snapshot: WorldSnapshot): void {
    if (
        snapshot === null ||
        typeof snapshot !== "object" ||
        !(snapshot.bytes instanceof Uint8Array) ||
        snapshot.state === null ||
        typeof snapshot.state !== "object"
    )
        throw new Error("physics: invalid world snapshot");

    if (!physicsWorld.isValid())
        throw new Error("physics: cannot restore a snapshot because its target World is not live");

    const state = physicsWorld.state;
    const identity = {
        ecsState: state.ecsState,
        worldId: state.worldId,
        generation: state.generation,
        maxCapacity: state.maxCapacity,
    };
    const saved = snapshot.state as SnapshotState;
    const stores: Record<StoreName, CheckpointStore> = {
        body: state.bodyStore,
        shape: state.shapeStore,
        manifold: state.manifoldStore,
        broadPhase: state.broadPhase,
        broadStore: state.broadPhase.store,
        query: queryColumns(state),
    };
    restoreClone(saved.world, new Map(), stores, state);
    // World identity and capacity belong to the target handle, not the snapshot's source handle.
    Object.assign(state, identity);
    for (const name of Object.keys(saved.checkpoints) as StoreName[]) {
        stores[name].restoreCheckpoint(restoreClone(saved.checkpoints[name], new Map(), stores));
    }
    const k = kernel(state.ecsState);
    const pointer = k.worldSnapshotBuffer(snapshot.bytes.byteLength);
    new Uint8Array(k.memory.buffer, pointer, snapshot.bytes.byteLength).set(snapshot.bytes);
    k.worldRestore(state.worldId);
    state.manifoldStore.refreshViews();
}
