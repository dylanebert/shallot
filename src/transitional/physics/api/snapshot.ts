import { kernel } from "../kernel/kernel";
import { liveWorldCount, type WorldState } from "../world/world";
import type { World } from "./world";

/** Plain, reusable snapshot data from a wasm-backed physics world. */
export interface WorldSnapshot {
    /** a detached copy of the logical world state */
    readonly state: unknown;
    /** the wasm linear-memory image at capture time */
    readonly bytes: Uint8Array;
}

type StoreName = "body" | "shape" | "manifold" | "broadPhase";
type StoreMarker = { readonly snapshotStore: StoreName };

const STORE_MARKERS: Record<StoreName, StoreMarker> = {
    body: Object.freeze({ snapshotStore: "body" }),
    shape: Object.freeze({ snapshotStore: "shape" }),
    manifold: Object.freeze({ snapshotStore: "manifold" }),
    broadPhase: Object.freeze({ snapshotStore: "broadPhase" }),
};

function snapshotStores(state: WorldState): Map<object, StoreName> {
    return new Map<object, StoreName>([
        [state.bodyStore, "body"],
        [state.shapeStore, "shape"],
        [state.manifoldStore, "manifold"],
        [state.broadPhase.store, "broadPhase"],
    ]);
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
        if (descriptor && "value" in descriptor)
            descriptor.value = clone(descriptor.value, seen, stores);
        if (descriptor) Object.defineProperty(out, key, descriptor);
    }
    return out as T;
}

function restoreClone<T>(
    value: T,
    seen: Map<object, unknown>,
    stores: Record<StoreName, object>,
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
    const out = Object.create(Object.getPrototypeOf(value)) as Record<PropertyKey, unknown>;
    seen.set(value as object, out);
    for (const key of Reflect.ownKeys(value)) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor && "value" in descriptor)
            descriptor.value = restoreClone(descriptor.value, seen, stores);
        if (descriptor) Object.defineProperty(out, key, descriptor);
    }
    return out as T;
}

/** Capture detached logical world state plus its own wasm linear-memory image. */
export function snapshot(world: World): WorldSnapshot {
    const state = world.state;
    return {
        // The ECS owner is identity, not solver data; snapshots never clone or retain it.
        state: clone(
            state,
            state.ecsState ? new Map<object, unknown>([[state.ecsState, null]]) : new Map(),
            snapshotStores(state),
        ),
        bytes: new Uint8Array(kernel(world.state.ecsState).memory.buffer).slice(),
    };
}

/** Restore into a live compatible World while its kernel has no other live World. */
export function restore(world: World, snapshot: WorldSnapshot): void {
    if (
        snapshot === null ||
        typeof snapshot !== "object" ||
        !(snapshot.bytes instanceof Uint8Array) ||
        snapshot.state === null ||
        typeof snapshot.state !== "object"
    )
        throw new Error("physics: invalid world snapshot");

    if (!world.isValid())
        throw new Error("physics: cannot restore a snapshot because its target World is not live");

    if (liveWorldCount(kernel(world.state.ecsState)) > 1)
        throw new Error(
            "physics: cannot restore a snapshot while other live Worlds share its kernel (WASM memory spans the whole kernel)",
        );

    const state = world.state;
    const restored = restoreClone(snapshot.state, new Map(), {
        body: state.bodyStore,
        shape: state.shapeStore,
        manifold: state.manifoldStore,
        broadPhase: state.broadPhase.store,
    }) as WorldState;
    // World identity and capacity belong to the target handle, not the snapshot's source handle.
    restored.ecsState = state.ecsState;
    restored.worldId = state.worldId;
    restored.generation = state.generation;
    restored.maxCapacity = state.maxCapacity;

    for (const key of Reflect.ownKeys(state)) {
        if (!Reflect.has(restored, key)) Reflect.deleteProperty(state, key);
    }
    for (const key of Reflect.ownKeys(restored)) {
        const descriptor = Object.getOwnPropertyDescriptor(restored, key);
        if (descriptor) Object.defineProperty(state, key, descriptor);
    }
    const memory = kernel(world.state.ecsState).memory;
    while (memory.buffer.byteLength < snapshot.bytes.byteLength) memory.grow(1);
    new Uint8Array(memory.buffer).set(snapshot.bytes);
    state.broadPhase.store.world = state;
    state.broadPhase.store.refreshViews();
    state.bodyStore.refreshViews();
    state.shapeStore.refreshViews();
    state.manifoldStore.refreshViews();
}
