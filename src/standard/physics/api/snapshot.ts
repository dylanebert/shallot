import { kernel } from "../kernel/kernel";
import type { WorldState } from "../world/world";
import type { World } from "./world";

/** Opaque binary snapshot of a wasm-backed physics world. */
export type WorldSnapshot = Uint8Array;

type SavedSnapshot = { state: WorldState; memory: Uint8Array };
const saved = new Map<number, SavedSnapshot>();
let nextId = 1;

function clone<T>(value: T, seen: Map<object, unknown>, opaque: Set<object>): T {
    if (value === null || typeof value !== "object") return value;
    if (opaque.has(value as object)) return value;
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
        for (const [k, v] of value) out.set(clone(k, seen, opaque), clone(v, seen, opaque));
        return out as T;
    }
    if (value instanceof Set) {
        const out = new Set();
        seen.set(value, out);
        for (const item of value) out.add(clone(item, seen, opaque));
        return out as T;
    }
    if (Array.isArray(value)) {
        const out: unknown[] = [];
        seen.set(value, out);
        for (const item of value) out.push(clone(item, seen, opaque));
        return out as T;
    }
    const out = Object.create(Object.getPrototypeOf(value)) as Record<PropertyKey, unknown>;
    seen.set(value as object, out);
    for (const key of Reflect.ownKeys(value)) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor && "value" in descriptor)
            descriptor.value = clone(descriptor.value, seen, opaque);
        if (descriptor) Object.defineProperty(out, key, descriptor);
    }
    return out as T;
}

function opaqueStores(state: WorldState): Set<object> {
    return new Set([
        state.bodyStore,
        state.shapeStore,
        state.manifoldStore,
        state.broadPhase.store,
    ]);
}

/** Capture all logical world state plus the current wasm linear-memory image. */
export function snapshot(world: World): WorldSnapshot {
    const id = nextId++;
    const state = world.state;
    const savedState = clone(state, new Map(), opaqueStores(state));
    const memory = new Uint8Array(kernel().memory.buffer).slice();
    saved.set(id, { state: savedState, memory });
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setBigUint64(0, BigInt(id), true);
    return bytes;
}

/** Restore a snapshot into the same world handle. Snapshots are immutable and may be replayed. */
export function restore(world: World, bytes: WorldSnapshot): void {
    if (bytes.byteLength !== 8) throw new Error("physics: invalid world snapshot");
    const id = Number(
        new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true),
    );
    const entry = saved.get(id);
    if (!entry) throw new Error("physics: unknown world snapshot");
    const state = world.state;
    const restored = clone(entry.state, new Map(), opaqueStores(state));
    for (const key of Reflect.ownKeys(state)) {
        if (!Reflect.has(restored, key)) Reflect.deleteProperty(state, key);
    }
    for (const key of Reflect.ownKeys(restored)) {
        const descriptor = Object.getOwnPropertyDescriptor(restored, key);
        if (descriptor) Object.defineProperty(state, key, descriptor);
    }
    const memory = kernel().memory;
    while (memory.buffer.byteLength < entry.memory.byteLength) memory.grow(1);
    new Uint8Array(memory.buffer).set(entry.memory);
    state.broadPhase.store.world = state;
    state.broadPhase.store.refreshViews();
    state.bodyStore.refreshViews();
    state.shapeStore.refreshViews();
    state.manifoldStore.refreshViews();
}
