import type { BroadStore } from "../kernel/broadcolumns";
import { kernel } from "../kernel/kernel";

export type HashSet = {
    store: BroadStore;
    initCapacity: number;
};

export function createSet(capacity: number, store: BroadStore): HashSet {
    return {
        store,
        initCapacity: capacity,
    };
}

export function ensureResident(set: HashSet): void {
    set.store.refreshIfStale();
    const k = kernel(set.store.ecsState);
    if (k.broadSetCap() === 0) k.broadCreateSet(set.initCapacity);
}

export function addKey(set: HashSet, a: number, b: number, child: number): boolean {
    ensureResident(set);
    return kernel(set.store.ecsState).broadAddPair(a, b, child) !== 0;
}
export function removeKey(set: HashSet, a: number, b: number, child: number): boolean {
    ensureResident(set);
    return kernel(set.store.ecsState).broadRemovePair(a, b, child) !== 0;
}
