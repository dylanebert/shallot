// Box3D id_pool.c.

/** A public opaque handle. Index and generation are owned by the corresponding kernel pool. */
export type EntityId = { index1: number; world0: number; generation: number };

// --- id pool --------------------------------------------------------------------------------

/** A recycling pool of dense integer ids: free ids are reused LIFO before extending the range. */
export type IdPool = { freeArray: number[]; nextIndex: number };

export function createIdPool(): IdPool {
    return { freeArray: [], nextIndex: 0 };
}

export function allocId(pool: IdPool): number {
    if (pool.freeArray.length > 0) {
        return pool.freeArray.pop() as number;
    }
    const id = pool.nextIndex;
    pool.nextIndex += 1;
    return id;
}

export function freeId(pool: IdPool, id: number): void {
    // b3FreeId retains this branch even though allocated ids are strictly below nextIndex.
    if (id === pool.nextIndex) {
        pool.nextIndex -= 1;
        return;
    }
    pool.freeArray.push(id);
}

export function idCount(pool: IdPool): number {
    return pool.nextIndex - pool.freeArray.length;
}
