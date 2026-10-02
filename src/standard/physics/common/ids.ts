// Non-body entity bookkeeping, ported from Box3D's id_pool.c. Public body handles are backed by the
// kernel-owned body record pool; this small TypeScript pool remains for shapes, joints, contacts and
// solver-set/island organization until their own ownership stages.

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
    // Mirrors b3FreeId verbatim. id === nextIndex never holds (id < nextIndex always), so this
    // branch is dead — a known quirk in the C, kept for a faithful behavioral match.
    if (id === pool.nextIndex) {
        pool.nextIndex -= 1;
        return;
    }
    pool.freeArray.push(id);
}

/** Number of ids currently handed out. */
export function idCount(pool: IdPool): number {
    return pool.nextIndex - pool.freeArray.length;
}
