// Opaque handle packing + the internal id pool, ported from Box3D's id.h + id_pool.c.
//
// Handles pack (index1: i32, world0: u16, generation: u16) into a u64 — exact 64-bit integer
// work, so keys are `bigint`, not `number` (index1 << 32 exceeds 2^53). The pool is pure
// integer bookkeeping: which slot a body/shape/joint gets is observable through the eventual
// world state, so alloc/free order is ported op-for-op.

const U16 = 0xffffn;
const U64 = 0xffffffffffffffffn;

/** A decoded opaque handle. Body/shape/joint ids share this layout (see id.h). */
export type EntityId = { index1: number; world0: number; generation: number };

/** Pack a handle into a u64, matching b3StoreBodyId/ShapeId/JointId. */
export function storeId(id: EntityId): bigint {
    const index = BigInt.asUintN(64, BigInt(id.index1));
    return ((index << 32n) | (BigInt(id.world0) << 16n) | BigInt(id.generation)) & U64;
}

/** Unpack a u64 into a handle, matching b3LoadBodyId/ShapeId/JointId. */
export function loadId(x: bigint): EntityId {
    return {
        index1: Number(BigInt.asIntN(32, x >> 32n)),
        world0: Number((x >> 16n) & U16),
        generation: Number(x & U16),
    };
}

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

/** The high-water range of allocated ids (dense upper bound). */
export function idCapacity(pool: IdPool): number {
    return pool.nextIndex;
}
