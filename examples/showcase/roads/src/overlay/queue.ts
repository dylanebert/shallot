// The overlay atlas's two pure, device-free mechanisms — split out of atlas.ts so the per-frame throttle
// and the layer-allocation policy are `bun test`-reachable without a GPU (`atlas.ts`'s `redraw` calls both,
// then does the real `writeTexture`/`writeBuffer` work these functions never touch).

/**
 * pop up to `n` ids off the front of `pending` (FIFO — oldest mark redraws first), removing each from
 * `pendingSet` too. Mutates both in place (the same queue+set pair `atlas.ts` owns across frames) and
 * returns the popped ids. Never pops more than `pending.length` has, so the caller's "how many did I
 * redraw this frame" is just the returned array's length.
 *
 * @example drain([1, 2, 3, 4], new Set([1, 2, 3, 4]), 2) // → [1, 2], pending left with [3, 4]
 */
export function drain(pending: number[], pendingSet: Set<number>, n: number): number[] {
    const count = Math.min(n, pending.length);
    const ids: number[] = [];
    for (let i = 0; i < count; i++) {
        const id = pending.shift() as number;
        pendingSet.delete(id);
        ids.push(id);
    }
    return ids;
}

/**
 * resolve tile `id`'s atlas layer against the indirection CPU mirror `cpu` (negative = unallocated):
 * already resident → its existing layer; otherwise pop the next free layer off `free` and write
 * `cpu[id]` in place. Throws when `free` is empty rather than silently overwriting a resident layer —
 * no plausible-fallback substitute for a real capacity limit. The free list is the complete shape
 * (`coding.md`): `release` pushes layers back, so an edit that releases old tiles before allocating new
 * ones never exhausts a layer pool the counter-with-a-reset stopgap could only flush wholesale.
 *
 * @example allocate(new Int32Array(4).fill(-1), 2, [0, 1, 2, 3], 4) // → 0, cpu[2] === 0
 */
export function allocate(cpu: Int32Array, id: number, free: number[], capacity: number): number {
    const existing = cpu[id];
    if (existing >= 0) return existing;
    if (free.length === 0) {
        throw new Error(
            `overlay atlas: capacity exceeded (${capacity} layers) allocating tile ${id}`,
        );
    }
    const layer = free.pop() as number;
    cpu[id] = layer;
    return layer;
}

/**
 * release tile `ids` back to the free list: clear each resident id's indirection to −1, push its layer
 * back onto `free`, and remove it from the dirty queue (`pending`/`pendingSet`) so a released tile is
 * never redrawn. Ids that are not resident (indirection already −1) are skipped — a release is
 * idempotent over the resident set. Mutates `cpu`/`free`/`pending`/`pendingSet` in place, the same
 * convention {@link drain} uses. This is the one path reseed and edit share: {@link invalidate} is
 * release-all over this mechanism, and `atlas.ts`'s `retile` is release-difference + mark-union-dirty.
 *
 * @example release(Int32Array.from([2, -1, 0]), [2], [3], [], new Set()) // cpu → [-1,-1,-1], free → [3, 0]
 */
export function release(
    cpu: Int32Array,
    ids: number[],
    free: number[],
    pending: number[],
    pendingSet: Set<number>,
): void {
    for (const id of ids) {
        const layer = cpu[id];
        if (layer >= 0) {
            free.push(layer);
            cpu[id] = -1;
        }
        if (pendingSet.has(id)) {
            pendingSet.delete(id);
            const idx = pending.indexOf(id);
            if (idx >= 0) pending.splice(idx, 1);
        }
    }
}

/**
 * release every resident tile and refill the free list to full `capacity` — `atlas.ts`'s document-swap
 * invalidation (`terrain.ts`'s `regenerate`, the F9 reseed control), called before the swapped-in
 * document's own tiles are marked dirty so they land in a freshly emptied atlas rather than packing onto
 * whatever the old document left resident. Mutates `cpu`/`free`/`pending`/`pendingSet` in place. The
 * refill is belt-and-braces: {@link release} already pushes every resident layer back, so the free list
 * would arrive at full capacity on its own, but an explicit refill guarantees the invariant
 * (resident + free = capacity) even if a caller hand-built a partial free list.
 *
 * @example invalidate(Int32Array.from([2, -1, 0]), [3], 4, [5], new Set([5])) // cpu → [-1,-1,-1], free → [3,2,1,0]
 */
export function invalidate(
    cpu: Int32Array,
    free: number[],
    capacity: number,
    pending: number[],
    pendingSet: Set<number>,
): void {
    const resident: number[] = [];
    for (let id = 0; id < cpu.length; id++) {
        if (cpu[id] >= 0) resident.push(id);
    }
    release(cpu, resident, free, pending, pendingSet);
    pending.length = 0;
    pendingSet.clear();
    free.length = 0;
    for (let i = capacity - 1; i >= 0; i--) free.push(i);
}
