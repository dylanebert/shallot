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

/** a resident-or-newly-assigned atlas layer for tile `id`, plus the free-running counter's next value. */
export interface Allocation {
    layer: number;
    nextLayer: number;
}

/**
 * resolve tile `id`'s atlas layer against the indirection CPU mirror `cpu` (negative = unallocated):
 * already resident → its existing layer, unchanged counter; otherwise the next free layer, `cpu[id]`
 * written in place. Throws when `nextLayer` would exceed `capacity` rather than silently overwriting a
 * resident layer or wrapping the counter — no plausible-fallback substitute for a real capacity limit.
 *
 * @example allocate(new Int32Array(4).fill(-1), 2, 0, 4) // → { layer: 0, nextLayer: 1 }, cpu[2] === 0
 */
export function allocate(
    cpu: Int32Array,
    id: number,
    nextLayer: number,
    capacity: number,
): Allocation {
    const existing = cpu[id];
    if (existing >= 0) return { layer: existing, nextLayer };
    if (nextLayer >= capacity) {
        throw new Error(
            `overlay atlas: capacity exceeded (${capacity} layers) allocating tile ${id}`,
        );
    }
    cpu[id] = nextLayer;
    return { layer: nextLayer, nextLayer: nextLayer + 1 };
}
