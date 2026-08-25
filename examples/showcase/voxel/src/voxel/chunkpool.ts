// The chunk-aware allocation glue over `RegionAllocator`: pure, device-free, CPU-exact per-chunk sizing
// against `facesInChunk`. `RegionAllocator` itself stays generic — this module is the one place that
// knows a "region" means one chunk's exact face-pool slice.
//
// Two entry points. `fullRemesh` builds a fresh pool from scratch, allocating every non-empty chunk in
// ascending slot order into an empty allocator — first-fit against one contiguous free block packs them
// back-to-back with no gaps, so the pool's high-water mark (`tail`) is exact and no freed region is ever
// orphaned (there's nothing to free). `planRemesh` resizes only the chunks the caller names (a touched+halo
// set), freeing and reallocating a chunk whose count changed and leaving one alone whose count didn't
// (its geometry can still differ — a face flips at a shared seam without changing the count — but the
// caller re-emits it regardless; only the *region* is stable). Exhaustion during `planRemesh` (no free
// region big enough) signals the caller to fall back to `fullRemesh` over the whole grid — `pool` is left
// as whatever partial mutation happened, which is fine because the caller discards it wholesale on that
// signal (`mesher.ts`'s `remeshFull` builds a brand new pool, never repairs this one).

import { facesInChunk, SLOT_COUNT } from "./grid";
import { type Region, RegionAllocator } from "./pool";

export interface ChunkPool {
    readonly allocator: RegionAllocator;
    /** the live region for each chunk slot, `null` where the chunk currently has no exposed faces. */
    readonly region: (Region | null)[];
    /** high-water mark of allocated pool space — the single indirect draw covers `[0, tail)`. */
    tail: number;
}

export function createChunkPool(capacity: number): ChunkPool {
    return {
        allocator: new RegionAllocator(capacity),
        region: new Array(SLOT_COUNT).fill(null),
        tail: 0,
    };
}

export interface RemeshPlan {
    /** regions released during this plan (freed outright, or replaced by a differently-sized region) —
     *  the caller zero-fills their index range so a stale face doesn't render as a hole reopens elsewhere
     *  in the pool. Empty when {@link exhausted}. */
    readonly freed: readonly Region[];
    /** no free region fit a resized chunk — `pool` reflects a partial, discardable attempt; the caller
     *  falls back to {@link fullRemesh} over the whole grid rather than repairing this one. */
    readonly exhausted: boolean;
}

/** resize exactly the chunks in `slots` (already touched+halo-expanded by the caller) against their
 *  current {@link facesInChunk} count, mutating `pool` in place. */
export function planRemesh(
    pool: ChunkPool,
    data: Float32Array,
    slots: readonly number[],
): RemeshPlan {
    const freed: Region[] = [];
    for (const slot of slots) {
        const size = facesInChunk(data, slot);
        const existing = pool.region[slot];
        if (existing && existing.size === size) continue; // same footprint — caller still re-emits it
        if (existing) {
            pool.allocator.free(existing.start);
            pool.region[slot] = null;
            freed.push(existing);
        }
        if (size === 0) continue; // now empty — no region needed
        const granted = pool.allocator.alloc(size);
        if (!granted) return { freed, exhausted: true };
        pool.region[slot] = granted;
        pool.tail = Math.max(pool.tail, granted.start + granted.size);
    }
    return { freed, exhausted: false };
}

/** rebuild the whole pool from `data`: every one of the 512 chunks, in ascending slot order, into a fresh
 *  empty allocator. First-fit against one contiguous free block can only fail to grant a slot when the
 *  grid's total exposed-face count exceeds `capacity` — the same overflow the emit kernel's own per-chunk
 *  capacity guard already clips, so a failed grant here is silently skipped (capacity 0, matching the
 *  guard) rather than treated as a second exhaustion signal. */
export function fullRemesh(capacity: number, data: Float32Array): ChunkPool {
    const pool = createChunkPool(capacity);
    for (let slot = 0; slot < SLOT_COUNT; slot++) {
        const size = facesInChunk(data, slot);
        if (size === 0) continue;
        const granted = pool.allocator.alloc(size);
        if (!granted) continue;
        pool.region[slot] = granted;
        pool.tail = granted.start + granted.size; // ascending, gap-free allocation ⇒ monotonic tail
    }
    return pool;
}

/** the GPU chunk-table payload: `vec2<u32>` per slot (`region.start`, `region.size`), zero for an
 *  unallocated (empty) chunk — the kernel's exact-capacity guard reads `size` as the write bound. */
export function buildTable(pool: ChunkPool): Uint32Array {
    const out = new Uint32Array(SLOT_COUNT * 2);
    for (let slot = 0; slot < SLOT_COUNT; slot++) {
        const region = pool.region[slot];
        out[slot * 2] = region ? region.start : 0;
        out[slot * 2 + 1] = region ? region.size : 0;
    }
    return out;
}
