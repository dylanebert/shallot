// A pure, device-free first-fit free-list allocator over the face-pool number line (the CPU-side half of
// scoped remesh): each touched chunk gets an exactly-sized contiguous region of `[0, capacity)`, sized by
// the caller's own exact face count (`facesInChunk` in `grid.ts`) — no worst-case reservation. `alloc`
// returns null on exhaustion (no first-fit region large enough) rather than throwing; the caller falls
// back to full realloc + full re-emit through the same machinery on that signal — always correct, rare.
// Regions merge back into the free list on `free`, so churn from repeated carves doesn't fragment into
// unusable slivers faster than it has to.

export interface Region {
    readonly start: number;
    readonly size: number;
}

export class RegionAllocator {
    private _freeList: Region[];
    private readonly _live = new Map<number, Region>();

    constructor(readonly capacity: number) {
        this._freeList = capacity > 0 ? [{ start: 0, size: capacity }] : [];
    }

    /** the free regions, sorted by start, coalesced — read-only, for tests and diagnostics. */
    regions(): readonly Region[] {
        return this._freeList;
    }

    /** first-fit: the first free region large enough for `size`, split if it overshoots. Null signals
     *  exhaustion — no single free region can satisfy the request (including full fragmentation). */
    alloc(size: number): Region | null {
        if (size <= 0) throw new Error("RegionAllocator: alloc size must be positive");
        const i = this._freeList.findIndex((r) => r.size >= size);
        if (i === -1) return null;
        const region = this._freeList[i];
        const granted: Region = { start: region.start, size };
        if (region.size === size) this._freeList.splice(i, 1);
        else this._freeList[i] = { start: region.start + size, size: region.size - size };
        this._live.set(granted.start, granted);
        return granted;
    }

    /** release a region previously returned by {@link alloc}, coalescing with adjacent free neighbours so
     *  free capacity never fragments below what a later alloc could otherwise find. */
    free(start: number): void {
        const region = this._live.get(start);
        if (!region) throw new Error(`RegionAllocator: free of unknown region at ${start}`);
        this._live.delete(start);

        let i = this._freeList.findIndex((r) => r.start > region.start);
        if (i === -1) i = this._freeList.length;
        this._freeList.splice(i, 0, region);

        // coalesce with the following neighbour first — its index doesn't shift by doing so.
        if (i + 1 < this._freeList.length) {
            const next = this._freeList[i + 1];
            if (region.start + region.size === next.start) {
                this._freeList[i] = { start: region.start, size: region.size + next.size };
                this._freeList.splice(i + 1, 1);
            }
        }
        // coalesce with the preceding neighbour.
        if (i > 0) {
            const prev = this._freeList[i - 1];
            const cur = this._freeList[i];
            if (prev.start + prev.size === cur.start) {
                this._freeList[i - 1] = { start: prev.start, size: prev.size + cur.size };
                this._freeList.splice(i, 1);
            }
        }
    }
}
