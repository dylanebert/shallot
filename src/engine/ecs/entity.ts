export type Entity = number;

/**
 * world entity allocator. sparse-set membership + freelist for ID reuse.
 * iterate alive entities via `dense` up to `count`.
 */
export class Entities {
    private _dense: number[] = [];
    private _sparse: number[] = [];
    private _stamp: number[] = [];
    private _count = 0;
    private _nextId = 1;
    // freed ids in the first `_freeCount` slots; a pop or `length` write would release the backing store
    // and the next free would allocate it again, so the list keeps its high-water capacity.
    private _freelist: number[] = [];
    private _freeCount = 0;

    add(): Entity {
        const eid = this._freeCount > 0 ? this._freelist[--this._freeCount] : this._nextId++;
        this._sparse[eid] = this._count;
        this._dense[this._count++] = eid;
        // bump on every allocation (fresh or recycled) so a held (eid, stamp) pair detects a realias
        this._stamp[eid] = (this._stamp[eid] ?? 0) + 1;
        return eid;
    }

    stamp(eid: Entity): number {
        return this._stamp[eid] ?? 0;
    }

    remove(eid: Entity): void {
        const idx = this._sparse[eid];
        if (idx === undefined || idx < 0 || idx >= this._count || this._dense[idx] !== eid) return;
        this._count--;
        const last = this._dense[this._count];
        this._dense[idx] = last;
        this._sparse[last] = idx;
        this._sparse[eid] = -1;
        this._freelist[this._freeCount++] = eid;
    }

    exists(eid: Entity): boolean {
        const idx = this._sparse[eid];
        return idx !== undefined && idx >= 0 && idx < this._count && this._dense[idx] === eid;
    }

    all(): readonly number[] {
        return this._dense.slice(0, this._count);
    }

    get dense(): readonly number[] {
        return this._dense;
    }

    get count(): number {
        return this._count;
    }
}
