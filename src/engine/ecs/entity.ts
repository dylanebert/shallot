export type Entity = number;

/**
 * A safe-integer reference that survives frames and storage growth, not destruction or
 * eid reuse. Local checkpoint recovery restores their identity too. 0 means missing.
 * Valid only in the World and run that made it, not saves.
 * The low 32 bits hold the eid; the upper 21 hold its generation. After 2^21 reuses
 * of one eid the generation wraps, warns once per World, and can alias an old reference.
 */
export type EntityRef = number & { readonly __entityRef: unique symbol };

/**
 * world entity allocator. sparse-set membership + freelist for ID reuse.
 * iterate alive entities via `dense` up to `count`.
 */
export class Entities {
    private _dense: number[] = [];
    private _sparse: number[] = [];
    private _generation: number[] = [];
    private _count = 0;
    private _warnedWrap = false;
    private _nextId = 1;
    // freed ids in the first `_freeCount` slots; a pop or `length` write would release the backing store
    // and the next free would allocate it again, so the list keeps its high-water capacity.
    private _freelist: number[] = [];
    private _freeCount = 0;

    add(): Entity {
        const eid = this._freeCount > 0 ? this._freelist[--this._freeCount] : this._nextId++;
        this._sparse[eid] = this._count;
        this._dense[this._count++] = eid;
        // bump on every allocation (fresh or recycled) so a held (eid, generation) pair detects a realias
        const generation = ((this._generation[eid] ?? 0) + 1) % 2 ** 21;
        this._generation[eid] = generation;
        if (generation === 0 && !this._warnedWrap) {
            this._warnedWrap = true;
            console.warn(
                "Entity reference generation wrapped; old references may alias live entities",
            );
        }
        return eid;
    }

    /** @internal Allocator image, including inactive slots that determine reuse. */
    checkpoint() {
        return {
            dense: this._dense.slice(),
            sparse: this._sparse.slice(),
            generation: this._generation.slice(),
            count: this._count,
            nextId: this._nextId,
            freelist: this._freelist.slice(),
            freeCount: this._freeCount,
        };
    }

    /** @internal Restore identity and subsequent allocation; wrap warnings remain host diagnostics. */
    restore(state: ReturnType<Entities["checkpoint"]>): void {
        this._dense = state.dense.slice();
        this._sparse = state.sparse.slice();
        this._generation = state.generation.slice();
        this._count = state.count;
        this._nextId = state.nextId;
        this._freelist = state.freelist.slice();
        this._freeCount = state.freeCount;
    }

    generation(eid: Entity): number {
        return this._generation[eid] ?? 0;
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
