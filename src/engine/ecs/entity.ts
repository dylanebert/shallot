/** World-local entity id. Ids start at 1, so 0 is never live; 0xFFFFFFFF is reserved as the invalid id
 * shaders read for a producer row. */
export type Entity = number;

const INVALID_ID = 0xffffffff;

/**
 * A safe-integer reference that survives frames and storage growth, not destruction or
 * eid reuse. Local snapshot recovery restores their identity too. 0 means missing.
 * Valid only in the World and run that made it, not saves.
 * The low 32 bits hold the eid; the upper 21 hold its generation. After 2^21 reuses
 * of one eid the generation wraps, warns once per World, and can alias an old reference.
 */
export type EntityRef = number & { readonly __entityRef: unique symbol };

/**
 * Reuses freed IDs. Only `dense[0..count)` contains live entities.
 */
export class Entities {
    private _dense: number[] = [];
    private _sparse: number[] = [];
    private _generation: number[] = [];
    private _count = 0;
    private _warnedWrap = false;
    private _nextId = 1;
    // Retain high-water backing capacity to avoid reallocating during entity churn.
    private _freelist: number[] = [];
    private _freeCount = 0;

    add(): Entity {
        if (this._freeCount === 0 && this._nextId >= INVALID_ID)
            throw new RangeError("Entity id range is exhausted; 0xFFFFFFFF is the invalid id");
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
    snapshot() {
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
    restore(state: ReturnType<Entities["snapshot"]>): void {
        if (state.nextId > INVALID_ID || state.dense.slice(0, state.count).includes(INVALID_ID))
            throw new RangeError(
                "Entity restore: id range exceeded or 0xFFFFFFFF invalid id is live",
            );
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
