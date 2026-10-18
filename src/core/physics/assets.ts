/** World-local assets keyed by numeric handles, never by their display names.
 * Definitions are immutable after addition. A removed handle may be reissued by the next add;
 * remove its consumers too, or they will reference its successor. */
export class PhysicsAssets<T> implements Iterable<T> {
    private readonly _items = new Map<number, T>();
    private _freeHandle = 0;

    add(value: T): number {
        const handle = this._freeHandle;
        this._items.set(handle, value);
        while (this._items.has(this._freeHandle)) this._freeHandle++;
        return handle;
    }

    get(handle: number): T | undefined {
        return this._items.get(handle);
    }

    has(handle: number): boolean {
        return this._items.has(handle);
    }

    remove(handle: number): T | undefined {
        const value = this._items.get(handle);
        if (this._items.delete(handle)) this._freeHandle = Math.min(this._freeHandle, handle);
        return value;
    }

    clear(): void {
        this._items.clear();
        this._freeHandle = 0;
    }

    get size(): number {
        return this._items.size;
    }

    entries(): MapIterator<[number, T]> {
        return this._items.entries();
    }

    values(): MapIterator<T> {
        return this._items.values();
    }

    [Symbol.iterator](): MapIterator<T> {
        return this.values();
    }

    /** Values are retained by reference; recovery owners copy mutable data. */
    snapshot(): { entries: Map<number, T>; freeHandle: number } {
        return { entries: new Map(this._items), freeHandle: this._freeHandle };
    }

    /** Restore live handles and allocation state in place. */
    restore(state: ReturnType<PhysicsAssets<T>["snapshot"]>): void {
        this.clear();
        for (const [handle, value] of state.entries) this._items.set(handle, value);
        this._freeHandle = state.freeHandle;
    }
}
