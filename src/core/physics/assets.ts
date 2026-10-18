import { HandleAllocator } from "../../engine/utils";

/** World-local assets keyed by generation-checked handles, never by their display names.
 * Definitions are immutable after addition. Removal and clear invalidate handles; a later add
 * cannot make a stale handle resolve to a different definition. Handles fit nonnegative i32. */
export class PhysicsAssets<T> implements Iterable<T> {
    private readonly _handles = new HandleAllocator();
    private _items: (T | undefined)[] = [];
    private _order: number[] = [];

    add(value: T): number {
        const handle = this._handles.allocate();
        this._items[this._handles.index(handle)!] = value;
        this._order.push(handle);
        return handle;
    }

    get(handle: number): T | undefined {
        const index = this._handles.index(handle);
        return index === undefined ? undefined : this._items[index];
    }

    has(handle: number): boolean {
        return this._handles.index(handle) !== undefined;
    }

    remove(handle: number): T | undefined {
        const index = this._handles.index(handle);
        if (index === undefined) return undefined;
        const value = this._items[index];
        this._handles.release(handle);
        this._items[index] = undefined;
        this._order.splice(this._order.indexOf(handle), 1);
        return value;
    }

    clear(): void {
        this._handles.clear();
        this._items.length = 0;
        this._order.length = 0;
    }

    get size(): number {
        return this._order.length;
    }

    *entries(): IterableIterator<[number, T]> {
        for (const handle of this._order) yield [handle, this.get(handle)!];
    }

    *values(): IterableIterator<T> {
        for (const handle of this._order) yield this.get(handle)!;
    }

    [Symbol.iterator](): IterableIterator<T> {
        return this.values();
    }

    /** Values are retained by reference; recovery owners copy mutable data. */
    snapshot() {
        return {
            handles: this._handles.snapshot(),
            items: this._items.slice(),
            order: this._order.slice(),
        };
    }

    /** Restore live handles and generations in place, replacing the allocation timeline. */
    restore(state: ReturnType<PhysicsAssets<T>["snapshot"]>): void {
        this._handles.restore(state.handles);
        this._items = state.items.slice();
        this._order = state.order.slice();
    }
}
