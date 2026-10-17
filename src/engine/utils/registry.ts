/**
 * Names receive monotonic IDs in first-registration order. Deleting or overwriting an entry
 * preserves its ID; {@link clear} resets the ID space and {@link restore} replaces it.
 */
export class Registry<T extends { name: string }> {
    private readonly _ids = new Map<string, number>();
    private readonly _names: string[] = [];
    private readonly _values = new Map<string, T>();

    /** Container image including deleted entries' reserved IDs. Values are retained by reference;
     * an owner recovering mutable entries must copy them at capture and restore. */
    snapshot(): { names: string[]; values: Map<string, T> } {
        return { names: this._names.slice(), values: new Map(this._values) };
    }

    /** Restore IDs and live entries in place; value ownership remains the caller's. */
    restore(state: ReturnType<Registry<T>["snapshot"]>): void {
        this.clear();
        for (let id = 0; id < state.names.length; id++) {
            const name = state.names[id];
            this._names.push(name);
            this._ids.set(name, id);
        }
        for (const [name, value] of state.values) this._values.set(name, value);
    }

    /** register or overwrite an entry. Returns its (stable) ID */
    register(spec: T): number {
        let id = this._ids.get(spec.name);
        if (id === undefined) {
            id = this._names.length;
            this._ids.set(spec.name, id);
            this._names.push(spec.name);
        }
        this._values.set(spec.name, spec);
        return id;
    }

    /** remove an entry. The ID slot stays reserved — re-registering reuses it */
    delete(name: string): boolean {
        return this._values.delete(name);
    }

    /**
     * Remove every entry and reset the ID space, including slots reserved by {@link delete}.
     */
    clear(): void {
        this._ids.clear();
        this._names.length = 0;
        this._values.clear();
    }

    get(name: string): T | undefined {
        return this._values.get(name);
    }

    has(name: string): boolean {
        return this._values.has(name);
    }

    /** numeric ID for a registered name, or `undefined` if never registered */
    id(name: string): number | undefined {
        return this._ids.get(name);
    }

    /** name at a given ID, or `undefined` if out of range */
    name(id: number): string | undefined {
        return this._names[id];
    }

    /** number of live (non-deleted) entries */
    get size(): number {
        return this._values.size;
    }

    /** Visit live entries without an iterator, in Map insertion order, not numeric-ID order. */
    forEach(callback: (value: T) => void): void {
        this._values.forEach(callback);
    }

    values(): IterableIterator<T> {
        return this._values.values();
    }

    [Symbol.iterator](): IterableIterator<T> {
        return this.values();
    }
}
