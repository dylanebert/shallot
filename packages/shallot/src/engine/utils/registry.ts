/**
 * named-thing registry with stable numeric IDs. Each entry has a unique
 * `name`; the registry assigns a monotonic ID by registration order. IDs stay
 * stable for the lifetime of the registry (re-registering reuses the prior
 * ID; deleting an entry keeps the ID slot reserved). Producers consume IDs
 * for compact GPU-side routing; consumers reference entries by name
 *
 * @example
 * const Surfaces = new Registry<Surface>();
 * const id = Surfaces.register({ name: "checker", wgsl, bindings });
 * Surfaces.get("checker");      // the value
 * Surfaces.id("checker");       // the ID
 * Surfaces.name(id);            // "checker"
 */
export class Registry<T extends { name: string }> {
    private readonly _ids = new Map<string, number>();
    private readonly _names: string[] = [];
    private readonly _values = new Map<string, T>();

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
     * wipe every entry and reset the ID space. The reload seam — a rebuild re-registers from a clean
     * registry (ecs.md "clear then rebuild"), so a plugin toggled off in the editor leaves no stale entry
     * (a draw against torn-down buffers). Unlike {@link delete}, this frees the ID slots too; IDs are
     * stable only within one build generation.
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

    values(): IterableIterator<T> {
        return this._values.values();
    }

    [Symbol.iterator](): IterableIterator<T> {
        return this._values.values();
    }
}
