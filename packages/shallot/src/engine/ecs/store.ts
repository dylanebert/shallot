export type Entity = number;

export class SparseSet implements Iterable<number> {
    _dense: number[] = [];
    _sparse: number[] = [];
    _count = 0;

    add(val: number): boolean {
        const idx = this._sparse[val];
        if (idx !== undefined && idx >= 0 && idx < this._count && this._dense[idx] === val)
            return false;
        this._sparse[val] = this._count;
        this._dense[this._count++] = val;
        return true;
    }

    remove(val: number): boolean {
        const idx = this._sparse[val];
        if (idx === undefined || idx < 0 || idx >= this._count || this._dense[idx] !== val)
            return false;
        this._count--;
        const last = this._dense[this._count];
        this._dense[idx] = last;
        this._sparse[last] = idx;
        this._sparse[val] = -1;
        return true;
    }

    has(val: number): boolean {
        const idx = this._sparse[val];
        return idx !== undefined && idx >= 0 && idx < this._count && this._dense[idx] === val;
    }

    get dense(): readonly number[] {
        return this._dense;
    }

    get count(): number {
        return this._count;
    }

    [Symbol.iterator](): Iterator<number> {
        let i = 0;
        const dense = this._dense;
        const count = this._count;
        return {
            next() {
                return i < count
                    ? { value: dense[i++], done: false }
                    : { done: true, value: undefined as any };
            },
        };
    }
}

export class EntityPool {
    private _set = new SparseSet();
    private _nextId = 1;
    private _freelist: number[] = [];

    add(): Entity {
        const eid = this._freelist.length > 0 ? this._freelist.pop()! : this._nextId++;
        this._set.add(eid);
        return eid;
    }

    remove(eid: Entity): void {
        if (this._set.remove(eid)) this._freelist.push(eid);
    }

    exists(eid: Entity): boolean {
        return this._set.has(eid);
    }

    all(): readonly number[] {
        return this._set._dense.slice(0, this._set._count);
    }

    get dense(): readonly number[] {
        return this._set._dense;
    }

    get alive(): number {
        return this._set._count;
    }

    get count(): number {
        return this._set._count;
    }
}

const BITS_PER_GEN = 31;

interface ComponentMeta {
    gen: number;
    bit: number;
}

export class ComponentStore {
    private _nextBit = 0;
    private _gen = 0;
    private _map = new Map<any, ComponentMeta>();
    private _masks: number[][] = [[]];

    has(eid: Entity, component: any): boolean {
        const meta = this._map.get(component);
        if (!meta) return false;
        return ((this._masks[meta.gen][eid] ?? 0) & meta.bit) !== 0;
    }

    add(eid: Entity, component: any): boolean {
        const meta = this.ensure(component);
        const prev = this._masks[meta.gen][eid] ?? 0;
        if (prev & meta.bit) return false;
        this._masks[meta.gen][eid] = prev | meta.bit;
        return true;
    }

    remove(eid: Entity, component: any): boolean {
        const meta = this._map.get(component);
        if (!meta) return false;
        const prev = this._masks[meta.gen][eid] ?? 0;
        if (!(prev & meta.bit)) return false;
        this._masks[meta.gen][eid] = prev & ~meta.bit;
        return true;
    }

    getAll(eid: Entity): any[] {
        const result: any[] = [];
        for (const [component, meta] of this._map) {
            if ((this._masks[meta.gen][eid] ?? 0) & meta.bit) result.push(component);
        }
        return result;
    }

    clear(eid: Entity): void {
        for (let g = 0; g <= this._gen; g++) this._masks[g][eid] = 0;
    }

    private ensure(component: any): ComponentMeta {
        let meta = this._map.get(component);
        if (meta) return meta;
        if (this._nextBit >= BITS_PER_GEN) {
            this._gen++;
            this._nextBit = 0;
            this._masks.push([]);
        }
        meta = { gen: this._gen, bit: 1 << this._nextBit++ };
        this._map.set(component, meta);
        return meta;
    }
}

type ObservableCallback = (eid: Entity, ...args: any[]) => void;

export interface HookInstance {
    type: "add" | "remove";
    terms: any[];
}

function hookDef(type: "add" | "remove") {
    return (...terms: any[]): HookInstance => ({ type, terms });
}

/**
 * hook that fires when all specified components are present
 * @params ...terms
 */
export const onAdd = hookDef("add");
/**
 * hook that fires before a component is removed
 * @params ...terms
 */
export const onRemove = hookDef("remove");

export class ObservableStore {
    private _observers: { hook: HookInstance; callback: ObservableCallback }[] = [];

    subscribe(hook: HookInstance, callback: ObservableCallback): () => void {
        const entry = { hook, callback };
        this._observers.push(entry);
        return () => {
            const idx = this._observers.indexOf(entry);
            if (idx >= 0) this._observers.splice(idx, 1);
        };
    }

    notifyAdd(eid: Entity, component: any, components: ComponentStore): void {
        for (const { hook, callback } of this._observers) {
            if (hook.type !== "add") continue;
            if (this.triggered(eid, hook.terms, component, components)) callback(eid);
        }
    }

    notifyRemove(eid: Entity, component: any, components: ComponentStore): void {
        for (const { hook, callback } of this._observers) {
            if (hook.type !== "remove") continue;
            if (this.triggered(eid, hook.terms, component, components)) callback(eid);
        }
    }

    private triggered(
        eid: Entity,
        terms: any[],
        trigger: any,
        components: ComponentStore,
    ): boolean {
        let hasTrigger = false;
        for (const term of terms) {
            if (term === trigger) hasTrigger = true;
            else if (!components.has(eid, term)) return false;
        }
        return hasTrigger;
    }
}

export type { ObservableCallback };
