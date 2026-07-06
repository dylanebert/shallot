import { type Components, idOf } from "./component";
import type { Entities, Entity } from "./entity";

const $op = Symbol("op");

interface QueryOp {
    [$op]: "not" | "and" | "or";
    components: any[];
}

function isOp(x: unknown): x is QueryOp {
    return x != null && typeof x === "object" && $op in x;
}

// op-interning + query-hash key off the same stable component id as membership
// (idOf, from component.ts) — one id system, and a reloaded component handle
// hashes identically so a re-run query resolves to its already-populated set.
const _opCache = new Map<string, QueryOp>();
function opKey(kind: string, components: any[]): string {
    const ids: number[] = new Array(components.length);
    for (let i = 0; i < components.length; i++) ids[i] = idOf(components[i]);
    return kind + ":" + ids.sort((a, b) => a - b).join(",");
}

// Interned op factory — `not(C)` returns the same QueryOp object on every call
// for the same C. Stable identity lets `Queries` resolve by terms-array element
// walk instead of recomputing the structural hash per query call. Single-arg
// path (the hot one) hits a per-kind WeakMap; multi-arg falls through to the
// shared structural-key Map.
function makeOp(kind: "not" | "and" | "or") {
    const cache1 = new WeakMap<object, QueryOp>();
    return (...components: any[]): QueryOp => {
        if (components.length === 1) {
            const c = components[0];
            let op = cache1.get(c);
            if (!op) cache1.set(c, (op = { [$op]: kind, components }));
            return op;
        }
        const key = opKey(kind, components);
        let op = _opCache.get(key);
        if (!op) _opCache.set(key, (op = { [$op]: kind, components }));
        return op;
    };
}

/** exclude entities with these components */
export const not = makeOp("not");
/** require all of these components */
export const and = makeOp("and");
/** match entities with at least one of these components */
export const or = makeOp("or");

// A pooled query iterator. The `next` object and its single reused IteratorResult are borrowed from
// the owning query's free-list and returned to it when the loop completes or breaks, so
// `for…of state.query([...])` allocates nothing after warmup (V8 doesn't reliably elide the per-loop
// iterator object). It snapshots count + the live `_dense` reference exactly as a fresh iterator
// would, so iteration-during-mutation is unchanged: a swap-remove of the current eid still visits
// every original member once (the swap only overwrites already-visited slots; the snapshotted count
// reads the original tail values at their original indices).
class QueryIterator implements Iterator<number> {
    private _i = 0;
    private _count = 0;
    private _dense: number[] = [];
    private _active = false;
    private readonly _r = { value: 0, done: false };

    constructor(private readonly _pool: QueryIterator[]) {}

    reset(dense: number[], count: number): void {
        this._dense = dense;
        this._count = count;
        this._i = 0;
        this._r.done = false;
        this._active = true;
    }

    next(): IteratorResult<number> {
        const r = this._r;
        if (this._i < this._count) r.value = this._dense[this._i++];
        else this.reclaim();
        return r;
    }

    // for…of calls return() on an early break/throw — reclaim there too so a broken-out loop's
    // iterator returns to the pool. The _active guard makes reclaim idempotent (a manual caller that
    // calls next() past done, or return() after a completed loop, can't double-push the same state).
    return(): IteratorResult<number> {
        this.reclaim();
        return this._r;
    }

    private reclaim(): void {
        this._r.done = true;
        if (this._active) {
            this._active = false;
            this._pool.push(this);
        }
    }
}

/**
 * matched-entity set + iteration for one registered query. owns sparse-set
 * state inline (no wrapper class). iteration snapshots count at start so
 * callers can mutate during iteration, e.g. `for (const eid of state.query([Spawn,
 * not(Initialized)])) state.add(eid, Initialized)`.
 */
export class RegisteredQuery implements Iterable<number> {
    readonly required: any[] = [];
    readonly excluded: any[] = [];
    readonly orGroups: any[][] = [];
    readonly all = new Set<any>();
    private _dense: number[] = [];
    private _sparse: number[] = [];
    private _count = 0;
    private _iterPool: QueryIterator[] = [];

    constructor(terms: readonly unknown[]) {
        for (const term of terms) {
            if (isOp(term)) {
                const k = term[$op];
                if (k === "not") this.excluded.push(...term.components);
                else if (k === "and") this.required.push(...term.components);
                else this.orGroups.push(term.components);
            } else {
                this.required.push(term);
            }
        }
        for (const c of this.required) this.all.add(c);
        for (const c of this.excluded) this.all.add(c);
        for (const g of this.orGroups) for (const c of g) this.all.add(c);
    }

    matches(eid: Entity, components: Components): boolean {
        for (const c of this.required) if (!components.has(eid, c)) return false;
        for (const c of this.excluded) if (components.has(eid, c)) return false;
        for (const group of this.orGroups) {
            let matched = false;
            for (const c of group) {
                if (components.has(eid, c)) {
                    matched = true;
                    break;
                }
            }
            if (!matched) return false;
        }
        return true;
    }

    add(eid: Entity): void {
        const idx = this._sparse[eid];
        if (idx !== undefined && idx >= 0 && idx < this._count && this._dense[idx] === eid) return;
        this._sparse[eid] = this._count;
        this._dense[this._count++] = eid;
    }

    remove(eid: Entity): void {
        const idx = this._sparse[eid];
        if (idx === undefined || idx < 0 || idx >= this._count || this._dense[idx] !== eid) return;
        this._count--;
        const last = this._dense[this._count];
        this._dense[idx] = last;
        this._sparse[last] = idx;
        this._sparse[eid] = -1;
    }

    [Symbol.iterator](): Iterator<number> {
        const it = this._iterPool.pop() ?? new QueryIterator(this._iterPool);
        it.reset(this._dense, this._count);
        return it;
    }
}

// Multi-level term-cache node. Each `terms` array element addresses one Map
// hop; the leaf carries the RegisteredQuery on a Symbol key. Walk by element
// identity skips parse + hash + string allocation on hot-path calls.
const $leaf = Symbol("leaf");
type TermNode = Map<unknown, TermNode> & { [$leaf]?: RegisteredQuery };

function hashOf(rq: RegisteredQuery): string {
    const req = rq.required.map(idOf).sort((a, b) => a - b);
    const exc = rq.excluded.map(idOf).sort((a, b) => a - b);
    const orr = rq.orGroups
        .map((g) =>
            g
                .map(idOf)
                .sort((a, b) => a - b)
                .join(","),
        )
        .sort();
    return `${req.join(",")};${exc.join(",")};${orr.join("|")}`;
}

export class Queries {
    private _all: RegisteredQuery[] = [];
    private _byHash = new Map<string, RegisteredQuery>();
    private _byTerms: TermNode = new Map() as TermNode;
    // keyed by component id (idOf), not the object — array-by-id, like membership
    private _byComponent: RegisteredQuery[][] = [];

    /**
     * resolve `terms` to a registered query, registering on first sight.
     * fast path — interned ops give terms stable element identity, so a
     * previously registered query resolves via a Map walk with no parse,
     * hash, or string allocation.
     */
    find(terms: readonly unknown[], components: Components, entities: Entities): RegisteredQuery {
        let node: TermNode | undefined = this._byTerms;
        for (let i = 0; i < terms.length; i++) {
            node = node.get(terms[i]) as TermNode | undefined;
            if (!node) break;
        }
        if (node) {
            const cached = node[$leaf];
            if (cached) return cached;
        }
        return this._register(terms, components, entities);
    }

    onComponentChanged(eid: Entity, component: any, components: Components): void {
        const queries = this._byComponent[idOf(component)];
        if (!queries) return;
        for (let i = 0; i < queries.length; i++) {
            const rq = queries[i];
            if (rq.matches(eid, components)) rq.add(eid);
            else rq.remove(eid);
        }
    }

    onEntityRemoved(eid: Entity): void {
        for (let i = 0; i < this._all.length; i++) this._all[i].remove(eid);
    }

    clear(): void {
        this._all.length = 0;
        this._byHash.clear();
        this._byTerms.clear();
        this._byComponent.length = 0;
    }

    private _register(
        terms: readonly unknown[],
        components: Components,
        entities: Entities,
    ): RegisteredQuery {
        const rq = new RegisteredQuery(terms);
        const hash = hashOf(rq);
        const existing = this._byHash.get(hash);
        if (existing) {
            this._cacheTerms(terms, existing);
            return existing;
        }
        for (let i = 0; i < entities.count; i++) {
            const eid = entities.dense[i];
            if (rq.matches(eid, components)) rq.add(eid);
        }
        this._all.push(rq);
        this._byHash.set(hash, rq);
        this._cacheTerms(terms, rq);
        for (const c of rq.all) {
            const id = idOf(c);
            let list = this._byComponent[id];
            if (!list) this._byComponent[id] = list = [];
            list.push(rq);
        }
        return rq;
    }

    private _cacheTerms(terms: readonly unknown[], rq: RegisteredQuery): void {
        let node: TermNode = this._byTerms;
        for (let i = 0; i < terms.length; i++) {
            const t = terms[i];
            let next = node.get(t) as TermNode | undefined;
            if (!next) node.set(t, (next = new Map() as TermNode));
            node = next;
        }
        node[$leaf] = rq;
    }
}
