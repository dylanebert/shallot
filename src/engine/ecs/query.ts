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

interface QueryOrder {
    dense: number[];
    active: number;
}

const ascending = (a: number, b: number) => a - b;

// Pool both the iterator and its result; unchanged iterations need neither sorting nor allocation.
class QueryIterator implements Iterator<number> {
    private _i = 0;
    private _count = 0;
    private _dense: number[] = [];
    private _order!: QueryOrder;
    private _active = false;
    private readonly _r = { value: 0, done: false };
    private readonly _pool: QueryIterator[];

    constructor(pool: QueryIterator[]) {
        this._pool = pool;
    }

    reset(order: QueryOrder): void {
        this._order = order;
        this._dense = order.dense;
        this._count = order.dense.length;
        order.active++;
        this._i = 0;
        this._r.done = false;
        this._active = true;
    }

    next(): IteratorResult<number> {
        const r = this._r;
        const dense = this._dense;
        while (this._i < this._count) {
            const eid = dense[this._i++];
            if (!eid) continue;
            r.value = eid;
            return r;
        }
        this.reclaim();
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
            this._order.active--;
            this._pool.push(this);
        }
    }
}

/**
 * When no iterator of this query is active, starting an iteration compacts and sorts changed
 * membership in place into ascending eid order. A nested iteration never reorders the shared array:
 * it visits the current order, skipping removed members. Sorting waits until every active iterator
 * completes or returns. Unchanged iterations allocate nothing after iterator-pool warmup.
 * Removing a member never reorders the remaining members. Each iteration visits its starting
 * memberships at most once: removing the current or an already visited member leaves later visits
 * unchanged; removing an unvisited member skips it. Additions, including removed/re-added members
 * and reused eids, wait for an iteration started after the addition.
 * Iterators and their result objects are borrowed until completion or return; do not retain results.
 */
export class RegisteredQuery implements Iterable<number> {
    readonly required: any[] = [];
    readonly excluded: any[] = [];
    readonly orGroups: any[][] = [];
    readonly all = new Set<any>();
    private readonly _order: QueryOrder = { dense: [], active: 0 };
    private _sparse: number[] = [];
    private _dirty = false;
    // dense[0, _sorted) stays ascending apart from tombstones, so a rebuild sorts only later appends.
    private _sorted = 0;
    private readonly _tail: number[] = [];
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
        const dense = this._order.dense;
        const idx = this._sparse[eid];
        if (idx !== undefined && idx >= 0 && dense[idx] === eid) return;
        this._sparse[eid] = dense.length;
        dense.push(eid);
        this._dirty = true;
    }

    remove(eid: Entity): void {
        const idx = this._sparse[eid];
        if (idx === undefined || idx < 0 || this._order.dense[idx] !== eid) return;
        // A tombstone skips an unvisited member without moving another member into its position.
        this._order.dense[idx] = 0;
        this._sparse[eid] = -1;
        this._dirty = true;
    }

    /** @internal Rebuild retained queries from restored membership, not historical order. */
    restore(members: readonly number[]): void {
        this._order.dense.length = 0;
        this._sparse.length = 0;
        this._sorted = 0;
        for (const eid of members) this.add(eid);
        this._dirty = true;
    }

    [Symbol.iterator](): Iterator<number> {
        if (this._dirty && this._order.active === 0) this.rebuild();
        const it = this._iterPool.pop() ?? new QueryIterator(this._iterPool);
        it.reset(this._order);
        return it;
    }

    // Sorting a retained tail preserves the prefix; merging backward keeps the dense array in place.
    private rebuild(): void {
        const dense = this._order.dense;
        const sparse = this._sparse;
        const tail = this._tail;
        let live = 0;
        for (let i = 0; i < this._sorted; i++) {
            const eid = dense[i];
            if (!eid) continue;
            if (live !== i) {
                dense[live] = eid;
                sparse[eid] = live;
            }
            live++;
        }
        let added = 0;
        for (let i = this._sorted; i < dense.length; i++) if (dense[i]) tail[added++] = dense[i];
        tail.length = added;
        tail.sort(ascending);
        let write = live + added;
        dense.length = write;
        let read = live - 1;
        for (let t = added - 1; t >= 0; t--) {
            const eid = tail[t];
            while (read >= 0 && dense[read] > eid) {
                const moved = dense[read--];
                dense[--write] = moved;
                sparse[moved] = write;
            }
            dense[--write] = eid;
            sparse[eid] = write;
        }
        this._sorted = dense.length;
        this._dirty = false;
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

    /** @internal Query order is derived from membership, including queries registered after capture. */
    restore(components: Components, entities: Entities): void {
        const alive = entities.all();
        for (const query of this._all)
            query.restore(alive.filter((eid) => query.matches(eid, components)));
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
