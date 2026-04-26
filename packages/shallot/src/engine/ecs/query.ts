import { SparseSet, type ComponentStore, type Entity } from "./store";

export { SparseSet } from "./store";
export type { Entity } from "./store";

const $qop = Symbol("qop");
const $hop = Symbol("hop");

interface QueryOp {
    [$qop]: "not" | "and" | "or";
    components: any[];
}

interface HierarchyOp {
    [$hop]: true;
    relation: any;
}

function isQueryOp(x: unknown): x is QueryOp {
    return x != null && typeof x === "object" && $qop in x;
}

function isHierarchyOp(x: unknown): x is HierarchyOp {
    return x != null && typeof x === "object" && $hop in x;
}

/** exclude entities with these components */
export function not(...components: any[]): QueryOp {
    return { [$qop]: "not", components };
}

/** require all of these components */
export function and(...components: any[]): QueryOp {
    return { [$qop]: "and", components };
}

/** match entities with at least one of these components */
export function or(...components: any[]): QueryOp {
    return { [$qop]: "or", components };
}

/** sort results by relation depth (parents before children) */
export function hierarchy(relation: any): HierarchyOp {
    return { [$hop]: true, relation };
}

interface CompiledQuery {
    required: any[];
    excluded: any[];
    orGroups: any[][];
    hierarchy: HierarchyOp | null;
}

export function compileQuery(terms: any[]): CompiledQuery {
    const required: any[] = [];
    const excluded: any[] = [];
    const orGroups: any[][] = [];
    let hierarchy: HierarchyOp | null = null;

    for (const term of terms) {
        if (isHierarchyOp(term)) {
            hierarchy = term;
        } else if (isQueryOp(term)) {
            const op = term[$qop];
            if (op === "not") excluded.push(...term.components);
            else if (op === "and") required.push(...term.components);
            else if (op === "or") orGroups.push(term.components);
        } else {
            required.push(term);
        }
    }

    return { required, excluded, orGroups, hierarchy };
}

function matchesQuery(eid: Entity, query: CompiledQuery, components: ComponentStore): boolean {
    for (const component of query.required) if (!components.has(eid, component)) return false;
    for (const component of query.excluded) if (components.has(eid, component)) return false;
    for (const group of query.orGroups) {
        let matched = false;
        for (const component of group) {
            if (components.has(eid, component)) {
                matched = true;
                break;
            }
        }
        if (!matched) return false;
    }
    return true;
}

interface RegisteredQuery {
    compiled: CompiledQuery;
    set: SparseSet;
    allComponents: Set<any>;
    sortedCache: number[] | null;
    sortedDirty: boolean;
}

export class QueryCache {
    private _queries: RegisteredQuery[] = [];
    private _hashMap = new Map<string, RegisteredQuery>();
    private _componentIds = new Map<any, number>();
    private _nextComponentId = 0;

    register(
        compiled: CompiledQuery,
        components: ComponentStore,
        entityDense: readonly number[],
        entityAlive: number,
    ): RegisteredQuery {
        const hash = this.hash(compiled);
        let rq = this._hashMap.get(hash);
        if (rq) return rq;

        const allComponents = new Set<any>();
        for (const c of compiled.required) allComponents.add(c);
        for (const c of compiled.excluded) allComponents.add(c);
        for (const group of compiled.orGroups) {
            for (const c of group) allComponents.add(c);
        }

        const set = new SparseSet();
        for (let i = 0; i < entityAlive; i++) {
            const eid = entityDense[i];
            if (matchesQuery(eid, compiled, components)) set.add(eid);
        }

        rq = { compiled, set, allComponents, sortedCache: null, sortedDirty: true };
        this._queries.push(rq);
        this._hashMap.set(hash, rq);
        return rq;
    }

    onComponentChanged(eid: Entity, component: any, components: ComponentStore): void {
        for (let i = 0; i < this._queries.length; i++) {
            const rq = this._queries[i];
            if (!rq.allComponents.has(component)) continue;
            if (matchesQuery(eid, rq.compiled, components)) {
                if (rq.set.add(eid)) rq.sortedDirty = true;
            } else {
                if (rq.set.remove(eid)) rq.sortedDirty = true;
            }
        }
    }

    onEntityRemoved(eid: Entity): void {
        for (let i = 0; i < this._queries.length; i++) {
            const rq = this._queries[i];
            if (rq.set.remove(eid)) rq.sortedDirty = true;
        }
    }

    clear(): void {
        this._queries.length = 0;
        this._hashMap.clear();
        this._componentIds.clear();
        this._nextComponentId = 0;
    }

    private componentId(component: any): number {
        let id = this._componentIds.get(component);
        if (id === undefined) {
            id = this._nextComponentId++;
            this._componentIds.set(component, id);
        }
        return id;
    }

    private hash(compiled: CompiledQuery): string {
        const req = compiled.required.map((c) => this.componentId(c)).sort((a, b) => a - b);
        const exc = compiled.excluded.map((c) => this.componentId(c)).sort((a, b) => a - b);
        const orr = compiled.orGroups
            .map((g) =>
                g
                    .map((c) => this.componentId(c))
                    .sort((a, b) => a - b)
                    .join(","),
            )
            .sort();
        return `${req.join(",")};${exc.join(",")};${orr.join("|")}`;
    }
}
