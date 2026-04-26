import type { ComponentStore, Entity } from "./store";
import { toKebabCase } from "./strings";

const $rel = Symbol("rel");
const $target = Symbol("target");
const $pair = Symbol("pair");
const $exclusive = Symbol("exclusive");
const $autoRemove = Symbol("autoRemove");

export type PairKey<_T = unknown> = ((target: Entity | typeof Wildcard) => any) & {
    [$rel]: true;
    [$exclusive]?: boolean;
    [$autoRemove]?: boolean;
};

const pairCache = new Map<any, Map<Entity | typeof Wildcard, any>>();
const _emptyTargets: readonly number[] = Object.freeze([]);

function pairFor(relation: PairKey, target: Entity | typeof Wildcard): any {
    let byTarget = pairCache.get(relation);
    if (!byTarget) {
        byTarget = new Map();
        pairCache.set(relation, byTarget);
    }
    let pair = byTarget.get(target);
    if (!pair) {
        pair = { [$pair]: true, [$rel]: relation, [$target]: target };
        byTarget.set(target, pair);
    }
    return pair;
}

export interface RelationOptions {
    readonly exclusive?: boolean;
    readonly autoRemoveSubject?: boolean;
}

export function createPairKey<T = unknown>(options?: RelationOptions): PairKey<T> {
    const relation = ((target: Entity | typeof Wildcard): any =>
        pairFor(relation as PairKey, target)) as PairKey<T>;
    (relation as any)[$rel] = true;
    if (options?.exclusive) (relation as any)[$exclusive] = true;
    if (options?.autoRemoveSubject) (relation as any)[$autoRemove] = true;
    return relation;
}

/** create a pair component linking a relation to a target */
export function pair(relation: PairKey | any, target: Entity | typeof Wildcard): any {
    if (typeof relation === "function" && relation[$rel]) return relation(target);
    return pairFor(relation as PairKey, target);
}

export function isPairComponent(component: unknown): boolean {
    return component != null && typeof component === "object" && (component as any)[$pair] === true;
}

export function pairRelation(component: any): PairKey | undefined {
    return component?.[$rel];
}

export function pairTarget(component: any): Entity | typeof Wildcard | undefined {
    return component?.[$target];
}

/** matches any target in a relation query */
export const Wildcard: PairKey = createPairKey();

export interface RelationHost {
    readonly components: ComponentStore;
    entityExists(eid: Entity): boolean;
    removeEntity(eid: Entity): void;
    notifyAdd(eid: Entity, component: any): void;
    notifyRemove(eid: Entity, component: any): void;
    notifyQueryChanged(eid: Entity, component: any): void;
}

export class RelationStore {
    private _relations = new Map<Entity, Map<PairKey, Set<Entity>>>();
    private _reverse = new Map<Entity, Set<Entity>>();
    private _host: RelationHost;

    constructor(host: RelationHost) {
        this._host = host;
    }

    add(subject: Entity, relation: PairKey, target: Entity): void {
        if ((relation as any)[$exclusive]) {
            for (const old of this.targets(subject, relation)) {
                if (old !== target) this.remove(subject, relation, old);
            }
        }

        let relMap = this._relations.get(subject);
        if (!relMap) {
            relMap = new Map();
            this._relations.set(subject, relMap);
        }
        let targetSet = relMap.get(relation);
        if (!targetSet) {
            targetSet = new Set();
            relMap.set(relation, targetSet);
        }
        targetSet.add(target);

        let subjectSet = this._reverse.get(target);
        if (!subjectSet) {
            subjectSet = new Set();
            this._reverse.set(target, subjectSet);
        }
        subjectSet.add(subject);

        const pairComponent = pairFor(relation, target);
        const wildcardComponent = pairFor(relation, Wildcard);
        this._host.components.add(subject, pairComponent);
        this._host.components.add(subject, wildcardComponent);
        this._host.notifyQueryChanged(subject, pairComponent);
        this._host.notifyQueryChanged(subject, wildcardComponent);
        this._host.notifyAdd(subject, pairComponent);
        this._host.notifyAdd(subject, wildcardComponent);
    }

    remove(subject: Entity, relation: PairKey, target: Entity): void {
        const relMap = this._relations.get(subject);
        if (!relMap) return;
        const targetSet = relMap.get(relation);
        if (!targetSet?.has(target)) return;
        targetSet.delete(target);

        const pairComponent = pairFor(relation, target);
        this._host.notifyRemove(subject, pairComponent);
        this._host.components.remove(subject, pairComponent);
        this._host.notifyQueryChanged(subject, pairComponent);

        if (targetSet.size === 0) {
            relMap.delete(relation);
            const wildcardComponent = pairFor(relation, Wildcard);
            this._host.notifyRemove(subject, wildcardComponent);
            this._host.components.remove(subject, wildcardComponent);
            this._host.notifyQueryChanged(subject, wildcardComponent);
        }

        this._reverse.get(target)?.delete(subject);
    }

    targets(subject: Entity, relation: PairKey): readonly number[] {
        const targetSet = this._relations.get(subject)?.get(relation);
        if (!targetSet) return _emptyTargets;
        const arr: number[] = [];
        for (const t of targetSet) arr.push(t);
        return arr;
    }

    onEntityRemoved(eid: Entity): void {
        const relMap = this._relations.get(eid);
        if (relMap) {
            for (const [relation, targetSet] of relMap) {
                for (const target of targetSet) {
                    const pair = pairFor(relation, target);
                    this._host.notifyRemove(eid, pair);
                    this._host.components.remove(eid, pair);
                    this._reverse.get(target)?.delete(eid);
                }
            }
            this._relations.delete(eid);
        }

        const subjects = this._reverse.get(eid);
        if (!subjects) return;

        const cascade: Entity[] = [];
        for (const subject of subjects) {
            const subRelMap = this._relations.get(subject);
            if (!subRelMap) continue;
            for (const [relation, targetSet] of subRelMap) {
                if (!targetSet.has(eid)) continue;
                targetSet.delete(eid);
                const pair = pairFor(relation, eid);
                this._host.notifyRemove(subject, pair);
                this._host.components.remove(subject, pair);
                this._host.notifyQueryChanged(subject, pair);

                if (targetSet.size === 0) {
                    subRelMap.delete(relation);
                    const wildcardComponent = pairFor(relation, Wildcard);
                    this._host.notifyRemove(subject, wildcardComponent);
                    this._host.components.remove(subject, wildcardComponent);
                    this._host.notifyQueryChanged(subject, wildcardComponent);
                }

                if ((relation as any)[$autoRemove]) cascade.push(subject);
            }
        }
        this._reverse.delete(eid);

        for (const subject of cascade) {
            if (this._host.entityExists(subject)) this._host.removeEntity(subject);
        }
    }
}

export interface Relation {
    readonly name: string;
    readonly relation: PairKey<unknown>;
    readonly exclusive?: boolean;
    readonly autoRemoveSubject?: boolean;
}

const registry = new Map<string, Relation>();

/** define a named relation type */
export function relation(name: string, options?: RelationOptions): Relation {
    const relation = createPairKey({
        exclusive: options?.exclusive,
        autoRemoveSubject: options?.autoRemoveSubject,
    });

    const rel: Relation = {
        name: toKebabCase(name),
        relation,
        exclusive: options?.exclusive,
        autoRemoveSubject: options?.autoRemoveSubject,
    };

    registry.set(rel.name, rel);
    return rel;
}

export function getRelation(name: string): Relation | undefined {
    return registry.get(toKebabCase(name));
}

export function registerRelation(def: Relation): void {
    registry.set(def.name, def);
}

export function clearRelations(): void {
    registry.clear();
}

/** parent-child relation (exclusive, cascading delete) */
export const ChildOf = relation("child-of", {
    exclusive: true,
    autoRemoveSubject: true,
});

/** exclusive targeting relation */
export const Target = relation("target", { exclusive: true });
