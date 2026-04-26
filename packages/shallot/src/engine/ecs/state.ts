import {
    EntityPool,
    ComponentStore,
    ObservableStore,
    type HookInstance,
    type ObservableCallback,
} from "./store";
import type { SparseSet, Entity } from "./store";
import { compileQuery, QueryCache } from "./query";
import {
    RelationStore,
    isPairComponent,
    pairRelation,
    pairTarget,
    type RelationHost,
    type Relation,
    type PairKey,
} from "./relation";
import { Scheduler, Time, type System } from "./scheduler";
import type { Plugin } from "./plugin";
import { registerComponent, getTraits } from "./component";
import type { Resource } from "./resource";
import { clearAllEvents } from "./events";
import { grow, reset } from "./capacity";

/**
 * ecs state passed to every system
 * @expand
 * @example
 * const MySystem: System = {
 *     update(state) {
 *         // state passed in every frame
 *     },
 * };
 */
export class State implements RelationHost {
    readonly entities = new EntityPool();
    readonly components = new ComponentStore();
    readonly relations = new RelationStore(this);
    readonly observables = new ObservableStore();
    readonly queryCache = new QueryCache();
    readonly scheduler = new Scheduler();

    private _resources = new Map<symbol, unknown>();
    private _disposed = false;
    private _max = 0;
    private _disposeHooks: (() => void)[] = [];

    /** current frame time and delta */
    get time(): Readonly<Time> {
        return this.scheduler.time;
    }

    /** highest entity ID in use */
    get max(): number {
        return this._max;
    }

    /** store a global resource */
    setResource<T>(key: Resource<T>, value: T): void {
        this._resources.set(key, value);
    }

    /** retrieve a global resource */
    getResource<T>(key: Resource<T>): T | undefined {
        return this._resources.get(key) as T | undefined;
    }

    /** delete a global resource */
    deleteResource<T>(key: Resource<T>): boolean {
        return this._resources.delete(key);
    }

    /** register a plugin or system */
    register(pluginOrSystem: Plugin | System): void {
        if (
            "update" in pluginOrSystem ||
            "setup" in pluginOrSystem ||
            "dispose" in pluginOrSystem
        ) {
            this.scheduler.register(pluginOrSystem as System);
        } else {
            const plugin = pluginOrSystem as Plugin;
            if (plugin.components) {
                for (const [name, component] of Object.entries(plugin.components)) {
                    registerComponent(name, component);
                }
            }
            if (plugin.systems) {
                for (const system of plugin.systems) {
                    this.scheduler.register(system, plugin.name);
                }
            }
        }
    }

    /** remove a system */
    unregister(system: System): void {
        this.scheduler.unregister(system);
    }

    /** advance one frame */
    step(deltaTime = Time.DEFAULT_DT): void {
        this.scheduler.step(this, deltaTime);
        clearAllEvents(this);
    }

    /** create a new entity, returns its ID */
    addEntity(): number {
        const eid = this.entities.add();
        grow(eid + 1);
        if (eid > this._max) this._max = eid;
        return eid;
    }

    /** destroy an entity and its components/relations */
    removeEntity(eid: number): void {
        if (!this.entities.exists(eid)) return;
        this.relations.onEntityRemoved(eid);
        for (const component of this.components.getAll(eid)) {
            this.observables.notifyRemove(eid, component, this.components);
        }
        this.queryCache.onEntityRemoved(eid);
        this.components.clear(eid);
        this.entities.remove(eid);
        if (eid === this._max) {
            while (this._max > 0 && !this.entities.exists(this._max)) this._max--;
        }
    }

    /** true if entity ID is alive */
    entityExists(eid: number): boolean {
        return this.entities.exists(eid);
    }

    getAllEntities(): readonly number[] {
        return this.entities.all();
    }

    /**
     * find entities matching component terms
     * @example
     * for (const eid of state.query([Health, not(Dead)])) {
     *     Health.current[eid] -= 1;
     * }
     */
    query(terms: any[]): SparseSet | number[] {
        const compiled = compileQuery(terms);
        const rq = this.queryCache.register(
            compiled,
            this.components,
            this.entities.dense,
            this.entities.alive,
        );
        if (!compiled.hierarchy) return rq.set;
        if (!rq.sortedDirty && rq.sortedCache) return rq.sortedCache;
        const results: number[] = [];
        const dense = rq.set.dense;
        const count = rq.set.count;
        for (let i = 0; i < count; i++) results.push(dense[i]);
        rq.sortedCache = this.sortByDepth(results, compiled.hierarchy.relation);
        rq.sortedDirty = false;
        return rq.sortedCache;
    }

    /**
     * find exactly one entity, warns if multiple match
     * @example
     * const player = state.only([Player]);
     */
    only(terms: any[]): number {
        let result = -1;
        let count = 0;
        for (const eid of this.query(terms)) {
            if (count === 0) result = eid;
            count++;
            if (count > 1) break;
        }
        if (count > 1) {
            console.warn("state.only: expected 1 match, found multiple");
        }
        return result;
    }

    getEntityComponents(eid: number): any[] {
        return this.components.getAll(eid);
    }

    /**
     * attach a component to an entity
     * @example
     * state.addComponent(eid, Health);
     * Health.current[eid] = 100;
     */
    addComponent<T>(eid: number, component: T): void {
        if (isPairComponent(component)) {
            const relation = pairRelation(component)!;
            const target = pairTarget(component)!;
            if (typeof target === "number") {
                this.relations.add(eid, relation, target);
                return;
            }
        }

        if (this.components.add(eid, component)) {
            this.queryCache.onComponentChanged(eid, component, this.components);
            this.notifyAdd(eid, component);
        }

        this.applyDefaults(eid, component);
    }

    /** detach a component from an entity */
    removeComponent(eid: number, component: any): void {
        if (isPairComponent(component)) {
            const relation = pairRelation(component)!;
            const target = pairTarget(component)!;
            if (typeof target === "number") {
                this.relations.remove(eid, relation, target);
                return;
            }
        }
        this.notifyRemove(eid, component);
        this.components.remove(eid, component);
        this.queryCache.onComponentChanged(eid, component, this.components);
    }

    /** true if entity has the component */
    hasComponent<T>(eid: number, component: T): boolean {
        return this.components.has(eid, component);
    }

    /** component if present, undefined otherwise */
    getComponent<T>(eid: number, component: T) {
        return this.components.has(eid, component) ? component : undefined;
    }

    /** link two entities with a relation */
    addRelation(subject: number, relation: Relation, target: number): void {
        this.addComponent(subject, relation.relation(target));
    }

    /** remove a relation between two entities */
    removeRelation(subject: number, relation: Relation, target: number): void {
        this.removeComponent(subject, relation.relation(target));
    }

    /** true if a relation exists between two entities */
    hasRelation(subject: number, relation: Relation, target: number): boolean {
        return this.hasComponent(subject, relation.relation(target));
    }

    /** all targets of a relation from a subject */
    getRelationTargets(subject: number, relation: Relation): readonly number[] {
        return this.relations.targets(subject, relation.relation);
    }

    /** first relation target, or -1 */
    getFirstRelationTarget(subject: number, relation: Relation): number {
        const targets = this.relations.targets(subject, relation.relation);
        return targets.length > 0 ? targets[0] : -1;
    }

    /**
     * subscribe to a lifecycle hook, returns unsubscribe
     * @example
     * state.observe(onAdd(Health), (eid) => {
     *     Health.current[eid] = Health.max[eid];
     * });
     */
    observe(hook: HookInstance, callback: ObservableCallback): () => void {
        return this.observables.subscribe(hook, callback);
    }

    notifyAdd(eid: Entity, component: any): void {
        this.observables.notifyAdd(eid, component, this.components);
    }

    notifyRemove(eid: Entity, component: any): void {
        this.observables.notifyRemove(eid, component, this.components);
    }

    notifyQueryChanged(eid: Entity, component: any): void {
        this.queryCache.onComponentChanged(eid, component, this.components);
    }

    /** register a cleanup hook */
    onDispose(hook: () => void): void {
        this._disposeHooks.push(hook);
    }

    /** tear down the world */
    dispose(): void {
        if (this._disposed) return;
        for (const hook of this._disposeHooks) hook();
        this._disposeHooks.length = 0;
        for (const system of this.scheduler.systems) {
            system.dispose?.(this);
        }
        this.queryCache.clear();
        clearAllEvents(this);
        reset();
        this._disposed = true;
    }

    private applyDefaults(eid: number, component: any): void {
        const traits = getTraits(component as Record<string, unknown>);
        if (!traits?.defaults) return;
        const defaults = traits.defaults();
        const data = component as Record<string, number[] | Float32Array | Uint32Array>;
        for (const [field, value] of Object.entries(defaults)) {
            const arr = data[field];
            if (arr != null) arr[eid] = value;
        }
    }

    private sortByDepth(eids: number[], relation: PairKey): number[] {
        const depths = new Map<number, number>();
        const depthOf = (eid: Entity): number => {
            const cached = depths.get(eid);
            if (cached !== undefined) return cached;
            const targets = this.relations.targets(eid, relation);
            const depth = targets.length === 0 ? 0 : depthOf(targets[0]) + 1;
            depths.set(eid, depth);
            return depth;
        };
        for (const eid of eids) depthOf(eid);
        eids.sort((a, b) => (depths.get(a) ?? 0) - (depths.get(b) ?? 0));
        return eids;
    }
}
