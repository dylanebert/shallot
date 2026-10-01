import type { Component } from "./component";
import { idOf, intern, isType, lanes } from "./component";
import { kebab } from "./reflection";
import type { World } from "./state";

/** defaults and enforced relationships declared per component */
export interface Traits {
    /**
     * derived outputs this component produces. `Transform` and `Body` provide `GlobalTransform`.
     * Adding a provider with `world.add` attaches `GlobalTransform`; removing a provider with
     * `world.remove` retains it while another provider owns it, otherwise deferring removal until
     * reconciliation.
     */
    provides?: Component[];
    /**
     * components that cannot coexist on the same entity. Symmetric — declaring
     * `A.excludes = [B]` is equivalent to declaring `B.excludes = [A]`; both
     * directions are enforced at `world.add`
     */
    excludes?: Component[];
    /**
     * default field values, applied on `world.add`. Values are scalars for
     * ScalarField fields and per-lane arrays for direct {@link Vector2Field}/{@link Vector4Field}
     * fields (`{ translation: [0, 0, 0, 0] }`). Dotted keys (`{ "translation.x": 0 }`)
     * address a single lane of a parent Vector2Field/Vector4Field
     */
    defaults?: (world: World) => Record<string, number | readonly number[]>;
}

interface DefaultsPlan {
    fields: { name: string; values: number[] }[];
}

interface Entry {
    component: Component;
    name: string;
    traits?: Traits;
    /** lazy-compiled defaults writer. undefined = unbuilt, null = no defaults. */
    plan?: DefaultsPlan | null;
}

export class ComponentRegistry {
    private readonly _byName = new Map<string, Entry>();
    // keyed by stable component id, so a handle held across reloads resolves this world's registration
    private readonly _byId = new Map<number, Entry>();
    private _exclusions: Map<number, Set<Component>> | null = null;

    /** register a component under a name, with optional traits */
    register(name: string, component: Component, traits?: Traits): void {
        const k = kebab(name);
        const id = intern(component, k);
        const entry: Entry = traits ? { component, name: k, traits } : { component, name: k };
        this._byName.set(k, entry);
        this._byId.set(id, entry);
        this._exclusions = null;
    }

    /** components that may not coexist with `component`. Symmetric over all declarations */
    getExclusions(component: Component): ReadonlySet<Component> | undefined {
        this._exclusions ??= this.buildExclusions();
        return this._exclusions.get(idOf(component));
    }

    /** whether this component declares the given component as a runtime producer output. */
    provides(component: Component, output: Component): boolean {
        return (
            this._byId
                .get(idOf(component))
                ?.traits?.provides?.some((item) => idOf(item) === idOf(output)) ?? false
        );
    }

    getName(component: Component): string | undefined {
        return this._byId.get(idOf(component))?.name;
    }

    /** iterate every component registered in this world with its name and traits */
    entries(): IterableIterator<Entry> {
        return this._byName.values();
    }

    /** write default values into this world's field columns. */
    applyDefaults(world: World, component: Component, eid: number): void {
        const entry = this._byId.get(idOf(component));
        if (!entry) return;
        let plan = entry.plan;
        if (plan === undefined) plan = entry.plan = compilePlan(entry, world);
        if (!plan) return;
        const storage = world.storage(component) as Record<
            string,
            { set(eid: number, ...values: number[]): void }
        >;
        for (const { name, values } of plan.fields) storage[name].set(eid, ...values);
    }

    /** clear registrations owned by this world. */
    clear(): void {
        this._byName.clear();
        this._byId.clear();
        this._exclusions = null;
    }

    private buildExclusions(): Map<number, Set<Component>> {
        const map = new Map<number, Set<Component>>();
        const link = (id: number, other: Component) => {
            let set = map.get(id);
            if (!set) map.set(id, (set = new Set()));
            set.add(other);
        };
        for (const entry of this._byName.values()) {
            for (const declared of entry.traits?.excludes ?? []) {
                const other = this._byId.get(idOf(declared))?.component ?? declared;
                link(idOf(entry.component), other);
                link(idOf(other), entry.component);
            }
        }
        return map;
    }
}

/** registration and reflection helpers always resolve through the owning World. */

export const getName = (world: World, component: Component) => world.registry.getName(component);

export const applyDefaults = (world: World, component: Component, eid: number) =>
    world.registry.applyDefaults(world, component, eid);

const LANE_INDEX: Record<string, number> = { x: 0, y: 1, z: 2, w: 3 };

function compilePlan(entry: Entry, world: World): DefaultsPlan | null {
    const defaults = entry.traits?.defaults;
    if (!defaults) return null;
    const dict = defaults(world);
    const schema = entry.component as Record<string, unknown>;
    const fields = new Map<string, number[]>();

    for (const [key, value] of Object.entries(dict)) {
        const dot = key.indexOf(".");
        if (dot >= 0) {
            const name = key.slice(0, dot);
            const target = schema[name];
            const width = lanes(target);
            const lane = LANE_INDEX[key.slice(dot + 1)];
            if (
                !isType(target) ||
                (width !== 2 && width !== 4) ||
                lane === undefined ||
                lane >= width
            ) {
                throw new Error(
                    `defaults key "${key}" on component "${entry.name}" does not target a valid vector lane`,
                );
            }
            if (typeof value !== "number") {
                throw new Error(
                    `defaults value for "${key}" on component "${entry.name}" is not a number`,
                );
            }
            const values = fields.get(name) ?? new Array(width).fill(0);
            values[lane] = value;
            fields.set(name, values);
            continue;
        }

        const target = schema[key];
        if (!isType(target)) {
            throw new Error(
                `defaults key "${key}" on component "${entry.name}" does not match a typed field`,
            );
        }
        const width = lanes(target);
        if (Array.isArray(value)) {
            const values = new Array(width).fill(0);
            for (let lane = 0; lane < width; lane++) values[lane] = value[lane] ?? 0;
            fields.set(key, values);
        } else if (typeof value === "number") {
            const values = fields.get(key) ?? new Array(width).fill(0);
            values[0] = value;
            fields.set(key, values);
        } else {
            throw new Error(
                `defaults value for "${key}" on component "${entry.name}" is not numeric`,
            );
        }
    }

    return fields.size === 0
        ? null
        : { fields: [...fields].map(([name, values]) => ({ name, values })) };
}
