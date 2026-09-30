import type { Alias, Input } from "../utils";
import type { Component } from "./component";
import { idOf, intern, isType, lanes } from "./component";
import { kebab } from "./reflection";
import type { State } from "./state";

/** parse-time metadata declared per component */
export interface Traits {
    requires?: Component[];
    /**
     * components this one stands in for — an entity carrying it satisfies another component's
     * `requires` of any listed component, without holding that component itself. `Body.provides =
     * [Transform]` (physics owns the entity's world transform, so `Body` excludes `Transform` yet a
     * `Part` on the same entity still renders). Directional (the counterpart to `requires`), read by
     * scene validation only, not enforced at `state.add`
     */
    provides?: Component[];
    /** one instance per scene (lights, the active camera). Informational — surfaced through
     * reflection, not enforced at `state.add` */
    singleton?: boolean;
    /**
     * runtime-derived decoration — a system owns its membership and values (for example
     * `GlobalTransform`), so scenes never author it: `serialize` skips it, authoring tooling
     * hides it, and `diagnose` flags an authored attr. Registration
     * still allocates its columns, and an always-mode system may add/remove it
     * freely, since nothing serialized sees it
     */
    derived?: boolean;
    /**
     * components that cannot coexist on the same entity. Symmetric — declaring
     * `A.excludes = [B]` is equivalent to declaring `B.excludes = [A]`; both
     * directions are enforced at `state.add` and during scene validation
     */
    excludes?: Component[];
    /**
     * default field values, applied on `state.add`. Values are scalars for
     * Single fields and per-lane arrays for direct {@link Pair}/{@link Quad}
     * fields (`{ pos: [0, 0, 0, 0] }`). Dotted keys (`{ "pos.x": 0 }`)
     * address a single lane of a parent Pair/Quad
     */
    defaults?: () => Record<string, number | readonly number[]>;
    /** per-field authoring aliases — a stored vector field edited in an alternate representation */
    aliases?: Record<string, Alias>;
    parse?: Record<string, (value: string) => number | undefined>;
    format?: Record<string, (value: number) => string | undefined>;
    enums?: Record<string, Record<string, number>>;
    /** per-field input widget — a stored field shown through a richer control (a `toggle`
     * checkbox, an `angle` unit switcher). Display-only; storage is unchanged */
    inputs?: Record<string, Input>;
    annotations?: Record<string, unknown>;
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
        const expanded = traits ? expandEnums(traits) : undefined;
        const entry: Entry = expanded
            ? { component, name: k, traits: expanded }
            : { component, name: k };
        this._byName.set(k, entry);
        this._byId.set(id, entry);
        this._exclusions = null;
    }

    /** components that may not coexist with `component`. Symmetric over all declarations */
    getExclusions(component: Component): ReadonlySet<Component> | undefined {
        this._exclusions ??= this.buildExclusions();
        return this._exclusions.get(idOf(component));
    }

    /** the registered component handle for a name, or `undefined` if none is registered under it */
    getComponent(name: string): Component | undefined {
        return this._byName.get(kebab(name))?.component;
    }

    /** the parse-time `Traits` registered with a component name, or `undefined` if none */
    getTraits(name: string): Traits | undefined {
        return this._byName.get(kebab(name))?.traits;
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
    applyDefaults(state: State, component: Component, eid: number): void {
        const entry = this._byId.get(idOf(component));
        if (!entry) return;
        let plan = entry.plan;
        if (plan === undefined) plan = entry.plan = compilePlan(entry);
        if (!plan) return;
        const storage = state.of(component) as Record<
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

function expandEnums(t: Traits): Traits {
    if (!t.enums) return t;
    const parse: NonNullable<Traits["parse"]> = { ...t.parse };
    const format: NonNullable<Traits["format"]> = { ...t.format };
    for (const [field, enumObj] of Object.entries(t.enums)) {
        const fwd = new Map<string, number>();
        const rev = new Map<number, string>();
        for (const [key, val] of Object.entries(enumObj)) {
            const k = kebab(key);
            fwd.set(k, val);
            rev.set(val, k);
        }
        parse[field] ??= (value: string) => fwd.get(value);
        format[field] ??= (value: number) => rev.get(value);
    }
    return { ...t, parse, format };
}

/** registration and reflection helpers always resolve through the owning State. */
export const register = (state: State, name: string, component: Component, traits?: Traits): void =>
    state.registry.register(name, component, traits);
export const getExclusions = (state: State, component: Component) =>
    state.registry.getExclusions(component);
export const getComponent = (state: State, name: string) => state.registry.getComponent(name);
export const getTraits = (state: State, name: string) => state.registry.getTraits(name);
export const getName = (state: State, component: Component) => state.registry.getName(component);
export const entries = (state: State) => state.registry.entries();
export const applyDefaults = (state: State, component: Component, eid: number) =>
    state.registry.applyDefaults(state, component, eid);
export const clear = (state: State): void => state.registry.clear();

const LANE_INDEX: Record<string, number> = { x: 0, y: 1, z: 2, w: 3 };

function compilePlan(entry: Entry): DefaultsPlan | null {
    const defaults = entry.traits?.defaults;
    if (!defaults) return null;
    const dict = defaults();
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
