import type { Component, ComponentValues } from "./component";
import { declaration, idOf, isType, lanes } from "./component";
import type { World } from "./world";

interface Entry {
    key: string;
    component: Component;
    /** Components added when missing on insertion; removing this component removes nothing. */
    requires?: Component[];
    /**
     * Default field values applied on `world.add`: numbers, text, and per-lane arrays
     * for vectors. An omitted vector lane is 0.
     */
    defaults?: ComponentValues<Component> | ((world: World) => ComponentValues<Component>);
    /** lazy-compiled defaults writer. undefined = unbuilt, null = no defaults. */
    plan?: DefaultsPlan | null;
}

interface DefaultsPlan {
    fields: { name: string; write(eid: number): void }[];
}

const noRequirements: readonly Component[] = [];

export class ComponentRegistry {
    private readonly _byName = new Map<string, Entry>();
    // keyed by stable component id, so a handle held across reloads resolves this world's registration
    private readonly _byId = new Map<number, Entry>();

    /** @internal Changes to registration invalidate local snapshots. */
    revision = 0;

    register(component: Component, plugin = "ComponentRegistry"): void {
        const metadata = declaration(component, plugin);
        const { key } = metadata;
        const id = idOf(component);
        const entry: Entry = { ...metadata };
        this.revision++;
        this._byName.set(key, entry);
        this._byId.set(id, entry);
    }

    /** Required companions for insertion. */
    getRequirements(component: Component): readonly Component[] {
        return this._byId.get(idOf(component))?.requires ?? noRequirements;
    }

    /** iterate every component registered in this world with its key and options */
    entries(): IterableIterator<Entry> {
        return this._byName.values();
    }

    /** write default values into this world's field columns. */
    applyDefaults(
        world: World,
        component: Component,
        eid: number,
        supplied?: Record<string, unknown>,
    ): void {
        const entry = this._byId.get(idOf(component));
        if (!entry) return;
        let plan = entry.plan;
        if (plan === undefined) plan = entry.plan = compilePlan(entry, world);
        if (!plan) return;
        for (const field of plan.fields) {
            if (supplied?.[field.name] === undefined) field.write(eid);
        }
    }

    clear(): void {
        this.revision++;
        this._byName.clear();
        this._byId.clear();
    }
}

function compilePlan(entry: Entry, world: World): DefaultsPlan | null {
    const defaults = entry.defaults;
    if (!defaults) return null;
    const dict = typeof defaults === "function" ? defaults(world) : defaults;
    const schema = entry.component as Record<string, unknown>;
    const fields = new Map<string, number[] | string>();

    for (const [key, value] of Object.entries(dict)) {
        if (value === undefined) continue;
        const target = schema[key];
        if (!isType(target)) {
            throw new Error(
                `defaults key "${key}" on component "${entry.key}" does not match a typed field`,
            );
        }
        if (target.byteCapacity !== undefined) {
            if (typeof value !== "string") throw new Error(`Default for "${key}" must be text`);
            fields.set(key, value);
            continue;
        }
        const width = lanes(target);
        if (Array.isArray(value)) {
            const values = new Array(width).fill(0);
            for (let lane = 0; lane < width; lane++) values[lane] = value[lane] ?? 0;
            fields.set(key, values);
        } else if (typeof value === "number") {
            const values = new Array(width).fill(0);
            values[0] = value;
            fields.set(key, values);
        } else {
            throw new Error(
                `defaults value for "${key}" on component "${entry.key}" is not numeric`,
            );
        }
    }

    if (fields.size === 0) return null;
    const storage = world.storage(entry.component) as Record<
        string,
        { set(eid: number, x: number | string, y?: number, z?: number, w?: number): void }
    >;
    return {
        fields: [...fields].map(([name, values]) => {
            const field = storage[name];
            if (typeof values === "string")
                return { name, write: (eid: number) => field.set(eid, values) };
            const [x, y, z, w] = values;
            const write =
                values.length === 1
                    ? (eid: number) => field.set(eid, x)
                    : values.length === 2
                      ? (eid: number) => field.set(eid, x, y)
                      : (eid: number) => field.set(eid, x, y, z, w);
            return { name, write };
        }),
    };
}
