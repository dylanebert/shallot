import type { Component, ComponentValues } from "./component";
import { declaration, idOf, isType, lanes } from "./component";
import type { World } from "./world";

interface Entry {
    key: string;
    component: Component;
    /** Components added when missing on insertion; removing this component removes nothing. */
    requires?: Component[];
    /**
     * default field values, applied on `world.add`. Values are scalars for
     * ScalarField fields and per-lane arrays for direct `Vector2Field`/`Vector4Field`
     * fields (`{ translation: [0, 0, 0, 0] }`).
     */
    defaults?: (world: World) => ComponentValues<Component>;
    /** lazy-compiled defaults writer. undefined = unbuilt, null = no defaults. */
    plan?: DefaultsPlan | null;
}

interface DefaultsPlan {
    fields: { name: string; values: number[] }[];
}

export class ComponentRegistry {
    private readonly _byName = new Map<string, Entry>();
    // keyed by stable component id, so a handle held across reloads resolves this world's registration
    private readonly _byId = new Map<number, Entry>();

    /** @internal Changes to registration invalidate local checkpoints. */
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
        return this._byId.get(idOf(component))?.requires ?? [];
    }

    /** iterate every component registered in this world with its key and options */
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

    clear(): void {
        this.revision++;
        this._byName.clear();
        this._byId.clear();
    }
}

function compilePlan(entry: Entry, world: World): DefaultsPlan | null {
    const defaults = entry.defaults;
    if (!defaults) return null;
    const dict = defaults(world);
    const schema = entry.component as Record<string, unknown>;
    const fields = new Map<string, number[]>();

    for (const [key, value] of Object.entries(dict)) {
        const target = schema[key];
        if (!isType(target)) {
            throw new Error(
                `defaults key "${key}" on component "${entry.key}" does not match a typed field`,
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
                `defaults value for "${key}" on component "${entry.key}" is not numeric`,
            );
        }
    }

    return fields.size === 0
        ? null
        : { fields: [...fields].map(([name, values]) => ({ name, values })) };
}
