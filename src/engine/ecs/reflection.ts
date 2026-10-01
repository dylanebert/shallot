import type { Component, ScalarField, Vector2Field, Vector4Field } from "./component";
import { lanes } from "./component";
import type { World } from "./state";

/** Internal registration and dump spelling; not a public component lookup key. */
export function kebab(str: string): string {
    return str
        .replace(/([a-z])([A-Z])/g, "$1-$2")
        .replace(/[\s_]+/g, "-")
        .toLowerCase();
}

/** a flat map of one entity's field values for a component, vec fields split into dotted lanes
 * (`translation.x`, `translation.y`) */
export interface FieldValues {
    [field: string]: number | string | readonly number[];
}

/** one entity's live component values: its `eid` and every attached component's `FieldValues` */
export interface EntityData {
    eid: number;
    components: Record<string, FieldValues>;
}

/** read every field of `component` on `eid` into a flat map, vec fields split into dotted lanes
 * (`translation.x`, `translation.y`); the row values tooling shows */
export function readFields(world: World, component: Component, eid: number): FieldValues {
    const fields: FieldValues = {};
    const storage = world.storage(component) as Record<string, unknown>;
    for (const [field, store] of Object.entries(storage)) {
        const n = lanes(store);
        if (n === 4) {
            const q = store as Vector4Field;
            fields[`${field}.x`] = q.x.get(eid);
            fields[`${field}.y`] = q.y.get(eid);
            fields[`${field}.z`] = q.z.get(eid);
            fields[`${field}.w`] = q.w.get(eid);
        } else if (n === 2) {
            const p = store as Vector2Field;
            fields[`${field}.x`] = p.x.get(eid);
            fields[`${field}.y`] = p.y.get(eid);
        } else if (n === 1) {
            fields[field] = (store as ScalarField).get(eid);
        } else if (ArrayBuffer.isView(store) || Array.isArray(store)) {
            fields[field] = (store as number[])[eid] ?? 0;
        }
    }
    return fields;
}

/**
 * every component on a live entity with its field values, or `null` if the entity isn't alive.
 * @example
 * const data = inspect(world, eid);
 * data?.components; // { transform: { "translation.x": 0, ... }, orbit: { ... } }
 */
export function inspect(world: World, eid: number): EntityData | null {
    if (!world.exists(eid)) return null;
    const components: Record<string, FieldValues> = {};
    for (const { component, name } of world.registry.entries()) {
        if (world.has(eid, component as never)) {
            components[name] = readFields(world, component, eid);
        }
    }
    return { eid, components };
}

/** every live entity's components and values: the whole world as `EntityData`, for tooling, saves, and
 * debugging */
export function snapshot(world: World): EntityData[] {
    const out: EntityData[] = [];
    for (const eid of world.entities()) {
        const data = inspect(world, eid);
        if (data) out.push(data);
    }
    return out;
}

/**
 * format an entity's components and field values as a human-readable string, for logging.
 * @example
 * console.log(dump(world, eid));
 */
export function dump(world: World, eid: number): string {
    const data = inspect(world, eid);
    if (!data) return `Entity ${eid}: not found`;

    const lines = [`Entity ${eid}:`];
    for (const [name, fields] of Object.entries(data.components)) {
        const parts = Object.entries(fields)
            .map(([k, v]) => `${kebab(k)}: ${v}`)
            .join(", ");
        lines.push(`  ${name}: ${parts}`);
    }
    return lines.join("\n");
}
