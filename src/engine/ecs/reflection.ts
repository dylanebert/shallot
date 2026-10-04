import type { Component, ScalarField, Vector2Field, Vector4Field } from "./component";
import { lanes } from "./component";
import type { World } from "./world";

/** Declared field names with scalar numbers and vector arrays. */
export interface FieldValues {
    [field: string]: number | readonly number[];
}

/** one entity's live component values: its `eid` and every attached component's `FieldValues` */
export interface EntityData {
    eid: number;
    components: Record<string, FieldValues>;
}

/** Read declared fields on `eid`, keeping vectors as arrays. */
export function readFields(world: World, component: Component, eid: number): FieldValues {
    const fields: FieldValues = {};
    const storage = world.storage(component) as Record<string, unknown>;
    for (const [field, store] of Object.entries(storage)) {
        const n = lanes(store);
        if (n === 4) {
            const q = store as Vector4Field;
            fields[field] = [q.x.get(eid), q.y.get(eid), q.z.get(eid), q.w.get(eid)];
        } else if (n === 2) {
            const p = store as Vector2Field;
            fields[field] = [p.x.get(eid), p.y.get(eid)];
        } else if (n === 1) {
            fields[field] = (store as ScalarField).get(eid);
        }
    }
    return fields;
}

/**
 * each component this world registers that a live entity carries, with its field values, or `null` if the
 * entity isn't alive. An attached but unregistered record is omitted.
 */
export function inspect(world: World, eid: number): EntityData | null {
    if (!world.exists(eid)) return null;
    const components: Record<string, FieldValues> = {};
    for (const { component, key } of world.registry.entries()) {
        if (world.has(eid, component as never)) {
            components[key] = readFields(world, component, eid);
        }
    }
    return { eid, components };
}

/** every live entity's registered components and values as `EntityData`, for tooling, saves, and
 * debugging; unregistered records are omitted, as in {@link inspect} */
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
 */
export function dump(world: World, eid: number): string {
    const data = inspect(world, eid);
    if (!data) return `Entity ${eid}: not found`;

    const lines = [`Entity ${eid}:`];
    for (const [name, fields] of Object.entries(data.components)) {
        const parts = Object.entries(fields)
            .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
            .join(", ");
        lines.push(`  ${name}: ${parts}`);
    }
    return lines.join("\n");
}
