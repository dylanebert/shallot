import type { FieldUnit } from "../utils";
import type { Component, ScalarField, Vector2Field, Vector4Field } from "./component";
import { entity, lanes } from "./component";
import type { World } from "./state";
import { getName, type Traits } from "./traits";
/** convert a name to kebab-case (`OrbitCamera` → `orbit-camera`), the canonical component-name form
 * registries and scene attributes key on */
export function kebab(str: string): string {
    return str
        .replace(/([a-z])([A-Z])/g, "$1-$2")
        .replace(/[\s_]+/g, "-")
        .toLowerCase();
}

/** convert a kebab-case name back to camelCase (`orbit-camera` → `orbitCamera`), the inverse of `kebab` */
export function camel(str: string): string {
    return str.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

/** the presentation category of a component field, how it's shown and edited: a `vec4` as four lanes, an
 * `enum` as a dropdown, an `entity` as an `@name` reference, a `unit` through a unit switcher */
export type FieldKind = "float" | "vec2" | "vec4" | "color" | "enum" | "unit" | "entity";

/** one field's reflected shape: its name, `kind`, default, and kind-specific extras (enum `options`,
 * vec lane keys in `fields`, the `units` menu) */
export interface FieldInfo {
    name: string;
    kind: FieldKind;
    default?: number | string | readonly number[];
    options?: Record<string, number>;
    fields?: string[];
    /** the unit menu, when `kind` is `unit` — `[0]` is the unit shown by default */
    units?: FieldUnit[];
}

/** a component's reflected field layout: its kebab `name` and the ordered `fields` a tooling surface
 * renders as rows */
export interface Schema {
    name: string;
    fields: FieldInfo[];
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

const VEC_KIND: Record<number, FieldKind> = { 2: "vec2", 4: "vec4" };

function isColor(key: string, traits: Traits | undefined): boolean {
    return (traits?.format?.[key] as { kind?: string } | undefined)?.kind === "color";
}

/**
 * reflect a registered component's field layout by name, or `null` if nothing is registered under it.
 * @example
 * const s = schema(world, "orbit");
 * s?.fields.map((f) => f.name); // ["distance", "yaw", "pitch", ...]
 */
export function schema(world: World, name: string): Schema | null {
    const component = world.registry.getComponent(name);
    if (!component) return null;
    const traits = world.registry.getTraits(name);
    const defaults = traits?.defaults?.(world) ?? {};

    const handled = new Set<string>();
    const fields: FieldInfo[] = [];

    for (const key of Object.keys(component)) {
        if (handled.has(key)) continue;
        handled.add(key);

        // direct Vector2Field/Vector4Field: a single key on the component backed by 2 or 4 lanes
        const direct = lanes(component[key]);
        if (direct === 2 || direct === 4) {
            const dotKeys =
                direct === 4
                    ? [`${key}.x`, `${key}.y`, `${key}.z`, `${key}.w`]
                    : [`${key}.x`, `${key}.y`];
            const arrDefault = defaults[key];
            const hasArrDefault = Array.isArray(arrDefault);
            const hasDotDefaults = dotKeys.every((m) => defaults[m] !== undefined);
            let defaultLanes: readonly number[] | undefined;
            if (hasArrDefault) defaultLanes = arrDefault as number[];
            else if (hasDotDefaults) defaultLanes = dotKeys.map((m) => defaults[m] as number);
            fields.push({
                name: key,
                kind: VEC_KIND[direct],
                fields: dotKeys,
                default: defaultLanes,
            });
            continue;
        }

        const input = traits?.inputs?.[key];
        const enumDef = traits?.enums?.[key];
        if (input?.kind === "unit") {
            fields.push({
                name: key,
                kind: "unit",
                default: defaults[key] as number,
                units: input.units,
            });
        } else if (enumDef) {
            fields.push({
                name: key,
                kind: "enum",
                default: defaults[key] as number,
                options: enumDef,
            });
        } else if (isColor(key, traits)) {
            fields.push({ name: key, kind: "color", default: defaults[key] as number });
        } else if (component[key] === entity) {
            // Ref-ness lives on the bare type declaration — surface it so tooling
            // shows an `@name` reference, not a number
            fields.push({ name: key, kind: "entity", default: defaults[key] as number });
        } else {
            fields.push({ name: key, kind: "float", default: defaults[key] as number });
        }
    }

    return { name: kebab(name), fields };
}

export function schemas(world: World): Schema[] {
    const out: Schema[] = [];
    for (const { name } of world.registry.entries()) {
        const s = schema(world, name);
        if (s) out.push(s);
    }
    return out;
}

/** the component names a component requires (its `requires` trait), empty for an unknown component or
 * one with no requirements */
export function requiredComponents(world: World, name: string): string[] {
    const traits = world.registry.getTraits(name);
    if (!traits?.requires) return [];
    const out: string[] = [];
    for (const req of traits.requires) {
        const reqName = getName(world, req);
        if (reqName) out.push(reqName);
    }
    return out;
}

/** the registered names of this component's derived outputs (`Transform` and `Body` provide
 * `GlobalTransform`). Scene validation counts them as satisfying `requires` on the same entity.
 * Empty for an unknown component or one with no `provides` trait */
export function provides(world: World, name: string): string[] {
    const traits = world.registry.getTraits(name);
    if (!traits?.provides) return [];
    const out: string[] = [];
    for (const p of traits.provides) {
        const pName = getName(world, p);
        if (pName) out.push(pName);
    }
    return out;
}

/** true if the component declares the `singleton` trait: one instance per scene (lights, the active
 * camera). reflection metadata, not enforced; false for an unknown component */
export function isSingleton(world: World, name: string): boolean {
    return world.registry.getTraits(name)?.singleton ?? false;
}

/** the component names that may not coexist with this one (the symmetric `excludes` trait); empty for
 * an unknown component or one with no exclusions */
export function exclusions(world: World, name: string): string[] {
    const component = world.registry.getComponent(name);
    if (!component) return [];
    const set = world.registry.getExclusions(component);
    if (!set) return [];
    const out: string[] = [];
    for (const other of set) {
        const otherName = getName(world, other);
        if (otherName) out.push(otherName);
    }
    return out;
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

/**
 * every live entity carrying the named component, each as `EntityData`; empty for an unknown component.
 * @example
 * find(world, "point-light").length; // how many point lights are in the scene
 */
export function find(world: World, name: string): EntityData[] {
    const component = world.registry.getComponent(name);
    if (!component) return [];
    const out: EntityData[] = [];
    for (const eid of world.query([component as never])) {
        const data = inspect(world, eid);
        if (data) out.push(data);
    }
    return out;
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
