import type { State } from "./state";
import {
    getComponent,
    getComponentName,
    getComponents,
    getFieldLayout,
    isStringField,
    type Component,
    type FieldLayout,
    type FieldProxy,
    type Traits,
    type Derived,
} from "./component";
import { toKebabCase, formatHex } from "./strings";

export type FieldKind = "float" | "vec2" | "vec3" | "vec4" | "color" | "enum" | "string";

export interface FieldInfo {
    name: string;
    kind: FieldKind;
    default?: number | string;
    options?: Record<string, number>;
    fields?: string[];
    /** memory layout for direct chunk access. omitted for string and derived fields. */
    layout?: FieldLayout | FieldLayout[];
}

export interface Schema {
    name: string;
    fields: FieldInfo[];
    derived: FieldInfo[];
}

export function detectVec2(component: Component, base: string): boolean {
    return `${base}X` in component && `${base}Y` in component;
}

export function detectVec3(component: Component, base: string): boolean {
    return detectVec2(component, base) && `${base}Z` in component;
}

export function detectVec4(component: Component, base: string): boolean {
    return detectVec3(component, base) && `${base}W` in component;
}

function isColorField(key: string, traits: Traits | undefined): boolean {
    return traits?.format?.[key] === formatHex;
}

function vecSize(component: Component, base: string): 2 | 3 | 4 | 0 {
    if (detectVec4(component, base)) return 4;
    if (detectVec3(component, base)) return 3;
    if (detectVec2(component, base)) return 2;
    return 0;
}

const VEC_SUFFIXES: Record<number, string[]> = {
    2: ["X", "Y"],
    3: ["X", "Y", "Z"],
    4: ["X", "Y", "Z", "W"],
};

function layoutOf(component: Component, key: string): FieldLayout | undefined {
    const val = component[key];
    if (!val || typeof val !== "object") return undefined;
    return getFieldLayout(val as FieldProxy);
}

export function schema(name: string): Schema | null {
    const reg = getComponent(name);
    if (!reg) return null;

    const { component, traits } = reg;
    const defaults = traits?.defaults?.() ?? {};
    const handled = new Set<string>();
    const fields: FieldInfo[] = [];

    const deriveds = traits?.annotations?.derived as Record<string, Derived> | undefined;
    const derivedKeys = deriveds ? new Set(Object.keys(deriveds)) : new Set<string>();

    for (const key of Object.keys(component)) {
        if (handled.has(key) || derivedKeys.has(key)) continue;

        if (isStringField(component, key)) {
            fields.push({ name: key, kind: "string", default: defaults[key] });
            handled.add(key);
            continue;
        }

        if (key.endsWith("X")) {
            const base = key.slice(0, -1);
            const size = vecSize(component, base);
            if (size > 0) {
                const suffixes = VEC_SUFFIXES[size];
                const fieldNames = suffixes.map((s) => base + s);
                for (const f of fieldNames) handled.add(f);

                const kind = (size === 4 ? "vec4" : size === 3 ? "vec3" : "vec2") as FieldKind;
                const info: FieldInfo = { name: base, kind, fields: fieldNames };

                const defVals = fieldNames.map((f) => defaults[f]);
                if (defVals.every((v) => v !== undefined)) {
                    info.default = defVals[0];
                }

                const layouts = fieldNames
                    .map((f) => layoutOf(component, f))
                    .filter((l): l is FieldLayout => l !== undefined);
                if (layouts.length === fieldNames.length) info.layout = layouts;

                fields.push(info);
                continue;
            }
        }

        const enumDef = traits?.enums?.[key];
        if (enumDef) {
            const info: FieldInfo = {
                name: key,
                kind: "enum",
                default: defaults[key],
                options: enumDef,
            };
            const layout = layoutOf(component, key);
            if (layout) info.layout = layout;
            fields.push(info);
        } else if (isColorField(key, traits)) {
            const info: FieldInfo = { name: key, kind: "color", default: defaults[key] };
            const layout = layoutOf(component, key);
            if (layout) info.layout = layout;
            fields.push(info);
            for (const suffix of ["R", "G", "B"]) {
                handled.add(key + suffix);
            }
        } else {
            const info: FieldInfo = { name: key, kind: "float", default: defaults[key] };
            const layout = layoutOf(component, key);
            if (layout) info.layout = layout;
            fields.push(info);
        }
        handled.add(key);
    }

    const derived: FieldInfo[] = [];
    if (deriveds) {
        const virtHandled = new Set<string>();
        for (const key of Object.keys(deriveds)) {
            if (virtHandled.has(key)) continue;

            if (key.endsWith("X")) {
                const base = key.slice(0, -1);
                const yKey = base + "Y";
                const zKey = base + "Z";
                if (deriveds[yKey] && deriveds[zKey]) {
                    virtHandled.add(key);
                    virtHandled.add(yKey);
                    virtHandled.add(zKey);
                    derived.push({
                        name: base,
                        kind: "vec3",
                        fields: [key, yKey, zKey],
                    });
                    continue;
                }
            }
            derived.push({ name: key, kind: "float" });
            virtHandled.add(key);
        }
    }

    return { name: reg.name, fields, derived };
}

export function dependencies(name: string): string[] {
    const reg = getComponent(name);
    if (!reg?.traits?.requires) return [];
    const result: string[] = [];
    for (const req of reg.traits.requires) {
        const reqName = getComponentName(req);
        if (reqName) result.push(reqName);
    }
    return result;
}

export function schemas(): Schema[] {
    return getComponents()
        .map((entry) => schema(entry.name))
        .filter((s): s is Schema => s !== null);
}

export interface FieldValues {
    [field: string]: number | string;
}

export interface EntityData {
    eid: number;
    components: Record<string, FieldValues>;
}

export function readFields(component: Component, eid: number): FieldValues {
    const fields: FieldValues = {};
    for (const [field, store] of Object.entries(component)) {
        if (isStringField(component, field)) {
            const val = (store as Record<number, string>)[eid];
            if (val !== undefined) fields[field] = val;
        } else if (ArrayBuffer.isView(store) || Array.isArray(store)) {
            fields[field] = (store as number[])[eid];
        }
    }
    return fields;
}

export function inspect(state: State, eid: number): EntityData | null {
    if (!state.entityExists(eid)) return null;

    const components: Record<string, FieldValues> = {};
    for (const entry of getComponents()) {
        if (state.hasComponent(eid, entry.component as never)) {
            components[entry.name] = readFields(entry.component, eid);
        }
    }
    return { eid, components };
}

export function find(state: State, name: string): EntityData[] {
    const reg = getComponent(name);
    if (!reg) return [];

    const results: EntityData[] = [];
    for (const eid of state.query([reg.component as never])) {
        const data = inspect(state, eid);
        if (data) results.push(data);
    }
    return results;
}

export function snapshot(state: State): EntityData[] {
    const results: EntityData[] = [];
    for (const eid of state.getAllEntities()) {
        const data = inspect(state, eid);
        if (data) results.push(data);
    }
    return results;
}

export function dump(state: State, eid: number): string {
    const data = inspect(state, eid);
    if (!data) return `Entity ${eid}: not found`;

    const lines = [`Entity ${eid}:`];
    for (const [name, fields] of Object.entries(data.components)) {
        const parts = Object.entries(fields)
            .map(([k, v]) => `${toKebabCase(k)}: ${v}`)
            .join(", ");
        lines.push(`  ${name}: ${parts}`);
    }
    return lines.join("\n");
}
