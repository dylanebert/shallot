import type { Component, Pair, Quad, Single, State } from "../ecs";
import {
    camel,
    dependencies,
    entries,
    exclusions,
    getComponent,
    getTraits,
    kebab,
    lanes,
    provides,
    readFields,
    refs,
    type Traits,
} from "../ecs/core";
import type { Attr, Node, ParseError } from "./xml";

interface Registered {
    component: Component;
    name: string;
    traits?: Traits;
}

function lookup(rawName: string): Registered | undefined {
    const component = getComponent(rawName);
    if (!component) return undefined;
    return { component, name: kebab(rawName), traits: getTraits(rawName) };
}

interface Ref {
    attr: string;
    target: string;
}

interface PendingFieldRef {
    eid: number;
    component: Component;
    field: string;
    targetName: string;
}

function levenshtein(a: string, b: string): number {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    const matrix: number[][] = [];
    for (let i = 0; i <= b.length; i++) {
        matrix[i] = [i];
    }
    for (let j = 0; j <= a.length; j++) {
        matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            const cost = a[j - 1] === b[i - 1] ? 0 : 1;
            matrix[i][j] = Math.min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            );
        }
    }

    return matrix[b.length][a.length];
}

function findClosestMatch(input: string, candidates: string[]): string | null {
    const inputKebab = kebab(input);

    let bestMatch: string | null = null;
    let bestScore = Infinity;

    for (const candidate of candidates) {
        const candidateKebab = kebab(candidate);

        if (inputKebab === candidateKebab) {
            return candidate;
        }

        if (inputKebab.endsWith(candidateKebab) || inputKebab.endsWith("-" + candidateKebab)) {
            return candidate;
        }

        const distance = levenshtein(inputKebab, candidateKebab);
        const maxLen = Math.max(inputKebab.length, candidateKebab.length);
        const threshold = Math.ceil(maxLen * 0.5);

        if (distance < bestScore && distance <= threshold) {
            bestScore = distance;
            bestMatch = candidate;
        }
    }

    return bestMatch;
}

/**
 * builds ECS state from a parsed scene: one entity per node, one component per registered attribute, each
 * entity's scene `id` recorded on `state.identity` (so `serialize` round-trips refs by name). `@name`
 * field refs resolve to their target eid in a second pass. Throws on any unknown component or unresolved
 * ref, joined into one message. `run()` calls this; a custom loader calls `parse` then `load`.
 *
 * @example
 * const map = load(parse(xml), state);
 */
export function load(nodes: Node[], state: State): Map<Node, number> {
    const nameToEntity = new Map<string, number>();
    const nodeToEntity = new Map<Node, number>();
    const errors: ParseError[] = [];
    const pendingFieldRefs: PendingFieldRef[] = [];

    for (const node of nodes) {
        const eid = state.create();
        if (node.id) nameToEntity.set(node.id, eid);
        nodeToEntity.set(node, eid);
        state.identity.author(eid, node.id);
    }

    for (const node of nodes) {
        const eid = nodeToEntity.get(node)!;
        const { componentAttrs, refs } = categorizeAttrs(node.attrs);

        if (refs.length > 0) {
            errors.push({
                message: `Unknown attribute "${refs[0].attr}": top-level entity-ref attrs are no longer supported; use field-property syntax instead`,
            });
        }

        for (const attr of componentAttrs) {
            applyComponent(state, eid, attr, errors, pendingFieldRefs);
        }
    }

    for (const ref of pendingFieldRefs) {
        const targetEid = nameToEntity.get(ref.targetName);
        if (targetEid === undefined) {
            errors.push({ message: `Unknown entity: "@${ref.targetName}"` });
            continue;
        }
        setFieldValue(ref.component, ref.field, ref.eid, targetEid);
    }

    if (errors.length > 0) {
        throw new Error(errors.map((e) => e.message).join("\n"));
    }

    return nodeToEntity;
}

/**
 * reads one live component instance back to its scene attribute string (fields at their trait default
 * elide). The single per-component readback the on-demand `serialize` and the editor's per-frame
 * `ReadbackSystem` share, so the two paths can't drift.
 *
 * `resolveRef` (passed only by `serialize`) turns an `entity`-typed field's
 * stored eid into the target's scene `id`, so the field formats as `@<id>`: a
 * ref keyed on a stable name survives the creation-order eid reshuffle a reload
 * causes. The editor's `ReadbackSystem` omits it on purpose, leaving `@name`
 * attrs untouched.
 */
export function readComponent(
    name: string,
    component: Component,
    eid: number,
    resolveRef?: (target: number) => string | undefined,
): string {
    const defaults = getTraits(name)?.defaults?.() ?? {};
    const fields = readFields(component, eid);
    const merged: Record<string, number | string | readonly number[]> = { ...defaults, ...fields };
    if (resolveRef) {
        for (const field of refs(component)) {
            const value = merged[field];
            if (typeof value !== "number") continue;
            const id = resolveRef(value);
            if (id !== undefined) merged[field] = `@${id}`;
        }
    }
    return formatFields(name, merged);
}

/**
 * reads a live `State` back to a node tree, the on-demand inverse of `load`: one node per entity, one
 * attribute per registered component it has. `stringify` the result for the scene text (save /
 * survive-reload), or feed it back to `load` to rebuild. Pay-for-what-you-use, not a per-frame cost.
 *
 * By default it serializes the **authored** set — the entities `load` created.
 * `warm`-derived entities (orrstead's trees) are absent by construction and
 * rebuilt by `warm` on the next build, so a restore never doubles them; pass an
 * explicit `eids` to serialize a different set (entities spawned outside load).
 * Each entity keeps its scene `id`, and a `refs` field round-trips as `@<id>`
 * (a target lacking a scene id is minted one). A round-trip preserves
 * codec-representable component state; GPU buffers and derived entities are
 * rebuilt, not serialized (`ecs.md`).
 *
 * @example
 * const xml = stringify(serialize(state));
 */
export function serialize(state: State, eids?: Iterable<number>): Node[] {
    const list = (eids ? [...eids] : [...state.identity.authored]).filter((e) => state.exists(e));
    const set = new Set(list);

    const ids = new Map<number, string>();
    const used = new Set<string>();
    for (const eid of list) {
        const id = state.identity.id(eid);
        if (id !== undefined) {
            ids.set(eid, id);
            used.add(id);
        }
    }
    const mint = (eid: number): string => {
        const existing = ids.get(eid);
        if (existing !== undefined) return existing;
        let id = `e${eid}`;
        for (let n = 2; used.has(id); n++) id = `e${eid}_${n}`;
        ids.set(eid, id);
        used.add(id);
        return id;
    };

    // a ref target that lacks a scene id needs one minted before any node emits, so its @-ref resolves on reload
    for (const eid of list) {
        for (const { component, traits } of entries()) {
            if (traits?.derived) continue;
            if (!state.has(eid, component as never)) continue;
            for (const field of refs(component)) {
                const target = (component[field] as Single).get(eid);
                if (target > 0 && set.has(target)) mint(target);
            }
        }
    }

    const resolveRef = (target: number): string | undefined =>
        target > 0 && set.has(target) ? ids.get(target) : undefined;

    const nodes: Node[] = [];
    for (const eid of list) {
        const attrs: Attr[] = [];
        for (const { component, name, traits } of entries()) {
            // a derived decoration is a system's runtime state (union-relative ids), never scene truth
            if (traits?.derived) continue;
            if (!state.has(eid, component as never)) continue;
            attrs.push({ name, value: readComponent(name, component, eid, resolveRef) });
        }
        nodes.push({ id: ids.get(eid), attrs, children: [] });
    }
    return nodes;
}

interface CategorizedAttrs {
    componentAttrs: { name: string; value: string; def: Registered }[];
    refs: Ref[];
    unknown: { name: string; value: string }[];
}

function categorizeAttrs(attrs: Attr[]): CategorizedAttrs {
    const componentAttrs: { name: string; value: string; def: Registered }[] = [];
    const refs: Ref[] = [];
    const unknown: { name: string; value: string }[] = [];

    for (const attr of attrs) {
        if (attr.value.startsWith("@") && attr.value.length > 1) {
            refs.push({ attr: attr.name, target: attr.value.slice(1) });
            continue;
        }

        const registered = lookup(attr.name);
        if (registered) {
            componentAttrs.push({ name: attr.name, value: attr.value, def: registered });
            continue;
        }

        unknown.push({ name: attr.name, value: attr.value });
    }

    return { componentAttrs, refs, unknown };
}

function applyComponent(
    state: State,
    eid: number,
    attr: { name: string; value: string; def: Registered },
    errors: ParseError[],
    pendingFieldRefs: PendingFieldRef[],
): void {
    const { def, value } = attr;
    const { component, name, traits } = def;

    state.add(eid, component as never);

    const defaults = traits?.defaults?.() ?? {};
    for (const [field, val] of Object.entries(defaults)) {
        setFieldValue(component, field, eid, val as number | number[]);
    }

    const props: Record<string, string> = {};
    if (value !== "") {
        props["_value"] = value;
    }

    const result = parseAttrs(def, props);
    const values = result.values;
    const entityRefs = result.entityRefs;
    for (const err of result.errors) {
        errors.push({ message: `<${name}> ${err}` });
    }

    for (const [field, val] of Object.entries(values)) {
        setFieldValue(component, field, eid, val);
    }

    for (const ref of entityRefs) {
        pendingFieldRefs.push({
            eid,
            component,
            field: ref.field,
            targetName: ref.targetName,
        });
    }
}

function parseAttrs(
    def: Registered,
    props: Record<string, string>,
): {
    values: Record<string, number>;
    entityRefs: { field: string; targetName: string }[];
    errors: string[];
} {
    const allValues: Record<string, number> = {};
    const allEntityRefs: { field: string; targetName: string }[] = [];
    const allErrors: string[] = [];

    if (props._value) {
        if (isCSSAttrSyntax(props._value)) {
            const result = parsePropertyString(def, props._value);
            Object.assign(allValues, result.values);
            allEntityRefs.push(...result.entityRefs);
            allErrors.push(...result.errors);
        }
    }

    for (const [propName, propValue] of Object.entries(props)) {
        if (propName === "_value") continue;
        if (!propValue) continue;

        if (isCSSAttrSyntax(propValue)) {
            const result = parsePropertyString(def, propValue);
            Object.assign(allValues, result.values);
            allEntityRefs.push(...result.entityRefs);
            allErrors.push(...result.errors);
        } else {
            const result = parsePropertyString(def, `${propName}: ${propValue}`);
            Object.assign(allValues, result.values);
            allEntityRefs.push(...result.entityRefs);
            allErrors.push(...result.errors);
        }
    }

    return { values: allValues, entityRefs: allEntityRefs, errors: allErrors };
}

/**
 * write a scene-parsed value into a component field. Handles:
 *
 * - `field = "pos"`, `value = number` — Single, or first lane of a Pair/Quad
 * - `field = "pos"`, `value = number[]` — Pair/Quad bulk lane write
 * - `field = "pos.x"`, `value = number` — single lane of a parent Pair/Quad
 * - `field = "posX"`, `value = number` — legacy lane Single (split-suffix
 *   storage). Path retires when the last split-suffix component migrates
 */
export function setFieldValue(
    component: Component,
    field: string,
    eid: number,
    value: number | number[],
): void {
    const dotIdx = field.indexOf(".");
    if (dotIdx !== -1) {
        const base = field.slice(0, dotIdx);
        const laneKey = field.slice(dotIdx + 1);
        const parent = component[base];
        if (parent == null) return;
        const lane = (parent as Record<string, unknown>)[laneKey] as Single | undefined;
        if (lane && typeof lane.set === "function" && typeof value === "number") {
            lane.set(eid, value);
        }
        return;
    }

    const target = component[field];
    if (target == null) return;
    const n = lanes(target);

    if (Array.isArray(value)) {
        if (n === 4) {
            const q = target as Quad;
            q.set(eid, value[0] ?? 0, value[1] ?? 0, value[2] ?? 0, value[3] ?? 0);
        } else if (n === 2) {
            const p = target as Pair;
            p.set(eid, value[0] ?? 0, value[1] ?? 0);
        } else if (n === 1) {
            (target as Single).set(eid, value[0] ?? 0);
        } else if (ArrayBuffer.isView(target) || Array.isArray(target)) {
            (target as number[])[eid] = value[0] ?? 0;
        }
        return;
    }

    if (ArrayBuffer.isView(target) || Array.isArray(target)) {
        (target as number[])[eid] = value;
    } else if (typeof (target as Single).set === "function") {
        (target as Single).set(eid, value);
    } else {
        console.warn(`Scene: cannot assign number to non-array field "${field}"`);
    }
}

function parseNumber(value: string): number | null {
    value = value.trim();

    if (value.startsWith("0x") || value.startsWith("0X")) {
        return parseInt(value, 16);
    }

    if (value.startsWith("#")) {
        const hex = value.slice(1);
        if (!/^[0-9a-fA-F]+$/.test(hex)) return null;
        return parseInt(hex, 16);
    }

    if (value === "true") return 1;
    if (value === "false") return 0;

    const num = parseFloat(value);
    return Number.isNaN(num) ? null : num;
}

function parseValues(valueStr: string): (number | null)[] {
    const result: (number | null)[] = [];
    const trimmed = valueStr.trim();
    let start = 0;
    for (let i = 0; i <= trimmed.length; i++) {
        const isWhitespace = i < trimmed.length && /\s/.test(trimmed[i]);
        const isEnd = i === trimmed.length;
        if (isWhitespace || isEnd) {
            if (start < i) {
                result.push(parseNumber(trimmed.slice(start, i)));
            }
            start = i + 1;
        }
    }
    return result;
}

function splitProperties(str: string): string[] {
    const result: string[] = [];
    let start = 0;
    for (let i = 0; i <= str.length; i++) {
        if (i === str.length || str[i] === ";") {
            const prop = str.slice(start, i).trim();
            if (prop) result.push(prop);
            start = i + 1;
        }
    }
    return result;
}

// the dotted lane key (`params.x`) a named axis of an identity-lane alias resolves to (`metallic` →
// `params.x`). Identity = one axis per lane; euler's 3-axis-over-4-lane alias fails the length check and
// stays positional. Drives both named parse (here) and named serialize (formatFields).
function identityLaneKey(
    traits: Traits | undefined,
    component: Component,
    axis: string,
): string | undefined {
    const aliases = traits?.aliases;
    if (!aliases) return undefined;
    for (const field in aliases) {
        const axes = aliases[field].axes;
        if (axes.length !== lanes(component[field])) continue;
        const i = axes.indexOf(axis);
        if (i !== -1) return `${field}.${"xyzw"[i]}`;
    }
    return undefined;
}

function parsePropertyString(
    entry: Registered,
    propertyString: string,
): {
    values: Record<string, number>;
    entityRefs: { field: string; targetName: string }[];
    errors: string[];
} {
    const { component, name: componentName, traits } = entry;
    const values: Record<string, number> = {};
    const entityRefs: { field: string; targetName: string }[] = [];
    const errors: string[] = [];

    const properties = splitProperties(propertyString);

    for (const prop of properties) {
        const colonIdx = prop.indexOf(":");
        if (colonIdx === -1) {
            errors.push(`Invalid syntax: "${prop}" (expected "field: value")`);
            continue;
        }

        const rawName = prop.slice(0, colonIdx).trim();
        const valueStr = prop.slice(colonIdx + 1).trim();

        if (!rawName || !valueStr) {
            errors.push(`Invalid syntax: "${prop}" (empty field or value)`);
            continue;
        }

        const name = camel(rawName);

        const dotIdx = name.indexOf(".");
        if (dotIdx !== -1) {
            const base = name.slice(0, dotIdx);
            const laneKey = name.slice(dotIdx + 1);
            const parent = component[base];
            const parentLanes = lanes(parent);
            if (parentLanes >= 2 && laneKey in (parent as Record<string, unknown>)) {
                const parsedLane = parseValues(valueStr);
                if (parsedLane.length !== 1 || parsedLane[0] === null) {
                    errors.push(`${componentName}.${rawName}: expected 1 value`);
                    continue;
                }
                values[name] = parsedLane[0];
                continue;
            }
            errors.push(`${componentName}: unknown field "${rawName}"`);
            continue;
        }

        if (valueStr.startsWith("@") && valueStr.length > 1) {
            if (name in component) {
                entityRefs.push({ field: name, targetName: valueStr.slice(1) });
            } else {
                const fieldNames = Object.keys(component);
                const suggestion = findClosestMatch(rawName, fieldNames);
                if (suggestion) {
                    errors.push(
                        `${componentName}: unknown field "${rawName}", did you mean "${kebab(suggestion)}"?`,
                    );
                } else {
                    errors.push(`${componentName}: unknown field "${rawName}"`);
                }
            }
            continue;
        }

        const parsed = parseValues(valueStr);

        if (parsed.some((v) => v === null)) {
            const parseFn = traits?.parse?.[name];
            if (parseFn) {
                const resolved = parseFn(valueStr.trim());
                if (resolved !== undefined) {
                    values[name] = resolved;
                    continue;
                }
            }
            errors.push(`Invalid number in "${prop}"`);
            continue;
        }

        const nums = parsed as number[];

        const direct = lanes(component[name]);
        if (direct === 4) {
            if (nums.length === 4) {
                values[`${name}.x`] = nums[0];
                values[`${name}.y`] = nums[1];
                values[`${name}.z`] = nums[2];
                values[`${name}.w`] = nums[3];
            } else if (nums.length === 3) {
                values[`${name}.x`] = nums[0];
                values[`${name}.y`] = nums[1];
                values[`${name}.z`] = nums[2];
            } else if (nums.length === 1) {
                values[`${name}.x`] = nums[0];
                values[`${name}.y`] = nums[0];
                values[`${name}.z`] = nums[0];
                values[`${name}.w`] = nums[0];
            } else {
                errors.push(
                    `${componentName}.${rawName}: expected 1, 3, or 4 values, got ${nums.length}`,
                );
            }
            continue;
        }

        if (direct === 2) {
            if (nums.length === 2) {
                values[`${name}.x`] = nums[0];
                values[`${name}.y`] = nums[1];
            } else if (nums.length === 1) {
                values[`${name}.x`] = nums[0];
                values[`${name}.y`] = nums[0];
            } else {
                errors.push(
                    `${componentName}.${rawName}: expected 1 or 2 values, got ${nums.length}`,
                );
            }
            continue;
        }

        if (name in component) {
            if (nums.length === 1) {
                values[name] = nums[0];
            } else {
                errors.push(`${componentName}.${rawName}: expected 1 value, got ${nums.length}`);
            }
            continue;
        }

        // named lane of an identity-lane alias: `metallic: 1` writes one lane of a packed Quad/Pair
        const laneKey = identityLaneKey(traits, component, name);
        if (laneKey) {
            if (nums.length === 1) {
                values[laneKey] = nums[0];
            } else {
                errors.push(`${componentName}.${rawName}: expected 1 value, got ${nums.length}`);
            }
            continue;
        }

        const fieldNames = Object.keys(component);
        const suggestion = findClosestMatch(rawName, fieldNames);
        if (suggestion) {
            errors.push(
                `${componentName}: unknown field "${rawName}", did you mean "${kebab(suggestion)}"?`,
            );
        } else {
            errors.push(`${componentName}: unknown field "${rawName}"`);
        }
    }

    return { values, entityRefs, errors };
}

/**
 * parses one component's attribute string into a field-value record, the inverse of `formatFields`. Vector
 * fields expand to dotted lanes (`"pos: 0 5 0"` → `{ "pos.x": 0, "pos.y": 5, "pos.z": 0 }`); an `@name`
 * ref stays a string. Throws on an unknown field or malformed value. The editor inspector parses an edit
 * through this before writing it back.
 *
 * @example
 * parseFields("transform", "pos: 0 5 0"); // { "pos.x": 0, "pos.y": 5, "pos.z": 0 }
 */
export function parseFields(
    componentName: string,
    attrValue: string,
): Record<string, number | string> {
    const registered = lookup(componentName);
    if (!registered) {
        throw new Error(`Unknown component "${componentName}"`);
    }

    const result = parsePropertyString(registered, attrValue);
    if (result.errors.length > 0) {
        throw new Error(result.errors.join("\n"));
    }

    const fields: Record<string, number | string> = {};
    for (const [k, v] of Object.entries(result.values)) {
        fields[k] = v;
    }
    for (const ref of result.entityRefs) {
        fields[ref.field] = `@${ref.targetName}`;
    }
    return fields;
}

/**
 * expand any array-form values on direct Pair/Quad fields into dotted lane
 * keys. `{ pos: [1, 2, 3, 4] }` → `{ "pos.x": 1, "pos.y": 2, ... }`. Used to
 * normalize the merged-defaults+fields record before formatting
 */
function normalizeFields(
    component: Component,
    fields: Record<string, number | string | readonly number[]>,
): Record<string, number | string> {
    const out: Record<string, number | string> = {};
    for (const [k, v] of Object.entries(fields)) {
        if (Array.isArray(v)) {
            const n = lanes(component[k]);
            if (n === 4) {
                out[`${k}.x`] = v[0] ?? 0;
                out[`${k}.y`] = v[1] ?? 0;
                out[`${k}.z`] = v[2] ?? 0;
                out[`${k}.w`] = v[3] ?? 0;
            } else if (n === 2) {
                out[`${k}.x`] = v[0] ?? 0;
                out[`${k}.y`] = v[1] ?? 0;
            } else if (v.length > 0) {
                out[k] = v[0];
            }
        } else {
            out[k] = v as number | string;
        }
    }
    return out;
}

/**
 * formats a field-value record into a component's canonical attribute string, the inverse of `parseFields`.
 * Vectors collapse (`{ "pos.x": 1, "pos.y": 2, "pos.z": 3 }` → `"pos: 1 2 3"`) and fields at their trait
 * default elide. Pass `{ stripDefaults: false }` to keep every field.
 *
 * @example
 * formatFields("transform", { "pos.x": 0, "pos.y": 5, "pos.z": 0 }); // "pos: 0 5 0"
 */
export function formatFields(
    componentName: string,
    fieldsInput: Record<string, number | string | readonly number[]>,
    options?: { stripDefaults?: boolean },
): string {
    const registered = lookup(componentName);
    if (!registered) {
        throw new Error(`Unknown component "${componentName}"`);
    }

    const { component, traits } = registered;
    const rawDefaults = traits?.defaults?.() ?? {};
    const defaults = normalizeFields(component, rawDefaults) as Record<string, number>;
    const format = traits?.format;
    const stripDefaults = options?.stripDefaults !== false;
    const fields = normalizeFields(component, fieldsInput);

    const remaining = new Set(Object.keys(fields));
    const parts: string[] = [];

    const handled = new Set<string>();

    // identity-lane aliases: emit each lane as an independent named scalar (`metallic: 1; roughness: 0.2`)
    // and claim its dotted key, so the positional Pair/Quad loop below skips it. Each lane elides
    // independently at its default — a packed material has no vec3-shaped trailing-lane semantics.
    const aliases = traits?.aliases;
    if (aliases) {
        for (const field in aliases) {
            const axes = aliases[field].axes;
            const n = lanes(component[field]);
            if (axes.length !== n) continue;
            for (let i = 0; i < n; i++) {
                const dotKey = `${field}.${"xyzw"[i]}`;
                if (!remaining.has(dotKey)) continue;
                handled.add(dotKey);
                remaining.delete(dotKey);
                const value = fields[dotKey] as number;
                if (stripDefaults && defaults[dotKey] !== undefined && value === defaults[dotKey]) {
                    continue;
                }
                parts.push(`${kebab(axes[i])}: ${formatNumber(value)}`);
            }
        }
    }

    // direct Pair/Quad fields keyed by dotted lane (`pos.x`, `pos.y` …).
    // Emit the longest contiguous lane prefix the user actually supplied —
    // omitted trailing lanes parse-back from the trait default, so dropping
    // them keeps `pos: 1 2 3` (vec3-shaped Quad) roundtripping cleanly
    // through `normalizeAttr`. Trim-trailing-default still operates within
    // the prefix when stripDefaults is on.
    for (const field of [...remaining]) {
        if (handled.has(field)) continue;
        const dotIdx = field.indexOf(".");
        if (dotIdx === -1) continue;
        const base = field.slice(0, dotIdx);
        const direct = lanes(component[base]);
        if (direct !== 2 && direct !== 4) continue;
        const laneNames = direct === 4 ? ["x", "y", "z", "w"] : ["x", "y"];
        const dotKeys = laneNames.map((l) => `${base}.${l}`);
        let prefix = 0;
        while (prefix < dotKeys.length && remaining.has(dotKeys[prefix])) prefix++;
        // smallest non-splat partial-set: Quad parses {1, 3, 4}, Pair parses
        // {1, 2}. Trim to 2 on a Quad would emit a length the parser rejects;
        // trim to 1 splats on re-parse and loses lane intent.
        const minPartial = direct === 4 ? 3 : 2;
        if (prefix < minPartial) continue;
        const values = dotKeys.slice(0, prefix).map((k) => fields[k] as number);
        const defaultValues = dotKeys.slice(0, prefix).map((k) => defaults[k]);
        const allDefault =
            stripDefaults &&
            values.every((v, i) => defaultValues[i] !== undefined && v === defaultValues[i]);
        if (!allDefault) {
            let trimEnd = values.length;
            if (stripDefaults) {
                while (
                    trimEnd > minPartial &&
                    defaultValues[trimEnd - 1] !== undefined &&
                    values[trimEnd - 1] === defaultValues[trimEnd - 1]
                ) {
                    trimEnd--;
                }
            }
            const emitted = values.slice(0, trimEnd);
            const allEqual = emitted.every((v) => v === emitted[0]);
            const k = kebab(base);
            // `pos: 5` splats to every lane at parse; only collapse when
            // emitting the full lane count, else `pos: 5 5 5` on a Quad
            // would silently set lane w to 5
            if (allEqual && emitted.length > 1 && emitted.length === direct) {
                parts.push(`${k}: ${formatNumber(emitted[0])}`);
            } else {
                parts.push(`${k}: ${emitted.map((v) => formatNumber(v)).join(" ")}`);
            }
        }
        for (let i = 0; i < prefix; i++) {
            handled.add(dotKeys[i]);
            remaining.delete(dotKeys[i]);
        }
    }

    for (const field of remaining) {
        if (handled.has(field)) continue;
        const value = fields[field];
        const def = defaults[field];

        // NaN is a valid sentinel default (e.g. Tween.from = "capture at runtime") but NaN !== NaN, so
        // a plain `value === def` never elides it and emits an unparseable `from: NaN`. Treat a field
        // sitting at its NaN default as default, so it elides and re-parses back to the sentinel.
        if (
            stripDefaults &&
            typeof value === "number" &&
            def !== undefined &&
            (value === def || (Number.isNaN(value) && Number.isNaN(def)))
        )
            continue;

        const k = kebab(field);

        if (typeof value === "string") {
            parts.push(`${k}: ${value}`);
            continue;
        }

        const formatFn = format?.[field];
        if (formatFn) {
            const formatted = formatFn(value as number);
            if (formatted !== undefined) {
                parts.push(`${k}: ${formatted}`);
                continue;
            }
        }

        parts.push(`${k}: ${formatNumber(value as number)}`);
    }

    return parts.join("; ");
}

/**
 * normalize a scene attribute value to its canonical form: parse, then re-format the way the live
 * `serialize` path does (`stripDefaults` on, so a field sitting at its trait default elides). The scene
 * formatter (`scripts/format.ts`) runs every `.scene` through this, so a formatted file is the same
 * minimal bytes the editor's save path and `serialize(state)` emit: one canonical form, no divergence
 * between hand-authored and editor-written scenes. Returns null for an empty value or unregistered
 * component (left untouched).
 */
export function normalizeAttr(name: string, value: string): string | null {
    if (!value) return null;
    if (!getComponent(name)) return null;
    try {
        const fields = parseFields(name, value);
        return formatFields(name, fields);
    } catch {
        return null;
    }
}

function formatNumber(n: number): string {
    if (Number.isInteger(n)) return n.toString();
    return +n.toPrecision(7) + "";
}

function isCSSAttrSyntax(value: string): boolean {
    // dotted-field syntax (e.g. `pos.x: 5`) opts in just like `field: value`
    return value.includes(":") && (value.includes(";") || /^[\w-]+(\.[a-z])?\s*:/.test(value));
}

/** one scene validation issue: the offending `node` and `attr`, a `kind` tag (`"unregistered"` / `"missing-requires"` / `"excluded-with"`), and a human-readable `message`. */
export interface Diagnostic {
    readonly node: Node;
    readonly attr: string;
    readonly kind: string;
    readonly message: string;
}

/**
 * validates a parsed scene against the registered components: an unknown component (with a did-you-mean
 * suggestion), an unmet `requires` trait, or a violated `excludes`. Returns every issue found; `run()`
 * warns each to the console, the editor surfaces them in the inspector. Empty means the scene is clean.
 *
 * @example
 * for (const d of diagnose(parse(xml))) console.warn(d.message);
 */
export function diagnose(nodes: Node[]): Diagnostic[] {
    const results: Diagnostic[] = [];
    const registered = [...entries()].map((e) => e.name);
    for (const node of nodes) {
        const attrNames = new Set(node.attrs.map((a) => a.name));
        // a component that `provides` X satisfies another's `requires` X on the same entity (Body
        // provides Transform), so fold every attr's provisions into the satisfied set
        const satisfied = new Set(attrNames);
        for (const name of attrNames) for (const p of provides(name)) satisfied.add(p);
        for (const attr of node.attrs) {
            if (attr.value.startsWith("@") && attr.value.length > 1) continue;
            const reg = getComponent(attr.name);
            if (!reg) {
                const suggestion = findClosestMatch(attr.name, registered);
                const message = suggestion
                    ? `"${attr.name}" is not registered, did you mean "${suggestion}"?`
                    : `"${attr.name}" is not registered`;
                results.push({ node, attr: attr.name, kind: "unregistered", message });
                continue;
            }
            if (getTraits(attr.name)?.derived) {
                results.push({
                    node,
                    attr: attr.name,
                    kind: "derived",
                    message: `"${attr.name}" is runtime-derived — a system owns it, so the authored value is overwritten`,
                });
            }
            for (const reqName of dependencies(attr.name)) {
                if (!satisfied.has(reqName)) {
                    results.push({
                        node,
                        attr: attr.name,
                        kind: "missing-requires",
                        message: `"${attr.name}" requires "${reqName}"`,
                    });
                }
            }
            for (const excName of exclusions(attr.name)) {
                if (attrNames.has(excName) && excName > attr.name) {
                    results.push({
                        node,
                        attr: attr.name,
                        kind: "excluded-with",
                        message: `"${attr.name}" cannot coexist with "${excName}"`,
                    });
                }
            }
        }
    }
    return results;
}
