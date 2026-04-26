import { ChildOf, type State, type Component } from "../ecs";
import {
    getComponent,
    getComponentName,
    getComponents,
    getRelation,
    getTraits,
    isStringField,
    toKebabCase,
    toCamelCase,
    detectVec2,
    detectVec3,
    detectVec4,
    type ComponentEntry,
} from "../ecs/core";
import type { Node, Attr, ParseError } from "./xml";

export {
    parse,
    serialize,
    findParent,
    findNodeById,
    type Node,
    type Attr,
    type ParseError,
} from "./xml";

export interface Ref {
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
    const inputKebab = toKebabCase(input);

    let bestMatch: string | null = null;
    let bestScore = Infinity;

    for (const candidate of candidates) {
        const candidateKebab = toKebabCase(candidate);

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

interface QueuedEntity {
    node: Node;
    eid: number;
    parent?: number;
}

export function load(nodes: Node[], state: State): Map<Node, number> {
    const nameToEntity = new Map<string, number>();
    const nodeToEntity = new Map<Node, number>();
    const errors: ParseError[] = [];
    const queue: QueuedEntity[] = [];
    const pendingFieldRefs: PendingFieldRef[] = [];

    for (const node of nodes) {
        createEntityTree(state, node, nameToEntity, nodeToEntity, undefined, queue);
    }

    for (const { node, eid, parent } of queue) {
        if (parent !== undefined) {
            state.addRelation(eid, ChildOf, parent);
        }

        const { componentAttrs, refs } = categorizeAttrs(node.attrs);

        for (const ref of refs) {
            applyRelation(state, eid, ref, nameToEntity, errors);
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

interface CategorizedAttrs {
    componentAttrs: { name: string; value: string; def: ComponentEntry }[];
    refs: Ref[];
    unknown: { name: string; value: string }[];
}

function categorizeAttrs(attrs: Attr[]): CategorizedAttrs {
    const componentAttrs: { name: string; value: string; def: ComponentEntry }[] = [];
    const refs: Ref[] = [];
    const unknown: { name: string; value: string }[] = [];

    for (const attr of attrs) {
        if (attr.value.startsWith("@") && attr.value.length > 1) {
            refs.push({ attr: attr.name, target: attr.value.slice(1) });
            continue;
        }

        const registered = getComponent(attr.name);
        if (registered) {
            componentAttrs.push({ name: attr.name, value: attr.value, def: registered });
            continue;
        }

        unknown.push({ name: attr.name, value: attr.value });
    }

    return { componentAttrs, refs, unknown };
}

function createEntityTree(
    state: State,
    node: Node,
    nameToEntity: Map<string, number>,
    nodeToEntity: Map<Node, number>,
    parent: number | undefined,
    queue: QueuedEntity[],
): number {
    const eid = state.addEntity();

    if (node.id) {
        nameToEntity.set(node.id, eid);
    }
    nodeToEntity.set(node, eid);

    queue.push({ node, eid, parent });

    for (const child of node.children) {
        createEntityTree(state, child, nameToEntity, nodeToEntity, eid, queue);
    }

    return eid;
}

function applyRelation(
    state: State,
    eid: number,
    ref: Ref,
    nameToEntity: Map<string, number>,
    errors: ParseError[],
): void {
    const rel = getRelation(ref.attr);
    if (!rel) {
        errors.push({ message: `Unknown relation: "${ref.attr}"` });
        return;
    }

    const targetEid = nameToEntity.get(ref.target);
    if (targetEid === undefined) {
        errors.push({ message: `Unknown entity: "@${ref.target}"` });
        return;
    }

    state.addRelation(eid, rel, targetEid);
}

function applyComponent(
    state: State,
    eid: number,
    attr: { name: string; value: string; def: ComponentEntry },
    errors: ParseError[],
    pendingFieldRefs: PendingFieldRef[],
): void {
    const { def, value } = attr;
    const { component, name, traits } = def;

    state.addComponent(eid, component as never);

    const defaults = traits?.defaults?.() ?? {};
    for (const [field, val] of Object.entries(defaults)) {
        setFieldValue(component, field, eid, val as number);
    }

    const props: Record<string, string> = {};
    if (value !== "") {
        props["_value"] = value;
    }

    const result = parseAttrs(def, props);
    const values = result.values;
    const strings = result.strings;
    const entityRefs = result.entityRefs;
    for (const err of result.errors) {
        errors.push({ message: `<${name}> ${err}` });
    }

    for (const [field, val] of Object.entries(values)) {
        setFieldValue(component, field, eid, val);
    }

    for (const [field, val] of Object.entries(strings)) {
        setString(component, field, eid, val);
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
    def: ComponentEntry,
    props: Record<string, string>,
): {
    values: Record<string, number>;
    strings: Record<string, string>;
    entityRefs: { field: string; targetName: string }[];
    errors: string[];
} {
    const allValues: Record<string, number> = {};
    const allStrings: Record<string, string> = {};
    const allEntityRefs: { field: string; targetName: string }[] = [];
    const allErrors: string[] = [];

    if (props._value) {
        if (isCSSAttrSyntax(props._value)) {
            const result = parsePropertyString(def.name, props._value, def.component);
            Object.assign(allValues, result.values);
            Object.assign(allStrings, result.strings);
            allEntityRefs.push(...result.entityRefs);
            allErrors.push(...result.errors);
        }
    }

    for (const [propName, propValue] of Object.entries(props)) {
        if (propName === "_value") continue;
        if (!propValue) continue;

        if (isCSSAttrSyntax(propValue)) {
            const result = parsePropertyString(def.name, propValue, def.component);
            Object.assign(allValues, result.values);
            Object.assign(allStrings, result.strings);
            allEntityRefs.push(...result.entityRefs);
            allErrors.push(...result.errors);
        } else {
            const result = parsePropertyString(
                def.name,
                `${propName}: ${propValue}`,
                def.component,
            );
            Object.assign(allValues, result.values);
            Object.assign(allStrings, result.strings);
            allEntityRefs.push(...result.entityRefs);
            allErrors.push(...result.errors);
        }
    }

    return { values: allValues, strings: allStrings, entityRefs: allEntityRefs, errors: allErrors };
}

export function setFieldValue(
    component: Component,
    field: string,
    eid: number,
    value: number,
): void {
    const arr = component[field];
    if (arr == null) return;
    if (ArrayBuffer.isView(arr) || Array.isArray(arr)) {
        (arr as number[])[eid] = value;
    } else {
        console.warn(`Scene: cannot assign number to non-array field "${field}"`);
    }
}

export { isStringField } from "../ecs/core";

export function setString(component: Component, field: string, eid: number, value: string): void {
    (component[field] as Record<number, string>)[eid] = value;
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

function parsePropertyString(
    componentName: string,
    propertyString: string,
    component: Component,
): {
    values: Record<string, number>;
    strings: Record<string, string>;
    entityRefs: { field: string; targetName: string }[];
    errors: string[];
} {
    const values: Record<string, number> = {};
    const strings: Record<string, string> = {};
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

        const name = toCamelCase(rawName);

        if (valueStr.startsWith("@") && valueStr.length > 1) {
            if (name in component) {
                entityRefs.push({ field: name, targetName: valueStr.slice(1) });
            } else {
                const fieldNames = Object.keys(component);
                const suggestion = findClosestMatch(rawName, fieldNames);
                if (suggestion) {
                    errors.push(
                        `${componentName}: unknown field "${rawName}", did you mean "${toKebabCase(suggestion)}"?`,
                    );
                } else {
                    errors.push(`${componentName}: unknown field "${rawName}"`);
                }
            }
            continue;
        }

        const parsed = parseValues(valueStr);

        if (parsed.some((v) => v === null)) {
            const rawValue = valueStr.trim();

            if (name in component && isStringField(component, name)) {
                strings[name] = rawValue;
                continue;
            }

            if (parsed.length === 1) {
                const traits = getTraits(component);
                const parseFn = traits?.parse?.[name];
                if (parseFn) {
                    const resolved = parseFn(rawValue);
                    if (resolved !== undefined) {
                        values[name] = resolved;
                        continue;
                    }
                }
            }
            errors.push(`Invalid number in "${prop}"`);
            continue;
        }

        const nums = parsed as number[];

        if (detectVec4(component, name)) {
            if (nums.length === 4) {
                values[`${name}X`] = nums[0];
                values[`${name}Y`] = nums[1];
                values[`${name}Z`] = nums[2];
                values[`${name}W`] = nums[3];
            } else if (nums.length === 1) {
                values[`${name}X`] = nums[0];
                values[`${name}Y`] = nums[0];
                values[`${name}Z`] = nums[0];
                values[`${name}W`] = nums[0];
            } else {
                errors.push(
                    `${componentName}.${rawName}: expected 1 or 4 values, got ${nums.length}`,
                );
            }
            continue;
        }

        if (detectVec3(component, name)) {
            if (nums.length === 3) {
                values[`${name}X`] = nums[0];
                values[`${name}Y`] = nums[1];
                values[`${name}Z`] = nums[2];
            } else if (nums.length === 1) {
                values[`${name}X`] = nums[0];
                values[`${name}Y`] = nums[0];
                values[`${name}Z`] = nums[0];
            } else {
                errors.push(
                    `${componentName}.${rawName}: expected 1 or 3 values, got ${nums.length}`,
                );
            }
            continue;
        }

        if (detectVec2(component, name)) {
            if (nums.length === 2) {
                values[`${name}X`] = nums[0];
                values[`${name}Y`] = nums[1];
            } else if (nums.length === 1) {
                values[`${name}X`] = nums[0];
                values[`${name}Y`] = nums[0];
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

        const fieldNames = Object.keys(component);
        const suggestion = findClosestMatch(rawName, fieldNames);
        if (suggestion) {
            errors.push(
                `${componentName}: unknown field "${rawName}", did you mean "${toKebabCase(suggestion)}"?`,
            );
        } else {
            errors.push(`${componentName}: unknown field "${rawName}"`);
        }
    }

    return { values, strings, entityRefs, errors };
}

export function parseFields(
    componentName: string,
    attrValue: string,
): Record<string, number | string> {
    const registered = getComponent(componentName);
    if (!registered) {
        throw new Error(`Unknown component "${componentName}"`);
    }

    const result = parsePropertyString(registered.name, attrValue, registered.component);
    if (result.errors.length > 0) {
        throw new Error(result.errors.join("\n"));
    }

    const fields: Record<string, number | string> = {};
    for (const [k, v] of Object.entries(result.values)) {
        fields[k] = v;
    }
    for (const [k, v] of Object.entries(result.strings)) {
        fields[k] = v;
    }
    return fields;
}

export function formatFields(
    componentName: string,
    fields: Record<string, number | string>,
    options?: { stripDefaults?: boolean },
): string {
    const registered = getComponent(componentName);
    if (!registered) {
        throw new Error(`Unknown component "${componentName}"`);
    }

    const { component, traits } = registered;
    const defaults = traits?.defaults?.() ?? {};
    const format = traits?.format;
    const stripDefaults = options?.stripDefaults !== false;

    const remaining = new Set(Object.keys(fields));
    const parts: string[] = [];

    const vecSizes = [4, 3, 2] as const;
    const detectors = {
        4: detectVec4,
        3: detectVec3,
        2: detectVec2,
    };
    const suffixes = {
        4: ["X", "Y", "Z", "W"],
        3: ["X", "Y", "Z"],
        2: ["X", "Y"],
    };

    const handled = new Set<string>();
    for (const field of remaining) {
        if (handled.has(field)) continue;

        for (const size of vecSizes) {
            for (const suffix of suffixes[size]) {
                if (field.endsWith(suffix)) {
                    const base = field.slice(0, -1);
                    if (detectors[size](component, base) && !handled.has(base)) {
                        const suf = suffixes[size];
                        const fieldNames = suf.map((s) => base + s);
                        const allPresent = fieldNames.every((f) => remaining.has(f));
                        if (allPresent) {
                            const values = fieldNames.map((f) => fields[f] as number);
                            const defaultValues = fieldNames.map((f) => defaults[f]);
                            const allDefault =
                                stripDefaults &&
                                values.every(
                                    (v, i) =>
                                        defaultValues[i] !== undefined && v === defaultValues[i],
                                );
                            if (!allDefault) {
                                const allEqual = values.every((v) => v === values[0]);
                                const kebab = toKebabCase(base);
                                if (allEqual) {
                                    parts.push(`${kebab}: ${formatNumber(values[0])}`);
                                } else {
                                    parts.push(
                                        `${kebab}: ${values.map((v) => formatNumber(v)).join(" ")}`,
                                    );
                                }
                            }
                            for (const f of fieldNames) {
                                handled.add(f);
                                remaining.delete(f);
                            }
                        }
                    }
                }
            }
        }
    }

    for (const field of remaining) {
        if (handled.has(field)) continue;
        const r = field + "R";
        const g = field + "G";
        const b = field + "B";
        if (r in component && g in component && b in component) {
            handled.add(r);
            handled.add(g);
            handled.add(b);
            remaining.delete(r);
            remaining.delete(g);
            remaining.delete(b);
        }
    }

    for (const field of remaining) {
        if (handled.has(field)) continue;
        const value = fields[field];
        const def = defaults[field];

        if (stripDefaults && typeof value === "number" && def !== undefined && value === def)
            continue;

        const kebab = toKebabCase(field);

        if (typeof value === "string") {
            parts.push(`${kebab}: ${value}`);
            continue;
        }

        const formatFn = format?.[field];
        if (formatFn) {
            const formatted = formatFn(value as number);
            if (formatted !== undefined) {
                parts.push(`${kebab}: ${formatted}`);
                continue;
            }
        }

        parts.push(`${kebab}: ${formatNumber(value as number)}`);
    }

    return parts.join("; ");
}

export function normalizeAttr(name: string, value: string): string | null {
    if (!value) return null;
    if (!getComponent(name)) return null;
    try {
        const fields = parseFields(name, value);
        return formatFields(name, fields, { stripDefaults: false });
    } catch {
        return null;
    }
}

function formatNumber(n: number): string {
    if (Number.isInteger(n)) return n.toString();
    return +n.toPrecision(7) + "";
}

function isCSSAttrSyntax(value: string): boolean {
    return value.includes(":") && (value.includes(";") || /^[\w-]+\s*:/.test(value));
}

export interface Diagnostic {
    readonly node: Node;
    readonly attr: string;
    readonly kind: string;
    readonly message: string;
}

export function diagnose(nodes: Node[]): Diagnostic[] {
    const results: Diagnostic[] = [];
    const registered = getComponents().map((r) => r.name);
    function walk(list: Node[]) {
        for (const node of list) {
            const attrNames = new Set(node.attrs.map((a) => a.name));
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
                if (reg.traits?.requires) {
                    for (const req of reg.traits.requires) {
                        const reqName = getComponentName(req);
                        if (reqName && !attrNames.has(reqName)) {
                            results.push({
                                node,
                                attr: attr.name,
                                kind: "missing-requires",
                                message: `"${attr.name}" requires "${reqName}"`,
                            });
                        }
                    }
                }
            }
            walk(node.children);
        }
    }
    walk(nodes);
    return results;
}
