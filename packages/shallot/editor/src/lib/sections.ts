import type { Component, Diagnostic, State, Unit } from "@dylanebert/shallot";
import {
    entries,
    getComponent,
    getTraits,
    kebab,
    readFields,
    schema,
    type Traits,
} from "@dylanebert/shallot/ecs/core";
import type { Node } from "@dylanebert/shallot/editor";
import { parseFields } from "@dylanebert/shallot/scene/core";

/** one editable axis of a `vec` {@link FieldEntry}. `key` is a dotted lane (`pos.x`) for a raw field,
 * or `alias:<base>:<axis>` when edits route through an alias. `mixed` marks a multi-select where this
 * axis differs across the selection ({@link multiSections}). */
type VecAxis = { label: string; key: string; value: number; mixed?: boolean };

/** `mixed` (multi-select only) marks a field whose value differs across the selected entities — the
 * inspector shows a placeholder and an edit writes the new value to every selected entity. */
export type FieldEntry =
    | { type: "float"; key: string; label: string; value: number; mixed?: boolean }
    | { type: "vec"; base: string; label: string; aliased: boolean; axes: VecAxis[] }
    | { type: "color"; key: string; label: string; value: number; mixed?: boolean }
    | {
          type: "enum";
          key: string;
          label: string;
          value: number;
          options: Record<string, number>;
          mixed?: boolean;
      }
    | { type: "unit"; key: string; label: string; value: number; units: Unit[]; mixed?: boolean }
    | { type: "other"; label: string; value: string };

/** a field that claims the full inspector column rather than packing two per row — its control is too
 * wide for the 2-split (a vector's axes, a unit field's dropdown). */
export function wide(field: FieldEntry): boolean {
    return field.type === "vec" || field.type === "unit";
}

/**
 * a field-value record. Single fields key by name (`fov`); Pair/Quad lanes key dotted (`pos.x`) the
 * way parseFields + readFields emit them. Array-form trait defaults (`pos: [0,0,0,0]`) ride the same
 * record until a lane read resolves them, so the value type carries the readonly-number[] shape too.
 */
export type FieldMap = Record<string, number | string | readonly number[]>;

export type ComponentSection = {
    name: string;
    fields: FieldEntry[];
    parsed: FieldMap | null;
    registered: boolean;
    diagnosticMessage?: string;
};

export function lookup(name: string): { component: Component; traits?: Traits } | null {
    const component = getComponent(name);
    if (!component) return null;
    return { component, traits: getTraits(name) };
}

function round(v: number): number {
    return Math.round(v * 1000) / 1000;
}

/** flatten a {@link FieldMap} to dotted numeric lanes — array-form vectors (`pos: [..]`) become
 * `pos.x`… so an alias's `read`/`write` see the same shape whatever the source. */
export function dotted(parsed: FieldMap): Record<string, number> {
    const axes = ["x", "y", "z", "w"];
    const out: Record<string, number> = {};
    // arrays first, numbers second — an explicit dotted lane (`rot.x`) overrides the array form (`rot`)
    for (const [k, v] of Object.entries(parsed)) {
        if (Array.isArray(v))
            for (let i = 0; i < v.length; i++) if (axes[i]) out[`${k}.${axes[i]}`] = v[i];
    }
    for (const [k, v] of Object.entries(parsed)) {
        if (typeof v === "number") out[k] = v;
    }
    return out;
}

function isCustomField(
    key: string,
    traits: { format?: Record<string, unknown>; parse?: Record<string, unknown> } | undefined,
): boolean {
    return !!(traits?.format?.[key] || traits?.parse?.[key]);
}

/**
 * resolve one lane of a Pair/Quad. parseFields + readFields emit dotted keys (`pos.x`); a trait
 * default may instead carry the whole vector as an array at the base key (`pos: [0,0,0,0]`). Read
 * either, so a field shows its real values whether it came from the ECS, a scene attr, or just defaults.
 */
function lane(parsed: FieldMap, base: string, axis: "x" | "y" | "z" | "w", i: number): number {
    const dotted = parsed[`${base}.${axis}`];
    if (typeof dotted === "number") return dotted;
    const arr = parsed[base];
    return Array.isArray(arr) ? (arr[i] ?? 0) : 0;
}

function buildFields(name: string, parsed: FieldMap): FieldEntry[] {
    const s = schema(name);
    if (!s) return [];
    const traits = getTraits(name);
    const fields: FieldEntry[] = [];

    const Axes = ["x", "y", "z", "w"] as const;

    for (const info of s.fields) {
        switch (info.kind) {
            case "vec4":
            case "vec2": {
                // a stored quaternion (or any field with an alias) edits through its alias's axes;
                // a raw vec4 shows xyz (lane w is padding in every component vector), a vec2 shows xy
                const alias = traits?.aliases?.[info.name];
                if (alias) {
                    const vals = alias.read(dotted(parsed));
                    fields.push({
                        type: "vec",
                        base: info.name,
                        label: kebab(info.name),
                        aliased: true,
                        axes: alias.axes.map((label, i) => ({
                            label,
                            key: `alias:${info.name}:${i}`,
                            value: round(vals[i] ?? 0),
                        })),
                    });
                    break;
                }
                if (isCustomField(info.fields![0], traits)) {
                    fields.push({
                        type: "other",
                        label: kebab(info.name),
                        value: info.fields!.map((k) => parsed[k]).join(" "),
                    });
                    break;
                }
                const n = info.kind === "vec2" ? 2 : 3;
                fields.push({
                    type: "vec",
                    base: info.name,
                    label: kebab(info.name),
                    aliased: false,
                    axes: Axes.slice(0, n).map((axis, i) => ({
                        label: axis,
                        key: `${info.name}.${axis}`,
                        value: round(lane(parsed, info.name, axis, i)),
                    })),
                });
                break;
            }
            case "enum": {
                const val = parsed[info.name];
                if (typeof val === "number" && info.options) {
                    fields.push({
                        type: "enum",
                        key: info.name,
                        label: kebab(info.name),
                        value: val,
                        options: info.options,
                    });
                }
                break;
            }
            case "color": {
                const val = parsed[info.name];
                if (typeof val === "number") {
                    fields.push({
                        type: "color",
                        key: info.name,
                        label: kebab(info.name),
                        value: val,
                    });
                }
                break;
            }
            case "unit": {
                const val = parsed[info.name];
                if (typeof val === "number" && info.units) {
                    // raw stored value — the inspector rounds in the shown unit after converting
                    fields.push({
                        type: "unit",
                        key: info.name,
                        label: kebab(info.name),
                        value: val,
                        units: info.units,
                    });
                }
                break;
            }
            case "entity": // an eid edits as a number; an `@name` attr falls to the text branch
            case "float": {
                const val = parsed[info.name];
                if (typeof val === "number") {
                    fields.push({
                        type: "float",
                        key: info.name,
                        label: kebab(info.name),
                        value: round(val),
                    });
                } else {
                    fields.push({ type: "other", label: kebab(info.name), value: String(val) });
                }
                break;
            }
        }
    }

    return fields;
}

function fromAttr(
    node: Node,
    attr: { name: string; value: string },
    diagnostics: readonly Diagnostic[],
): ComponentSection {
    const reg = lookup(attr.name);
    if (!reg) {
        const diag = diagnostics.find((d) => d.node === node && d.attr === attr.name);
        return {
            name: attr.name,
            fields: [],
            parsed: null,
            registered: false,
            diagnosticMessage: diag?.message,
        };
    }

    let parsed: FieldMap;
    try {
        const defaults = reg.traits?.defaults?.() ?? {};
        parsed = attr.value ? { ...defaults, ...parseFields(attr.name, attr.value) } : defaults;
    } catch {
        return { name: attr.name, fields: [], parsed: null, registered: true };
    }

    return { name: attr.name, fields: buildFields(attr.name, parsed), parsed, registered: true };
}

function fromEcs(name: string, eid: number): ComponentSection {
    const reg = lookup(name);
    if (!reg) return { name, fields: [], parsed: null, registered: false };

    const defaults = reg.traits?.defaults?.() ?? {};
    const parsed = { ...defaults, ...readFields(reg.component, eid) };

    return { name, fields: buildFields(name, parsed), parsed, registered: true };
}

/**
 * the inspector's reflection→UI derivation: a selected node's components become editable sections.
 * Reads from the live ECS when the node is mapped to an entity (`eid` + `ecs`), else from the scene
 * attrs. ECS-only components (present on the entity but not yet written to an attr) append after the
 * authored ones. Pure: same inputs, same sections — `bun test`-covered against synthetic components.
 */
export function sections(
    node: Node,
    eid: number | undefined,
    ecs: State | null,
    diagnostics: readonly Diagnostic[],
): ComponentSection[] {
    const useEcs = eid !== undefined && ecs !== null;
    const result: ComponentSection[] = [];
    const seen = new Set<string>();
    for (const attr of node.attrs) {
        seen.add(attr.name);
        result.push(useEcs ? fromEcs(attr.name, eid!) : fromAttr(node, attr, diagnostics));
    }
    if (useEcs) {
        for (const { name, component, traits } of entries()) {
            if (seen.has(name)) continue;
            // a derived decoration (glTF route sync's Textured/Skin) is a system's runtime state, not
            // an inspectable component of the node
            if (traits?.derived) continue;
            if (!ecs!.has(eid!, component as never)) continue;
            result.push(fromEcs(name, eid!));
        }
    }
    return result;
}

function allEqual(values: number[]): boolean {
    return values.every((v) => v === values[0]);
}

// merge one component's field across the selected nodes (same schema → identical field structure, only
// values differ): keep the active (first) node's value and flag `mixed` where the selection disagrees.
function mergeField(peers: FieldEntry[]): FieldEntry {
    const f = peers[0];
    switch (f.type) {
        case "float":
        case "color":
        case "enum":
        case "unit":
            return { ...f, mixed: !allEqual(peers.map((p) => (p as { value: number }).value)) };
        case "vec":
            return {
                ...f,
                axes: f.axes.map((ax, i) => ({
                    ...ax,
                    mixed: !allEqual(
                        peers.map((p) => (p as Extract<FieldEntry, { type: "vec" }>).axes[i].value),
                    ),
                })),
            };
        default:
            return f;
    }
}

/**
 * the inspector view for a multi-node selection: the components present (and registered) on EVERY
 * selected node, each field showing the active (last) node's value and flagged `mixed` where the
 * selection disagrees. The active node is the last entry (insertion order = last-picked), matching the
 * pivot's Active mode and the gizmo's local frame. A single-node selection routes through
 * {@link sections} unchanged (no `mixed`). `nodes`/`eids` are index-aligned. Pure — `bun test`-covered.
 */
export function multiSections(
    nodes: Node[],
    eids: (number | undefined)[],
    ecs: State | null,
    diagnostics: readonly Diagnostic[],
): ComponentSection[] {
    if (nodes.length === 0) return [];
    const perNode = nodes.map((n, i) => sections(n, eids[i], ecs, diagnostics));
    if (perNode.length === 1) return perNode[0];

    const active = perNode[perNode.length - 1];
    const rest = perNode.slice(0, -1);
    const out: ComponentSection[] = [];
    for (const sec of active) {
        if (!sec.registered) continue; // an unregistered/plugin-missing component isn't multi-editable
        const peers = rest.map((ns) => ns.find((s) => s.name === sec.name));
        if (peers.some((p) => !p?.registered)) continue; // absent on a node → outside the intersection
        const sections_ = [sec, ...(peers as ComponentSection[])];
        out.push({
            ...sec,
            fields: sec.fields.map((_, i) => mergeField(sections_.map((s) => s.fields[i]))),
        });
    }
    return out;
}
