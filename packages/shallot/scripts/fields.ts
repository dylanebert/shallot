import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Glob } from "bun";
import type { Component } from "../src/engine/ecs/component";
import { entries, getTraits, kebab, register, schema, type Traits } from "../src/engine/ecs/core";
import { findDefinitionLine, parseComponentFields, parseJSDoc } from "./jsdoc";

/**
 * FIELDS markers — the reflection-generated twin of the API/CORE tables.
 *
 * `<!-- FIELDS:component -->` expands to a component's field table (field, type, default) walked from
 * the same `schema()` + `getTraits()` reflection the editor inspector derives its rows from
 * (`editor/src/lib/sections.ts`). The table the docs show is the table the inspector shows, so it can't
 * drift: rename or retype a field and the marker re-renders. Workflow prose stays hand-written; only the
 * field schema is generated. The defaults come straight from `traits.defaults()`, so the table is the
 * inspector's at its default pose.
 */

export interface FieldRow {
    field: string;
    type: string;
    default: string;
}

type Defaults = Record<string, number | readonly number[]>;

/** round the way the inspector does (`sections.ts`) so a generated default reads as the inspector shows it. */
function round(v: number): number {
    return Math.round(v * 1000) / 1000;
}

/** flatten array-form vector defaults (`pos: [..]`) to dotted lanes (`pos.x`), mirroring `sections.ts` `dotted`. */
function flatten(defaults: Defaults): Record<string, number> {
    const axes = ["x", "y", "z", "w"];
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(defaults)) {
        if (Array.isArray(v))
            for (let i = 0; i < v.length; i++) if (axes[i]) out[`${k}.${axes[i]}`] = v[i];
    }
    for (const [k, v] of Object.entries(defaults)) if (typeof v === "number") out[k] = v;
    return out;
}

/** one lane of a vec default — dotted key first, array form second (mirrors `sections.ts` `lane`). */
function lane(
    defaults: Defaults,
    dot: Record<string, number>,
    base: string,
    axis: string,
    i: number,
): number {
    const d = dot[`${base}.${axis}`];
    if (typeof d === "number") return d;
    const arr = defaults[base];
    return Array.isArray(arr) ? (arr[i] ?? 0) : 0;
}

/** the inspector's field rows for `name` at its default pose, or `null` if the component isn't registered. */
export function fieldRows(name: string): FieldRow[] | null {
    const s = schema(name);
    if (!s) return null;
    const traits = getTraits(name);
    const defaults = traits?.defaults?.() ?? {};
    const dot = flatten(defaults);
    const Axes = ["x", "y", "z", "w"];
    const rows: FieldRow[] = [];

    for (const info of s.fields) {
        const field = kebab(info.name);
        switch (info.kind) {
            case "vec2":
            case "vec4": {
                const alias = traits?.aliases?.[info.name];
                if (alias) {
                    const vals = alias.read(dot);
                    rows.push({
                        field,
                        type: `vec${alias.axes.length}`,
                        default: alias.axes.map((_, i) => round(vals[i] ?? 0)).join(" "),
                    });
                    break;
                }
                const n = info.kind === "vec2" ? 2 : 3;
                rows.push({
                    field,
                    type: `vec${n}`,
                    default: Axes.slice(0, n)
                        .map((axis, i) => round(lane(defaults, dot, info.name, axis, i)))
                        .join(" "),
                });
                break;
            }
            case "enum": {
                const options = Object.keys(info.options ?? {}).map(kebab);
                const def = Object.entries(info.options ?? {}).find(
                    ([, v]) => v === info.default,
                )?.[0];
                rows.push({
                    field,
                    type: options.join(" | "),
                    default: def ? kebab(def) : "",
                });
                break;
            }
            case "color": {
                const format = traits?.format?.[info.name];
                const def =
                    typeof info.default === "number"
                        ? (format?.(info.default) ??
                          `0x${(info.default >>> 0).toString(16).padStart(6, "0")}`)
                        : "";
                rows.push({ field, type: "color", default: def });
                break;
            }
            case "unit": {
                const us = info.units ?? [];
                const u = us[0];
                const def =
                    u && typeof info.default === "number"
                        ? `${round(u.to(info.default))} ${u.label}`
                        : "";
                rows.push({ field, type: us.map((x) => x.label).join(" | "), default: def });
                break;
            }
            default:
                rows.push({
                    field,
                    type: info.kind === "entity" ? "entity" : "float",
                    default: typeof info.default === "number" ? String(round(info.default)) : "",
                });
        }
    }

    return rows;
}

/** a markdown field table for `name`, or an `error` (no period — joined into a `warning:` line) when unresolved. */
export function fieldTable(name: string): { table: string; error?: string } {
    const rows = fieldRows(name);
    if (!rows) return { table: "", error: `component "${name}" is not registered` };
    if (rows.length === 0) return { table: "", error: `component "${name}" has no fields` };
    // escape `|` here, not in fieldRows — the rows are data (read clean by the editor's UI reference);
    // the pipe escape is a markdown-table-cell concern that belongs at the markdown emit site.
    const cell = (s: string) => s.replace(/\|/g, "\\|");
    const body = rows
        .map((r) => `| \`${r.field}\` | ${cell(r.type)} | ${cell(r.default)} |`)
        .join("\n");
    return { table: `| Field | Type | Default |\n| --- | --- | --- |\n${body}` };
}

let registered = false;

/**
 * populate the component registry the FIELDS markers read. The public plugin barrels reference WebGPU
 * constants at module scope, so install the constants-only `bun-webgpu` globals (no adapter) before the
 * dynamic import, then register every collected plugin's components + traits — `schema()` reads pure data,
 * no device. Idempotent: registering a name twice overwrites with the same definition.
 */
export async function registerComponents(): Promise<void> {
    if (registered) return;
    registered = true;
    const { globals } = await import("bun-webgpu");
    Object.assign(globalThis, globals());
    const barrels = await Promise.all([import("../src/index"), import("../src/extras/index")]);
    type PluginShape = { components: Record<string, Component>; traits?: Record<string, Traits> };
    const plugins = new Set<PluginShape>();
    for (const mod of barrels) {
        for (const v of Object.values(mod)) {
            if (v && typeof v === "object" && "components" in v && (v as PluginShape).components) {
                plugins.add(v as PluginShape);
            }
        }
    }
    for (const plugin of plugins) {
        for (const [name, component] of Object.entries(plugin.components)) {
            register(name, component, plugin.traits?.[name]);
        }
    }
}

/**
 * Annotation-sourced UI reference data — the build-side generator behind `editor/src/lib/fielddocs.json`.
 *
 * Per registered component: its one-line summary (the component JSDoc) and, per field, the complete hover
 * data — `type` + `default` from the same `fieldRows` reflection the FIELDS table renders, plus the
 * `description` parsed from the field's doc comment. Keyed kebab — component by its registered name, fields
 * by the inspector's label — so the editor inspector, the FIELDS table, and the docs reference all read one
 * truth. `type`/`default` are derivable from `schema()`; `description` is the source-only half that a
 * browser can't reach. A field with no doc comment has `description: null`; both are never invented.
 */
export interface FieldDoc {
    description: string | null;
    type: string;
    default: string;
}

export interface ComponentDoc {
    summary: string | null;
    fields: Record<string, FieldDoc>;
}

/** map a component's kebab name → its source file + export identifier, by scanning `src/` for the
 * `export const Name = { … }` declaration (systems/plugins carry a `: System`/`: Plugin` annotation, so
 * the no-annotation pattern matches components only). Non-component consts are filtered by the registry. */
function componentFiles(): Map<string, { exportName: string; file: string }> {
    const root = resolve(import.meta.dir, "../src");
    const out = new Map<string, { exportName: string; file: string }>();
    for (const rel of new Glob("**/*.ts").scanSync(root)) {
        if (rel.endsWith(".test.ts")) continue;
        const file = resolve(root, rel);
        for (const line of readFileSync(file, "utf-8").split("\n")) {
            const m = line.match(/^export const (\w+) = \{/);
            if (m) out.set(kebab(m[1]), { exportName: m[1], file });
        }
    }
    return out;
}

/** the UI-reference doc data for every registered component, keyed by kebab name (sorted for stable
 * output). `fieldRows` is the authoritative field set + type/default (so the artifact's fields are the
 * inspector's fields); the source supplies the summary + per-field descriptions, `null` where unwritten. */
export async function componentDocs(): Promise<Record<string, ComponentDoc>> {
    await registerComponents();
    const files = componentFiles();
    const out: Record<string, ComponentDoc> = {};
    for (const { name } of [...entries()].sort((a, b) => a.name.localeCompare(b.name))) {
        const rows = fieldRows(name) ?? [];
        const src = files.get(name);
        const summary = src
            ? parseJSDoc(src.file, findDefinitionLine(src.file, src.exportName)).description
            : null;
        const descs: Record<string, string> = {};
        if (src) {
            const raw = parseComponentFields(src.file, src.exportName);
            for (const field of Object.keys(raw)) descs[kebab(field)] = raw[field];
        }
        const fields: Record<string, FieldDoc> = {};
        for (const r of rows) {
            fields[r.field] = {
                description: descs[r.field] ?? null,
                type: r.type,
                default: r.default,
            };
        }
        out[name] = { summary, fields };
    }
    return out;
}

/** the committed `fielddocs.json` bytes — JSON so biome (which formats only `.ts`) leaves it stable, and
 * kebab field keys stay legal. Shared by the writer (`scripts/fielddocs.ts`) and the staleness gate. */
export function serializeDocs(docs: Record<string, ComponentDoc>): string {
    return `${JSON.stringify(docs, null, 2)}\n`;
}
