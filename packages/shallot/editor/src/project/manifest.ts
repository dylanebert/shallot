// The `shallot.json` project manifest: the single source of truth for plugin enablement, read by both
// the editor and the production boot. A committed data file the editor owns (like a scene file), never
// code it rewrites. These are the pure parse / resolve / edit helpers — `bun test`-covered so the
// enablement contract can't drift between the editor's read and the generator's read of the same file.

/**
 * a plugin's manifest value. Every form maps to a proven config idiom:
 * - `true` / `false` — an **engine** plugin's enablement, the `{ name: source }` dependency-map bool
 *   (npm / Unity `manifest.json`); resolved `${Name}Plugin` from the barrel by name.
 * - a module specifier `string` — a **local/external** plugin, enabled; the module **default-exports** a
 *   Plugin (Babel/PostCSS shorthand string; a relative `./src/x`, or a package subpath `orrstead/grid`).
 * - `[spec, false]` — the same local/external plugin kept but **disabled** (PostCSS's `["plugin", false]`
 *   disable, the Babel `[name, options]` tuple with `false` in the options slot). The spec survives the toggle.
 */
export type PluginValue = boolean | string | [string, boolean];

/**
 * the on-disk manifest, tolerant-parsed — the serialized form of the runtime `Config`, read identically
 * by the editor and a standalone boot. `plugins` is name → {@link PluginValue}; `capacity` is the fixed
 * entity capacity (a session invariant, like `new State({ capacity })`), omitted to take the engine default.
 */
export interface Manifest {
    /** JSON Schema pointer for editor autocomplete/validation (`@dylanebert/shallot/shallot.schema.json`);
     *  preserved across the editor's round-trip writes, ignored by resolution. */
    $schema?: string;
    scene?: string;
    plugins?: Record<string, PluginValue>;
    capacity?: number;
}

/** where a resolved plugin came from — drives the menu's grouping + the generator's import form. */
export type PluginSource = "default" | "extra" | "project";

/**
 * one resolved plugin: its name, where it came from, its module specifier (null for an engine plugin
 * the engine resolves by name), and whether it's enabled. The editor's read model and the boot's input.
 */
export interface ResolvedEntry {
    readonly name: string;
    readonly source: PluginSource;
    readonly spec: string | null;
    readonly enabled: boolean;
}

/** a problem found resolving a manifest — a bool naming an unknown engine plugin, a malformed value. */
export interface ManifestDiagnostic {
    readonly source: string;
    readonly message: string;
}

export interface Resolution {
    readonly entries: ResolvedEntry[];
    readonly diagnostics: ManifestDiagnostic[];
}

/** parse persisted manifest JSON, tolerating absent or corrupt storage (a first run, a hand-edit) */
export function normalize(raw: string | null): Manifest {
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw ?? "{}");
    } catch {
        return {};
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
    const obj = parsed as Record<string, unknown>;
    const manifest: Manifest = {};
    // $schema first so a re-serialize keeps it at the top, where editors expect the pointer
    if (typeof obj.$schema === "string") manifest.$schema = obj.$schema;
    if (typeof obj.scene === "string") manifest.scene = obj.scene;
    if (typeof obj.plugins === "object" && obj.plugins !== null && !Array.isArray(obj.plugins)) {
        manifest.plugins = obj.plugins as Record<string, PluginValue>;
    }
    if (typeof obj.capacity === "number") manifest.capacity = obj.capacity;
    return manifest;
}

/** serialize a manifest to the stable on-disk form (2-space JSON + trailing newline). */
export function serialize(manifest: Manifest): string {
    return JSON.stringify(manifest, null, 2) + "\n";
}

/** read a local plugin's specifier + enabled from its manifest value, or null when it isn't one (an engine bool). */
export function localOf(value: PluginValue): { spec: string; enabled: boolean } | null {
    if (typeof value === "string") return { spec: value, enabled: true };
    if (Array.isArray(value) && typeof value[0] === "string") {
        return { spec: value[0], enabled: value[1] !== false };
    }
    return null;
}

/**
 * resolve a manifest against the engine's known plugin names into the editor read model. Defaults are
 * on unless a `name: false` disables them; extras are off unless a `name: true` enables them; a key
 * naming neither — with a string / `{ use }` value — is a local plugin (source "project", its spec).
 * Pure over (manifest, defaults, extras); `enabled ⊆ available` by construction (a bare bool can only
 * name a known engine plugin, a local must carry a real specifier). A bool naming an unknown plugin,
 * or a default/extra given a specifier, is a diagnostic.
 */
export function resolve(
    manifest: Manifest,
    defaults: readonly string[],
    extras: readonly string[],
): Resolution {
    const plugins = manifest.plugins ?? {};
    const entries: ResolvedEntry[] = [];
    const diagnostics: ManifestDiagnostic[] = [];
    const engine = new Set([...defaults, ...extras]);

    for (const name of defaults) {
        const v = plugins[name];
        if (typeof v === "string" || (typeof v === "object" && v !== null)) {
            diagnostics.push({
                source: name,
                message: `"${name}" is a default engine plugin — use true/false, not a specifier`,
            });
        }
        entries.push({ name, source: "default", spec: null, enabled: v !== false });
    }
    for (const name of extras) {
        const v = plugins[name];
        if (typeof v === "string" || (typeof v === "object" && v !== null)) {
            diagnostics.push({
                source: name,
                message: `"${name}" is an engine plugin — use true/false, not a specifier`,
            });
        }
        entries.push({ name, source: "extra", spec: null, enabled: v === true });
    }
    for (const [name, value] of Object.entries(plugins)) {
        if (engine.has(name)) continue;
        const local = localOf(value);
        if (!local) {
            diagnostics.push({
                source: name,
                message: `"${name}" is not a known engine plugin — give it a module specifier to declare a local plugin`,
            });
            continue;
        }
        entries.push({ name, source: "project", spec: local.spec, enabled: local.enabled });
    }
    return { entries, diagnostics };
}

/**
 * fold one name→on/off edit back into a manifest — the write the editor persists. The encoding keeps
 * diffs minimal: a default at its on-default and an extra at its off-default are absent. A local plugin
 * keeps its specifier across a toggle (passed in), so disabling then re-enabling is lossless.
 */
export function setEnabled(
    manifest: Manifest,
    name: string,
    on: boolean,
    source: PluginSource,
    spec?: string | null,
): Manifest {
    const plugins = { ...(manifest.plugins ?? {}) };
    if (source === "default") {
        if (on) delete plugins[name];
        else plugins[name] = false;
    } else if (source === "extra") {
        if (on) plugins[name] = true;
        else delete plugins[name];
    } else {
        if (!spec) throw new Error(`a local plugin toggle needs its specifier ("${name}")`);
        plugins[name] = on ? spec : [spec, false];
    }
    return { ...manifest, plugins };
}
