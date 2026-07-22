// The `shallot.json` project manifest: the single source of truth for plugin enablement, read by the CLI
// toolchain and the production boot. A committed data file the toolchain reads (like a scene file), never
// code it rewrites. These are the pure parse helpers, `bun test`-covered.

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
 * by the toolchain and a standalone boot. `plugins` is name → {@link PluginValue}; `capacity` is the fixed
 * entity capacity (a session invariant, like `new State({ capacity })`), omitted to take the engine default.
 */
export interface Manifest {
    /** JSON Schema pointer for IDE autocomplete/validation (`@dylanebert/shallot/shallot.schema.json`);
     *  preserved by `normalize`, ignored by `plan()`. */
    $schema?: string;
    scene?: string;
    plugins?: Record<string, PluginValue>;
    capacity?: number;
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
    if (typeof obj.$schema === "string") manifest.$schema = obj.$schema;
    if (typeof obj.scene === "string") manifest.scene = obj.scene;
    if (typeof obj.plugins === "object" && obj.plugins !== null && !Array.isArray(obj.plugins)) {
        manifest.plugins = obj.plugins as Record<string, PluginValue>;
    }
    if (typeof obj.capacity === "number") manifest.capacity = obj.capacity;
    return manifest;
}

/** read a local plugin's specifier + enabled from its manifest value, or null when it isn't one (an engine bool). */
export function localOf(value: PluginValue): { spec: string; enabled: boolean } | null {
    if (typeof value === "string") return { spec: value, enabled: true };
    if (Array.isArray(value) && typeof value[0] === "string") {
        return { spec: value[0], enabled: value[1] !== false };
    }
    return null;
}
