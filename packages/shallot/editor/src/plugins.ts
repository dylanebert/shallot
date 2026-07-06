import * as shallot from "@dylanebert/shallot";
import {
    DEFAULT_PLUGINS,
    InputPlugin,
    MirrorPlugin,
    OrbitPlugin,
    type Plugin,
    RenderPlugin,
} from "@dylanebert/shallot";
import { PickPlugin } from "./lib/pick";
import { GizmosPlugin } from "./lib/viewport";
import { type Manifest, type PluginSource, type ResolvedEntry, resolve } from "./project/manifest";

// the engine's zero-config set — what a default / scene-only project enables. deriving from
// DEFAULT_PLUGINS (vs a hand-kept copy) avoids the drift that silently dropped SlabPlugin:
// TransformsPlugin depends on it, so build() would skip Transforms and render nothing.
export const STANDARD_PLUGINS: Plugin[] = [...DEFAULT_PLUGINS];

const DEFAULT_SET = new Set(STANDARD_PLUGINS.map((p) => p.name));

function isPlugin(value: unknown): value is Plugin {
    return (
        typeof value === "object" && value !== null && typeof (value as Plugin).name === "string"
    );
}

// every non-default `*Plugin` the barrel exports — the engine surface a manifest can name, derived from
// the barrel, not a hand list (the `${name}Plugin` convention `catalog.test` gates). One source of truth:
// a manifest enabling physics / character / fog by name resolves to its object here with no editor edit,
// and a plugin added to the engine appears in the menu automatically. The editor bundle holds them all
// (it can't import on demand the way the lean production build does), so toggling stays client-side.
export const SHALLOT_PLUGINS: Plugin[] = Object.entries(shallot)
    .filter(
        ([key, value]) => key.endsWith("Plugin") && isPlugin(value) && !DEFAULT_SET.has(value.name),
    )
    .map(([, value]) => value as Plugin);

// the engine plugins the editor resolves a manifest against + offers in its menu: the defaults (always on)
// plus the opt-in extras. A manifest names one of these by its `.name`; the editor maps the name back to
// the object here — it can't, like the lean production build, import it on demand. Source-tagged for the menu.
const CATALOG = new Map<string, { plugin: Plugin; source: PluginSource }>([
    ...STANDARD_PLUGINS.map(
        (plugin) => [plugin.name, { plugin, source: "default" as const }] as const,
    ),
    ...SHALLOT_PLUGINS.map(
        (plugin) => [plugin.name, { plugin, source: "extra" as const }] as const,
    ),
]);

const DEFAULT_NAMES = STANDARD_PLUGINS.map((p) => p.name);
const EXTRA_NAMES = SHALLOT_PLUGINS.map((p) => p.name);

/** the names of the engine plugins the editor catalogs (defaults + extras), for `resolve`'s name sets. */
export const ENGINE_NAMES = { defaults: DEFAULT_NAMES, extras: EXTRA_NAMES };

/** a resolved manifest entry carrying its plugin object — null when the name resolves to nothing known. */
export interface EditorEntry extends ResolvedEntry {
    readonly plugin: Plugin | null;
}

/**
 * resolve a manifest into the editor's plugin entries: every catalog plugin (defaults + extras) with its
 * enabled state, plus the project's declared locals. The menu renders these; an enabled entry's `plugin`
 * builds the State. Engine plugins map back to the catalog object; locals to the objects the generated
 * `virtual:project` imported (the editor can't import a project file itself).
 */
export function entriesFor(
    manifest: Manifest,
    locals: readonly { name: string; plugin: Plugin }[],
): EditorEntry[] {
    const localMap = new Map(locals.map((l) => [l.name, l.plugin]));
    return resolve(manifest, DEFAULT_NAMES, EXTRA_NAMES).entries.map((e) => ({
        ...e,
        plugin:
            e.source === "project"
                ? (localMap.get(e.name) ?? null)
                : (CATALOG.get(e.name)?.plugin ?? null),
    }));
}

/** the enabled plugin objects to build — engine + local, in catalog-then-local order. */
export function enabledPlugins(entries: readonly EditorEntry[]): Plugin[] {
    return entries
        .filter((e): e is EditorEntry & { plugin: Plugin } => e.enabled && e.plugin !== null)
        .map((e) => e.plugin);
}

/**
 * the project's own enabled (local) plugins — the hot-reload swap pairs these by name before and after a
 * code reload. Engine plugins don't change on a project edit, so they're excluded; a local added or removed
 * shows up as a name-set change the swap rejects, falling back to a rebuild.
 */
export function localPlugins(entries: readonly EditorEntry[]): Plugin[] {
    return entries
        .filter(
            (e): e is EditorEntry & { plugin: Plugin } =>
                e.source === "project" && e.enabled && e.plugin !== null,
        )
        .map((e) => e.plugin);
}

/**
 * a plugin is editor tooling if any of its systems declares the tooling layer (`annotations.layer`) —
 * the "layer" axis decomposed out of the overloaded `mode`. Tooling is injected into the editor host and
 * excluded from a shipped game build; an app plugin (the default) ships.
 */
export function isTooling(plugin: Plugin): boolean {
    return plugin.systems?.some((s) => s.annotations?.layer === "tooling") ?? false;
}

// the editor's own tooling plugins — the grid + selection outline (`GizmosPlugin`) and viewport picking
// (`PickPlugin`) today, transform handles later. {@link compose} spreads these into an *edit* build alongside
// the app's plugins; a game build never imports this module, so they tree-shake out. New tooling joins by
// marking its systems `layer: "tooling"` (see {@link isTooling}).
export const TOOLING_PLUGINS: Plugin[] = [GizmosPlugin, PickPlugin];

// the engine substrates the editor shares with the app by reference — one input source, one render pipeline.
// Both are `always`-mode (they run in edit), so the edit viewport has input + rendering even for an app that
// declares neither; `compose` adds them to an edit build only when the app lacks them. Never forced into
// play, where the build mirrors the app's declaration exactly.
export const EDITOR_SUBSTRATES: Plugin[] = [InputPlugin, RenderPlugin];

/**
 * the plugin set to build for a mode — the editor composing *over* the app's State, not forking it.
 *
 * **play** is the app's resolved plugins verbatim: a faithful preview that runs exactly what ships, nothing
 * the editor needs bolted on (a `defaults: false` app with no renderer draws nothing in play, as it would
 * shipped). **edit** is that set ∪ the editor foundation, each member expressed through the engine's own
 * `mode` / `layer` axes rather than force-prepended:
 * - Input + Render ({@link EDITOR_SUBSTRATES}) — `always`-mode substrates, added only when the app lacks them.
 * - the edit camera's `OrbitPlugin` — reused edit-scoped until edit-nav lifts into its own tooling plugin.
 * - the `layer: "tooling"` set ({@link TOOLING_PLUGINS}: gizmos + pick), stripped from any shipped game.
 *   `MirrorPlugin` rides in as `PickPlugin`'s readback dependency (`build()` skips a plugin whose dependency
 *   is absent, so Pick needs Mirror present) — it's Pick's, not a standalone foundation member.
 *
 * Pure over (mode, app) and `bun test`-covered, so the per-mode split can't regress to a force-prepend that
 * makes play silently run editor machinery. Deduped by name: an app declaring orbit / mirror keeps its one
 * instance, the editor never adds a second.
 */
export function compose(mode: "edit" | "play", app: Plugin[]): Plugin[] {
    if (mode === "play") return app;
    const out = [...app];
    const seen = new Set(app.map((p) => p.name));
    const add = (plugin: Plugin) => {
        if (seen.has(plugin.name)) return;
        seen.add(plugin.name);
        out.push(plugin);
    };
    for (const plugin of EDITOR_SUBSTRATES) add(plugin);
    add(OrbitPlugin);
    add(MirrorPlugin);
    for (const plugin of TOOLING_PLUGINS) add(plugin);
    return out;
}
