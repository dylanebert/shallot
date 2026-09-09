// The command half of the project host: a project directory in, an exit code and (on success) one
// resolved plan out, plus the loaders that turn that plan's names into real Plugin objects. This is the
// single module the package's own tooling (`bin/tui.ts`, `bin/toolchain.ts`) reaches the project through
// — it exists so a command never repeats resolution, never reads this package's `exports` map and never
// deep-imports engine source to find a plugin.
//
// Two properties this module owes, both gated in `command.test.ts`:
//   1. **Pure to import.** No Vite, no browser API, no GPU global enters the module graph when this file
//      is imported in a bare `bun` process. Every engine reach below is a lazy `import()` inside a
//      function, because importing the barrel evaluates GPU-touching module code.
//   2. **Nothing loads before validation passes.** `planProject` resolves and validates; only then may a
//      caller run `loadLocalPlugins`/`loadEnginePlugins`. A missing or invalid dependency exits with a
//      code and no module of the project has been evaluated, so there is nothing to clean up.
//
// Presentation stays with the tool: this module says which plugins a headless run enables
// (`headlessEngineNames`, Glaze dropped — it composites a swapchain no headless run owns) and never
// touches a terminal, canvas, encoder or frame loop.

import type { Plugin } from "@dylanebert/shallot";
import { SUBPATH_PLUGIN_MODULES } from "./engine";
import {
    headlessEngineNames,
    isProject,
    localModuleErrors,
    missingProjectMessage,
    type ProjectIo,
    type ProjectPlan,
    readProject,
} from "./host";

export {
    discoverScenes,
    emptyPlan,
    headlessEngineNames,
    isProject,
    localModuleErrors,
    missingProjectMessage,
    type PlannedLocal,
    type ProjectPlan,
    readProject,
} from "./host";

// exit codes: 0 and the setup code every sibling command already uses for "bad input, never reached the
// real work" (`bin/tui.ts`'s EXIT_SETUP, `bin/verify.ts`'s own numbering).
export const EXIT_OK = 0;
export const EXIT_SETUP = 2;

/** the outcome of planning one project directory: an exit code the caller returns, the plan when there
 *  is one, and the diagnostics to print. `plan` is null exactly when `code` is non-zero. */
export interface ProjectCommand {
    readonly code: number;
    readonly plan: ProjectPlan | null;
    readonly errors: readonly string[];
}

/**
 * plan a project directory for a command: discovery (is this a project at all), resolution (its manifest,
 * scenes and plugin set) and dependency validation (every enabled local plugin resolves from the project
 * root), in that order, with no module of the project evaluated. Returns an exit code rather than
 * exiting, so the command entry composes.
 */
export function planProject(dir: string, io?: ProjectIo): ProjectCommand {
    if (!isProject(dir))
        return { code: EXIT_SETUP, plan: null, errors: missingProjectMessage(dir) };
    const plan = readProject(dir, io);
    const errors = localModuleErrors(plan);
    if (errors.length > 0) return { code: EXIT_SETUP, plan: null, errors };
    return { code: EXIT_OK, plan, errors: [] };
}

/** the engine plugins that ship on their own subpath rather than the main barrel, as lazy literal
 *  imports of the modules the export map publishes for them (`./avbd`) — resolved here rather than by
 *  reading this package's `exports` map at runtime. `catalog.test.ts` gates the keys against
 *  `SUBPATH_PLUGIN_MODULES`, so a new backend plugin cannot land on only one of the two. */
const SUBPATH_PLUGIN_IMPORTERS: Record<string, () => Promise<Record<string, unknown>>> = {
    Avbd: () => import("@dylanebert/shallot/avbd"),
};

/** the subpath-plugin names this module can load — the catalog gate's other side. */
export const SUBPATH_PLUGIN_LOADERS: readonly string[] = Object.keys(SUBPATH_PLUGIN_IMPORTERS);

/**
 * resolve engine plugin names to real Plugin objects: the main barrel first (every default and most
 * extras), then a backend plugin's own published subpath module. Throws naming the plugin rather than
 * handing back a silent `undefined`. Lazy by construction — the barrel evaluates GPU-touching module
 * code, so it is imported here, at call time, never at module scope.
 */
export async function loadEnginePlugins(names: readonly string[]): Promise<Plugin[]> {
    const barrel = (await import("@dylanebert/shallot")) as unknown as Record<string, unknown>;
    const plugins: Plugin[] = [];
    for (const name of names) {
        const direct = barrel[`${name}Plugin`];
        if (direct) {
            plugins.push(direct as Plugin);
            continue;
        }
        const importer = SUBPATH_PLUGIN_IMPORTERS[name];
        const subpath = importer ? ((await importer()) as Record<string, unknown>) : null;
        const plugin = subpath?.[`${name}Plugin`];
        if (!plugin) {
            throw new Error(
                `shallot.json enables unknown engine plugin "${name}" (no export named "${name}Plugin"${
                    SUBPATH_PLUGIN_MODULES[name] ? ` on ${SUBPATH_PLUGIN_MODULES[name]}` : ""
                })`,
            );
        }
        plugins.push(plugin as Plugin);
    }
    return plugins;
}

/**
 * import a plan's enabled local plugins, in manifest order. Each module's **default export** is the
 * Plugin — the same runtime contract the generated browser module enforces — and a module that resolved
 * but default-exports something else fails loud, naming the manifest key. A disabled plugin is not in
 * the plan, so it is never imported and its module code never runs.
 */
export async function loadLocalPlugins(plan: ProjectPlan): Promise<Plugin[]> {
    const plugins: Plugin[] = [];
    for (const local of plan.locals) {
        const mod = (await import(local.path)) as { default?: Plugin };
        const plugin = mod.default;
        if (!plugin || typeof plugin.name !== "string") {
            throw new Error(
                `shallot.json plugin "${local.name}": its module must default-export a Plugin`,
            );
        }
        plugins.push(plugin);
    }
    return plugins;
}

/** the plugins a headless command runs: the plan's engine set minus Glaze, then its locals. One call so
 *  a tool never re-derives the headless set or the load order. */
export async function loadHeadlessPlugins(plan: ProjectPlan): Promise<Plugin[]> {
    const engine = await loadEnginePlugins(headlessEngineNames(plan));
    return [...engine, ...(await loadLocalPlugins(plan))];
}
