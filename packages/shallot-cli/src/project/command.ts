// The command half of the project host: a project directory in, an exit code and (on success) one
// resolved plan out, plus the loader that turns that plan's local plugins into real Plugin objects. This
// is the single module the package's own tooling (`bin/toolchain.ts`) reaches the project through, so a
// command never repeats resolution, never reads this package's `exports` map and never deep-imports
// engine source to find a plugin.
//
// Two properties this module owes, both gated in `command.test.ts`:
//   1. **Pure to import.** No Vite, no browser API, no GPU global enters the module graph when this file
//      is imported in a bare `bun` process.
//   2. **Nothing loads before validation passes.** `planProject` resolves and validates; only then may a
//      caller run `loadLocalPlugins`. A missing or invalid dependency exits with a code and no module of
//      the project has been evaluated, so there is nothing to clean up.

import type { Plugin } from "@dylanebert/shallot";
import {
    isProject,
    localModuleErrors,
    missingProjectMessage,
    type ProjectIo,
    type ProjectPlan,
    readProject,
    resolveLocalModules,
} from "./host";

export {
    discoverScenes,
    emptyPlan,
    isProject,
    localModuleErrors,
    missingProjectMessage,
    type PlannedLocal,
    type ProjectPlan,
    readProject,
} from "./host";

// exit codes: 0 and the setup code every sibling command already uses for "bad input, never reached the
// real work" (`bin/verify.ts`'s own numbering).
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

/**
 * import a plan's enabled local plugins, in manifest order. Each module's **default export** is the
 * Plugin — the same runtime contract the generated browser module enforces — and a module that resolved
 * but default-exports something else fails loud, naming the manifest key. A disabled plugin is not in
 * the plan, so it is never imported and its module code never runs.
 */
export async function loadLocalPlugins(plan: ProjectPlan): Promise<Plugin[]> {
    const plugins: Plugin[] = [];
    for (const local of resolveLocalModules(plan)) {
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
