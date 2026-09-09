// The project host: plan, discovery and resolution for a project directory, as pure data. One module
// answers "what is this project, and which plugins does it enable" for both consumers — the browser
// generator (`generate.ts` → `virtual:project`, through `vite.ts`) and the terminal command
// (`command.ts` → `bin/tui.ts`) — so the two cannot drift in how a manifest becomes a plugin set.
//
// Nothing here imports Vite, a browser API or a GPU global, and nothing here loads a plugin module: a
// plan is data the caller may inspect, log or refuse before any module evaluation happens
// (`command.ts` owns the loading half and runs it only after `localModuleErrors` is empty). That split
// is what lets a dependency mistake fail with an exit code instead of a half-imported project.
//
// Presentation is deliberately absent: the host names the enabled plugins and, for a headless run, says
// which of them a terminal presentation must drop (`headlessEngineNames` — Glaze composites a swapchain
// that does not exist headless). Terminal canvas/input/readback/encoding stay in `bin/tui.ts`.

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { manifestPath, readManifest } from "./assets";
import { DEFAULT_PLUGIN_NAMES } from "./engine";
import { localOf, type Manifest, normalize } from "./manifest";

/** one enabled local/external plugin: its manifest key, the specifier as authored, and the module the
 *  generator emits / the command imports — a project-relative spec absolutized against the project dir,
 *  a bare package or absolute path passed through (the project root's own resolver owns those). */
export interface PlannedLocal {
    readonly name: string;
    readonly spec: string;
    readonly path: string;
}

/** a project directory reduced to what both consumers need: its manifest, its scenes, and the plugin
 *  set the manifest enables. `dir` is null only for the host's own no-project fallback. */
export interface ProjectPlan {
    readonly dir: string | null;
    readonly manifest: Manifest;
    readonly scenes: readonly string[];
    /** engine plugin names to resolve as `${name}Plugin` (enabled defaults + declared extras) */
    readonly engine: readonly string[];
    readonly locals: readonly PlannedLocal[];
    /** manifest keys explicitly turned off — never resolved, never imported; kept so a caller can say so */
    readonly disabled: readonly string[];
}

/** the reads a plan is allowed to make, injectable so a test can pin the read set (the seam takes a
 *  project root and reads inside it — never this package's own layout or export map). */
export interface ProjectIo {
    /** the file's text, or null when it does not exist (a scene-only project has no manifest) */
    readFile(path: string): string | null;
    discoverScenes(dir: string): string[];
}

// a local specifier resolved for its consumers: project-relative → project-absolute (the generated
// virtual module resolves against the host root, not the project), a bare package or absolute path
// passed through.
function localPath(spec: string, absDir: string): string {
    return spec.startsWith(".") ? join(absDir, spec) : spec;
}

/** classify a manifest into the engine plugins, local plugins and explicitly disabled keys. */
export function plan(
    manifest: Manifest,
    absDir: string | null,
): { engine: string[]; locals: PlannedLocal[]; disabled: string[] } {
    const plugins = manifest.plugins ?? {};
    const defaults = new Set<string>(DEFAULT_PLUGIN_NAMES);
    const engine: string[] = [];
    const locals: PlannedLocal[] = [];
    const disabled: string[] = [];

    // every default is enabled unless explicitly turned off
    for (const name of DEFAULT_PLUGIN_NAMES) {
        if (plugins[name] !== false) engine.push(name);
        else disabled.push(name);
    }
    // then the declared entries: an engine extra (true), or a local (a specifier). defaults already handled.
    for (const [name, value] of Object.entries(plugins)) {
        if (defaults.has(name)) continue;
        if (value === true) engine.push(name);
        else if (value === false) disabled.push(name);
        else {
            const local = localOf(value);
            if (!local) continue;
            if (local.enabled)
                locals.push({ name, spec: local.spec, path: localPath(local.spec, absDir ?? "") });
            else disabled.push(name);
        }
    }
    return { engine, locals, disabled };
}

/** the engine plugins a headless presentation runs: the plan's set minus Glaze, which composites the
 *  rendered scene onto a swapchain no headless run owns. Unconditional, not a manifest toggle — a
 *  terminal run has no swapchain to disable it against. */
export function headlessEngineNames(plan: Pick<ProjectPlan, "engine">): string[] {
    return plan.engine.filter((name) => name !== "Glaze");
}

/** every `.scene` under `dir`, project-relative and sorted. */
export function discoverScenes(dir: string): string[] {
    const scenes: string[] = [];
    // per-directory try/catch, not one around the whole walk: an unreadable subtree (permissions, a
    // broken symlink) used to throw out of the recursive `walk`, which the outer catch swallowed —
    // silently truncating every sibling not yet visited at every ancestor level, not just the bad
    // subtree, with no warning that the scene list was incomplete.
    function walk(current: string) {
        let entries: string[];
        try {
            entries = readdirSync(current);
        } catch (e) {
            console.warn(`  ! scene discovery: skipping unreadable directory "${current}": ${e}`);
            return;
        }
        for (const entry of entries) {
            if (entry === "node_modules" || entry === "dist") continue;
            const full = join(current, entry);
            let isDirectory: boolean;
            try {
                isDirectory = statSync(full).isDirectory();
            } catch (e) {
                console.warn(`  ! scene discovery: skipping unreadable entry "${full}": ${e}`);
                continue;
            }
            if (isDirectory) walk(full);
            else if (entry.endsWith(".scene")) scenes.push(relative(dir, full));
        }
    }
    walk(dir);
    return scenes.sort();
}

/** dir holds a shallot project — a shallot.json manifest or a .scene file. */
export function isProject(dir: string): boolean {
    return existsSync(manifestPath(dir)) || discoverScenes(dir).length > 0;
}

/** the diagnostic for a directory that is no project — one message, printed by every command that
 *  needs one (`bin/toolchain.ts`'s `requireProject`, the terminal command's setup exit). */
export function missingProjectMessage(dir: string): string[] {
    return [
        `\n  ✗ No shallot project found at ${dir}`,
        "    Expected a shallot.json manifest or a .scene file\n",
        "    To create a project:",
        "      bun create shallot my-game",
        "      cd my-game && bun install",
        "      bunx shallot dev\n",
    ];
}

const REAL_IO: ProjectIo = {
    readFile(path) {
        try {
            return readFileSync(path, "utf-8");
        } catch {
            return null;
        }
    },
    discoverScenes,
};

/** read a project directory into a plan: its manifest (absent → a scene-only project's empty one), its
 *  scenes, and the classified plugin set. Reads only inside `dir`. */
export function readProject(dir: string, io: ProjectIo = REAL_IO): ProjectPlan {
    // the real path goes through `readManifest`, which also emits the manifest-boundary warnings; an
    // injected io (a test pinning the read set) parses the same text through the same `normalize`.
    const manifest = io === REAL_IO ? readManifest(dir) : normalize(io.readFile(manifestPath(dir)));
    const scenes = io.discoverScenes(dir);
    return { dir, manifest, scenes, ...plan(manifest, dir) };
}

/** the plan for no project at all — the generator's empty-manifest fallback (`projectPlugin()` with no
 *  dir), kept here so "no project" is one shape rather than an inline literal per consumer. */
export function emptyPlan(): ProjectPlan {
    return { dir: null, manifest: {}, scenes: [], ...plan({}, null) };
}

/** resolve a module specifier from the project root, or null when nothing resolves. */
function resolveFromProject(spec: string, dir: string): string | null {
    try {
        return Bun.resolveSync(spec, dir);
    } catch {
        return null;
    }
}

/**
 * the dependency errors of a plan: every enabled local plugin whose module does not resolve from the
 * project root — a missing install (`bare-plugin` never installed) or a missing file. Pure and
 * side-effect free: the caller refuses with an exit code before anything is imported, which is what
 * keeps a bad manifest from leaving a half-loaded project behind. A disabled plugin is never resolved.
 */
export function localModuleErrors(
    project: ProjectPlan,
    resolver: (spec: string, dir: string) => string | null = resolveFromProject,
): string[] {
    if (!project.dir) return [];
    const errors: string[] = [];
    for (const local of project.locals) {
        if (resolver(local.path, project.dir)) continue;
        errors.push(
            `shallot.json plugin "${local.name}": cannot resolve its module "${local.spec}" from ${project.dir}`,
        );
    }
    return errors;
}
