// The project host: plan, discovery and resolution for a project directory, as pure data. The
// `virtual:project` generator and native feature check share one plan for the manifest's plugins.
//
// Nothing here imports Vite, a browser API or a GPU global, and nothing here loads a plugin module: a
// plan is data the caller may inspect, log or refuse before any module evaluation happens
// (Bun loaders resolve the complete entry set again at load time). That split
// is what lets a dependency mistake fail with an exit code instead of a half-imported project.

import { readFileSync } from "node:fs";
import { join } from "node:path";
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

/** a project directory reduced to what both consumers need: its manifest and the plugin
 *  set the manifest enables. */
export interface ProjectPlan {
    readonly dir: string;
    readonly manifest: Manifest;
    /** engine plugin names to resolve as `${name}Plugin` (enabled defaults + declared extras) */
    readonly engine: readonly string[];
    readonly locals: readonly PlannedLocal[];
    /** manifest keys explicitly turned off — never resolved, never imported; kept so a caller can say so */
    readonly disabled: readonly string[];
}

/** the reads a plan is allowed to make, injectable so a test can pin the read set (the seam takes a
 *  project root and reads inside it — never this package's own layout or export map). */
export interface ProjectIo {
    /** the file's text, or null when it does not exist */
    readFile(path: string): string | null;
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

const REAL_IO: ProjectIo = {
    readFile(path) {
        try {
            return readFileSync(path, "utf-8");
        } catch {
            return null;
        }
    },
};

/** read a project directory into a plan: its manifest (empty when absent) and the classified plugin
 *  set. Reads only inside `dir`. */
export function readProject(dir: string, io: ProjectIo = REAL_IO): ProjectPlan {
    // the real path goes through `readManifest`, which also emits the manifest-boundary warnings; an
    // injected io (a test pinning the read set) parses the same text through the same `normalize`.
    const manifest = io === REAL_IO ? readManifest(dir) : normalize(io.readFile(manifestPath(dir)));
    return { dir, manifest, ...plan(manifest, dir) };
}

/** resolve a module specifier from the project root, or null when nothing resolves. */
function resolveFromProject(spec: string, dir: string): string | null {
    try {
        return Bun.resolveSync(spec, dir);
    } catch {
        return null;
    }
}

/** resolve every enabled entry before evaluation, retaining browser-authored paths in the plan. */
export function resolveLocalModules(project: Pick<ProjectPlan, "dir" | "locals">): PlannedLocal[] {
    const errors: string[] = [];
    const locals: PlannedLocal[] = [];
    for (const local of project.locals) {
        const path = resolveFromProject(local.path, project.dir);
        if (path) locals.push({ ...local, path });
        else
            errors.push(
                `shallot.json plugin "${local.name}": cannot resolve its module "${local.spec}" from ${project.dir}`,
            );
    }
    if (errors.length) throw new Error(errors.join("\n"));
    return locals;
}
