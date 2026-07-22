import { join } from "node:path";
import { DEFAULT_PLUGIN_NAMES, SUBPATH_PLUGIN_MODULES } from "./engine";
import { localOf, type Manifest } from "./manifest";

// Generates the `virtual:project` module source from a `shallot.json` manifest — the one place a manifest
// becomes static imports. Pure over (manifest, absDir, scenes), so `generate.test.ts` pins the emitted
// import lines without a running vite. Engine plugins resolve to a lean named import — the main
// barrel (`import { OrbitPlugin } from "@dylanebert/shallot"`, tree-shaken) for most, or a backend
// plugin's own subpath (`SUBPATH_PLUGIN_MODULES`, e.g. `AvbdPlugin` from `@dylanebert/shallot/avbd`) when
// it isn't barrel-listed; a local/external plugin is a module whose **default export** is the Plugin
// (Expo / Obsidian / Babel convention — the package declares its entry, e.g. a subpath `orrstead/grid`
// default-exporting GridPlugin). The runtime guard below fails loud when a default import resolved to
// something that isn't a Plugin (a default import is silently `undefined` otherwise), naming the manifest key.

const ENGINE = "@dylanebert/shallot";

/** the module specifier an engine plugin name imports from: its declared subpath, else the main barrel. */
function engineSource(name: string): string {
    return SUBPATH_PLUGIN_MODULES[name] ?? ENGINE;
}

// a local specifier resolved for the generated module: project-relative → project-absolute (the virtual
// module resolves against the host root, not the project), a bare package or absolute path → passed through.
function localPath(spec: string, absDir: string): string {
    return spec.startsWith(".") ? join(absDir, spec) : spec;
}

interface Plan {
    /** engine plugin names to import as `{ ${name}Plugin }` from the barrel (enabled defaults + extras) */
    readonly engine: string[];
    /** enabled local plugins, by name + resolved import path */
    readonly locals: { name: string; path: string }[];
}

/** classify a manifest into the engine + local plugins to statically import. */
export function plan(manifest: Manifest, absDir: string | null): Plan {
    const plugins = manifest.plugins ?? {};
    const defaults = new Set<string>(DEFAULT_PLUGIN_NAMES);
    const engine: string[] = [];
    const locals: Plan["locals"] = [];

    // every default is enabled unless explicitly turned off
    for (const name of DEFAULT_PLUGIN_NAMES) {
        if (plugins[name] !== false) engine.push(name);
    }
    // then the declared entries: an engine extra (true), or a local (a specifier). defaults already handled.
    for (const [name, value] of Object.entries(plugins)) {
        if (defaults.has(name)) continue;
        if (value === true) engine.push(name);
        else {
            const local = localOf(value);
            if (local?.enabled) locals.push({ name, path: localPath(local.spec, absDir ?? "") });
        }
    }
    return { engine, locals };
}

/**
 * build the `virtual:project` module source for a project dir with a (possibly empty) manifest. The
 * module is static imports + the project object, no HMR self-accept — vite full-reloads it on a plugin
 * edit, which the page reload cleans up (dev and a production build agree).
 */
export function generateModule(manifest: Manifest, dir: string | null, scenes: string[]): string {
    const { engine, locals } = plan(manifest, dir);
    const idents = engine.map((n) => `${n}Plugin`);
    const lines: string[] = [];

    // group by resolved source (barrel vs. a backend plugin's own subpath) so each import line pulls
    // only from the module that actually exports those names — preserves first-seen source order.
    const bySource = new Map<string, string[]>();
    for (const name of engine) {
        const source = engineSource(name);
        const identsForSource = bySource.get(source) ?? [];
        identsForSource.push(`${name}Plugin`);
        bySource.set(source, identsForSource);
    }
    for (const [source, sourceIdents] of bySource) {
        lines.push(`import { ${sourceIdents.join(", ")} } from ${JSON.stringify(source)};`);
    }
    // a local plugin is the module's default export (the package declares this entry). A wrong/missing
    // default is silently `undefined`, so the runtime guard below is what makes a mistake loud.
    for (let i = 0; i < locals.length; i++) {
        lines.push(`import _l${i} from ${JSON.stringify(locals[i].path)};`);
    }

    lines.push(`const engine = [${idents.join(", ")}];`);
    lines.push(
        `const locals = [${locals
            .map((l, i) => `{ name: ${JSON.stringify(l.name)}, plugin: _l${i} }`)
            .join(", ")}];`,
    );
    // a module that resolved but doesn't default-export a Plugin (no `name`) fails loud, naming the
    // manifest key + its specifier — never a silent drop.
    lines.push(
        `for (const l of locals) if (!l.plugin || typeof l.plugin.name !== "string") throw new Error("shallot.json plugin \\"" + l.name + "\\": its module must default-export a Plugin");`,
    );
    lines.push(`const manifest = ${JSON.stringify(manifest)};`);
    lines.push(`const scenes = ${JSON.stringify(scenes)};`);
    lines.push(`const scene = ${JSON.stringify(manifest.scene ?? null)};`);
    lines.push(`const capacity = ${JSON.stringify(manifest.capacity ?? null)};`);
    lines.push(`const dir = ${JSON.stringify(dir)};`);
    lines.push(
        `const project = { dir, scene, capacity, scenes, manifest, locals, plugins: [...engine, ...locals.map((l) => l.plugin)] };`,
    );
    lines.push(`export default project;`);

    return lines.join("\n");
}
