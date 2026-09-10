import { SUBPATH_PLUGIN_MODULES } from "./engine";
import { type ProjectPlan, plan } from "./host";
import type { Manifest } from "./manifest";

// Generates the `virtual:project` module source from a `shallot.json` manifest — the one place a manifest
// becomes static imports. Pure over (manifest, absDir, scenes), so `generate.test.ts` pins the emitted
// import lines without a running vite. Engine plugins resolve to a lean named import — the main
// barrel (`import { OrbitPlugin } from "@dylanebert/shallot"`, tree-shaken) for most, or a backend
// plugin's own subpath (`SUBPATH_PLUGIN_MODULES`) when
// it isn't barrel-listed; a local/external plugin is a module whose **default export** is the Plugin
// (Expo / Obsidian / Babel convention — the package declares its entry, e.g. a subpath `my-plugin/grid`
// default-exporting GridPlugin). The runtime guard below fails loud when a default import resolved to
// something that isn't a Plugin (a default import is silently `undefined` otherwise), naming the manifest key.

const ENGINE = "@dylanebert/shallot";

/** the module specifier an engine plugin name imports from: its declared subpath, else the main barrel. */
function engineSource(name: string): string {
    return SUBPATH_PLUGIN_MODULES[name] ?? ENGINE;
}

// Planning itself lives in `host.ts` — the browser generator and the command entry consume the same
// resolved plan, so a manifest classifies once. Re-exported here because the CLI's feature reader
// already imports `plan` through this module.
export { plan };

/**
 * build the `virtual:project` module source for a project dir with a (possibly empty) manifest. The
 * module is static imports + the project object, no HMR self-accept — vite full-reloads it on a plugin
 * edit, which the page reload cleans up (dev and a production build agree).
 */
export function generateModule(manifest: Manifest, dir: string | null, scenes: string[]): string {
    return generateModuleFromPlan({ dir, manifest, scenes, ...plan(manifest, dir) });
}

/** the same module source, built from an already-resolved {@link ProjectPlan} — the shape both consumers
 *  share, so the command entry and this generator provably run the same plugin set. */
export function generateModuleFromPlan(project: ProjectPlan): string {
    const { dir, manifest, scenes, engine, locals } = project;
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
    lines.push(`const pixelRatio = ${JSON.stringify(manifest.pixelRatio ?? null)};`);
    lines.push(`const dir = ${JSON.stringify(dir)};`);
    lines.push(
        `const project = { dir, scene, capacity, pixelRatio, scenes, manifest, locals, plugins: [...engine, ...locals.map((l) => l.plugin)] };`,
    );
    lines.push(`export default project;`);

    return lines.join("\n");
}
