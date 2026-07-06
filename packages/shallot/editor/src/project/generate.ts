import { join } from "node:path";
import { DEFAULT_PLUGIN_NAMES } from "./engine";
import { localOf, type Manifest } from "./manifest";

// Generates the `virtual:project` module source from a `shallot.json` manifest — the one place a manifest
// becomes static imports. Pure over (manifest, absDir, scenes), so `generate.test.ts` pins the emitted
// import + HMR lines without a running vite. Engine plugins resolve to a lean named barrel import
// (`import { OrbitPlugin } from "@dylanebert/shallot"`, tree-shaken); a local/external plugin is a module
// whose **default export** is the Plugin (Expo / Obsidian / Babel convention — the package declares its
// entry, e.g. a subpath `orrstead/grid` default-exporting GridPlugin). The runtime guard below fails loud
// when a default import resolved to something that isn't a Plugin (a default import is silently `undefined`
// otherwise), naming the manifest key.
//
// The module serves two readers: production boot reads `plugins` (the full enabled set); the editor reads
// `locals` (project plugin objects it can't import itself) + `manifest` (it resolves enablement against its
// own engine catalog). A local edit hot-swaps through the per-local `import.meta.hot.accept`.

const ENGINE = "@dylanebert/shallot";

// a local specifier resolved for the generated module: project-relative → project-absolute (the module
// lives at the editor root, not the project), a bare package or absolute path → passed through.
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
 * build the `virtual:project` module source for a project dir with a (possibly empty) manifest. `hot`
 * gates the per-local HMR block — on for the dev server (the editor swaps a local edit in place), off for
 * a production build, which has no editor `/src/project/reload` to import and no `import.meta.hot`.
 */
export function generateModule(
    manifest: Manifest,
    dir: string | null,
    scenes: string[],
    hot = true,
): string {
    const { engine, locals } = plan(manifest, dir);
    const idents = engine.map((n) => `${n}Plugin`);
    const localIdents = locals.map((_, i) => `_l${i}`);
    const wantHot = hot && locals.length > 0;
    const lines: string[] = [];

    if (idents.length > 0)
        lines.push(`import { ${idents.join(", ")} } from ${JSON.stringify(ENGINE)};`);
    // a local plugin is the module's default export (the package declares this entry). A wrong/missing
    // default is silently `undefined`, so the runtime guard below is what makes a mistake loud.
    for (let i = 0; i < locals.length; i++) {
        lines.push(`import _l${i} from ${JSON.stringify(locals[i].path)};`);
    }
    if (wantHot) lines.push(`import { emitProjectReload } from "/src/project/reload";`);

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

    if (wantHot) {
        const paths = locals.map((l) => JSON.stringify(l.path)).join(", ");
        lines.push(`if (import.meta.hot) {`);
        lines.push(`    const live = [${localIdents.join(", ")}];`);
        lines.push(`    import.meta.hot.accept([${paths}], (mods) => {`);
        lines.push(`        mods.forEach((m, i) => { if (m) live[i] = m.default; });`);
        lines.push(
            `        emitProjectReload({ locals: locals.map((l, i) => ({ name: l.name, plugin: live[i] })), manifest });`,
        );
        lines.push(`    });`);
        lines.push(`}`);
    }
    return lines.join("\n");
}
