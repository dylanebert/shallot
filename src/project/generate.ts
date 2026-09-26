import type { ProjectPlan } from "./host";

// Generates the `virtual:project` module source from a `shallot.json` manifest — the one place a manifest
// becomes static imports. Pure over (manifest, absDir, scenes), so `generate.test.ts` pins the emitted
// import lines without a running vite. Engine plugins resolve to a lean named import from the main
// barrel (`import { OrbitPlugin } from "@dylanebert/shallot"`, tree-shaken); a local/external plugin is a module whose **default export** is the Plugin
// (Expo / Obsidian / Babel convention — the package declares its entry, e.g. a subpath `my-plugin/grid`
// default-exporting GridPlugin). The runtime guard below fails loud when a default import resolved to
// something that isn't a Plugin (a default import is silently `undefined` otherwise), naming the manifest key.

const ENGINE = "@dylanebert/shallot";

/** Generate the virtual module from its resolved project plan. */
export function generateModuleFromPlan(project: ProjectPlan): string {
    const { dir, manifest, scenes, engine, locals } = project;
    const idents = engine.map((n) => `${n}Plugin`);
    const lines: string[] = [];

    if (idents.length > 0) {
        lines.push(`import { ${idents.join(", ")} } from ${JSON.stringify(ENGINE)};`);
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
