// Project the version-matched agent context into the tarball. `bun pm pack` runs this before packing
// (and `postpack.ts` removes the projection after), so the published package carries the recipes corpus
// + a shipped examples index an installed agent reads from `node_modules/@dylanebert/shallot/examples/`.
// The source of truth stays at the repo-root `examples/`; this is a gitignored copy, never edited by hand.
//
// The recipes ship as both reference to read and copy-out sources for `shallot recipe <name>`, so the
// project surface ships: plugin modules (`src/`), the scene + assets (`public/`), the manifest
// (`shallot.json`), and the `package.json` that makes it a project (its `workspace:*` engine dep is
// rewritten to the installed version at copy-out). What's stripped is the monorepo-only plumbing that
// can't resolve outside the workspace: `node_modules` and the tsconfig whose `extends` walks up to it,
// plus the `src/smoke.ts` dynamics-smoke plugins (CI scaffolding for `bun run recipes`, enabled by a
// `./src/smoke` manifest entry) — those must not land in a user's copied-out project, so both the file
// and its `shallot.json` entry are dropped here. The index drops the gym/showcase tiers, which don't
// ship in the tarball.
//
// One recipe's plugin implementation lives outside the recipe: `gpu-particles` consumes the private
// `shallot-gpu-particles` workspace by its package name, the same way an installed plugin resolves.
// That dependency is a workspace-only specifier — unpublished, so unresolvable from a copied-out
// project — so this projection inlines the package's own source into the recipe's `src/`, rewrites the
// manifest entry to that local path and drops the dependency. The tarball therefore carries one
// implementation projected from one owner, never a second edited copy (`VENDORED_PLUGINS`).

import {
    cpSync,
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { basename, resolve } from "node:path";
import { recipeRoot } from "./projections";

const ROOT = resolve(import.meta.dir, "../../.."); // packages/shallot/scripts → repo root
const SRC = resolve(ROOT, "examples");
const DEST = resolve(import.meta.dir, "..", recipeRoot); // packages/shallot/examples (gitignored projection)

// repo-only plumbing that can't resolve outside the workspace (package.json is kept — it's what makes a
// copied-out recipe a runnable project), plus the smoke-test plugin the copy-out must not carry
const STRIP = new Set(["node_modules", "tsconfig.json", "smoke.ts"]);

/** a recipe whose plugin implementation is owned by a private workspace package: at pack time the
 *  package's `src/` is projected into the recipe's own `src/<dir>/`, the manifest's bare-package plugin
 *  entry becomes that local module, and the workspace dependency is dropped. */
export interface VendoredPlugin {
    /** recipe directory name under `examples/recipes/` */
    readonly recipe: string;
    /** the private package's directory, repo-root-relative */
    readonly from: string;
    /** the package name the recipe's manifest and dependencies name */
    readonly dependency: string;
    /** the projected directory under the recipe's `src/`, and the manifest module path it becomes */
    readonly into: string;
}

export const VENDORED_PLUGINS: readonly VendoredPlugin[] = [
    {
        recipe: "gpu-particles",
        from: "packages/shallot-gpu-particles",
        dependency: "shallot-gpu-particles",
        into: "particles",
    },
];

// a plugin entry pointing at a smoke module — its final path segment is `smoke` (e.g. `./src/smoke`)
const isSmokeModule = (path: string) => path.split("/").pop() === "smoke";

// drop the smoke plugin from a projected recipe's manifest so the shipped shallot.json can't reference
// the `src/smoke.ts` STRIP just removed.
function stripSmokeEntry(shallotJsonPath: string): void {
    const manifest = JSON.parse(readFileSync(shallotJsonPath, "utf8"));
    const plugins = manifest.plugins;
    if (!plugins) return;
    let changed = false;
    for (const [name, value] of Object.entries(plugins)) {
        if (typeof value === "string" && isSmokeModule(value)) {
            delete plugins[name];
            changed = true;
        }
    }
    if (changed) writeFileSync(shallotJsonPath, `${JSON.stringify(manifest, null, 4)}\n`);
}

/** rewrite one manifest plugin entry from a bare package specifier to a projected local module. The
 *  entry may be authored as a string or as an `[spec, enabled]` pair, so both forms move. */
export function inlineManifestEntry(
    manifestText: string,
    dependency: string,
    local: string,
): string {
    const manifest = JSON.parse(manifestText) as { plugins?: Record<string, unknown> };
    for (const [name, value] of Object.entries(manifest.plugins ?? {})) {
        if (typeof value === "string" && value === dependency)
            (manifest.plugins as Record<string, unknown>)[name] = local;
        else if (Array.isArray(value) && value[0] === dependency)
            (manifest.plugins as Record<string, unknown>)[name] = [local, ...value.slice(1)];
    }
    return `${JSON.stringify(manifest, null, 4)}\n`;
}

/** drop a workspace-only dependency from a projected recipe's package.json — its source is inlined, so
 *  naming an unpublished package would make `bun install` fail in the copied-out project. */
export function dropDependency(pkgText: string, dependency: string): string {
    const pkg = JSON.parse(pkgText) as Record<string, Record<string, string> | undefined>;
    for (const field of ["dependencies", "devDependencies", "peerDependencies"] as const) {
        if (pkg[field]) delete pkg[field]?.[dependency];
    }
    return `${JSON.stringify(pkg, null, 4)}\n`;
}

/**
 * Inline one vendored plugin into an already-projected recipe. Returns the files written, so a caller
 * (and the test) can see the projection happened rather than assume it. Refuses a table entry whose
 * package, recipe or manifest reference has moved: a silent no-op here would ship a recipe whose
 * manifest names an unpublished package.
 */
export function inlineVendoredPlugin(
    root: string,
    destRecipes: string,
    entry: VendoredPlugin,
): string[] {
    const recipeDir = resolve(destRecipes, entry.recipe);
    const manifestPath = resolve(recipeDir, "shallot.json");
    const pkgPath = resolve(recipeDir, "package.json");
    const packageSrc = resolve(root, entry.from, "src");
    if (!existsSync(packageSrc))
        throw new Error(`prepack: vendored plugin source missing: ${entry.from}/src`);
    if (!existsSync(manifestPath))
        throw new Error(`prepack: vendored plugin recipe missing: recipes/${entry.recipe}`);
    const manifestText = readFileSync(manifestPath, "utf8");
    if (!manifestText.includes(`"${entry.dependency}"`))
        throw new Error(
            `prepack: recipes/${entry.recipe}/shallot.json no longer names ${entry.dependency}`,
        );

    const into = resolve(recipeDir, "src", entry.into);
    rmSync(into, { recursive: true, force: true });
    cpSync(packageSrc, into, { recursive: true, filter: (src) => !STRIP.has(basename(src)) });
    writeFileSync(
        manifestPath,
        inlineManifestEntry(manifestText, entry.dependency, `./src/${entry.into}/index`),
    );
    if (existsSync(pkgPath))
        writeFileSync(pkgPath, dropDependency(readFileSync(pkgPath, "utf8"), entry.dependency));
    return readdirSync(into, { recursive: true, withFileTypes: true })
        .filter((file) => file.isFile())
        .map((file) => resolve(file.parentPath, file.name));
}

/** the recipes corpus as the tarball carries it: stripped of monorepo plumbing, with every vendored
 *  plugin inlined. Pure over two directories so a fixture tree can exercise every clause. */
export function projectRecipes(
    root: string,
    src: string,
    dest: string,
    table: readonly VendoredPlugin[] = VENDORED_PLUGINS,
): void {
    const destRecipes = resolve(dest, "recipes");
    cpSync(resolve(src, "recipes"), destRecipes, {
        recursive: true,
        filter: (path) => !STRIP.has(basename(path)),
    });
    for (const name of readdirSync(destRecipes)) {
        const manifest = resolve(destRecipes, name, "shallot.json");
        if (existsSync(manifest)) stripSmokeEntry(manifest);
    }
    for (const entry of table) inlineVendoredPlugin(root, destRecipes, entry);
}

// The shipped index leads with the recipe corpus and nothing else. Grep-first framing, honest about the
// shipped context: these are read-and-adapt patterns under this directory, not runnable-in-place projects.
const HEADER = `# Examples

Problem-indexed patterns, shipped with the engine. Grep for the problem you have, then read that recipe's
source under this directory — plugin modules in \`src/\`, the scene in \`public/scenes/\`, plugin enablement
in \`shallot.json\`. They aren't wired to run in place: \`bunx shallot recipe <name> [dir]\` copies one out
into a runnable, version-matched project. The bench and showcase tiers, and the live corpus, are at
github.com/dylanebert/shallot.

`;

export function shippedIndex(src: string): string {
    const md = readFileSync(resolve(src, "AGENTS.md"), "utf8");
    const start = md.indexOf("## Recipes");
    const end = md.indexOf("## Bench");
    if (start < 0 || end < 0)
        throw new Error("examples/AGENTS.md: expected ## Recipes and ## Bench sections");
    return `${HEADER}${md.slice(start, end).trimEnd()}\n`;
}

if (import.meta.main) {
    rmSync(DEST, { recursive: true, force: true });
    mkdirSync(DEST, { recursive: true });
    projectRecipes(ROOT, SRC, DEST);
    writeFileSync(resolve(DEST, "AGENTS.md"), shippedIndex(SRC));
    console.log(`prepack: projected recipes + index → ${DEST}`);
}
