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

const ROOT = resolve(import.meta.dir, "../../.."); // packages/shallot/scripts → repo root
const SRC = resolve(ROOT, "examples");
const DEST = resolve(import.meta.dir, "../examples"); // packages/shallot/examples (gitignored projection)

// repo-only plumbing that can't resolve outside the workspace (package.json is kept — it's what makes a
// copied-out recipe a runnable project), plus the smoke-test plugin the copy-out must not carry
const STRIP = new Set(["node_modules", "tsconfig.json", "smoke.ts"]);

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

// The shipped index leads with the recipe corpus and nothing else. Grep-first framing, honest about the
// shipped context: these are read-and-adapt patterns under this directory, not runnable-in-place projects.
const HEADER = `# Examples

Problem-indexed patterns, shipped with the engine. Grep for the problem you have, then read that recipe's
source under this directory — plugin modules in \`src/\`, the scene in \`public/scenes/\`, plugin enablement
in \`shallot.json\`. They aren't wired to run in place: \`bunx shallot recipe <name> [dir]\` copies one out
into a runnable, version-matched project. The gym and showcase tiers, and the live corpus, are at
github.com/dylanebert/shallot.

`;

function shippedIndex(): string {
    const md = readFileSync(resolve(SRC, "AGENTS.md"), "utf8");
    const start = md.indexOf("## Recipes");
    const end = md.indexOf("## Gym");
    if (start < 0 || end < 0)
        throw new Error("examples/AGENTS.md: expected ## Recipes and ## Gym sections");
    return `${HEADER}${md.slice(start, end).trimEnd()}\n`;
}

rmSync(DEST, { recursive: true, force: true });
mkdirSync(DEST, { recursive: true });

const destRecipes = resolve(DEST, "recipes");
cpSync(resolve(SRC, "recipes"), destRecipes, {
    recursive: true,
    filter: (src) => !STRIP.has(basename(src)),
});
for (const name of readdirSync(destRecipes)) {
    const manifest = resolve(destRecipes, name, "shallot.json");
    if (existsSync(manifest)) stripSmokeEntry(manifest);
}
writeFileSync(resolve(DEST, "AGENTS.md"), shippedIndex());

console.log(`prepack: projected recipes + index → ${DEST}`);
