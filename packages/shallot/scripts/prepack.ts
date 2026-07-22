// Project the version-matched agent context into the tarball. `bun pm pack` runs this before packing
// (and `postpack.ts` removes the projection after), so the published package carries the recipes corpus
// + a shipped examples index an installed agent reads from `node_modules/@dylanebert/shallot/examples/`.
// The source of truth stays at the repo-root `examples/`; this is a gitignored copy, never edited by hand.
//
// The recipes ship as reference to read and adapt, not projects to build in place — so the repo-only
// plumbing (node_modules, package.json's `workspace:*`, tsconfig's `extends` up the monorepo) is stripped;
// what ships is the meaningful surface: plugin modules (`src/`), the scene + assets (`public/`), and the
// manifest (`shallot.json`). The index drops the gym/showcase tiers, which don't ship in the tarball.

import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "../../.."); // packages/shallot/scripts → repo root
const SRC = resolve(ROOT, "examples");
const DEST = resolve(import.meta.dir, "../examples"); // packages/shallot/examples (gitignored projection)

// repo-only plumbing that is meaningless or misleading inside a tarball
const STRIP = new Set(["node_modules", "package.json", "tsconfig.json"]);

// The shipped index leads with the recipe corpus and nothing else. Grep-first framing, honest about the
// shipped context: these are read-and-adapt patterns under this directory, not runnable-in-place projects.
const HEADER = `# Examples

Problem-indexed patterns, shipped with the engine. Grep for the problem you have, then read that recipe's
source under this directory — plugin modules in \`src/\`, the scene in \`public/scenes/\`, plugin enablement
in \`shallot.json\`. Read and adapt them into your own project; they aren't wired to run in place. The gym
and showcase tiers, and the live corpus, are at github.com/dylanebert/shallot.

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

cpSync(resolve(SRC, "recipes"), resolve(DEST, "recipes"), {
    recursive: true,
    filter: (src) => !STRIP.has(basename(src)),
});
writeFileSync(resolve(DEST, "AGENTS.md"), shippedIndex());

console.log(`prepack: projected recipes + index → ${DEST}`);
