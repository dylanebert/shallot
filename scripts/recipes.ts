import { existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

// `bun run recipes` — booted every recipe through `shallot verify`. Verify is archived and `check`
// replaces it in this version, so after validating the selector this reports the run unavailable and
// exits 2: nothing ran, and an unrun gate is not a pass.

/** Pure seam for the population guards: an error message if the population is empty or the selector is
 *  unknown, or null if the run should proceed. */
export function populationError(recipeDirs: string[], only?: string): string | null {
    if (only && !recipeDirs.includes(only)) {
        return `no recipe "${only}" — one of: ${recipeDirs.join(", ")}`;
    }
    if (!only && recipeDirs.length === 0) {
        return `no recipes derived from examples/recipes — the scan matched nothing`;
    }
    return null;
}

const recipesRoot = resolve(import.meta.dir, "../examples/recipes");
const RECIPES = existsSync(recipesRoot)
    ? readdirSync(recipesRoot, { withFileTypes: true })
          .filter(
              (e) => e.isDirectory() && existsSync(resolve(recipesRoot, e.name, "shallot.json")),
          )
          .map((e) => e.name)
          .sort()
    : [];

function main(): void {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run recipes [--recipe <name>]

Unavailable in this version: \`check\` replaces \`shallot verify\`, which booted each recipe.

Options:
  --recipe <name>   Select a single recipe by its directory name (e.g. moving-platform)`);
        process.exit(0);
    }
    const idx = args.indexOf("--recipe");
    const only = idx !== -1 ? args[idx + 1] : undefined;

    const popErr = populationError(RECIPES, only);
    if (popErr) {
        console.error(popErr);
        process.exit(only ? 2 : 1);
    }

    console.error(
        "bun run recipes: unavailable — `check` replaces `shallot verify` in this version",
    );
    process.exit(2);
}

if (import.meta.main) main();
