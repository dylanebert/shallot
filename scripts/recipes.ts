import { existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { skipReason, verify } from "./verify";

// `bun run recipes` — every recipe through the shipped `shallot verify`, gated on its own boot + settled
// nonblank render. `./verify` owns the display guard, so a headless seat skips honestly instead of
// reaching a software adapter.

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

async function runRecipe(dir: string): Promise<boolean> {
    console.log(`\n--- ${dir} ---`);
    const result = await verify(`examples/recipes/${dir}`, ["--headed", "--timeout", "60000"]);
    const ok = result?.pass === true;
    console.log(ok ? `PASS: ${dir}` : `FAIL: ${dir}`);
    return ok;
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run recipes [--recipe <name>]

Boots every recipe through \`shallot verify\` and requires a settled nonblank render. Display-gated
(native hardware only).

Options:
  --recipe <name>   Run a single recipe by its directory name (e.g. moving-platform)`);
        process.exit(0);
    }
    const idx = args.indexOf("--recipe");
    const only = idx !== -1 ? args[idx + 1] : undefined;

    const popErr = populationError(RECIPES, only);
    if (popErr) {
        console.error(popErr);
        process.exit(only ? 2 : 1);
    }

    const skip = skipReason();
    if (skip) {
        console.log(`bun run recipes needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    const list = only ? [only] : RECIPES;
    console.log(`Running ${list.length} recipe verification(s)...`);
    let allPass = true;
    for (const dir of list) allPass = (await runRecipe(dir)) && allPass;

    if (!allPass) {
        console.error("\nFAIL: recipe verification failed");
        process.exit(1);
    }
    console.log("\nPASS: recipes green");
    process.exit(0);
}

if (import.meta.main) {
    main().catch((err) => {
        console.error(err instanceof Error ? err.message : err);
        process.exit(1);
    });
}
