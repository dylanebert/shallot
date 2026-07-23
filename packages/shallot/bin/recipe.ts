import { cpSync, existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { RECIPE_TSCONFIG, recipeDoc } from "./scaffold";

// `shallot recipe [name] [dir]` — copy a recipe out of the installed package into a runnable project.
// The recipes ship in the tarball under this package's `examples/recipes/` (prepack projection); running
// one in place breaks its own dep resolution and users shouldn't edit inside node_modules, so copy-out is
// the path. The copy's `workspace:*` dep on the engine is rewritten to the installed version so a plain
// `bun install && bunx shallot dev` runs green. Paths resolve relative to this package, never cwd — the
// corpus lives beside the CLI (`bin/` and `examples/` are siblings at the package root).

const PACKAGE_ROOT = resolve(import.meta.dir, "..");
const ENGINE = "@dylanebert/shallot";

interface Env {
    recipesDir: string;
    version: string;
}

function env(): Env {
    const pkg = JSON.parse(readFileSync(resolve(PACKAGE_ROOT, "package.json"), "utf8"));
    return { recipesDir: resolve(PACKAGE_ROOT, "examples/recipes"), version: pkg.version };
}

/** recipe directory names available to copy — a dir is a recipe when it carries a `shallot.json`. */
export function listRecipes(recipesDir: string): string[] {
    if (!existsSync(recipesDir)) return [];
    return readdirSync(recipesDir)
        .filter((name) => existsSync(resolve(recipesDir, name, "shallot.json")))
        .sort();
}

/** true when `dest` is occupied — a non-empty dir, or a regular file — so the overwrite guard refuses it. */
export function occupied(dest: string): boolean {
    if (!existsSync(dest)) return false;
    const stat = statSync(dest);
    return stat.isDirectory() ? readdirSync(dest).length > 0 : true;
}

const DEP_FIELDS = [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
] as const;

/**
 * rewrite a `workspace:` engine dep to a concrete range so the copy installs from the registry, matching
 * bun's publish semantics: `workspace:*` (and bare `workspace:`) → the exact version, `workspace:^` /
 * `workspace:~` → `^<version>` / `~<version>`, and an explicit range (`workspace:^1.2.3`, `workspace:1.2.3`)
 * → the range verbatim with only the `workspace:` prefix stripped. The corpus is all `workspace:*`, so
 * today every path pins exact; the rest holds the rewrite honest for a future explicit dep.
 */
export function pinEngine(pkgText: string, version: string): string {
    const pkg = JSON.parse(pkgText);
    for (const field of DEP_FIELDS) {
        const dep = pkg[field]?.[ENGINE];
        if (typeof dep !== "string" || !dep.startsWith("workspace:")) continue;
        const marker = dep.slice("workspace:".length);
        pkg[field][ENGINE] =
            marker === "*" || marker === ""
                ? version
                : marker === "^" || marker === "~"
                  ? `${marker}${version}`
                  : marker;
    }
    return `${JSON.stringify(pkg, null, 4)}\n`;
}

export async function runRecipe(args: string[], e: Env = env()): Promise<number> {
    const { recipesDir, version } = e;
    const available = listRecipes(recipesDir);

    if (available.length === 0) {
        console.error(
            `no recipes found (looked in ${recipesDir}). Run this from an installed ${ENGINE} package.`,
        );
        return 1;
    }

    const name = args[0];
    if (name == null) {
        console.log("Available recipes:\n");
        for (const r of available) console.log(`  ${r}`);
        console.log("\nCopy one out with:\n  bunx shallot recipe <name> [dir]");
        return 0;
    }

    if (!available.includes(name)) {
        console.error(`unknown recipe: ${name}\n\nAvailable recipes:`);
        for (const r of available) console.error(`  ${r}`);
        return 1;
    }

    const dest = resolve(args[1] || name);
    if (occupied(dest)) {
        console.error(`refusing to copy into ${dest}: directory is not empty`);
        return 1;
    }

    cpSync(resolve(recipesDir, name), dest, {
        recursive: true,
        filter: (src) => basename(src) !== "node_modules",
    });

    const pkgPath = resolve(dest, "package.json");
    if (existsSync(pkgPath))
        writeFileSync(pkgPath, pinEngine(readFileSync(pkgPath, "utf8"), version));

    // emit the standalone scaffold the monorepo recipe lacks: the agent-surface pointer (AGENTS.md +
    // CLAUDE.md) that hands a harness the installed engine's contract, and a tsconfig for `bunx tsc`.
    // Don't clobber a recipe that ships its own.
    const doc = recipeDoc(name);
    for (const file of ["AGENTS.md", "CLAUDE.md"]) {
        const path = resolve(dest, file);
        if (!existsSync(path)) writeFileSync(path, doc);
    }
    const tsconfig = resolve(dest, "tsconfig.json");
    if (!existsSync(tsconfig)) writeFileSync(tsconfig, RECIPE_TSCONFIG);

    console.log(`copied ${name} → ${dest}`);
    console.log(`  cd ${args[1] || name} && bun install && bunx shallot dev`);
    return 0;
}
