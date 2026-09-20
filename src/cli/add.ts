import { cpSync, existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { CLAUDE_IMPORT, PROJECT_GITIGNORE, RECIPE_TSCONFIG, recipeDoc } from "./add-fragments";

// `shallot add [name] [dir]` — copy a recipe out of the installed package into a runnable project.
// The recipes ship in the tarball under this package's `examples/`; running
// one in place breaks its own dep resolution and users shouldn't edit inside node_modules, so copy-out is
// the path. The copy gains (or has its local dep rewritten to) the installed engine version so a plain
// `bun install && bunx shallot dev` runs green. Paths resolve relative to this package, never cwd — the
// corpus lives beside the CLI (`examples/` sits two levels above `src/cli/`, at the package root).

const PACKAGE_ROOT = resolve(import.meta.dir, "../..");
const ENGINE = "@dylanebert/shallot";
const ADD_USAGE = `
  shallot add [name] [dir]

  Without a name, lists available recipes.
  With a name, copies one recipe into a project.
  The destination defaults to the recipe name relative to the current directory.
  An occupied destination is refused.
`;

interface Env {
    recipesDir: string;
    version: string;
}

function env(): Env {
    const pkg = JSON.parse(readFileSync(resolve(PACKAGE_ROOT, "package.json"), "utf8"));
    return { recipesDir: resolve(PACKAGE_ROOT, "examples"), version: pkg.version };
}

interface Recipe {
    name: string;
    intent?: string;
}

function manifestText(value: unknown): string | undefined {
    return typeof value === "string" && value.trim() !== "" ? value.trim() : undefined;
}

function listRecipeEntries(recipesDir: string): Recipe[] {
    if (!existsSync(recipesDir)) return [];
    return readdirSync(recipesDir)
        .sort()
        .flatMap((name): Recipe[] => {
            const manifest = resolve(recipesDir, name, "shallot.json");
            if (!existsSync(manifest)) return [];
            try {
                const parsed = JSON.parse(readFileSync(manifest, "utf8"));
                if (parsed.kind !== "recipe") return [];
                return [
                    {
                        name,
                        intent: manifestText(parsed.problem) ?? manifestText(parsed.description),
                    },
                ];
            } catch {
                return [];
            }
        });
}

/** recipe directory names available to copy — a dir is a recipe when its `shallot.json` declares
 *  `"kind": "recipe"`; the directory layout says nothing. */
export function listRecipes(recipesDir: string): string[] {
    return listRecipeEntries(recipesDir).map((recipe) => recipe.name);
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
 * rewrite a local engine dep to a concrete range so the copy installs from the registry. `file:` and
 * `link:` (the corpus links the repo root by relative path) → the exact version. `workspace:` follows
 * bun's publish semantics: `workspace:*` (and bare `workspace:`) → the exact version, `workspace:^` /
 * `workspace:~` → `^<version>` / `~<version>`, and an explicit range (`workspace:^1.2.3`,
 * `workspace:1.2.3`) → the range verbatim with only the `workspace:` prefix stripped.
 */
export function pinEngine(pkgText: string, version: string): string {
    const pkg = JSON.parse(pkgText);
    for (const field of DEP_FIELDS) {
        const dep = pkg[field]?.[ENGINE];
        if (typeof dep !== "string") continue;
        if (dep.startsWith("file:") || dep.startsWith("link:")) {
            pkg[field][ENGINE] = version;
            continue;
        }
        if (!dep.startsWith("workspace:")) continue;
        const marker = dep.slice("workspace:".length);
        pkg[field][ENGINE] =
            marker === "*" || marker === ""
                ? version
                : marker === "^" || marker === "~"
                  ? `${marker}${version}`
                  : marker;
    }
    // in-repo members declare no engine dep (the root self-links); a standalone copy needs one
    if (!DEP_FIELDS.some((field) => typeof pkg[field]?.[ENGINE] === "string"))
        pkg.dependencies = { [ENGINE]: version, ...pkg.dependencies };
    return `${JSON.stringify(pkg, null, 4)}\n`;
}

export async function runAdd(args: string[], e: Env = env()): Promise<number> {
    if (args[0] === "--help" || args[0] === "-h") {
        console.log(ADD_USAGE);
        return 0;
    }

    const { recipesDir, version } = e;
    const available = listRecipeEntries(recipesDir);

    if (available.length === 0) {
        console.error(
            `no recipes found (looked in ${recipesDir}). Run this from an installed ${ENGINE} package.`,
        );
        return 1;
    }

    const name = args[0];
    const printRecipe = (recipe: Recipe, write: (line: string) => void) =>
        write(`  ${recipe.name}${recipe.intent ? ` — ${recipe.intent}` : ""}`);

    if (name == null) {
        console.log("Available recipes:\n");
        for (const recipe of available) printRecipe(recipe, console.log);
        console.log("\nCopy one out with:\n  bunx shallot add <name> [dir]");
        return 0;
    }

    if (!available.some((recipe) => recipe.name === name)) {
        console.error(`unknown recipe: ${name}\n\nAvailable recipes:`);
        for (const recipe of available) printRecipe(recipe, console.error);
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

    // emit the standalone scaffold the monorepo recipe lacks: the agent-surface pointer (AGENTS.md,
    // imported by CLAUDE.md) that hands a harness the installed engine's contract, the project ignore
    // (bun pack drops `.gitignore`), and a tsconfig for `bunx tsc`. Don't clobber a recipe that ships its own.
    for (const [file, content] of [
        ["AGENTS.md", recipeDoc(name)],
        ["CLAUDE.md", CLAUDE_IMPORT],
        [".gitignore", PROJECT_GITIGNORE],
    ] as const) {
        const path = resolve(dest, file);
        if (!existsSync(path)) writeFileSync(path, content);
    }
    const tsconfig = resolve(dest, "tsconfig.json");
    if (!existsSync(tsconfig)) writeFileSync(tsconfig, RECIPE_TSCONFIG);

    console.log(`copied ${name} → ${dest}`);
    console.log(`  cd ${args[1] || name} && bun install && bunx shallot dev`);
    return 0;
}
