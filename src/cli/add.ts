import {
    cpSync,
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { CLAUDE_IMPORT, PROJECT_GITIGNORE, recipeDoc } from "./add-fragments";

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

  Without a name, lists available examples.
  With a name, copies one example into a project.
  The destination defaults to the example name relative to the current directory.
  An occupied destination is refused.

  Common examples
    shallot add
    shallot add first-person
    shallot add first-person my-game

  Options
    -h, --help  Show this help
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

/** Give a copied project the exact engine version from the installed package. */
export function pinEngine(pkgText: string, version: string): string {
    const pkg = JSON.parse(pkgText);
    let found = false;
    for (const field of DEP_FIELDS) {
        if (typeof pkg[field]?.[ENGINE] !== "string") continue;
        pkg[field][ENGINE] = version;
        found = true;
    }
    if (!found) pkg.dependencies = { [ENGINE]: version, ...pkg.dependencies };
    return `${JSON.stringify(pkg, null, 4)}\n`;
}

function importsPackage(recipeDir: string, packageName: string): boolean {
    const sourceFile = /\.(?:[cm]?[jt]sx?)$/;
    const importPattern = new RegExp(
        `(?:\\bfrom\\s*|\\bimport\\s*(?:\\(\\s*)?)["']${packageName}(?:/[^"']*)?["']`,
    );
    const pending = [recipeDir];
    while (pending.length > 0) {
        const dir = pending.pop()!;
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            if (entry.name === "node_modules") continue;
            const path = join(dir, entry.name);
            if (entry.isDirectory()) pending.push(path);
            else if (entry.isFile() && sourceFile.test(entry.name)) {
                if (importPattern.test(readFileSync(path, "utf8"))) return true;
            }
        }
    }
    return false;
}

function recipePackage(dest: string, recipeDir: string): string {
    const engine = JSON.parse(readFileSync(resolve(PACKAGE_ROOT, "package.json"), "utf8"));
    const devDependencies: Record<string, string> = {
        "@types/bun": engine.devDependencies["@types/bun"],
        playwright: engine.devDependencies.playwright,
        typescript: engine.devDependencies.typescript,
        vite: engine.devDependencies.vite,
    };
    if (importsPackage(recipeDir, "typegpu"))
        devDependencies.typegpu = engine.peerDependencies.typegpu;
    return `${JSON.stringify(
        {
            name: basename(dest),
            version: "0.0.0",
            private: true,
            type: "module",
            scripts: {
                dev: "vite --host 127.0.0.1",
                build: "vite build",
                preview: "vite preview",
                "test:browser": "playwright test",
            },
            devDependencies,
        },
        null,
        4,
    )}\n`;
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
            `no examples found (looked in ${recipesDir}). Run this from an installed ${ENGINE} package.`,
        );
        return 1;
    }

    const name = args[0];
    const printRecipe = (recipe: Recipe, write: (line: string) => void) =>
        write(`  ${recipe.name}${recipe.intent ? ` — ${recipe.intent}` : ""}`);

    if (name == null) {
        console.log("Available examples:\n");
        for (const recipe of available) printRecipe(recipe, console.log);
        console.log("\nCopy an example with:\n  bunx shallot add <name> [dir]");
        return 0;
    }

    if (!available.some((recipe) => recipe.name === name)) {
        console.error(`unknown example: ${name}\n\nAvailable examples:`);
        for (const recipe of available) printRecipe(recipe, console.error);
        return 1;
    }

    const dest = resolve(args[1] || name);
    if (occupied(dest)) {
        console.error(`refusing to copy into ${dest}: directory is not empty`);
        return 1;
    }

    const recipeDir = resolve(recipesDir, name);
    cpSync(recipeDir, dest, {
        recursive: true,
        filter: (src) => basename(src) !== "node_modules",
    });

    const pkgPath = resolve(dest, "package.json");
    const pkgText = existsSync(pkgPath)
        ? readFileSync(pkgPath, "utf8")
        : recipePackage(dest, resolve(recipesDir, name));
    writeFileSync(pkgPath, pinEngine(pkgText, version));

    // Emit scaffolding absent from the copied directory: the agent-surface pointer (AGENTS.md, imported
    // by CLAUDE.md), the project ignore (bun pack drops `.gitignore`), and the Bun transform preload used
    // by project tests. Don't clobber a recipe that ships its own.
    for (const [file, content] of [
        ["AGENTS.md", recipeDoc(name)],
        ["CLAUDE.md", CLAUDE_IMPORT],
        [".gitignore", PROJECT_GITIGNORE],
    ] as const) {
        const path = resolve(dest, file);
        if (!existsSync(path)) writeFileSync(path, content);
    }
    const preload = resolve(dest, "tests/preload.ts");
    const bunfig = resolve(dest, "bunfig.toml");
    if (!existsSync(preload)) {
        mkdirSync(dirname(preload), { recursive: true });
        writeFileSync(
            preload,
            'import { plugin } from "bun";\nimport { shallot } from "@dylanebert/shallot/bun";\nplugin(shallot({ root: import.meta.dir }));\n',
        );
    }
    if (!existsSync(bunfig)) writeFileSync(bunfig, '[test]\npreload = ["./tests/preload.ts"]\n');
    console.log(`copied example ${name} → ${dest}`);
    console.log(`  cd ${args[1] || name} && bun install && bunx shallot dev`);
    return 0;
}
