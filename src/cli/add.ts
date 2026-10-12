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

// `shallot add [name] [dir]` — copy a recipe out of the installed package into a runnable project.
// The recipes ship in the tarball under this package's `examples/`; running
// one in place breaks its own dep resolution and users shouldn't edit inside node_modules, so copy-out is
// the path. The copy gains a manifest pinned to the installed engine version so a plain
// `bun install && bun run dev` runs green. Paths resolve relative to this package, never cwd — the
// corpus lives beside the CLI (`examples/` sits two levels above `src/cli/`, at the package root).

const PACKAGE_ROOT = resolve(import.meta.dir, "../..");
const ENGINE = "@dylanebert/shallot";
const PROJECT_PRELOAD = `import { shallot } from "@dylanebert/shallot/bun";
import { plugin } from "bun";

plugin(shallot({ root: import.meta.dir }));
`;
const PROJECT_GITIGNORE = "node_modules/\ndist/\nbuild/\n.artifacts/\ntest-results/\n";
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

function listRecipeEntries(recipesDir: string): Recipe[] {
    if (!existsSync(recipesDir)) return [];
    return readdirSync(recipesDir, { withFileTypes: true })
        .filter(
            (entry) =>
                entry.isDirectory() && existsSync(resolve(recipesDir, entry.name, "index.html")),
        )
        .map((entry) => {
            const page = readFileSync(resolve(recipesDir, entry.name, "index.html"), "utf8");
            const meta = page.match(/<meta\b[^>]*\bname=["']description["'][^>]*>/i)?.[0];
            const intent = meta?.match(/\bcontent=(["'])(.*?)\1/i)?.[2];
            return { name: entry.name, intent };
        })
        .sort((a, b) => a.name.localeCompare(b.name));
}

/** Every directory under examples with an index.html is available to copy. */
export function listRecipes(recipesDir: string): string[] {
    return listRecipeEntries(recipesDir).map((recipe) => recipe.name);
}

/** true when `dest` is occupied — a non-empty dir, or a regular file — so the overwrite guard refuses it. */
export function occupied(dest: string): boolean {
    if (!existsSync(dest)) return false;
    const stat = statSync(dest);
    return stat.isDirectory() ? readdirSync(dest).length > 0 : true;
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

function recipePackage(dest: string, recipeDir: string, version: string): string {
    const engine = JSON.parse(readFileSync(resolve(PACKAGE_ROOT, "package.json"), "utf8"));
    const devDependencies: Record<string, string> = {
        "@types/bun": engine.devDependencies["@types/bun"],
        playwright: engine.peerDependencies.playwright,
        vite: engine.devDependencies.vite,
    };
    if (importsPackage(recipeDir, `${ENGINE}/webgpu`))
        devDependencies.webgpu = engine.peerDependencies.webgpu;
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
                preview: "vite preview --host 127.0.0.1",
            },
            dependencies: { [ENGINE]: version },
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

    // Recipes ship without project metadata; copy-out writes the manifest, ignore rules and Bun preload.
    writeFileSync(resolve(dest, "package.json"), recipePackage(dest, recipeDir, version));
    writeFileSync(resolve(dest, ".gitignore"), PROJECT_GITIGNORE);
    const preload = resolve(dest, "tests/preload.ts");
    mkdirSync(dirname(preload), { recursive: true });
    writeFileSync(preload, PROJECT_PRELOAD);
    writeFileSync(resolve(dest, "bunfig.toml"), '[test]\npreload = ["./tests/preload.ts"]\n');
    console.log(`copied example ${name} → ${dest}`);
    console.log(`  cd ${args[1] || name} && bun install && bun run dev`);
    return 0;
}
