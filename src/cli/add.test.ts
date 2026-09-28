import { expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runAdd } from "./add";
import { PROJECT_GITIGNORE } from "./add-fragments";

function recipes(): string {
    const root = mkdtempSync(join(tmpdir(), "shallot-add-"));
    mkdirSync(join(root, "examples/demo"), { recursive: true });
    writeFileSync(join(root, "examples/demo/shallot.json"), '{"kind":"recipe"}\n');
    return root;
}

function discoveryRecipes(): string {
    const root = mkdtempSync(join(tmpdir(), "shallot-add-discovery-"));
    const manifests: Record<string, string> = {
        zeta: '{"kind":"recipe","description":"description fallback"}\n',
        alpha: '{"kind":"recipe","description":"unused description","problem":"declared problem"}\n',
        legacy: '{"kind":"recipe"}\n',
        showcase: '{"kind":"showcase","problem":"not a recipe"}\n',
    };
    for (const [name, manifest] of Object.entries(manifests)) {
        mkdirSync(join(root, "examples", name), { recursive: true });
        writeFileSync(join(root, "examples", name, "shallot.json"), manifest);
    }
    return root;
}

async function captureOutput<T>(
    body: () => Promise<T>,
): Promise<{ value: T; stdout: string; stderr: string }> {
    const stdout: string[] = [];
    const stderr: string[] = [];
    const log = console.log;
    const error = console.error;
    console.log = (...args: unknown[]) => stdout.push(args.join(" "));
    console.error = (...args: unknown[]) => stderr.push(args.join(" "));
    try {
        return { value: await body(), stdout: stdout.join("\n"), stderr: stderr.join("\n") };
    } finally {
        console.log = log;
        console.error = error;
    }
}

test("shallot add help succeeds with no recipe catalogue or destination writes", async () => {
    const root = mkdtempSync(join(tmpdir(), "shallot-add-help-"));
    try {
        for (const flag of ["--help", "-h"]) {
            const dest = join(root, `not-created-${flag.slice(1)}`);
            const output = await captureOutput(() =>
                runAdd([flag, dest], {
                    recipesDir: join(root, "empty"),
                    version: "0.0.0",
                }),
            );
            expect(output.value).toBe(0);
            expect(output.stderr).toBe("");
            expect(output.stdout).toContain("shallot add [name] [dir]");
            expect(output.stdout.trim()).toBe(
                `
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
    -h, --help  Show this help`.trim(),
            );
            expect(existsSync(dest)).toBe(false);
        }
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

test("shallot add lists every recipe in stable name order with problem, description, or name-only fallback", async () => {
    const root = discoveryRecipes();
    try {
        const output = await captureOutput(() =>
            runAdd([], {
                recipesDir: join(root, "examples"),
                version: "0.0.0",
            }),
        );
        expect(output.value).toBe(0);
        expect(output.stderr).toBe("");
        expect(output.stdout).toContain(
            "Available examples:\n\n  alpha — declared problem\n  legacy\n  zeta — description fallback\n\nCopy an example with:\n  bunx shallot add <name> [dir]",
        );
        expect(output.stdout).not.toContain("showcase");
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

test("shallot add declares TypeGPU when the copied source imports it", async () => {
    const root = recipes();
    const source = join(root, "examples/demo");
    mkdirSync(join(source, "src"));
    writeFileSync(join(source, "src/game.ts"), 'import tgpu from "typegpu";\nvoid tgpu;\n');
    try {
        const dest = join(root, "out");
        await captureOutput(() =>
            runAdd(["demo", dest], {
                recipesDir: join(root, "examples"),
                version: "0.0.0",
            }),
        );
        const pkg = JSON.parse(readFileSync(join(dest, "package.json"), "utf8"));
        const rootPackage = JSON.parse(
            readFileSync(join(import.meta.dir, "../../package.json"), "utf8"),
        );
        expect(pkg.devDependencies.typegpu).toBe(rootPackage.peerDependencies.typegpu);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

test("shallot add writes a Bun plugin preload for the generated project", async () => {
    const root = recipes();
    const source = join(root, "examples/demo");
    mkdirSync(join(source, "src"));
    writeFileSync(
        join(source, "src/demo.test.ts"),
        'import { test } from "bun:test";\ntest("demo", () => {});\n',
    );
    try {
        const dest = join(root, "out");
        await captureOutput(() =>
            runAdd(["demo", dest], {
                recipesDir: join(root, "examples"),
                version: "0.0.0",
            }),
        );
        expect(readFileSync(join(dest, "tests/preload.ts"), "utf8")).toBe(
            'import { plugin } from "bun";\nimport { shallot } from "@dylanebert/shallot/bun";\nplugin(shallot({ root: import.meta.dir }));\n',
        );
        expect(readFileSync(join(dest, "bunfig.toml"), "utf8")).toBe(
            '[test]\npreload = ["./tests/preload.ts"]\n',
        );
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

test("shallot add gives a copied recipe the canonical .gitignore", async () => {
    const root = recipes();
    try {
        const dest = join(root, "out");
        const output = await captureOutput(() =>
            runAdd(["demo", dest], {
                recipesDir: join(root, "examples"),
                version: "0.0.0",
            }),
        );
        expect(output.value).toBe(0);
        expect(output.stdout).toBe(
            `copied example demo → ${dest}\n  cd ${dest} && bun install && bunx shallot dev`,
        );
        expect(readFileSync(join(dest, ".gitignore"), "utf8")).toBe(PROJECT_GITIGNORE);
        expect(existsSync(join(dest, "tests/preload.ts"))).toBe(true);
        expect(readFileSync(join(dest, "bunfig.toml"), "utf8")).toContain(
            'preload = ["./tests/preload.ts"]',
        );
        const pkg = JSON.parse(readFileSync(join(dest, "package.json"), "utf8"));
        const engine = JSON.parse(
            readFileSync(join(import.meta.dir, "../../package.json"), "utf8"),
        );
        expect(pkg.dependencies["@dylanebert/shallot"]).toBe("0.0.0");
        expect(pkg.dependencies.vite).toBeUndefined();
        expect(pkg.devDependencies.vite).toBe(engine.devDependencies.vite);
        expect(pkg.devDependencies.playwright).toBe(engine.devDependencies.playwright);
        expect(pkg.devDependencies.typescript).toBeDefined();
        expect(pkg.devDependencies["@types/bun"]).toBeDefined();
        expect(pkg.devDependencies.typegpu).toBeUndefined();
        const agents = readFileSync(join(dest, "AGENTS.md"), "utf8");
        expect(agents).toContain("A Shallot example — a minimal project demonstrating one concept");
        expect(agents).toContain("The examples live at");
        expect(agents).toContain("bun test");
        expect(agents).not.toContain("recipe");
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
