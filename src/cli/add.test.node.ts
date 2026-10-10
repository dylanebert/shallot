import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runAdd } from "./add";

function recipes(): string {
    const root = mkdtempSync(join(tmpdir(), "shallot-add-"));
    mkdirSync(join(root, "examples/demo"), { recursive: true });
    writeFileSync(
        join(root, "examples/demo/index.html"),
        '<meta name="description" content="Demo">\n',
    );
    return root;
}

function discoveryRecipes(): string {
    const root = mkdtempSync(join(tmpdir(), "shallot-add-discovery-"));
    const descriptions = { zeta: "Zeta example", alpha: "Alpha example", legacy: "Legacy example" };
    for (const [name, description] of Object.entries(descriptions)) {
        mkdirSync(join(root, "examples", name), { recursive: true });
        writeFileSync(
            join(root, "examples", name, "index.html"),
            `<meta name="description" content="${description}">\n`,
        );
    }
    writeFileSync(join(root, "examples", "not-a-directory"), "ignored");
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

test("shallot add lists every example directory in stable name order with its meta description", async () => {
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
            "Available examples:\n\n  alpha — Alpha example\n  legacy — Legacy example\n  zeta — Zeta example\n\nCopy an example with:\n  bunx shallot add <name> [dir]",
        );
        expect(output.stdout).not.toContain("not-a-directory");
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

test("shallot add skips directories without a page without breaking the listing", async () => {
    const root = recipes();
    mkdirSync(join(root, "examples/no-page/node_modules"), { recursive: true });
    try {
        const output = await captureOutput(() =>
            runAdd([], {
                recipesDir: join(root, "examples"),
                version: "0.0.0",
            }),
        );
        expect(output.value).toBe(0);
        expect(output.stderr).toBe("");
        expect(output.stdout).toContain("  demo — Demo");
        expect(output.stdout).not.toContain("no-page");
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});

test("shallot add refuses an occupied destination and preserves its contents", async () => {
    const root = recipes();
    const dest = join(root, "out");
    mkdirSync(dest);
    const page = join(dest, "index.html");
    writeFileSync(page, "existing project\n");
    try {
        const output = await captureOutput(() =>
            runAdd(["demo", dest], {
                recipesDir: join(root, "examples"),
                version: "0.0.0",
            }),
        );
        expect(readFileSync(page, "utf8")).toBe("existing project\n");
        expect(output.value).toBe(1);
        expect(output.stderr).toBe(`refusing to copy into ${dest}: directory is not empty`);
        expect(existsSync(join(dest, "package.json"))).toBe(false);
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
        expect(readFileSync(join(dest, ".gitignore"), "utf8")).toBe(
            "node_modules/\ndist/\nbuild/\n.artifacts/\n",
        );
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
        expect(existsSync(join(dest, "AGENTS.md"))).toBe(false);
        expect(existsSync(join(dest, "CLAUDE.md"))).toBe(false);
    } finally {
        rmSync(root, { recursive: true, force: true });
    }
});
