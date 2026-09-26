import { expect } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
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

check(
    "add help aliases are informational",
    { claim: "shallot add help succeeds with no recipe catalogue or destination writes" },
    async () => {
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
    },
);

check(
    "add lists manifest intent without changing the recipe population",
    {
        claim: "shallot add lists every recipe in stable name order with problem, description, or name-only fallback",
    },
    async () => {
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
    },
);

check(
    "add writes the project ignore",
    { claim: "shallot add gives a copied recipe the canonical .gitignore" },
    async () => {
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
            const agents = readFileSync(join(dest, "AGENTS.md"), "utf8");
            expect(agents).toContain(
                "A Shallot example — a minimal project demonstrating one concept",
            );
            expect(agents).toContain("The examples live at");
            expect(agents).not.toContain("recipe");
        } finally {
            rmSync(root, { recursive: true, force: true });
        }
    },
);
