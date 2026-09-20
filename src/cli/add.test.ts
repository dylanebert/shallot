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
                expect(output.stdout).toContain("Without a name, lists available recipes.");
                expect(output.stdout).toContain("With a name, copies one recipe");
                expect(output.stdout).toContain("destination defaults to the recipe name");
                expect(output.stdout).toContain("An occupied destination is refused.");
                expect(existsSync(dest)).toBe(false);
            }
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
        const log = console.log;
        console.log = () => {};
        try {
            const dest = join(root, "out");
            expect(
                await runAdd(["demo", dest], {
                    recipesDir: join(root, "examples"),
                    version: "0.0.0",
                }),
            ).toBe(0);
            expect(readFileSync(join(dest, ".gitignore"), "utf8")).toBe(PROJECT_GITIGNORE);
        } finally {
            console.log = log;
            rmSync(root, { recursive: true, force: true });
        }
    },
);
