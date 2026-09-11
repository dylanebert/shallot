import { expect } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";

const ROOT = resolve(import.meta.dir, "..");
const SCRIPT = resolve(ROOT, "scripts/examples-index.ts");

type Result = { code: number; out: string; err: string };

function run(root: string, args: string[] = []): Result {
    const proc = Bun.spawnSync(["bun", SCRIPT, "--root", root, ...args], {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    return {
        code: proc.exitCode ?? -1,
        out: proc.stdout.toString(),
        err: proc.stderr.toString(),
    };
}

function git(root: string, args: string[]): void {
    const proc = Bun.spawnSync(["git", ...args], { cwd: root, stdout: "pipe", stderr: "pipe" });
    expect(proc.exitCode).toBe(0);
}

function fixture(): string {
    const root = mkdtempSync(join(tmpdir(), "shallot-examples-index-"));
    mkdirSync(join(root, "examples/visible-source"), { recursive: true });
    mkdirSync(join(root, "examples/ignored-output/dist"), { recursive: true });
    writeFileSync(join(root, ".gitignore"), "examples/ignored-output/dist/\n");
    writeFileSync(
        join(root, "examples/visible-source/shallot.json"),
        '{"kind":"recipe","description":"a visible fixture"}\n',
    );
    writeFileSync(join(root, "examples/visible-source/source.ts"), "export const source = true;\n");
    writeFileSync(join(root, "examples/ignored-output/dist/output.js"), "ignored output\n");
    git(root, ["init", "--quiet"]);
    git(root, ["add", ".gitignore", "examples/visible-source/shallot.json"]);
    return root;
}

check(
    "the examples index follows Git-visible source populations",
    {
        claim: "examples-index.ts indexes tracked and non-ignored source files, excludes ignored-only directories, and refuses missing Git evidence",
        size: "integration",
    },
    () => {
        const tree = fixture();
        try {
            const generated = run(tree);
            expect(generated.code).toBe(0);
            const index = readFileSync(join(tree, "examples/AGENTS.md"), "utf8");
            expect(index).toContain("`visible-source`");
            expect(index).not.toContain("ignored-output");
            expect(run(tree, ["--check"]).code).toBe(0);

            writeFileSync(
                join(tree, "examples/ignored-output/source.ts"),
                "export const source = true;\n",
            );
            const untracked = run(tree, ["--check"]);
            expect(untracked.code).not.toBe(0);
            expect(untracked.err).toContain("examples/ignored-output/ has no shallot.json");

            mkdirSync(join(tree, "examples/tracked-missing"), { recursive: true });
            writeFileSync(
                join(tree, "examples/tracked-missing/source.ts"),
                "export const source = true;\n",
            );
            git(tree, ["add", "examples/tracked-missing/source.ts"]);
            const tracked = run(tree, ["--check"]);
            expect(tracked.code).not.toBe(0);
            expect(tracked.err).toContain("examples/tracked-missing/ has no shallot.json");
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }

        const noGit = mkdtempSync(join(tmpdir(), "shallot-examples-index-no-git-"));
        try {
            const missingEvidence = run(noGit);
            expect(missingEvidence.code).not.toBe(0);
            expect(missingEvidence.err).toContain("`git ls-files` failed");
        } finally {
            rmSync(noGit, { recursive: true, force: true });
        }
    },
);
