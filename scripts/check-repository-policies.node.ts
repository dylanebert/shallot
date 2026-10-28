import { expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { execFileSync } from "node:child_process";
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { removeTestDirectory } from "./test-directory";

const ROOT = resolve(import.meta.dir, "..");

test("every CI job declares its timeout budget", async () => {
    for (const workflow of ["test.yml", "release.yml"]) {
        const source = await Bun.file(resolve(ROOT, ".github/workflows", workflow)).text();
        const { jobs } = Bun.YAML.parse(source) as {
            jobs: Record<string, { "timeout-minutes"?: number }>;
        };
        for (const [name, job] of Object.entries(jobs)) {
            expect(job["timeout-minutes"], `${workflow}: ${name}`).toBeGreaterThan(0);
        }
    }
});

/** Tracked files under `diagnostics/` that are not source tools. */
function diagnosticsOutput(root: string): string[] {
    return execFileSync("git", ["ls-files", "-z", "--", "diagnostics/"], {
        cwd: root,
        encoding: "utf8",
    })
        .split("\0")
        .filter((file) => file && !/\.(ts|mjs|c|cpp|h|py|rs)$/.test(file))
        .map((file) => `${file}: diagnostics keeps source tools, never a run's output`);
}

test("tracked diagnostics are source tools, never a run's output", () => {
    expect(diagnosticsOutput(ROOT)).toEqual([]);
});

test("the diagnostics policy refuses tracked run output and admits source tools", () => {
    const tree = mkdtempSync(join(tmpdir(), "shallot-diagnostics-policy-"));
    try {
        execFileSync("git", ["init", "--quiet"], { cwd: tree });
        mkdirSync(join(tree, "diagnostics/run"), { recursive: true });
        for (const extension of ["ts", "mjs", "c", "cpp", "h", "py", "rs"])
            writeFileSync(join(tree, `diagnostics/tool.${extension}`), "");
        const outputs = ["profile.txt", "profile.json", "junit.xml", "output.log", "REPORT.md"];
        for (const file of outputs) writeFileSync(join(tree, "diagnostics/run", file), "");
        execFileSync("git", ["add", "diagnostics"], { cwd: tree });
        writeFileSync(join(tree, "diagnostics/untracked.log"), "");
        expect(diagnosticsOutput(tree).sort()).toEqual(
            outputs
                .map(
                    (file) =>
                        `diagnostics/run/${file}: diagnostics keeps source tools, never a run's output`,
                )
                .sort(),
        );
    } finally {
        removeTestDirectory(tree);
    }
});

test("browser shards reject malformed or out-of-range selections", () => {
    for (const shard of ["", "1", "0/2", "3/2", "1/0", "-1/2", "1/2/3", "a/2", "01/2"]) {
        const result = Bun.spawnSync(["sh", "scripts/browser.sh", "--shard", shard], {
            cwd: ROOT,
            stdout: "pipe",
            stderr: "pipe",
        });
        expect(result.exitCode, shard).toBe(2);
        expect(result.stderr.toString()).toContain("Usage:");
    }
});

test("browser shards partition sorted subjects but validate every subject", () => {
    const tree = mkdtempSync(join(tmpdir(), "shallot-browser-shards-"));
    try {
        execFileSync("git", ["init", "--quiet"], { cwd: tree });
        for (const directory of ["scripts", "src", "bin", "examples/a", "examples/b", "examples/c"])
            mkdirSync(join(tree, directory), { recursive: true });
        copyFileSync(join(ROOT, "scripts/browser.sh"), join(tree, "scripts/browser.sh"));
        for (const subject of ["a", "b", "c"]) {
            for (const file of ["vite.config.ts", "playwright.config.ts", "check.e2e.ts"])
                writeFileSync(join(tree, "examples", subject, file), "");
        }
        // The stubbed bun keeps every subject as this host's and reads the browser timeout as 60000.
        const stubs = {
            bun: 'if [ "$1" = scripts/test-hosts.ts ]; then shift; printf \'%s\\n\' "$@"; else echo 60000; fi',
            bunx: "echo 60000",
        };
        for (const [command, script] of Object.entries(stubs)) {
            const path = join(tree, "bin", command);
            writeFileSync(path, `#!/bin/sh\n${script}\n`);
            chmodSync(path, 0o755);
        }
        const run = (...args: string[]) =>
            Bun.spawnSync(["sh", "scripts/browser.sh", ...args], {
                cwd: tree,
                env: {
                    ...process.env,
                    PATH: `${join(tree, "bin")}${process.platform === "win32" ? ";" : ":"}${process.env.PATH}`,
                },
                stdout: "pipe",
                stderr: "pipe",
            });
        const subjects = (shard: string) => {
            const result = run("--shard", shard);
            expect(result.exitCode).toBe(0);
            return result.stdout
                .toString()
                .split("\n")
                .filter((line) => line.startsWith("Browser subject:"));
        };
        expect(subjects("1/2")).toEqual(["Browser subject: examples/a"]);
        expect(subjects("2/2")).toEqual([
            "Browser subject: examples/b",
            "Browser subject: examples/c",
        ]);
        writeFileSync(join(tree, "src/orphan.e2e.ts"), "");
        expect(run("--shard", "2/2").stderr.toString()).toContain(
            "Orphan browser test: src/orphan.e2e.ts",
        );
        // A missing config outside this shard is still refused.
        writeFileSync(join(tree, "src/vite.config.ts"), "");
        expect(run("--shard", "2/2").stderr.toString()).toContain(
            "Missing browser config: src/playwright.config.ts",
        );
    } finally {
        removeTestDirectory(tree);
    }
});

test("browser subjects satisfy their repository policy", () => {
    const result = Bun.spawnSync(["sh", "scripts/browser.sh", "--check-only"], {
        cwd: ROOT,
        stdout: "pipe",
        stderr: "pipe",
    });
    if (!result.success) {
        throw new Error(
            [result.stdout.toString(), result.stderr.toString()].filter(Boolean).join("\n"),
        );
    }
});
