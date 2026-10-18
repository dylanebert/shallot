import { expect, setDefaultTimeout, test } from "bun:test";

import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
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
