import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../scripts/test-tiers";
import { readProjectPolicy } from "./policy";

setDefaultTimeout(CEILING.node);

test("project policy keeps tracked diagnostics source tools, never run output", () => {
    const tree = mkdtempSync(join(tmpdir(), "shallot-project-policy-diagnostics-"));
    try {
        execFileSync("git", ["init", "--quiet"], { cwd: tree });
        mkdirSync(join(tree, "diagnostics/run"), { recursive: true });
        for (const extension of ["ts", "mjs", "c", "h", "py", "rs"])
            writeFileSync(join(tree, `diagnostics/tool.${extension}`), "");
        const outputs = ["profile.txt", "profile.json", "junit.xml", "output.log", "REPORT.md"];
        for (const file of outputs) writeFileSync(join(tree, "diagnostics/run", file), "");
        execFileSync("git", ["add", "diagnostics"], { cwd: tree });
        writeFileSync(join(tree, "diagnostics/untracked.log"), "");
        expect(readProjectPolicy(tree).sort()).toEqual(
            outputs
                .map((file) => `diagnostics/run/${file}: diagnostics keeps source tools, never a run's output`)
                .sort(),
        );
    } finally {
        rmSync(tree, { recursive: true, force: true });
    }
});
