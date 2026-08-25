import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Arm for `scripts/format.ts`'s report-only mode — the mechanism that lets `check`
// stop writing the tree it judges (spec: shallot-gate-never-writes, Validation 1 & 2).
// `scripts/format.ts` scans `process.cwd()` for `**/*.scene`, so each test points the
// subprocess at an OS-tmpdir fixture tree (never a tracked file). The script's own
// imports resolve relative to the script file, not cwd, so the engine loads regardless.

const SCRIPT = join(import.meta.dir, "../../../scripts/format.ts");

// A .scene that normalization would change: no trailing newline. The script does
// `stringify(nodes) + "\n"`, so any scene missing the final newline is in the would-change set.
const UNFORMATTED_SCENE = "<scene>\n    <a part />\n</scene>";

function fixtureScene(): { dir: string; scenePath: string } {
    const dir = mkdtempSync(join(tmpdir(), "format-gate-"));
    const scenePath = join(dir, "test.scene");
    writeFileSync(scenePath, UNFORMATTED_SCENE);
    return { dir, scenePath };
}

function runFormat(args: string[], cwd: string) {
    return Bun.spawnSync(["bun", SCRIPT, ...args], {
        cwd,
        stdout: "pipe",
        stderr: "pipe",
    });
}

describe("format.ts report-only mode — Validation 1: check does not write", () => {
    test("--check leaves a would-change fixture byte-identical", () => {
        const { dir, scenePath } = fixtureScene();
        try {
            const before = readFileSync(scenePath, "utf-8");
            runFormat(["--check"], dir);
            const after = readFileSync(scenePath, "utf-8");
            expect(after).toBe(before);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    test("--check on an already-normalized scene exits zero and does not write", () => {
        const { dir, scenePath } = fixtureScene();
        try {
            // normalize first via write mode
            runFormat([], dir);
            const normalized = readFileSync(scenePath, "utf-8");

            const proc = runFormat(["--check"], dir);
            expect(proc.exitCode).toBe(0);
            expect(readFileSync(scenePath, "utf-8")).toBe(normalized);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
});

describe("format.ts report-only mode — Validation 2: gates and still writes", () => {
    test("--check reds (exits nonzero) on a fixture normalization would change", () => {
        const { dir } = fixtureScene();
        try {
            const proc = runFormat(["--check"], dir);
            expect(proc.exitCode).not.toBe(0);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    test("write mode (default) still writes that same fixture", () => {
        const { dir, scenePath } = fixtureScene();
        try {
            const before = readFileSync(scenePath, "utf-8");
            const proc = runFormat([], dir);
            expect(proc.exitCode).toBe(0);
            const after = readFileSync(scenePath, "utf-8");
            expect(after).not.toBe(before);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
});
