import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

// Arm for `scripts/format.ts`'s report-only mode — the mechanism that lets `check`
// stop writing the tree it judges (spec: shallot-gate-never-writes, Validation 1 & 2).
// `scripts/format.ts` scans from `import.meta.dir/..` (the repo root) for `**/*.scene`,
// so each test creates a temp fixture tree under the repo root (never a tracked file).
// The script's own imports resolve relative to the script file, not cwd, so the engine loads regardless.

const SCRIPT = join(import.meta.dir, "../../../scripts/format.ts");
const REPO_ROOT = resolve(import.meta.dir, "../../..");

// A .scene that normalization would change: no trailing newline. The script does
// `stringify(nodes) + "\n"`, so any scene missing the final newline is in the would-change set.
const UNFORMATTED_SCENE = "<scene>\n    <a part />\n</scene>";

function fixtureScene(): { dir: string; scenePath: string } {
    const dir = mkdtempSync(join(REPO_ROOT, "_format-gate-tmp-"));
    const scenePath = join(dir, "test.scene");
    writeFileSync(scenePath, UNFORMATTED_SCENE);
    return { dir, scenePath };
}

function runFormat(args: string[]) {
    // run from a subdirectory so the test proves the import.meta.dir anchor —
    // with the old process.cwd() scan, the fixture under REPO_ROOT would not be found
    return Bun.spawnSync(["bun", SCRIPT, ...args], {
        cwd: join(REPO_ROOT, "scripts"),
        stdout: "pipe",
        stderr: "pipe",
    });
}

describe("format.ts report-only mode — Validation 1: check does not write", () => {
    test("--check leaves a would-change fixture byte-identical", () => {
        const { dir, scenePath } = fixtureScene();
        try {
            const before = readFileSync(scenePath, "utf-8");
            runFormat(["--check"]);
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
            runFormat([]);
            const normalized = readFileSync(scenePath, "utf-8");

            const proc = runFormat(["--check"]);
            expect(proc.exitCode).toBe(0);
            expect(readFileSync(scenePath, "utf-8")).toBe(normalized);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
});

describe("format.ts report-only mode — Validation 2: gates and still writes", () => {
    // Mutation witness: reverting the import.meta.dir anchor to process.cwd() makes both tests
    // below fail — the subprocess runs from REPO_ROOT/scripts, so process.cwd() no longer reaches
    // the fixture under REPO_ROOT, and --check exits 0 (vacuously green) despite unformatted content.
    test("--check reds (exits nonzero) on a fixture normalization would change", () => {
        const { dir } = fixtureScene();
        try {
            const proc = runFormat(["--check"]);
            expect(proc.exitCode).not.toBe(0);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    test("write mode (default) still writes that same fixture", () => {
        const { dir, scenePath } = fixtureScene();
        try {
            const before = readFileSync(scenePath, "utf-8");
            const proc = runFormat([]);
            expect(proc.exitCode).toBe(0);
            const after = readFileSync(scenePath, "utf-8");
            expect(after).not.toBe(before);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
});

describe("format.ts ignore matching — segment, not substring", () => {
    // Mutation witness: reverting segment matching to path.includes(dir) makes this test fail —
    // "distortion-xxx/test.scene".includes("dist") is true, so the file is silently skipped and
    // --check exits 0 (green) despite unformatted content. The segment match (split on "/")
    // correctly distinguishes "distortion-xxx" from "dist".
    test("a dist-substring directory (distortion/) is not skipped by the dist ignore entry", () => {
        const dir = mkdtempSync(join(REPO_ROOT, "distortion-"));
        const scenePath = join(dir, "test.scene");
        writeFileSync(scenePath, UNFORMATTED_SCENE);
        try {
            const proc = runFormat(["--check"]);
            const out = proc.stdout.toString() + proc.stderr.toString();
            // the fixture path must appear in the would-change output — it was not skipped
            const relPath = scenePath.slice(REPO_ROOT.length + 1);
            expect(out).toContain(relPath);
            expect(proc.exitCode).not.toBe(0);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
});
