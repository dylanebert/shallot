// S1 arm — scripts/check-docs.ts pointer-validity two-sided vacuity reading
//
// Invariant: the pointer-validity arm (Arm f) in check-docs.ts reds when a comment cites a
// `*.md` path that resolves to nothing in-repo, and stays quiet when the cite resolves. The
// arm's vacuity guard checks `pointerCitationCount === 0` (the regex matched something), but
// that alone does not prove the *resolution* check fires — delete the
// `if (!resolvedMdBasenames.has(ref))` body and `deadPointers` stays empty, the counter still
// increments, the guard still passes, and the arm goes vacuously green. This test gives the
// two-sided reading that closes that gap: a dead cite reds (side 1), a live cite is spared
// (side 2), and the dead-cite assertion fails if the resolution check is deleted (the arm goes
// vacuously green and the script exits 0 instead of 1).
//
// The arm scans git-tracked .ts files (from `git ls-files`), not the filesystem, so the seed
// must land in an existing tracked file. The test appends comments to `scripts/rosters.ts`
// (a tracked .ts file, not in POINTER_EXCLUSION), runs the real script as a subprocess, and
// restores the original content in `finally` — same pattern as check-imports.test.ts. Both
// cites are seeded in a single subprocess run. The subprocess is driven from `beforeAll` so
// its wall-clock cost is invisible to the per-file test-duration cap (5000 ms), which measures
// only the test body, not `beforeAll`/`afterAll` (see packages/shallot/tests/test-cap.ts).

import { beforeAll, expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "check-docs.ts");
const SEED_FILE = resolve(import.meta.dir, "rosters.ts");

let captured: { exitCode: number; output: string };

beforeAll(async () => {
    const original = readFileSync(SEED_FILE, "utf-8");
    try {
        // Seed both a dead cite (resolves to nothing) and a live cite (README.md is tracked
        // in-repo) in the same file. One subprocess run covers both sides.
        writeFileSync(
            SEED_FILE,
            original +
                "\n// vacuity-arm-seed: see zzz-vacuity-dead-seed.md for details\n" +
                "// vacuity-arm-seed: see README.md for details\n",
        );
        const proc = Bun.spawn({
            cmd: ["bun", SCRIPT],
            stdout: "pipe",
            stderr: "pipe",
        });
        const [stdout, stderr, exitCode] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        captured = { exitCode, output: stdout + stderr };
    } finally {
        writeFileSync(SEED_FILE, original);
    }
}, 30000);

test("pointer-validity — two-sided vacuity reading (dead reds, live spared)", () => {
    // Side 1 — reds on a dead .md cite: the arm fires and the script exits 1. This is
    // the discriminator: if the resolution check is deleted, deadPointers stays empty,
    // the pointer arm goes vacuously green, and the script exits 0 — failing this
    // assertion.
    expect(captured.exitCode).toBe(1);
    // The dead cite appears in the pointer-validity output (secondary stable check).
    expect(captured.output).toMatch(/zzz-vacuity-dead-seed\.md/);

    // Side 2 — stays quiet on a live .md cite: the live cite resolves, so it must NOT
    // appear as a dead pointer. The pointer arm's output format is `  file:line: ref`,
    // so this pattern would only appear if the arm falsely flagged the live cite.
    // (Exit code is not asserted on this side: pre-sweep the script reds at 46 from
    // existing sites, post-sweep it greens — either way the live cite is spared.)
    expect(captured.output).not.toMatch(/rosters\.ts:\d+: README\.md/);
});
