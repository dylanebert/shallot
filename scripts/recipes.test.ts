// S3 arm — scripts/recipes.ts unknown-selector exit + empty-glob guard
//
// Two invariants pinned:
//   (1) An unknown --recipe selector exits nonzero (exit 2), ahead of the skipReason() display
//       skip. Before the S1 fix, the unknown selector was behind the skip, so a headless seat
//       could never observe it. This is behavioral: the arm runs the real script as a subprocess.
//   (2) An empty derived population (the glob matches nothing) reads red (exit 1), not vacuous
//       green. The S1 fix derived RECIPES from a glob and added the empty-guard. The real glob
//       matches real recipes, so the empty case can't be triggered hermetically — the guard is
//       extracted into populationError(), a pure seam the arm exercises directly with an empty
//       recipe list.

import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { populationError } from "./recipes";

const SCRIPT = resolve(import.meta.dir, "recipes.ts");

test("recipes — unknown --recipe exits 2 (ahead of the display skip)", async () => {
    const proc = Bun.spawn({
        cmd: ["bun", SCRIPT, "--recipe", "bogus"],
        stdout: "pipe",
        stderr: "pipe",
    });
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    const out = stdout + stderr;
    expect(exitCode).toBe(2);
    expect(out).toContain("no recipe");
});

test("recipes — empty derived population reads red (behavioral via the pure seam)", () => {
    // The empty-glob guard: if the glob matches nothing, the script must exit 1, not 0
    // "PASS: recipe smoke green". populationError returns a non-null message for an empty
    // population — the condition that triggers process.exit(1) in main().
    const err = populationError([], undefined);
    expect(err).not.toBeNull();
    expect(err).toContain("no recipes");
});

test("recipes — unknown selector reads red (behavioral via the pure seam)", () => {
    // The unknown-selector guard: populationError returns a non-null message for an unknown
    // selector — the condition that triggers process.exit(2) in main().
    const err = populationError(["moving-platform", "joints"], "bogus");
    expect(err).not.toBeNull();
    expect(err).toContain("no recipe");
});

test("recipes — a known selector with a populated list proceeds (green direction)", () => {
    // The green direction: a known selector with a populated list returns null (no error).
    const err = populationError(["moving-platform", "joints"], "joints");
    expect(err).toBeNull();
});

test("recipes — a populated list with no selector proceeds (green direction)", () => {
    const err = populationError(["moving-platform", "joints"], undefined);
    expect(err).toBeNull();
});

test("recipes — importing the module runs no gate side effect (import.meta.main guard)", async () => {
    // Before the S2a fix, `main()` ran unguarded at module scope, so any importer —
    // this very test file included — fired the real GPU dynamics smoke as a side effect.
    // A subprocess that only imports the module (never calls it as a script) must exit
    // clean and print none of the smoke's own output.
    const proc = Bun.spawn({
        cmd: ["bun", "-e", `import(${JSON.stringify(SCRIPT)})`],
        stdout: "pipe",
        stderr: "pipe",
    });
    const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    const out = stdout + stderr;
    expect(exitCode).toBe(0);
    expect(out).not.toContain("Running recipe dynamics smoke");
});
