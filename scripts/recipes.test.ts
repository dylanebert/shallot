// S3 arm — scripts/recipes.ts unknown-selector exit + empty-glob guard
//
// Two invariants pinned:
//   (1) An unknown --recipe selector exits nonzero (exit 2), ahead of the skipReason() display
//       skip. Before the S1 fix, the unknown selector was behind the skip, so a headless seat
//       could never observe it.
//   (2) An empty derived population (the glob matches nothing) reads red (exit 1), not vacuous
//       green. The S1 fix derived RECIPES from a glob and added the empty-guard. The empty-glob
//       guard is a structural check (the real glob matches real recipes, so it can't be triggered
//       hermetically), but the unknown-selector guard is behavioral (run as subprocess).

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const SCRIPT = resolve(import.meta.dir, "recipes.ts");
const src = readFileSync(SCRIPT, "utf8");

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

test("recipes — empty derived population reads red (the empty-glob guard is present)", () => {
    // The glob `examples/recipes/*/src/smoke.ts` derives the recipe list. If it matches nothing,
    // the script must exit 1, not 0 "PASS: recipe smoke green". The guard is:
    //   if (!only && RECIPES.length === 0) { ... process.exit(1); }
    // This is a structural check because the real glob matches real recipes — the empty case
    // can't be triggered hermetically without removing recipe dirs (not hermetic).
    expect(src).toMatch(/RECIPES\.length\s*===\s*0/);
    expect(src).toMatch(/process\.exit\(1\)/);
});

test("recipes — the recipe list is derived from a glob, not a hand list", () => {
    // The fix replaced a hand-maintained list with a glob derivation so a new smoke is gated
    // by construction — no hand list to drift.
    expect(src).toMatch(/Glob.*smoke\.ts/);
});
