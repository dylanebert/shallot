// S3 arm — evals/setup.ts stripTarball
//
// Invariant: the --bare arm strips AGENTS.md and MIGRATION.md from the tarball (not just
// examples/). The S2 fix added rmSync calls for both files inside stripTarball. stripTarball
// is a local function in setup.ts (not exported), and setup.ts is a top-level script that
// runs `bun pm pack` + `bun install` — neither is hermetic. So this arm reads the source and
// asserts the rmSync calls for both files are present, pinning the invariant structurally.
//
// A guard that is plainly absent (the fix never landed) may be reported as absent; this arm
// reds if either rmSync is missing, which is the signal the fix was reverted or never shipped.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const src = readFileSync(resolve(import.meta.dir, "setup.ts"), "utf8");

test("stripTarball — removes AGENTS.md from the tarball", () => {
    // The --bare arm withholds all shipped context. AGENTS.md is agent documentation, not
    // code/JSDoc/product-workflow, so it must be stripped at the tarball level (deleting only
    // from node_modules doesn't hold — bun add re-extracts).
    expect(src).toContain('rmSync(join(ex, "package/AGENTS.md")');
});

test("stripTarball — removes MIGRATION.md from the tarball", () => {
    // MIGRATION.md is migration documentation, same class as AGENTS.md — stripped at the tarball
    // so the context-withheld claim holds.
    expect(src).toContain('rmSync(join(ex, "package/MIGRATION.md")');
});

test("stripTarball — still removes examples/ (the original S2 fix's target)", () => {
    // The original fix removed examples/; the S2 extension added AGENTS.md + MIGRATION.md.
    // Both must be present — removing one must not silently drop the other.
    expect(src).toContain('rmSync(join(ex, "package/examples")');
});
