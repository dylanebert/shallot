// S3 arm — scripts/build-site.ts single-demo build preserves the roster and the index
//
// Invariant: a single-demo build (--demo <slug>) only clears that demo's slot (not all of
// out/site), and the index always lists the full roster. Before the S1 fix, --demo unconditionally
// rmSync'd all of out/site then built only the filtered roster — a single-demo build destroyed
// and under-emitted the site. The fix: --demo only clears that demo's slot, and siteIndex is
// always called with ROSTER (the full roster), not `demos` (the filtered list).
//
// build-site.ts needs `bun install` + `npm` to build (not hermetic), so this arm reads the source
// and asserts the two structural properties that pin the invariant.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const src = readFileSync(resolve(import.meta.dir, "build-site.ts"), "utf8");

test("build-site --demo — only clears that demo's slot, not all of out/site", () => {
    // The fix: when `only` is set, rmSync targets `resolve(outDir, only)` (that demo's slot),
    // not `outDir` (the entire site). Before the fix, --demo unconditionally rmSync'd outDir.
    expect(src).toMatch(/if\s*\(only\)\s*\{[^}]*rmSync.*resolve\(outDir,\s*only\)/s);
    // The else branch (no --demo) clears the full outDir.
    expect(src).toMatch(/else\s*\{[^}]*rmSync.*outDir/s);
});

test("build-site — the index always lists the full roster (ROSTER, not demos)", () => {
    // The fix: siteIndex is called with ROSTER (the full roster), not `demos` (the filtered list).
    // Before the fix, a single-demo build's index only listed the filtered demos.
    // The call site is `siteIndex(ROSTER, version, refShort)` — check the call, not the function
    // definition (whose parameter is named `demos`).
    expect(src).toMatch(/siteIndex\(ROSTER,\s*version/);
    // Ensure the call does NOT pass the filtered `demos` variable to siteIndex.
    // The filtered list is `const demos = only ? ROSTER.filter(...) : ROSTER` — the call must use ROSTER.
    expect(src).not.toMatch(/siteIndex\(demos,\s*version/);
});
