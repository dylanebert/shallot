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
//
// THIS SITE COUNTS AS UNARMED. A structural pin matches a commented-out guard as readily as a live
// one — measured: commenting the guard out leaves this arm GREEN. A source-text match cannot tell a
// guard from a comment, which is this spec's own defect class, so this file is a note and not
// coverage. Arming it behaviorally needs a hermetic build fixture; that cost was not paid here.

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { RUM_ENV_SNIPPET, RUM_ENV_SNIPPET_STAGING } from "../site/rum-config";
import {
    datadogInitSnippet,
    rewriteSiteDependencies,
    shallotDependencies,
    workspaceExtensionDependencies,
} from "./build-site";

const src = readFileSync(resolve(import.meta.dir, "build-site.ts"), "utf8");

// S1 (staging build mode) — datadogInitSnippet is a pure function, so its mode selection is
// armed behaviorally rather than by source-text match, unlike the arms below.
test("build-site — datadogInitSnippet selects the env constant by mode, mutually exclusive", () => {
    const prod = datadogInitSnippet("prod");
    const staging = datadogInitSnippet("staging");
    expect(prod).toContain(RUM_ENV_SNIPPET);
    expect(prod).not.toContain(RUM_ENV_SNIPPET_STAGING);
    expect(staging).toContain(RUM_ENV_SNIPPET_STAGING);
    expect(staging).not.toContain(RUM_ENV_SNIPPET);
    // no mode arg defaults to prod — the prod build path stays byte-unchanged for a caller that
    // never learns about `--staging`
    expect(datadogInitSnippet()).toBe(prod);
});

test("build-site — discovers workspace extensions and rewrites them in both engine modes", () => {
    const authored = {
        dependencies: {
            "@dylanebert/shallot": "workspace:*",
            "@dylanebert/shallot-wave": "workspace:*",
            "@dylanebert/shallot-fixed": "1.2.3",
            typegpu: "~0.12.4",
        },
    };
    expect(shallotDependencies(authored)).toEqual([
        ["@dylanebert/shallot", "workspace:*"],
        ["@dylanebert/shallot-fixed", "1.2.3"],
        ["@dylanebert/shallot-wave", "workspace:*"],
    ]);
    expect(workspaceExtensionDependencies(authored)).toEqual(["@dylanebert/shallot-wave"]);

    for (const enginePin of ["0.8.0", "file:/tmp/shallot.tgz"]) {
        const pkg = structuredClone(authored);
        rewriteSiteDependencies(
            pkg,
            enginePin,
            new Map([["@dylanebert/shallot-wave", "file:/tmp/wave.tgz"]]),
        );
        expect(pkg.dependencies["@dylanebert/shallot"]).toBe(enginePin);
        expect(pkg.dependencies["@dylanebert/shallot-wave"]).toBe("file:/tmp/wave.tgz");
        expect(pkg.dependencies["@dylanebert/shallot-fixed"]).toBe("1.2.3");
    }
});

test("build-site — refuses to leave a discovered workspace extension unpinned", () => {
    expect(() =>
        rewriteSiteDependencies(
            { dependencies: { "@dylanebert/shallot-wave": "workspace:*" } },
            "0.8.0",
            new Map(),
        ),
    ).toThrow("no packed tarball");
});

// The staging pack-and-`file:`-pin mechanism itself needs a real `bun pm pack` + `bun install`,
// which this file's other arms already document as too slow/non-hermetic for a source-text
// match (see the file header). Structural pins only, same discipline and the same caveat: a
// source-text match cannot tell a live guard from a commented-out one.
test("build-site --staging — packs the engine once and pins every demo to the tarball, not the version", () => {
    expect(src).toMatch(/bun pm pack/);
    // the per-demo rewrite must use the mode-selected `enginePin`, not the bare release `version`
    // — this is the line staging mode depends on to avoid ever pinning a prod release
    expect(src).toMatch(/@dylanebert\/shallot["']\]\s*=\s*enginePin/);
});

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
