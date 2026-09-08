/** The declared escapes from the distribution boundary, and their bounded reasons. `check-boundary.ts`
 *  reads this table both ways: an undeclared escape refuses, and a declaration naming a specifier no
 *  live file imports refuses too, so the ledger cannot outlive what it excuses. */

/** One declared reach from the package's own tooling (`bin/**`) into engine source that the export map
 *  does not publish. Keyed `<repo-relative file> <specifier>` — per call site, not per module, so moving
 *  one reader does not silently license the others. */
export const TOOLING_SEAMS: Record<string, string> = {
    // the project plan/discovery/resolution the CLI and TUI share. No published seam exposes it yet;
    // `shallot-release-boundaries` A1 extracts one and these entries retire with it.
    'packages/shallot/bin/features.ts "../src/project/generate"':
        "shared project plan, pending the A1 project-host seam",
    'packages/shallot/bin/features.ts "../src/project/manifest"':
        "manifest resolution, pending the A1 project-host seam",
    'packages/shallot/bin/native.ts "../src/project/manifest"':
        "manifest resolution, pending the A1 project-host seam",
    'packages/shallot/bin/tui.ts "../src/project/assets"':
        "manifest read, pending the A1 project-host seam",
    'packages/shallot/bin/tui.ts "../src/project/engine"':
        "subpath plugin modules, pending the A1 project-host seam",
    'packages/shallot/bin/tui.ts "../src/project/generate"':
        "shared project plan, pending the A1 project-host seam",
    // the TUI's own engine reach: the barrel type only. `../src` (the published `.` target) is what it
    // actually loads at runtime.
    'packages/shallot/bin/tui.ts "../src/engine"': "Plugin type for the loaded manifest plugins",
    // verify's node-side diagnostics. These are tool-facing readings with no author-facing contract, so
    // they stay unpublished rather than growing the surface.
    'packages/shallot/bin/verify.ts "../src/engine/runtime/gpu"':
        "adapter identity for the run's hardware line",
    'packages/shallot/bin/verify.ts "../src/engine/runtime/log"':
        "the log predicate the console reader shares",
    'packages/shallot/bin/verify.ts "../src/extras/profile/benchmark"':
        "the benchmark measurement shape the --json envelope carries",
    'packages/shallot/bin/verify.ts "../src/harness/degraded-boot"':
        "the degraded-boot predicate, published only through ./harness's barrel",
};

/** Files allowed to build a module specifier at runtime rather than name it literally, each with the
 *  bound that keeps it readable. Every other computed `import()`/`require()` refuses: a specifier this
 *  reader cannot resolve is a hole in the source cone, not a detail. */
export const COMPUTED_LOADERS: Record<string, string> = {
    "examples/showcase/roads/test/edit-safety.playwright.ts":
        "browser-evaluated /src/ URLs into Roads' own src/, served by playwright.config.ts webServer shallot dev .; non-literal spelling leaves imports to the browser instead of Playwright's CJS transform",
    "examples/showcase/roads/test/touch-smoke.playwright.ts":
        "browser-evaluated /src/ URLs into Roads' own src/, served by playwright.config.ts webServer shallot dev .; non-literal spelling leaves imports to the browser instead of Playwright's CJS transform",
    "packages/shallot/bin/tui.ts":
        "loads the project's own manifest-declared plugin files, resolved under the project root",
    "packages/shallot/bin/features.ts":
        "loads the project's own manifest-declared local plugins to read their feature declarations",
    "packages/shallot/bin/bun-native.ts":
        "loads the downloaded native projection, after its sha256 matches the pinned hash",
    "packages/shallot/bin/verify.ts":
        "loads the consumer project's own installed playwright, resolved from its package root",
};

/** Directories that carry a `package.json` but are deliberately not workspaces: install-time fixtures and
 *  eval harnesses a workspace install must not hoist. Listed so the completeness rule can tell a fixture
 *  apart from a workspace someone forgot to declare; an entry naming a directory that no longer exists
 *  refuses. */
export const NON_WORKSPACE_PACKAGES: Record<string, string> = {
    "evals/harness": "eval harness installed per run, never part of the repo workspace graph",
    "scripts/install-test/widget":
        "the synthetic malformed-install fixture the packed install gate publishes into a temp tree",
    "scripts/install-test/compat-0.9.5/scaffold":
        "the project create-shallot@0.9.5 emitted, frozen byte-for-byte as the compatibility baseline",
};
