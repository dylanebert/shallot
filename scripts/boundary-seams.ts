/** The declared escapes from the distribution boundary, and their bounded reasons. `check-boundary.ts`
 *  reads this table both ways: an undeclared escape refuses, and a declaration naming a specifier no
 *  live file imports refuses too, so the ledger cannot outlive what it excuses. */

/** One declared reach from the package's own tooling (`bin/**`) into engine source that the export map
 *  does not publish. Keyed `<repo-relative file> <specifier>` — per call site, not per module, so moving
 *  one reader does not silently license the others. */
export const TOOLING_SEAMS: Record<string, string> = {
    'packages/shallot-tooling/bin/bench.test.ts "../../../scripts/bench"':
        "the CLI parser's production repository runner",
    'packages/shallot-tooling/bin/recipe.test.ts "../../create-shallot/index"':
        "compare recipe scaffolding against the sole scaffold source",
    'packages/shallot-tooling/bin/create-shallot.test.ts "../../create-shallot/index"':
        "exercise the sole scaffold source",
    'packages/shallot-tooling/bin/verify.test.ts "../../../examples/gym/src/scenarios/timeouts"':
        "assert the actual gym scenario budget bindings",
    'packages/shallot-tooling/bin/verify.test.ts "../../../scripts/bench"':
        "exercise production batch/bench composition",
    'packages/shallot-tooling/bin/verify.test.ts "../../../scripts/boot-cost"':
        "exercise the diagnostic consumer's parser",
    'packages/shallot-tooling/bin/verify.test.ts "../../../scripts/install-test"':
        "exercise the packed-consumer diagnostic binding",
    'packages/shallot-tooling/bin/verify.test.ts "../../../scripts/verify"':
        "the repository transport's result types",
    'packages/shallot-tooling/bin/verify.test.ts "../../../site/rum-sampler"':
        "differential against the independently executed page sampler",
    // verify's node-side diagnostics. These are tool-facing readings with no author-facing contract, so
    // they stay unpublished rather than growing the surface.
    'packages/shallot-tooling/bin/verify.ts "../../shallot/src/engine/runtime/gpu"':
        "adapter identity for the run's hardware line",
    'packages/shallot-tooling/bin/verify.ts "../../shallot/src/engine/runtime/log"':
        "the log predicate the console reader shares",
    'packages/shallot-tooling/bin/verify.ts "../../shallot/src/extras/profile/benchmark"':
        "the benchmark measurement shape the --json envelope carries",
    'packages/shallot-tooling/bin/verify.ts "../../shallot/src/harness/degraded-boot"':
        "the degraded-boot predicate, published only through ./harness's barrel",
};

/** Files allowed to build a module specifier at runtime rather than name it literally, each with the
 *  bound that keeps it readable. Every other computed `import()`/`require()` refuses: a specifier this
 *  reader cannot resolve is a hole in the source cone, not a detail. */
export const COMPUTED_LOADERS: Record<string, string> = {
    "packages/shallot-tooling/src/project/command.ts":
        "loads enabled manifest plugins only after project planning resolves their paths",
    "packages/shallot-tooling/src/project/command.test.ts":
        "the bare-process isolation fixture imports the named command entry under test",
    "examples/showcase/roads/test/edit-safety.playwright.ts":
        "browser-evaluated /src/ URLs into Roads' own src/, served by playwright.config.ts webServer shallot dev .; non-literal spelling leaves imports to the browser instead of Playwright's CJS transform",
    "examples/showcase/roads/test/touch-smoke.playwright.ts":
        "browser-evaluated /src/ URLs into Roads' own src/, served by playwright.config.ts webServer shallot dev .; non-literal spelling leaves imports to the browser instead of Playwright's CJS transform",
    "packages/shallot-tooling/bin/features.ts":
        "loads the project's own manifest-declared local plugins to read their feature declarations",
    "packages/shallot-tooling/bin/bun-native.ts":
        "loads the downloaded native projection, after its sha256 matches the pinned hash",
    "packages/shallot-tooling/bin/verify.ts":
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
