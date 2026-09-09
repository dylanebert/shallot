/** The declared escapes from the distribution boundary, and their bounded reasons. `check-boundary.ts`
 *  reads this table both ways: an undeclared escape refuses, and a declaration naming a specifier no
 *  live file imports refuses too, so the ledger cannot outlive what it excuses. */

/** One declared reach from the package's own tooling (`bin/**`) into engine source that the export map
 *  does not publish. Keyed `<repo-relative file> <specifier>` — per call site, not per module, so moving
 *  one reader does not silently license the others. */
export const TOOLING_SEAMS: Record<string, string> = {
    'packages/shallot-cli/bin/bench.test.ts "../../../scripts/bench"':
        "the CLI parser's production repository runner",
    'packages/shallot-cli/bin/recipe.test.ts "../../create-shallot/index"':
        "compare recipe scaffolding against the sole scaffold source",
    'packages/shallot-cli/bin/create-shallot.test.ts "../../create-shallot/index"':
        "exercise the sole scaffold source",
    'packages/shallot-cli/bin/verify.test.ts "../../../examples/gym/src/scenarios/timeouts"':
        "assert the actual gym scenario budget bindings",
    'packages/shallot-cli/bin/verify.test.ts "../../../scripts/bench"':
        "exercise production batch/bench composition",
    'packages/shallot-cli/bin/verify.test.ts "../../../scripts/boot-cost"':
        "exercise the diagnostic consumer's parser",
    'packages/shallot-cli/bin/verify.test.ts "../../../scripts/install-test"':
        "exercise the packed-consumer diagnostic binding",
    'packages/shallot-cli/bin/verify.test.ts "../../../scripts/verify"':
        "the repository transport's result types",
    'packages/shallot-cli/bin/verify.test.ts "../../../site/rum-sampler"':
        "differential against the independently executed page sampler",
    // verify's node-side diagnostics. These are tool-facing readings with no author-facing contract, so
    // they stay unpublished rather than growing the surface.
    'packages/shallot-cli/bin/verify.ts "../../shallot-runtime/src/engine/runtime/gpu"':
        "adapter identity for the run's hardware line",
    'packages/shallot-cli/bin/verify.ts "../../shallot-runtime/src/engine/runtime/log"':
        "the log predicate the console reader shares",
    'packages/shallot-cli/bin/verify.ts "../../shallot-runtime/src/extras/profile/benchmark"':
        "the benchmark measurement shape the --json envelope carries",
    'packages/shallot-cli/bin/verify.ts "../../shallot-runtime/src/harness/degraded-boot"':
        "the degraded-boot predicate, published only through ./harness's barrel",
};

/** Files allowed to build a module specifier at runtime rather than name it literally, each with the
 *  bound that keeps it readable. Every other computed `import()`/`require()` refuses: a specifier this
 *  reader cannot resolve is a hole in the source cone, not a detail. */
export const COMPUTED_LOADERS: Record<string, string> = {
    "packages/shallot-tumble/scripts/gen-tumble-sample-golds.ts":
        "Frozen mint recipe: four loads name the absent retired sample base/registry and this owner's body/index files; refuses before loading without that checkout. Never part of shipped solver source.",
    "packages/shallot-tumble/src/standard/tumble/engine/pool.ts":
        "the Node-only branch loads the fixed node:worker_threads specifier with vite-ignore; the browser branch creates an embedded Blob worker",
    "packages/shallot-cli/src/project/command.ts":
        "eagerly resolves every enabled entry from the project root, then imports those resolved identities",
    "packages/shallot-cli/src/project/command.test.ts":
        "the bare-process isolation fixture imports the named command entry under test",
    "examples/showcase/roads/test/edit-safety.playwright.ts":
        "browser-evaluated /src/ URLs into Roads' own src/, served by playwright.config.ts webServer shallot dev .; non-literal spelling leaves imports to the browser instead of Playwright's CJS transform",
    "examples/showcase/roads/test/touch-smoke.playwright.ts":
        "browser-evaluated /src/ URLs into Roads' own src/, served by playwright.config.ts webServer shallot dev .; non-literal spelling leaves imports to the browser instead of Playwright's CJS transform",
    "packages/shallot-cli/bin/features.ts":
        "preflights all enabled project-root entry identities before engine/local evaluation, then reads required features from those identities",
    "packages/shallot-cli/bin/bun-native.ts":
        "loads the downloaded native projection, after its sha256 matches the pinned hash",
    "packages/shallot-cli/bin/verify.ts":
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
