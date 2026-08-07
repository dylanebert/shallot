// The stage 1 deliverable of `specs/shallot-cli-tests.md`: a registry over the CLI/toolchain layer's
// tier-of-truth ("Classify by tier-of-truth, then close what nothing reaches. No coverage floor" — the
// spec's Locked decision), asserted both directions against a mechanically walked file population
// (`examples/gym/src/scenarios/coverage.ts`'s pattern, re-housed rather than imported so this package
// doesn't reach into `examples/gym`). Granularity is the file, never a hand-declared region — the spec's
// own words: "Per-region detail lives in the row's prose reason, and the spec does not claim per-region
// completeness is asserted, because it isn't." So exactly one row per walked file; a file whose content
// splits across tiers still gets one row, whose `arm` is the *weakest* tier any part of the file's
// content reaches (unit/tier only when the whole file is genuinely reached; extract when the outstanding
// work is pulling pure logic out of an effectful body; gap otherwise — untested today for any other
// reason, including "by decision, permanently"). The row's reason carries the detail a second row would
// otherwise have held: what's covered and by what, plus what isn't and who closes it.

/** the four tiers of truth a row may claim, per the spec's Locked decision. */
export type Arm = "unit" | "tier" | "extract" | "gap";

export interface CoverageRow {
    /** path relative to the shallot package root, matching a file in {@link CLI_POPULATION_GLOBS}. */
    file: string;
    arm: Arm;
    /** the load-bearing property this row claims — never a structural shape ("barrel re-export, no
     *  logic of its own" was false twice in three uses, per `testing.md`). A `tier` row names the
     *  command and the flow/scenario; an `extract` row names the extraction; a `gap` row names its
     *  occupants by name. */
    reason: string;
    /** the later sub-stage (2-6) that discharges this row, when one owns it. Absent means either
     *  already-closed (`unit`/`tier`) or a permanent, undischarged `gap`. */
    stage?: 2 | 3 | 4 | 5 | 6;
}

/** the population globs: every non-test `.ts` file the spec's Goal names as the CLI/toolchain layer. */
export const CLI_POPULATION_GLOBS: readonly string[] = [
    "packages/shallot/bin/*.ts",
    "packages/shallot/src/project/*.ts",
    "packages/create-shallot/index.ts",
];

/** converts a `dir/*.ts`-style glob to a RegExp, re-housed from `coverage.ts` rather than imported —
 *  `examples/gym` is off limits to this package. `*` matches within one path segment (no `**` support:
 *  every glob above is one directory deep, so cross-segment matching is untested surface). */
export function globToRegExp(glob: string): RegExp {
    let pattern = "";
    for (let i = 0; i < glob.length; i++) {
        const c = glob[i];
        if (c === "*") {
            pattern += "[^/]*";
        } else if ("\\^$.|?+()[]{}".includes(c)) {
            pattern += `\\${c}`;
        } else {
            pattern += c;
        }
    }
    return new RegExp(`^${pattern}$`);
}

/** the four test-tier suffixes `testing.md` names (`.test.ts`, `.oracle.ts`, `.probes.ts`, `.lab.ts`) —
 *  a test file, at whatever tier, is not itself a population member needing a row; it's the instrument,
 *  not the thing measured. Excluding only `.test.ts` would leave `verify.probes.ts` in the population as
 *  if it were untested production code, when it is itself the by-path browser gate for two constants
 *  `verify.test.ts` already sentinels. */
const TEST_TIER_SUFFIXES = /\.(test|oracle|probes|lab)\.ts$/;

/** walks the real filesystem under `root` (the shallot repo root) and returns every path (relative to
 *  `root`, forward-slashed) matching {@link CLI_POPULATION_GLOBS}, excluding every test-tier suffix. */
export async function cliPopulation(root: string): Promise<string[]> {
    const globs = CLI_POPULATION_GLOBS.map(globToRegExp);
    const out: string[] = [];
    for await (const path of new Bun.Glob("**/*.ts").scan({ cwd: root })) {
        if (TEST_TIER_SUFFIXES.test(path)) continue;
        if (globs.some((re) => re.test(path))) out.push(path);
    }
    return out;
}

export type FindingKind =
    | "file-missing-row"
    | "row-names-unwalked-file"
    | "file-has-multiple-rows"
    | "row-missing-reason"
    | "undischarged-extract-missing-stage";

export interface Finding {
    kind: FindingKind;
    detail: string;
}

/** the both-directions completeness check: every walked file has exactly one row, every row names a
 *  walked file. A second row on a file is a finding (`file-has-multiple-rows`) — granularity is the
 *  file, never the region, so a file split into two rows is exactly the region-level split the spec's
 *  Locked decision forbids (it also silently weakens the registry: a reviewer proved this by deleting
 *  one row of a multi-row file and finding the check still green). Also checks every row carries a
 *  non-empty reason (a row with no reason is a structural checklist entry, not a load-bearing claim),
 *  and that every `extract` row names the stage that will discharge it (per the Locked decision: "every
 *  extract row converts to unit by close or moves to gap with its name recorded" — an extract with no
 *  owning stage is a promise nobody is keeping). Pure over the registry + population, so
 *  `cli-coverage.test.ts` red-proves it against fixtures with no filesystem.
 */
export function checkCliCoverage(
    rows: readonly CoverageRow[],
    population: readonly string[],
): Finding[] {
    const findings: Finding[] = [];
    const walked = new Set(population);
    const countByFile = new Map<string, number>();

    for (const row of rows) {
        if (!walked.has(row.file)) {
            findings.push({ kind: "row-names-unwalked-file", detail: row.file });
        }
        if (row.reason.trim() === "") {
            findings.push({ kind: "row-missing-reason", detail: row.file });
        }
        if (row.arm === "extract" && row.stage === undefined) {
            findings.push({ kind: "undischarged-extract-missing-stage", detail: row.file });
        }
        countByFile.set(row.file, (countByFile.get(row.file) ?? 0) + 1);
    }

    for (const file of population) {
        if (!countByFile.has(file)) {
            findings.push({ kind: "file-missing-row", detail: file });
        }
    }

    for (const [file, count] of countByFile) {
        if (count > 1) {
            findings.push({ kind: "file-has-multiple-rows", detail: `${file} (${count} rows)` });
        }
    }

    return findings;
}

/**
 * the registry itself. Exactly one row per file in {@link CLI_POPULATION_GLOBS}. Verified against the
 * file it names, not copied forward from the spec's own prose (`testing.md`: "a check is evidence only
 * if you've seen it fail" — the same discipline applies to a classification, not only a test).
 */
export const CLI_COVERAGE: readonly CoverageRow[] = [
    {
        file: "packages/shallot/bin/build.ts",
        arm: "extract",
        reason:
            "synthIndex (the synthesized <html> entry a manifest project owns none of) is directly " +
            "asserted by dev.test.ts's `describe(\"synthIndex\")` block. buildWeb's effectful body — the " +
            "synthesized entry write/cleanup, the viteBuild call, the public-dir copy — runs end to end on " +
            'every `bun run test:install` ("shallot build exits clean", a manifest project with a subpath ' +
            "engine plugin, a local plugin, the wasm asset). What isn't closed: buildWeb's inlined " +
            "build-config object literal (root/base/plugins/build) is the one place build.ts and dev.ts's " +
            "devConfig disagree in shape — dev.ts already factored the equivalent into an exported, " +
            "unit-tested devConfig. Stage 2 folds buildWeb's inline literal into a matching exported " +
            "buildConfig.",
        stage: 2,
    },
    {
        file: "packages/shallot/bin/cli.ts",
        arm: "extract",
        reason:
            "reached by nothing under `bun test` today — no test file imports cli.ts (verified: no " +
            '`from "./cli"` / `from "../cli"` in any *.test.ts). `bun run test:install` only shells out to ' +
            "the installed CLI as a subprocess (`node_modules/@dylanebert/shallot/bin/cli.ts`), which " +
            "proves the binary runs but attributes no coverage to this module. Stage 2 factors the flag-" +
            "parse loop + subcommand dispatch into a pure parseCliArgs(raw), the shape parseVerifyArgs " +
            "already models in verify.ts.",
        stage: 2,
    },
    {
        file: "packages/shallot/bin/dev.ts",
        arm: "tier",
        reason:
            "devConfig is directly unit-tested by dev.test.ts (plugin order, fs.allow set, the open " +
            "default, COOP/COEP headers). startDev and synthIndexPlugin's middleware — the rest of this " +
            "file's content — are exercised end to end by `bun run test:install`'s \"shallot dev boots a " +
            'server" rung, which requests `/` against a booted dev server over an installed project. No ' +
            "part of this file is reached by nothing, so no stage owns further work here.",
    },
    {
        file: "packages/shallot/bin/features.ts",
        arm: "unit",
        reason:
            "verdict's four target/portable branches and requiredFeatures' plugin-to-feature resolution " +
            "(a physics-only project, a Profile project needing timestamp-query) are both directly " +
            "asserted by features.test.ts — every exported function, every branch.",
    },
    {
        file: "packages/shallot/bin/gpu-globals.ts",
        arm: "gap",
        reason:
            "installGpuGlobals's write branch (`if (!(name in globalThis))`) never actually fires under " +
            "`bun test`: bun-webgpu's setupGlobals(), preloaded by tests/setup.ts for every run (verified: " +
            "`bun -e 'console.log(typeof GPUBufferUsage)'` prints undefined without the preload, a real " +
            "value with it), already defines GPUBufferUsage/GPUShaderStage/etc. as real globals, so the " +
            "guard short-circuits on every call this repo's own suite can make — only the shipped CLI's " +
            "plain-`bun` path (no bun-webgpu installed) takes the write. No stage in the spec's Approach " +
            "names gpu-globals.ts; occupant: installGpuGlobals.",
    },
    {
        file: "packages/shallot/bin/native.ts",
        arm: "gap",
        reason:
            "nativeOutDir (exported, pure, currently zero test callers), cargoTarget, a pure " +
            "resolveCargoInvocation carved out of cargoBuild's portable/target/WSL decision table, " +
            "trimMacLocales' delete-predicate, macHelperBin, macInfoPlist's two independent branches " +
            "(helper/LSUIElement, icon key — both are string-template conditionals with no external " +
            "effect), and findCefDir's CEF_PATH/build-tree search are pure or temp-dir-testable logic " +
            "stage 5 extracts and tests. What stays permanently untested, by decision, past stage 5: " +
            "devShellBuild (declared manual — its PowerShell DevShell-entry template would be restated " +
            "byte-for-byte by any test) and the subprocess-orchestrating bulk that calls real " +
            "cargo/sips/iconutil/strip (cargoBuild's execSync/spawnSync arms, bundleNativeMac, " +
            "bundleNativeWindows, bundleNativeLinux, ensureIcon, tryStrip, copyLocale, copyCefLibs, " +
            "copyCefDlls, prepareMacIcon, convertIconToIcns) — frozen out of the `unit` arm by the Locked " +
            'decision itself ("native.ts extracts and declares; it gets no real-cargo gate"). The one ' +
            "closed rung against this gap is the missing-crate guard (rust/window isn't shipped in " +
            "package.json's `files`, so an installed `--target <os>` build dies on a raw ENOENT instead of " +
            "a named diagnostic), asserted in the install gate rather than here.",
        stage: 5,
    },
    {
        file: "packages/shallot/bin/recipe.ts",
        arm: "unit",
        reason:
            "listRecipes, occupied, pinEngine's every workspace-marker form, and runRecipe's every branch " +
            "(absent corpus, bare list, unknown name, copy+pin, doc+tsconfig emission, refuse-nonempty) " +
            "are all directly asserted by recipe.test.ts against a temp-dir recipe corpus.",
    },
    {
        file: "packages/shallot/bin/run.ts",
        arm: "extract",
        reason:
            "reached by nothing under any tier in this repo today — no test file imports run.ts, and " +
            "neither test:install, bench, flows, nor recipes ever invokes `shallot run` (verified: no " +
            'grep hit for runProject/"shallot run" across those scripts). Its target dispatch (web/mac/' +
            "linux/windows) and each branch's env/command composition (LD_LIBRARY_PATH, the wslpath + " +
            "powershell.exe windows launch) are presently inline inside the effectful spawnSync calls. " +
            "Stage 2 yields the dispatch selection and env/command composition as pure functions — " +
            "explicitly not the spawn arms themselves, which stay unreached (opening a real native window " +
            "or preview server isn't a headless-testable act, the same boundary native.ts's bundle bodies " +
            "sit behind).",
        stage: 2,
    },
    {
        file: "packages/shallot/bin/scaffold.ts",
        arm: "unit",
        reason:
            'every export is directly asserted by recipe.test.ts\'s "scaffold pointer is one source" ' +
            'block: recipeDoc("orbit-camera") is called and checked to contain ENGINE_REFERENCE; ' +
            "CLAUDE_IMPORT is asserted with `toBe`; and RECIPE_TSCONFIG is asserted indirectly but " +
            "genuinely — runRecipe writes it verbatim to the copied recipe's tsconfig.json, and " +
            'recipe.test.ts\'s "emits the agent-surface pointer + a standalone tsconfig" test reads that ' +
            'file back and asserts `compilerOptions.types` contains "@webgpu/types".',
    },
    {
        file: "packages/shallot/bin/toolchain.ts",
        arm: "gap",
        reason:
            "flattenPlugins' array/promise/falsy-entry walk and composeViteConfig's merge + name-based " +
            "drop are both directly asserted by toolchain.test.ts. isProject, requireProject, and " +
            "loadProjectConfig are untested by `bun test` (requireProject calls process.exit on its " +
            "failure branch; loadProjectConfig awaits vite's loadConfigFromFile against real disk), though " +
            "all three run for real whenever `bun run test:install`'s dev/build rungs boot a project — no " +
            "test asserts on them directly. Stage 6 tests isProject/requireProject/loadProjectConfig by " +
            "temp dir.",
        stage: 6,
    },
    {
        file: "packages/shallot/bin/verify.ts",
        arm: "gap",
        reason:
            "the majority of this file's exported surface is already directly asserted by verify.test.ts: " +
            "parseVerifyArgs, buildUrl, resolveBatchQueries, batchPass, spansFromCheckpoints, " +
            "formatTimings, installHarnessProbe/harnessInstallMs, summarizeResourceTiming, " +
            "formatExtraTimings, structured/hasStructure/gridDiff, stepWait, coerceVerdict, withTimeout, " +
            "harnessPass, gpuLogChecks, failureArtifacts, settlePass, fitMemory, bootArm, flushStdout, " +
            "reportBatch, and report's rendered arms. verifyCommand's green path (serveDist/serveDev/" +
            "serveEjected booting a real project, driving a real Playwright page) is exercised on every " +
            "`bun bench`, `bun run flows`, and `bun run recipes` run. What's reached by nothing: " +
            "driveHarness's ready-timeout, noRender opt-out, probe-error, and hasRun-fallback verdict " +
            "arms; withGpuLog's `(result.verdict?.ok ?? result.pass) && !failed` merge; the three serve* " +
            "SetupError guards; report's unrendered arms (the LEAK line, compilationError); and " +
            "decodeSample/decodeRgba (pending stage 4's own purity check) — a genuine browser failure, " +
            "which bench/flows/recipes' green-path runs never produce. Stage 4 covers these with a " +
            "duck-typed page stub.",
        stage: 4,
    },
    {
        file: "packages/shallot/src/project/engine.ts",
        arm: "unit",
        reason:
            "DEFAULT_PLUGIN_NAMES, SUBPATH_PLUGIN_MODULES, and KNOWN_ENGINE_PLUGINS are each gated by " +
            "catalog.test.ts against the engine's real barrel/subpath exports (DEFAULT_PLUGINS, avbd's " +
            "AvbdPlugin, and the barrel's own *Plugin set) — KNOWN_ENGINE_PLUGINS is asserted with a " +
            "direct `toEqual` against the real union, which transitively pins EXTRA_PLUGIN_NAMES too (a " +
            "wrong entry there would break that equality).",
    },
    {
        file: "packages/shallot/src/project/generate.ts",
        arm: "unit",
        reason:
            "plan and generateModule's emitted import lines (engine-barrel grouping by source, local " +
            "default imports and their project-relative path resolution, the missing-default runtime " +
            "guard, capacity threading) are directly asserted by generate.test.ts.",
    },
    {
        file: "packages/shallot/src/project/manifest.ts",
        arm: "unit",
        reason:
            "normalize's tolerant-parse (absent/corrupt/non-object storage, $schema/capacity, each " +
            "PluginValue field) is directly asserted by manifest.test.ts. localOf has no row of its own " +
            "test but is exercised through every one of generate.test.ts's plan() cases with a local " +
            "plugin (Spin's bare-string enabled form, Off's disabled-tuple form, Pkg's bare-package form) " +
            "— verified: generate.ts's plan() calls localOf(value) directly for every non-default, " +
            "non-true manifest entry.",
    },
    {
        file: "packages/shallot/src/project/vite.ts",
        arm: "gap",
        reason:
            "assetSrc's public-dir-relative matching, manifestWarnings' corrupt-JSON and unknown-plugin " +
            "branches, and orphanedAssets' fixpoint asset-pruning are directly asserted by vite.test.ts. " +
            "projectPlugin's hooks (resolveId, load, configureServer's static-asset middleware carrying " +
            "the path-traversal guard and the no-store rule, handleHotUpdate, generateBundle) run against " +
            "a stub `this`/server today in production only, never a test; readManifest, discoverScenes, " +
            "findPublicDirs, manifestPath, and contentType are exercised only transitively through those " +
            "hooks. Stage 3 covers resolveId/load/configureServer's middleware/handleHotUpdate/" +
            "generateBundle's orphan deletion + byte accounting, plus the fs readers by temp dir and " +
            "contentType.",
        stage: 3,
    },
    {
        file: "packages/create-shallot/index.ts",
        arm: "extract",
        reason:
            "template's emitted file map is directly asserted by recipe.test.ts (imported cross-package: " +
            "the AGENTS.md/ENGINE_REFERENCE and CLAUDE.md/CLAUDE_IMPORT contents), which recipe.ts's own " +
            "corpus fixtures build from. scaffold and the `if (import.meta.main)` CLI block (arg parsing, " +
            "the existing-dir guard, the created-project console output) both run for real on every `bun " +
            'run test:install` ("bun create shallot (scaffold → install → build the starter)", a real ' +
            "subprocess invocation), but neither is asserted directly — no test calls scaffold or the CLI " +
            "block as a function. Stage 2 gives create-shallot a main(argv) so that block stops being " +
            "subprocess-only, and gives scaffold its own temp-dir test.",
        stage: 2,
    },
];
