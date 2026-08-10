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

/** the population globs: every non-test `.ts` file the spec's Goal names as the CLI/toolchain layer, plus
 *  stage 6's one named engine straggler — `extras/outline/**` — added to this registry rather than left a
 *  prose-only mention, so a third row can't drift from the walk the way a hand-declared region list would
 *  (the Locked decision's "granularity is the file, never the region"). `packFog` and `computeGlyphMetrics`
 *  stay outside this population: each is one pure function inside a file whose *other* content (the fog
 *  ECS/system/plugin half, `createGlyphAtlas`/`ensureString`'s real-device calls) would drag the whole
 *  file's weakest-arm down to `gap` the moment it joined this registry, for content this spec never asked
 *  this registry to carry — they get a direct test with no row, same as every other `.test.ts` addition
 *  this unit made without touching the registry. */
export const CLI_POPULATION_GLOBS: readonly string[] = [
    "packages/shallot/bin/*.ts",
    "packages/shallot/src/project/*.ts",
    "packages/create-shallot/index.ts",
    "packages/shallot/src/extras/outline/*.ts",
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
        arm: "gap",
        reason:
            'synthIndex is directly asserted by dev.test.ts\'s `describe("synthIndex")` block, and ' +
            "buildConfig (stage 2's fold of the inlined build-config literal into the shape dev.ts's " +
            'devConfig already used) is directly asserted by dev.test.ts\'s own `describe("buildConfig")` ' +
            "block (root, plugin order, base/build output shape). buildWeb's manifest-shape body — the " +
            "synthesized entry write/cleanup, the viteBuild call, the public-dir copy — runs end to end on " +
            'every `bun run test:install` ("shallot build exits clean"). Reached by nothing under any ' +
            'tier: buildWeb\'s ejected-shape branch (an owned index.html, `execSync("bunx vite build")`) ' +
            "— no test:install fixture ever has an index.html for buildWeb to find, the ejected fixture " +
            "boots through its own vite server via `verify()`, never `shallot build`; buildWeb's missing-" +
            "manifest/scene guard (`process.exit(1)`); and buildProject's native-target dispatch " +
            "(windows/mac/linux, each a thin call into native.ts's already-`gap` bundlers) — no " +
            "test:install/bench/flows/recipes run ever passes `--target`. No stage in the spec's Approach " +
            "owns these; occupants: buildWeb's ejected-shape branch, buildWeb's missing-manifest guard, " +
            "buildProject's native-target dispatch.",
    },
    {
        file: "packages/shallot/bin/cli.ts",
        arm: "gap",
        reason:
            "stage 2 factored the flag-parse loop + subcommand routing into a pure parseCliArgs(raw), " +
            "the shape parseVerifyArgs (verify.ts) models, directly asserted by cli.test.ts (verify/" +
            "recipe delegation, bare/--help/unrecognized-subcommand usage selection, dev/build/run flag " +
            "resolution, the unknown-option throw). The module-scope dispatch that consumes it — now " +
            "gated behind `if (import.meta.main)` so the pure parser is importable without running the " +
            "live CLI — is exercised for real by `bun run test:install`'s subprocess invocations of the " +
            "installed CLI for `verify`, `recipe`, `dev`, and `build` (`bun CLI build .`, `bun CLI dev . " +
            "--port N`, `bun CLI verify --help`, `bun CLI recipe joints dest`). Reached by nothing: the " +
            "`run` subcommand's dispatch line, the `dev --target <native>` branch, and the bare/`--help`/" +
            "unrecognized-subcommand usage-print paths — no test:install/bench/flows/recipes run ever " +
            "invokes `shallot run` or passes `--target` to `cli.ts`. No stage in the spec's Approach owns " +
            "these; occupants: the `run` subcommand dispatch, `dev`'s native-target branch, the usage-" +
            "print paths.",
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
            "stage 5 discharged the extraction: nativeOutDir, cargoTarget, resolveCargoInvocation (the " +
            "pure portable/target/WSL decision table cargoBuild now delegates to), missingCrateDiagnostic, " +
            "isStaleLocaleEntry (trimMacLocales' delete-predicate), macHelperBin, macInfoPlist's two " +
            "independent branches (helper/LSUIElement, icon key), and findCefDir's CEF_PATH/build-tree " +
            "search (by temp dir, both the short-circuit and the fall-through) are all now directly " +
            "asserted by native.test.ts. dropSwiftshader was fixed from a module-load constant to a " +
            "per-call env read (SHALLOT_DROP_SWIFTSHADER honored mid-process, not frozen at import) and is " +
            "tested for that property directly. What stays permanently untested, by decision: devShellBuild " +
            "(declared manual — its PowerShell DevShell-entry template would be restated byte-for-byte by " +
            "any test) and the subprocess-orchestrating bulk that calls real cargo/sips/iconutil/strip " +
            "(cargoBuild's execSync/spawnSync arms, bundleNativeMac, bundleNativeWindows, bundleNativeLinux, " +
            "ensureIcon, tryStrip, copyLocale, copyCefLibs, copyCefDlls, prepareMacIcon, convertIconToIcns) " +
            '— frozen out of the `unit` arm by the Locked decision itself ("native.ts extracts and ' +
            "declares; it gets no real-cargo gate\"). The missing-crate guard (rust/window isn't shipped in " +
            "package.json's `files`, so an installed `--target <os>` build used to die on a raw ENOENT) is " +
            "now closed end to end: requireRustCrate calls missingCrateDiagnostic before cargoBuild spawns " +
            "anything, and `bun run test:install`'s \"native build fails with the missing-crate diagnostic, " +
            'not a raw ENOENT" rung asserts it from a real installed package via `shallot build --target ' +
            "linux`, red-proved by removing the requireRustCrate() call (surfaces a raw ENOENT from " +
            "posix_spawn instead).",
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
        arm: "gap",
        reason:
            "stage 2 yielded the target dispatch selection (resolveRunTarget) and each branch's env/" +
            "command composition (linuxRunEnv's LD_LIBRARY_PATH prefix, windowsRunCommand's wslpath + " +
            "powershell.exe launch line) as pure functions, directly asserted by run.test.ts. runProject " +
            "itself — the orchestration that calls them plus every spawnSync/execSync/preview call — " +
            "stays reached by nothing under any tier: neither test:install, bench, flows, nor recipes " +
            'ever invokes `shallot run` (verified: no grep hit for runProject/"shallot run" across those ' +
            "scripts), and that was true before this stage's extraction too. Opening a real native window " +
            "or preview server isn't a headless-testable act, the same boundary native.ts's bundle bodies " +
            "sit behind. No stage in the spec's Approach owns it; occupant: runProject's dispatch body " +
            "(the web/mac/linux/windows spawn arms).",
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
            "drop are both directly asserted by toolchain.test.ts, which also covers isProject's " +
            "manifest/nested-.scene/neither cases, requireProject's returns-without-exiting path, and " +
            "loadProjectConfig's no-config null plus its flattened-plugins/overlay/path result, all by " +
            "temp dir. Permanent gap occupant: requireProject's failure branch, which calls process.exit(1) " +
            "and so cannot be driven without killing the test runner — the Locked decision forbids the " +
            "spy that would reach it. It runs for real on `bun run test:install`'s dev/build rungs.",
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
        file: "packages/shallot/src/extras/outline/index.ts",
        arm: "tier",
        reason:
            "the component, OutlineSystem's per-camera mask→JFA→composite dispatch, and OutlinePlugin's " +
            "warm/dispose resource lifecycle are exercised end to end on a real GPU by `bun bench " +
            "--scenario outline`'s three checks: highlighted entities actually run the outline:mask → " +
            'outline:jfa → outline:composite passes (hasPass(lit, "outline:")), nothing highlighted ' +
            "runs zero of them (the zero-cost gate — a bug here would keep paying the dispatch cost with " +
            "nothing on screen), and fog + outline both composite through the shared post-color seam in " +
            'one frame (hasPass(lit, "fog:march") && hasPass(lit, "outline:composite")) — the property ' +
            "that matters is the real pass graph a GPU timestamp query observes, not that the file " +
            "registers a plugin.",
    },
    {
        file: "packages/shallot/src/extras/outline/passes.ts",
        arm: "unit",
        reason:
            "jfaSteps' clamp/power-of-two ladder and groupByMesh's batching are directly asserted by " +
            "outline.test.ts. Every other export is a TGSL kernel or bind-group layout, reached through " +
            "the same file's maskWgsl/outlineWgsl resolved-WGSL structural tests (`testing.md` \"CPU " +
            'execution of pure TGSL kernels" / "resolved WGSL structure" — the logic-truth tier, not a ' +
            "real-device claim): maskWgsl(false)/maskWgsl(true) resolve maskVertex/maskFragment over both " +
            "maskLayoutPlain and maskLayoutOcclude (the reverse-Z occlusion compare, the vertex pull's " +
            "decode/transform chain), and outlineWgsl() resolves fullscreenVs/jfaFs/jfaLayout (the 3×3 " +
            "flood-fill neighbor loop) and compositeKernel/compositeLayout (the band's distance-to-alpha " +
            "math). No export is reached by nothing.",
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
            "assetSrc, manifestWarnings, orphanedAssets, contentType, readManifest, discoverScenes, and " +
            "findPublicDirs are all directly asserted by vite.test.ts against a temp dir. projectPlugin's " +
            "resolveId, load, configureServer's static-asset middleware, handleHotUpdate, and " +
            "generateBundle's orphan deletion + byte accounting are each exercised against a stub " +
            "`this`/server, per hook. The path-traversal guard was extracted into resolveAssetPath and " +
            "tested there directly, because its escape branch is empirically unreachable through any real " +
            'req.url: `new URL(req.url, "http://localhost")`\'s own WHATWG dot-segment normalization ' +
            "strips every crafted payload two independent sweeps tried (raw, percent- and double-encoded " +
            "dots, encoded slash, backslash, overlong/unicode dot forms, absolute-URL and protocol-relative " +
            "request targets, %00) before the guard ever runs, ~35 payloads with no exception found. The " +
            "guard asserts a *segment boundary*, not a string prefix — a bare startsWith(dir) also admits " +
            "a sibling extending dir's basename (`/a/public-secrets` under `/a/public`), which the " +
            "adversarial pass proved exploitable on the extracted function and which now has its own test. " +
            "typegpuPlugin is `tier`: it carries no logic of its own beyond a single-instance constraint, " +
            "and the property that matters — the TGSL transform actually ran over engine `.ts` — is " +
            "asserted at runtime by checkTgsl on every GPU boot, so each `bun bench` / `bun run flows` / " +
            "`bun run recipes` scenario reds if this plugin stops being wired into the synthesized config. " +
            "What stays reached by nothing: configureServer's " +
            "onProjectFile watcher callback (the dev-server glue mapping a live `.scene`/manifest file " +
            "event to invalidate+reload, and a model-asset event through assetSrc to a full-reload) — no " +
            "test:install/bench/flows/recipes run ever edits a file under a booted dev server's watch, and " +
            "this stage's stub server never drives a real chokidar event either. No stage in the spec's " +
            "Approach owns it; occupant: configureServer's onProjectFile callback.",
    },
    {
        file: "packages/create-shallot/index.ts",
        arm: "tier",
        reason:
            "template's emitted file map is directly asserted by recipe.test.ts (imported cross-package: " +
            "the AGENTS.md/ENGINE_REFERENCE and CLAUDE.md/CLAUDE_IMPORT contents), which recipe.ts's own " +
            "corpus fixtures build from. Stage 2 pulled the `if (import.meta.main)` CLI block (arg " +
            "parsing, the existing-dir guard, the created-project console output) out into an exported " +
            "main(argv) returning an exit code rather than calling process.exit itself, directly asserted " +
            'by create-shallot.test.ts\'s `describe("main")` block (no-name usage error, existing-dir ' +
            'refusal, success); scaffold gets its own temp-dir test in the same file (`describe("scaffold' +
            '")`, nested public/ dirs created). All real logic in the file is now directly unit-tested. ' +
            "What remains is the one-line `if (import.meta.main) process.exit(main(...))` entry adapter — " +
            "no logic of its own, but genuinely unreached by `bun test` (importing the module for its " +
            'exports never sets import.meta.main) — run for real on every `bun run test:install` ("bun ' +
            'create shallot (scaffold → install → build the starter)", `bun packages/create-shallot/' +
            "index.ts starter-app` as a real subprocess). Same shape as dev.ts's row: unit-tested logic, " +
            "tier-tested entry wiring, so the file's weakest reached tier is tier.",
    },
];
