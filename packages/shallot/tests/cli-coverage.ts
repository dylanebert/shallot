// A registry over the CLI/toolchain layer's
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

import { posix, resolve } from "node:path";
import { TEST_TIER_SUFFIX_NAMES, TEST_TIER_SUFFIXES } from "./test-tiers";

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
 *  the one named engine straggler — `extras/outline/**` — added to this registry rather than left a
 *  prose-only mention, so a third row can't drift from the walk the way a hand-declared region list would
 *  (the Locked decision's "granularity is the file, never the region"). `packFog` and `computeGlyphMetrics`
 *  stay outside this population: each is one pure function inside a file whose *other* content (the fog
 *  ECS/system/plugin half, `createGlyphAtlas`/`ensureString`'s real-device calls) is `src/standard/**`
 *  content this registry was scoped to the CLI/toolchain layer, never to carry — they get a direct test
 *  with no row, same as every other `.test.ts` addition this unit made without touching the registry. */
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

/** the test-tier suffix roster — re-exported from the shared `test-tiers.ts` constant so the tier
 *  list lives in one place. A test file, at whatever tier, is not itself a population member needing a
 *  row; it's the instrument, not the thing measured. Excluding only `.test.ts` would leave
 *  `verify.probes.ts` in the population as if it were untested production code, when it is itself the
 *  by-path browser gate for two constants `verify.test.ts` already sentinels — and omitting `.tier`
 *  would demand a `*.tier.ts` coverage row as production code. `scripts/check-docs.ts` asserts this
 *  constant against `testing.md`'s own enumeration. */
export { TEST_TIER_SUFFIX_NAMES, TEST_TIER_SUFFIXES };

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

/** the inverse of {@link cliPopulation}: every one of this layer's own test-tier files, over the same
 *  glob population — the mock-module standing check walks this rather than a second, hand-declared
 *  population, since two populations leave a seam. */
export async function cliTestFiles(root: string): Promise<string[]> {
    const globs = CLI_POPULATION_GLOBS.map(globToRegExp);
    const out: string[] = [];
    for await (const path of new Bun.Glob("**/*.ts").scan({ cwd: root })) {
        if (!TEST_TIER_SUFFIXES.test(path)) continue;
        if (globs.some((re) => re.test(path))) out.push(path);
    }
    return out;
}

/** the filesystem-derived half of the per-arm link checks, gathered by {@link cliCoverageLinks} and
 *  passed in as plain data so `cli-coverage.test.ts` red-proves the check against fixtures with no
 *  filesystem. Both fields travel together in one value the check requires, rather than two the caller
 *  may supply half of (two values that must agree travel as one). */
export interface CoverageLinks {
    /** every population file imported by at least one test-tier file anywhere in the repo. */
    importedByTest: readonly string[];
    /** the exported identifier names of each population file, keyed by the same path the rows use. */
    exports: Readonly<Record<string, readonly string[]>>;
}

/** every `from "…"` / `import("…")` specifier in a source file. Deliberately lexical, matching inside
 *  string literals and comments too — over-matching costs a spurious *link*, and a link only ever makes
 *  the check more permissive for the rowed file it names, never less. */
const IMPORT_SPECIFIER = /(?:from|import|require)\s*\(?\s*["']([^"']+)["']/g;

/** each `export`ed identifier name: the declaration forms (`export const|function|class|type|…`) plus
 *  named export lists, where `x as y` exports `y`. `export default` names nothing and is skipped. */
const EXPORT_DECL =
    /^export\s+(?:declare\s+)?(?:async\s+)?(?:function\s*\*?|const|let|var|abstract\s+class|class|interface|type|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)/gm;
const EXPORT_LIST = /^export\s+(?:type\s+)?\{([^}]*)\}/gm;

function exportsOf(source: string): string[] {
    const names = new Set<string>();
    for (const [, name] of source.matchAll(EXPORT_DECL)) if (name) names.add(name);
    for (const [, list] of source.matchAll(EXPORT_LIST)) {
        for (const entry of (list ?? "").split(",")) {
            const parts = entry
                .trim()
                .replace(/^type\s+/, "")
                .split(/\s+as\s+/);
            const name = parts[parts.length - 1]?.trim();
            if (name && /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)) names.add(name);
        }
    }
    return [...names];
}

/** gathers the filesystem half of the per-arm link checks: which population files any test-tier file in
 *  the repo imports, and what each population file exports. Kept beside {@link cliPopulation} and
 *  {@link cliTestFiles} — the pure {@link checkCliCoverage} takes the result as plain data.
 *
 *  The import walk spans the whole repo, not {@link cliTestFiles}'s glob-scoped inverse: `catalog.test.ts`
 *  sits beside `src/project/engine.ts` and inside the population, but `cli-coverage.test.ts` and
 *  `create-shallot.test.ts` don't, and a row's test may live anywhere. Only relative specifiers resolve —
 *  a package specifier (`@dylanebert/shallot`) reaches this layer's files only through a barrel, which is
 *  not the direct import a `unit` row claims. */
export async function cliCoverageLinks(
    root: string,
    population: readonly string[],
): Promise<CoverageLinks> {
    const walked = new Set(population);
    const importedByTest = new Set<string>();
    for await (const path of new Bun.Glob("**/*.ts").scan({ cwd: root })) {
        if (path.includes("node_modules/") || !TEST_TIER_SUFFIXES.test(path)) continue;
        const dir = posix.dirname(path);
        const source = await Bun.file(resolve(root, path)).text();
        for (const [, specifier] of source.matchAll(IMPORT_SPECIFIER)) {
            if (!specifier?.startsWith(".")) continue;
            const base = posix.join(dir, specifier);
            for (const candidate of [base, `${base}.ts`, `${base}/index.ts`]) {
                if (walked.has(candidate)) importedByTest.add(candidate);
            }
        }
    }

    const exports: Record<string, string[]> = {};
    for (const file of population) {
        exports[file] = exportsOf(await Bun.file(resolve(root, file)).text());
    }
    return { importedByTest: [...importedByTest], exports };
}

export type FindingKind =
    | "file-missing-row"
    | "row-names-unwalked-file"
    | "file-has-multiple-rows"
    | "row-missing-reason"
    | "undischarged-extract-missing-stage"
    | "unit-row-file-untested"
    | "gap-row-names-no-export";

export interface Finding {
    kind: FindingKind;
    detail: string;
}

/** whether `reason`'s prose contains any of `exports` as a whole word. The boundary is the identifier
 *  character set (`$` and `_` included), not `\b` — `\b` would let a row naming `buildWebEjected` satisfy
 *  an export called `buildWeb`. A file with no exports can satisfy nothing, which is the intended answer:
 *  a `gap` row over a file whose exported surface is empty has nothing left to name. */
function namesAnExport(reason: string, exports: readonly string[]): boolean {
    return exports.some((name) => {
        const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        return new RegExp(`(?<![A-Za-z0-9_$])${escaped}(?![A-Za-z0-9_$])`).test(reason);
    });
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
 *
 *  Two per-arm links make a stale classification red rather than merely wrong, the analogue of
 *  `coverage.ts`'s resolve-the-glob rule. A `unit` row's file must be imported by at least one test-tier
 *  file (`unit-row-file-untested`), and a `gap` row's reason must name at least one identifier the file
 *  actually exports (`gap-row-names-no-export`). **Neither proves the arm.** An import is not an
 *  assertion — a test importing a file for one export leaves the rest of it untested, and the `unit`
 *  link cannot see that. Naming an export is not a claim that the export is uncovered: `testing.md`'s
 *  "a 'references the symbol' check is satisfiable by a mention" applies in full, and this is weaker
 *  than even that, since an export name appearing anywhere in the prose satisfies it — including inside
 *  a sentence describing what *is* covered. What they catch is drift: a `unit` row left standing after
 *  its test file is deleted, and a `gap` row whose named occupants were renamed out of the file. The
 *  `tier` and `extract` arms are deliberately unlinked — a `tier` row's evidence is a bench or install
 *  run no static read of this repo can see, and an `extract` row already owes its discharging stage.
 */
export function checkCliCoverage(
    rows: readonly CoverageRow[],
    population: readonly string[],
    links: CoverageLinks,
): Finding[] {
    const findings: Finding[] = [];
    const walked = new Set(population);
    const imported = new Set(links.importedByTest);
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
        if (row.arm === "unit" && !imported.has(row.file)) {
            findings.push({ kind: "unit-row-file-untested", detail: row.file });
        }
        if (row.arm === "gap" && !namesAnExport(row.reason, links.exports[row.file] ?? [])) {
            findings.push({ kind: "gap-row-names-no-export", detail: row.file });
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
            "installGpuGlobals is called incidentally — it's the first line of requiredFeatures, which " +
            "features.test.ts calls directly — but no test asserts anything about its effect. bun-webgpu's " +
            "setupGlobals(), preloaded by tests/setup.ts, already defines GPUBufferUsage/GPUTextureUsage/" +
            "GPUShaderStage/GPUMapMode as real globals, so the read guard (`name in globalThis`) is true " +
            "for those four; it does not cover GPUColorWrite (verified: `bun --preload " +
            "./packages/shallot/tests/setup.ts -e 'console.log(\"GPUColorWrite\" in globalThis)'` prints " +
            "false), so the write branch fires for GPUColorWrite on every features.test.ts run — yet " +
            "nothing checks that GPUColorWrite lands, or observes either branch's outcome directly. No " +
            "stage in the spec's Approach names gpu-globals.ts; occupant: installGpuGlobals.",
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
            "package.json's `files`, so an installed `--target <os>` build would die on a raw ENOENT) is " +
            "closed end to end: requireRustCrate calls missingCrateDiagnostic before cargoBuild spawns " +
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
            "the majority of this file's exported surface is directly asserted by verify.test.ts: " +
            "parseVerifyArgs, buildUrl, resolveBatchQueries, batchPass, spansFromCheckpoints, " +
            "formatTimings, installHarnessProbe/harnessInstallMs, summarizeResourceTiming, " +
            "formatExtraTimings, structured/hasStructure/gridDiff, stepWait, coerceVerdict, withTimeout, " +
            "harnessPass, gpuLogChecks, failureArtifacts, settlePass, fitMemory, bootArm, flushStdout, " +
            "reportBatch, and report's rendered AND unrendered arms (the LEAK line, compilationError). " +
            "Stage 4 closed the failure-arm gap a red verdict from any tier routes through, by a duck-" +
            "typed page stub over the pre-existing Page parameter (never a module mock): driveHarness's " +
            "ready-timeout/noRender-opt-out/probe-error/hasRun-fallback verdicts, withGpuLog's " +
            "`(result.verdict?.ok ?? result.pass) && !failed` merge and check concatenation (both the " +
            "`??`-vs-`||` and the `!failed` term proved to change the outcome), and the serveDist/serveDev " +
            "SetupError guards by temp dir (asserting the throw class, never message prose). " +
            "verifyCommand's green path — serveDist/serveDev/serveEjected booting a real project and " +
            "driving a real Playwright page end to end — stays `tier`: exercised on every `bun bench`, " +
            "`bun run flows`, and `bun run recipes` run, with no unit test standing in for the real boot-" +
            "and-navigate sequence. What stays reached by nothing, permanently: decodeSample and " +
            "decodeRgba. The stage's open question — whether they're pure pixel math over a captured " +
            'buffer — is settled false: both construct `new Image()`, `document.createElement("canvas")`, ' +
            "and call `getImageData`, genuine in-page DOM serialized into the browser by Playwright's " +
            "`page.evaluate`, not a CPU-callable function a `bun test` could drive. They run for real only " +
            "inside a genuine harness/settle pass under bench/flows/recipes.",
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
        file: "packages/shallot/src/project/assets.ts",
        arm: "unit",
        reason:
            "the internal sibling holding the `./vite` entry's pure readers (`exports.md` \"Barrel " +
            'rules": a module-internal export shared across sibling files lives in a sibling imported ' +
            "directly, never re-exported from the published subpath — the shape sear/codegen.ts uses). " +
            "Every export is a pure function over (path | raw text) with no vite, server, or hook " +
            "surface, and every one is directly asserted by assets.test.ts against a temp dir: " +
            "manifestPath through readManifest's absent/valid/corrupt cases, manifestWarnings' " +
            "unparseable + unknown-plugin-key + silent-on-valid arms, contentType's mapped/case-" +
            "insensitive/unmapped arms, and resolveAssetPath's resolve, missing, directory, traversal, " +
            "and sibling-prefix arms. manifestPath and manifestWarnings stay on the published subpath " +
            "by re-export from vite.ts (bin/build.ts, bin/features.ts and bin/toolchain.ts resolve " +
            "manifestPath through it).",
    },
    {
        file: "packages/shallot/src/project/vite.ts",
        arm: "gap",
        reason:
            "assetSrc, orphanedAssets, discoverScenes, and findPublicDirs are all directly asserted by " +
            "vite.test.ts against a temp dir; the pure readers readManifest, " +
            "manifestWarnings, contentType, resolveAssetPath live in the internal sibling assets.ts " +
            "and are asserted by assets.test.ts (own row below). projectPlugin's " +
            "resolveId, load, configureServer's static-asset middleware, handleHotUpdate, and " +
            "generateBundle's orphan deletion + byte accounting are each exercised against a stub " +
            "`this`/server, per hook. The path-traversal guard lives in resolveAssetPath (assets.ts) and " +
            "is tested there directly, because its escape branch is empirically unreachable through any real " +
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
