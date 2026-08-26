/**
 * Arms for the per-file test-duration cap (`packages/shallot/tests/test-cap.ts`) and for the
 * configuration that reaches it (the repo-root `bunfig.toml`).
 *
 * Four load-bearing arms, each pinning a property a single-file arm cannot see:
 *
 *   (a) the per-file window — mutating the `Bun.main` latch to a run-scoped one must red, since
 *       every other arm in this file runs a single file where per-file and per-run are
 *       indistinguishable and that leaves the per-file property unpinned;
 *   (b) a fixture slow file reds, with the cap's message asserted **structurally over its own
 *       enumeration** (numbering, count-word agreement, line count, suffixes derived through the
 *       exempt predicate) and never by substring — a third sanctioned move in that message would
 *       ship green under the substring form;
 *   (c) a fast file passes and the exemption is derived for every by-path suffix with its extent pinned
 *       (an exclusion is a deletion primitive and owes the tighter proof);
 *   (d) the configuration-reach control — removing the preload line from `bunfig.toml` must red,
 *       because a gate supplied by configuration fails open when the configuration is wrong.
 *
 * Shallot's `bunfig.toml` carries a `root = "."` key (widened from `"packages/shallot"` in S3
 * so `bun run test` discovers the `scripts/` and `evals/` arms), which scopes bun's test
 * discovery to the repo root — a property this repo's `bunfig.toml` adds and the reason this
 * repo owes its own configuration-reach arm: the cap must fire from the tracked preload, not
 * from a wrapper or ambient config, and the `root` key's discovery scoping is part of what that
 * arm pins.
 *
 * Every duration arm drives the cap with a tiny `TEST_FILE_CAP_MS` instead of a genuinely slow
 * fixture: a real sleep would make this suite pay the thing the cap exists to stop. The default
 * (5000, no env set) is pinned separately — once on `resolveCapMs`, and once end-to-end by running
 * the same fixture with the variable absent and reading green.
 */
import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    capMessage,
    DEFAULT_CAP_MS,
    isCapExempt,
    PROMOTION_DESTINATIONS,
    resolveCapMs,
} from "./test-cap";
import { TEST_TIER_SUFFIX_NAMES } from "./test-tiers";

const REPO_ROOT = join(import.meta.dir, "../../..");
const PRELOAD = join(import.meta.dir, "test-cap.ts");
const OVER_FIXTURE_REL = "packages/shallot/tests/fixtures/test-cap-over.fixture.ts";
const OVER_FIXTURE = join(REPO_ROOT, OVER_FIXTURE_REL);

type Run = { exitCode: number; output: string };

/** Env for a child `bun test`, with `TEST_FILE_CAP_MS` set or explicitly absent. */
function childEnv(capMs: string | null): Record<string, string> {
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) if (v !== undefined) env[k] = v;
    delete env.TEST_FILE_CAP_MS;
    if (capMs !== null) env.TEST_FILE_CAP_MS = capMs;
    return env;
}

function run(cwd: string, args: string[], capMs: string | null): Run {
    const proc = Bun.spawnSync(["bun", "test", ...args], {
        cwd,
        env: childEnv(capMs),
        stdout: "pipe",
        stderr: "pipe",
    });
    return { exitCode: proc.exitCode, output: proc.stdout.toString() + proc.stderr.toString() };
}

/**
 * A temp dir holding the tracked over-cap fixture's exact bytes under `name`. The cap is
 * supplied by an explicit `--preload`, NOT by a bunfig — a copy of the root bunfig would prove
 * nothing about the real one, which is the configuration-reach arm's job below.
 */
function hermetic(name: string): { dir: string; name: string; file: string } {
    const dir = mkdtempSync(join(tmpdir(), "shallot-test-cap-"));
    const file = join(dir, name);
    writeFileSync(file, readFileSync(OVER_FIXTURE, "utf8"));
    return { dir, name, file };
}

function runHermetic(name: string, capMs: string | null): Run {
    const { dir } = hermetic(name);
    return run(dir, ["--preload", PRELOAD, `./${name}`], capMs);
}

describe("cap resolution and the derived exemption", () => {
    /**
     * Witnessed red: `if (!Number.isFinite(parsed) || parsed < 0)` mutated to `if (!parsed || …)`
     * — the falsy-zero form — failed here `Expected: 0 / Received: 5000` at the `"0"` line, and
     * red every child-process arm below with it, since those drive the cap with `"0"`. That is
     * the defect worth the arm: a falsy guard reads an explicit 0 as unset, silently restores
     * 5000, and makes the duration arms vacuous.
     */
    test("the default is 5000 with no env set, and an explicit 0 is honoured, not read as unset", () => {
        expect(DEFAULT_CAP_MS).toBe(5000);
        expect(resolveCapMs({})).toBe(5000);
        expect(resolveCapMs({ TEST_FILE_CAP_MS: "" })).toBe(5000);
        expect(resolveCapMs({ TEST_FILE_CAP_MS: "0" })).toBe(0);
        expect(resolveCapMs({ TEST_FILE_CAP_MS: "250" })).toBe(250);
        expect(resolveCapMs({ TEST_FILE_CAP_MS: "not-a-number" })).toBe(5000);
        expect(resolveCapMs({ TEST_FILE_CAP_MS: "-1" })).toBe(5000);
    });

    /**
     * Witnessed red: `TIER_SUFFIXES` set to `[]` failed here at the `.oracle.ts` line
     * (`Expected: true / Received: false`) and red the both-ways arm below with it.
     */
    test("exemption is derived from the suffix, so no default-tier path can be exempt", () => {
        expect(isCapExempt("packages/shallot/tests/avbd/oracle.oracle.ts")).toBe(true);
        expect(isCapExempt("packages/shallot/bin/verify.probes.ts")).toBe(true);
        expect(isCapExempt("examples/showcase/roads/src/editCorridor.tier.ts")).toBe(true);
        expect(isCapExempt("packages/shallot/tests/foo.lab.ts")).toBe(true);
        expect(isCapExempt("packages/shallot/tests/conformance.test.ts")).toBe(false);
        expect(isCapExempt(OVER_FIXTURE_REL)).toBe(false);
        expect(isCapExempt("packages/shallot/tests/oracle.ts")).toBe(false);
    });

    /**
     * The exemption's *extent* over the live tree, not just its mechanism — an exclusion is a
     * deletion primitive, so measure and pin its extent in the diff that adds it. The claim that has
     * real content in shallot: bun discovers only `.test.`/`.spec.` files, so no admitted (exempt)
     * path may also carry a `.test.` or `.spec.` segment in its basename — a file named
     * `foo.test.oracle.ts` would be both discovered by `bun test` and exempt by suffix, which is the
     * actual hole this pin exists to exclude. The old form (`!collectedByBunTest || isCapExempt(path)`)
     * was vacuous: the filter that built `admitted` already guaranteed `isCapExempt(path)` true, so the
     * second disjunct held by construction.
     *
     * The uncovered direction is derived, never listed: nothing here names a file.
     *
     * Witnessed red (reviewer): `TIER_SUFFIXES` replaced by `[]` reds on the first admitted path.
     * Witnessed red (counter-example): a hand-made path `foo.test.oracle.ts` reds the basename check.
     */
    test("the exemption's extent: no admitted path carries a .test. or .spec. segment in its basename", () => {
        const tracked = Bun.spawnSync(["git", "ls-files"], {
            cwd: REPO_ROOT,
            stdout: "pipe",
            stderr: "pipe",
        })
            .stdout.toString()
            .split("\n")
            .filter(Boolean);
        const admitted = tracked.filter((p) => isCapExempt(p));
        expect(
            admitted.length,
            "the suffixes admit no paths — the extent claim below would be vacuous",
        ).toBeGreaterThan(0);
        for (const path of admitted) {
            const basename = path.split("/").pop()!;
            // bun discovers only .test. or .spec. files, so no admitted (exempt) path may also
            // carry a .test. or .spec. segment in its basename — foo.test.oracle.ts would be both
            // discovered by bun test and exempt by suffix, the hole this pin excludes.
            expect({
                path,
                hasTestSegment: /\.test\./.test(basename) || /\.spec\./.test(basename),
            }).toEqual({
                path,
                hasTestSegment: false,
            });
        }
    });

    /**
     * `resolveCapMs` can only ever *loosen* the cap — a finite value above 5000 disables it — so
     * the one thing keeping the env override from being a per-file escape is that nothing in the
     * tree sets it. `test-cap.ts`'s docblock claims exactly that, and a docblock claim is an
     * untested claim (`testing.md`), so this is the drift arm for it: no tracked file may
     * *assign* `TEST_FILE_CAP_MS` except this file, which drives child processes with it.
     *
     * The population is derived from `git ls-files` — the tracked-file list, independent of the
     * predicate under test and of `resolveCapMs` — not from the cap's own resolver.
     *
     * Witnessed red: with `"packages/shallot/tests/test-cap.test.ts"` removed from `Allowed`, this
     * arm reds naming `packages/shallot/tests/test-cap.test.ts` — `childEnv`'s own
     * `env.TEST_FILE_CAP_MS = capMs;` is a real assignment in a real tracked file, so the
     * predicate fires on production text, not on a fixture. Witnessed red (temporary assignment):
     * a `TEST_FILE_CAP_MS=10000` line added to `bunfig.toml` reds this arm naming `bunfig.toml`.
     */
    test("nothing in the tree sets TEST_FILE_CAP_MS, so the env override cannot be a standing escape", () => {
        const Allowed = ["packages/shallot/tests/test-cap.test.ts"];
        const tracked = Bun.spawnSync(["git", "ls-files"], {
            cwd: REPO_ROOT,
            stdout: "pipe",
            stderr: "pipe",
        })
            .stdout.toString()
            .split("\n")
            .filter(Boolean);
        const readable = [".ts", ".toml", ".json", ".sh", ".md"];
        const candidates = tracked.filter((p) => readable.some((ext) => p.endsWith(ext)));
        expect(
            candidates.length,
            "the tracked-file scan yielded too few candidates — the setter check below would be vacuous",
        ).toBeGreaterThan(500);
        const setters = candidates.filter(
            (p) =>
                !Allowed.includes(p) &&
                /TEST_FILE_CAP_MS\s*[=:]/.test(readFileSync(join(REPO_ROOT, p), "utf8")),
        );
        expect(setters).toEqual([]);
    });

    /**
     * "Only the two permitted moves" asserted **structurally over the message's own enumeration**,
     * not as a list of `not.toContain` spellings. What carries it:
     *
     *   - the enumerated moves are parsed out of the message and their numbering must be exactly
     *     `[1, 2]`, so a third `(3)` reds and a renumbering reds;
     *   - the message's own count word is parsed and must equal the number enumerated, so a message
     *     that says "Three responses" while listing two reds, and vice versa;
     *   - the message has no line that is not the reading, the header, or an enumerated move, so a
     *     third move smuggled in as prose reds on the line count;
     *   - every file-suffix literal the message names must be one production's own `isCapExempt`
     *     actually exempts — the destination is derived from the mechanism, not restated.
     *
     * Witnessed red, four ways, each run: appending a third enumerated move reds the numbering;
     * changing the header to "Three responses" reds the count-word agreement; appending an unnumbered
     * prose line reds the line count; and changing `*.tier.ts` to `*.slow.ts` reds the suffix
     * derivation.
     */
    test("the message names the file, the reading, the cap, and only the two permitted moves", () => {
        const msg = capMessage("packages/shallot/tests/conformance.test.ts", 5321.4, 5000);
        expect(msg).toContain("packages/shallot/tests/conformance.test.ts");
        expect(msg).toMatch(/ran 5321\.4ms of tests against a 5000ms cap/);

        const lines = msg.split("\n");
        const enumerated = lines.flatMap((line) => {
            const m = /^\((\d+)\)\s+(.+)$/.exec(line);
            return m ? [{ n: Number(m[1]), text: m[2] }] : [];
        });
        expect(enumerated.map((move) => move.n)).toEqual([1, 2]);

        const CountWords: Record<string, number> = { One: 1, Two: 2, Three: 3, Four: 4 };
        const claimed = CountWords[/^(\w+) responses/m.exec(msg)?.[1] ?? ""];
        expect(
            claimed,
            "the message's count word must be one this arm can read, or the agreement below is vacuous",
        ).toBeDefined();
        expect(claimed).toBe(enumerated.length);

        // the reading, the header, and one line per move — nothing else, so a third move as prose reds
        expect(lines.length).toBe(2 + enumerated.length);

        // each move names its own destination
        expect(enumerated[0].text).toContain("derive the scan's size");
        expect(enumerated[1].text).toContain("by-path tier");
        expect(enumerated[1].text).toContain("reason in its header");

        // and the tier suffixes it offers are the promotion destinations, pinned against the
        // constant the message is built from — NOT `isCapExempt`, which admits every exempt
        // suffix and would green a message offering `.probes.ts` or `.lab.ts` as a destination.
        const offered = [...msg.matchAll(/\*(\.[a-z.]*ts)\b/g)].map((m) => m[1]);
        expect(offered.length).toBeGreaterThan(0);
        expect(offered).toEqual(PROMOTION_DESTINATIONS);

        expect(msg.toLowerCase()).not.toContain("raise");
        expect(msg.toLowerCase()).not.toContain("exempt");
        expect(msg).not.toContain("TEST_FILE_CAP_MS");
    });
});

describe("the cap in a real bun test child", () => {
    /**
     * Witnessed red: with the `afterEach` throw replaced by `void capMessage(…)` this arm read
     * `Expected: 1 / Received: 0` on the child's exit code.
     */
    test("a file over the cap reds with the cap's message, once for the whole file", () => {
        const { exitCode, output } = runHermetic("over.test.ts", "0");
        expect(exitCode).toBe(1);
        expect(output).toContain("per-file test cap exceeded: over.test.ts");
        expect(output).toContain("derive the scan's size");
        expect(output).toContain("promote the slow arms to a by-path tier");
        expect(output).toMatch(/\n 1 pass\n/);
        expect(output).toMatch(/\n 1 fail\n/);
    });

    /**
     * The same fixture with the variable absent: the default cap must not red a fast file. This
     * is the end-to-end half of the 5000 default — `resolveCapMs({})` above is the unit half.
     * Witnessed red: `DEFAULT_CAP_MS` set to 0 read `Expected: 0 / Received: 1` on the child's
     * exit code here.
     */
    test("a fast file passes under the default cap, with no env set", () => {
        const { exitCode, output } = runHermetic("over.test.ts", null);
        expect(exitCode).toBe(0);
        expect(output).not.toContain("per-file test cap exceeded");
        expect(output).toMatch(/\n 2 pass\n/);
    });

    /**
     * The exemption proven both ways off ONE body of bytes: the same fixture content reds as
     * `.test.ts` and passes as every by-path tier suffix in `TEST_TIER_SUFFIX_NAMES`, so the only
     * difference between the red and the green is the extension. Witnessed red: `TIER_SUFFIXES = []`
     * (which is `isCapExempt` false for every path) read `Expected: 0 / Received: 1` on the tier
     * child's exit code.
     *
     * The same-bytes precondition is asserted against the **tracked fixture on disk**, not by
     * comparing two `hermetic()` outputs to each other: an equality between two outputs of one
     * expression is an assertion about `hermetic`, the helper, and would hold if the helper wrote
     * the wrong bytes or no bytes at all. `OVER_FIXTURE` is that outside side — the same committed
     * file the configuration-reach arm below runs in place.
     */
    test("the same bytes red as .test.ts and pass as every by-path tier suffix", () => {
        const fixtureBytes = readFileSync(OVER_FIXTURE, "utf8");
        expect(
            fixtureBytes.length,
            "the tracked over-cap fixture is empty — the byte-identity claim below would be vacuous",
        ).toBeGreaterThan(0);
        const red = hermetic("subject.test.ts");
        // Derive the by-path tiers from the shared roster the same way `test-cap.ts` does — by
        // dropping `.test`, the default tier the cap applies to — so the arm covers the roster by
        // construction and a seventh tier is covered the day it is added.
        const byPathSuffixes = TEST_TIER_SUFFIX_NAMES.filter((name) => name !== "test");
        const greenTiers = byPathSuffixes.map((name) => hermetic(`subject.${name}.ts`));
        expect(readFileSync(red.file, "utf8")).toBe(fixtureBytes);
        for (const tier of greenTiers) {
            expect(readFileSync(tier.file, "utf8")).toBe(fixtureBytes);
        }

        const redRun = run(red.dir, ["--preload", PRELOAD, "./subject.test.ts"], "0");
        expect(redRun.exitCode).toBe(1);
        expect(redRun.output).toContain("per-file test cap exceeded");

        for (const tier of greenTiers) {
            const tierRun = run(tier.dir, ["--preload", PRELOAD, `./${tier.name}`], "0");
            expect(tierRun.exitCode).toBe(0);
            expect(tierRun.output).not.toContain("per-file test cap exceeded");
        }
    });

    /**
     * The per-file property itself — per file, not per test, not per
     * suite. Two copies of the same fixture in ONE child under a 0 ms cap: the window must reopen at
     * the second file, so both files red and each message names its own path. Every other child arm in
     * this file runs a single file, where a run-scoped window and a file-scoped one are
     * indistinguishable.
     *
     * Witnessed red (reviewer, mutation applied to a `/tmp` copy of the preload): with
     * `if (Bun.main === currentFile) return;` mutated to `if (currentFile !== "") return;` — a
     * run-scoped window instead of a file-scoped one — this arm reads one cap message and `1 fail`
     * instead of two and `2 fail`, while the over-cap, default-cap and exemption arms above all stay
     * green under that same mutation.
     */
    test("the window is per file, not per run: two files in one child each red on their own reading", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-test-cap-"));
        const bytes = readFileSync(OVER_FIXTURE, "utf8");
        for (const name of ["first.test.ts", "second.test.ts"])
            writeFileSync(join(dir, name), bytes);
        const { exitCode, output } = run(
            dir,
            ["--preload", PRELOAD, "./first.test.ts", "./second.test.ts"],
            "0",
        );
        expect(exitCode).toBe(1);
        expect(output).toContain("per-file test cap exceeded: first.test.ts");
        expect(output).toContain("per-file test cap exceeded: second.test.ts");
        expect(output).toMatch(/\n 2 pass\n/);
        expect(output).toMatch(/\n 2 fail\n/);
    });

    /**
     * The configuration-reach arm — the load-bearing one. `testing.md` § Per-file speed cap:
     * the cap is a preload in `bunfig.toml` — a gate supplied by configuration, so its presence is
     * asserted at the boundary and never inferred from a green run, because a configuration gate
     * fails open when the configuration is wrong.
     *
     * No `--preload` here, and the cwd is the real shallot root, so the ONLY thing that can inject
     * the cap into this child is the tracked `bunfig.toml`'s `[test] preload`. Witnessed by
     * hand, once: with the `preload = [...]` line deleted from that file (and nothing else
     * changed), the child read exit 0 and `2 pass / 0 fail` with no cap message, so this arm was
     * the ONLY one to red — every other arm supplies the preload itself on the command line. That
     * asymmetry is what makes this a boundary assertion rather than a second copy of the arm above.
     *
     * The fixture is passed as an explicit path; bun runs an explicitly named file even when it
     * does not match the default test-file pattern (measured), which is why `.fixture.ts` keeps
     * it out of `bun test` and still reachable here. Shallot's `bunfig.toml` carries
     * `root = "."` (widened in S3), which scopes discovery but does not block an explicit path.
     */
    test("the root bunfig.toml's preload reaches a bun test run from the shallot root", () => {
        const { exitCode, output } = run(REPO_ROOT, [`./${OVER_FIXTURE_REL}`], "0");
        expect(exitCode).toBe(1);
        expect(output).toContain(`per-file test cap exceeded: ${OVER_FIXTURE_REL}`);
        expect(output).toContain("promote the slow arms to a by-path tier");
    });

    /**
     * The control the arm above needs: its red proves *some* configuration injected the preload, not
     * that the tracked one did — a `$HOME`-level bunfig on one machine would green it while the tracked
     * file was broken. bun reads `bunfig.toml` from the cwd only (measured, bun 1.3.14: it does not
     * walk ancestors), so the same tracked fixture, same absolute path, same 0 ms cap, run from a cwd
     * that holds no bunfig must be GREEN. The pair discriminates: red from the root and green from
     * elsewhere means the tracked file is what carried the cap.
     *
     * Witnessed red: pointing this arm's `cwd` at `REPO_ROOT` (i.e. making it a copy of the arm above)
     * reds it at the message-absence assertion — `Expected to not contain: "per-file test cap exceeded"`
     * — which is the fail-open direction stated as a red: a cap arriving from somewhere other than the
     * cwd's own bunfig.
     */
    test("the cap comes from the tracked root bunfig, not from ambient machine config: no cap from a cwd with no bunfig", () => {
        const elsewhere = mkdtempSync(join(tmpdir(), "shallot-test-cap-nobunfig-"));
        const { exitCode, output } = run(elsewhere, [OVER_FIXTURE], "0");
        expect(output).not.toContain("per-file test cap exceeded");
        expect(exitCode).toBe(0);
        expect(output).toMatch(/\n 2 pass\n/);
    });

    /**
     * The bare-discovery safety arm. `bunfig.toml`'s `root = "."` scopes `bun test` discovery to
     * the repo root, so a bare `bun test` collects every `.test.ts`/`.spec.ts` file under the repo
     * — including `examples/gym/src` and `examples/showcase/{roads,voxel}/src`, which the default
     * gate (`bun run test`) excludes by passing explicit paths. The safety property: every
     * `.test.ts`/`.spec.ts` file bare `bun test` would discover — tracked **and** untracked — is
     * run by some declared gate (the default gate or a documented by-path tier). A file in neither
     * is an orphan — discovered by bare `bun test` but run by no gate, so it can rot silently.
     *
     * The cone comes from sources **independent of the population being checked**:
     *
     *   - Default-gate paths: derived from the root `package.json`'s `test` script — the entry
     *     point, never hand-listed.
     *   - By-path tier paths: listed literally from the documented tier split, each with a comment
     *     naming where it is declared. They are NOT derived from the test files on disk — deriving
     *     the cone from the population it checks would make the arm green by construction: any
     *     tracked test file added anywhere silently extends the cone, and an untracked test file
     *     added to a directory that already holds one is also green.
     *
     * One known orphan is excluded with a stated comment: `site/rum-sampler.test.ts` — no gate runs
     * it (not in the default gate, not in any documented by-path tier, not in any `package.json`
     * script). Reported as a finding for the coordinator to book.
     *
     * Witnessed red (three-armed mutation proof, each run + reverted by deleting only the created
     * path):
     *   (a) an untracked `.test.ts` in an undeclared directory → red;
     *   (b) a tracked (`git add`ed, uncommitted) `.test.ts` in an undeclared directory → red — the
     *       case the old arm could not see, since it derived the cone from tracked test files;
     *   (c) an untracked `.test.ts` inside a declared by-path tier directory → green, proving the
     *       arm is not merely counting files.
     */
    test("bare-discovery population matches the documented tier split: no test file outside the declared cone", () => {
        // Default-gate cone: derived from the root package.json's `test` script — the entry point.
        // Re-derive: `node -e "console.log(JSON.parse(require('fs').readFileSync('package.json','utf8')).scripts.test)"`
        const rootPkg = JSON.parse(readFileSync(join(REPO_ROOT, "package.json"), "utf8"));
        const testScript: string = rootPkg.scripts.test;
        const defaultGatePaths = testScript
            .replace(/^bun\s+test\s*/, "")
            .split(/\s+/)
            .filter(Boolean);

        // By-path tier paths: listed literally from the documented tier split, NOT derived from
        // test files on disk. Each entry names where it is declared and how a reader re-derives it.
        const byPathTierPaths = [
            // Declared in `.claude/rules/testing.md` line 37: "run `bun test ./examples/gym/src`".
            // Re-derive: `grep -n 'bun test ./examples/gym/src' .claude/rules/testing.md`
            "examples/gym/src",
            // Declared in `examples/showcase/roads/package.json` `test` script:
            //   "bun test --cwd ../../.. ./examples/showcase/roads/src"
            // Re-derive: `node -e "console.log(JSON.parse(require('fs').readFileSync('examples/showcase/roads/package.json','utf8')).scripts.test)"`
            "examples/showcase/roads/src",
            // Declared in `examples/showcase/voxel/package.json` `test` script:
            //   "bun test --cwd ../../.. ./examples/showcase/voxel/src"
            // Re-derive: `node -e "console.log(JSON.parse(require('fs').readFileSync('examples/showcase/voxel/package.json','utf8')).scripts.test)"`
            "examples/showcase/voxel/src",
        ];

        const declaredCone = [...defaultGatePaths, ...byPathTierPaths];

        function isInCone(path: string, cone: string[]): boolean {
            return cone.some((c) => path === c || path.startsWith(c + "/"));
        }

        // Known orphan: `site/rum-sampler.test.ts` — discovered by bare `bun test` (root = ".")
        // but run by no gate. Not in the default gate's explicit paths, not in any documented
        // by-path tier, not in any `package.json` script. Reported as a finding for the
        // coordinator to book; not moved, deleted, or given a tier here (out of scope).
        const orphanExclusions = ["site/rum-sampler.test.ts"];

        // All .test.ts/.spec.ts files — tracked + untracked (git ls-files --cached --others).
        const allFiles = Bun.spawnSync(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            { cwd: REPO_ROOT, stdout: "pipe", stderr: "pipe" },
        )
            .stdout.toString()
            .split("\n")
            .filter(Boolean);
        const allTestFiles = allFiles.filter((p) => /\.test\.ts$/.test(p) || /\.spec\.ts$/.test(p));

        expect(
            allTestFiles.length,
            "the scan yielded no .test.ts/.spec.ts files — the population check below would be vacuous",
        ).toBeGreaterThan(0);

        const outside = allTestFiles.filter(
            (p) => !isInCone(p, declaredCone) && !orphanExclusions.includes(p),
        );
        expect(outside, "test files outside the declared cone (excluding known orphans)").toEqual(
            [],
        );
    });
});
