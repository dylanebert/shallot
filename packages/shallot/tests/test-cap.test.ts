/**
 * Arms for the per-file test-duration cap (`packages/shallot/tests/test-cap.ts`) and for the
 * configuration that reaches it (the repo-root `bunfig.toml`).
 *
 * Shallot's own copy of stage 1's four load-bearing arms. Kex's `harness/test-cap.test.ts` reads
 * kex's `bunfig.toml`, and shallot's carries a `root = "packages/shallot"` key kex's deliberately
 * omits — so the load-bearing configuration-reach property is unpinned in shallot until an arm here
 * asserts it. The four arms:
 *
 *   (a) the per-file window — mutating the `Bun.main` latch to a run-scoped one must red, since
 *       stage 1's original arms all ran a single file where per-file and per-run are
 *       indistinguishable and that left the spec's central property unpinned;
 *   (b) a fixture slow file reds, with the cap's message asserted **structurally over its own
 *       enumeration** (numbering, count-word agreement, line count, suffixes derived through the
 *       exempt predicate) and never by substring — a crashed child once left a third sanctioned move
 *       in that message and the substring form would have shipped it green;
 *   (c) a fast file passes and the exemption is derived for all four suffixes with its extent pinned
 *       (an exclusion is a deletion primitive and owes the tighter proof);
 *   (d) the configuration-reach control — removing the preload line from `bunfig.toml` must red,
 *       because a gate supplied by configuration fails open when the configuration is wrong.
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
import { capMessage, DEFAULT_CAP_MS, isCapExempt, resolveCapMs } from "./test-cap";

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
     * deletion primitive, so measure and pin its extent in the diff that adds it, and an exemption
     * is a defect until every entry is asserted both ways (a real member, and genuinely uncovered).
     * This arm enumerates every path the four suffixes admit from `git ls-files` and asserts each
     * is either a by-path tier run by path (no `bun:test` import — so the cap could not have
     * measured it) or is collected by `bun test` but the cap exempts it by suffix.
     *
     * The uncovered direction is derived, never listed: nothing here names a file.
     *
     * Witnessed red (reviewer): `TIER_SUFFIXES` replaced by `[]` reds on the first admitted path.
     */
    test("the exemption's extent: every path its suffixes admit is never collected by bun test or is exempt by suffix", () => {
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
            const collectedByBunTest = readFileSync(join(REPO_ROOT, path), "utf8").includes(
                'from "bun:test"',
            );
            // every admitted path is either not collected by bun test (no bun:test import) or is exempt by suffix
            expect({ path, covered: !collectedByBunTest || isCapExempt(path) }).toEqual({
                path,
                covered: true,
            });
        }
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

        // and the tier suffixes it offers are the ones production exempts, derived through `isCapExempt`
        const offered = [...msg.matchAll(/\*(\.[a-z.]*ts)\b/g)].map((m) => m[1]);
        expect(offered.length).toBeGreaterThan(0);
        expect(offered.filter((suffix) => !isCapExempt(`some-file${suffix}`))).toEqual([]);

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
     * `.test.ts` and passes as `.oracle.ts` / `.probes.ts` / `.tier.ts` / `.lab.ts`, so the only
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
    test("the same bytes red as .test.ts and pass as all four by-path tier suffixes", () => {
        const fixtureBytes = readFileSync(OVER_FIXTURE, "utf8");
        expect(
            fixtureBytes.length,
            "the tracked over-cap fixture is empty — the byte-identity claim below would be vacuous",
        ).toBeGreaterThan(0);
        const red = hermetic("subject.test.ts");
        const greenOracle = hermetic("subject.oracle.ts");
        const greenProbes = hermetic("subject.probes.ts");
        const greenTier = hermetic("subject.tier.ts");
        const greenLab = hermetic("subject.lab.ts");
        expect(readFileSync(red.file, "utf8")).toBe(fixtureBytes);
        expect(readFileSync(greenOracle.file, "utf8")).toBe(fixtureBytes);
        expect(readFileSync(greenTier.file, "utf8")).toBe(fixtureBytes);
        expect(readFileSync(greenLab.file, "utf8")).toBe(fixtureBytes);

        const redRun = run(red.dir, ["--preload", PRELOAD, "./subject.test.ts"], "0");
        expect(redRun.exitCode).toBe(1);
        expect(redRun.output).toContain("per-file test cap exceeded");

        for (const tier of [greenOracle, greenProbes, greenTier, greenLab]) {
            const tierRun = run(tier.dir, ["--preload", PRELOAD, `./${tier.name}`], "0");
            expect(tierRun.exitCode).toBe(0);
            expect(tierRun.output).not.toContain("per-file test cap exceeded");
        }
    });

    /**
     * The per-file property itself — the spec's Locked decision, "per file, not per test, not per
     * suite". Two copies of the same fixture in ONE child under a 0 ms cap: the window must reopen at
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
     * The configuration-reach arm — the load-bearing one. `checks.md`: a permission gate
     * supplied by configuration fails open when the configuration is wrong, so its presence is
     * asserted at the boundary and never inferred from a green run.
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
     * `root = "packages/shallot"`, which scopes discovery but does not block an explicit path.
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
});
