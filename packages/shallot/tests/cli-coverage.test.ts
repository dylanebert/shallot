import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    CLI_COVERAGE,
    CLI_POPULATION_GLOBS,
    checkCliCoverage,
    cliCoverageLinks,
    cliPopulation,
    cliTestFiles,
    globToRegExp,
} from "./cli-coverage";

// fixture-only red-proofs (a check is evidence only if you've seen it fail): each of these
// was run against a broken registry first — see the spec's Live log / commit for the real-file mutations
// this stage red-proved (deleting a row, adding an unrowed file, adding a second row for an already-
// rowed file, adding an oracle/lab scratch file). None of these fixture cases touch the filesystem or
// the real registry.
describe("checkCliCoverage (fixtures)", () => {
    // the cases below predate the per-arm link checks and exercise the structural rules only; they assert
    // with `toContainEqual`, so an extra link finding on the same fixture row can't mask them.
    const NoLinks = { importedByTest: [], exports: {} };

    test("a walked file with no row is a finding", () => {
        const findings = checkCliCoverage([], ["real/file.ts"], NoLinks);
        expect(findings).toContainEqual({ kind: "file-missing-row", detail: "real/file.ts" });
    });

    test("a row naming a file outside the walked population is a finding", () => {
        const findings = checkCliCoverage(
            [{ file: "ghost.ts", arm: "unit", reason: "not real" }],
            [],
            NoLinks,
        );
        expect(findings).toContainEqual({ kind: "row-names-unwalked-file", detail: "ghost.ts" });
    });

    test("two rows for the same file is a finding, even with different arms — granularity is the file", () => {
        const rows = [
            { file: "a.ts", arm: "unit" as const, reason: "the pure half" },
            { file: "a.ts", arm: "gap" as const, reason: "occupant: theEffectfulHalf" },
        ];
        const findings = checkCliCoverage(rows, ["a.ts"], NoLinks);
        expect(findings).toContainEqual({
            kind: "file-has-multiple-rows",
            detail: "a.ts (2 rows)",
        });
    });

    test("a row with an empty reason is a finding", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "gap", reason: "  " }],
            ["a.ts"],
            NoLinks,
        );
        expect(findings).toContainEqual({ kind: "row-missing-reason", detail: "a.ts" });
    });

    test("an extract row with no discharging stage is a finding", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "extract", reason: "pure logic trapped in an effectful body" }],
            ["a.ts"],
            NoLinks,
        );
        expect(findings).toContainEqual({
            kind: "undischarged-extract-missing-stage",
            detail: "a.ts",
        });
    });

    test("an extract row naming its discharging stage is clean", () => {
        const findings = checkCliCoverage(
            [
                {
                    file: "a.ts",
                    arm: "extract",
                    reason: "pure logic trapped in an effectful body",
                    stage: 2,
                },
            ],
            ["a.ts"],
            NoLinks,
        );
        expect(findings).toEqual([]);
    });

    test("a fully clean registry against its own population is clean", () => {
        const rows = [
            { file: "a.ts", arm: "unit" as const, reason: "directly asserted by a.test.ts" },
            { file: "b.ts", arm: "tier" as const, reason: "covered by `bun bench`'s foo scenario" },
        ];
        expect(
            checkCliCoverage(rows, ["a.ts", "b.ts"], {
                importedByTest: ["a.ts"],
                exports: { "a.ts": ["thing"], "b.ts": ["other"] },
            }),
        ).toEqual([]);
    });

    // the two per-arm link checks. Neither proves the arm — a test may import a file and assert nothing
    // about it, and a reason may name an export it doesn't actually leave uncovered. Both make a *stale*
    // row red instead of merely wrong.
    test("a unit row whose file no test imports is a finding", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "unit", reason: "directly asserted by a.test.ts" }],
            ["a.ts"],
            { importedByTest: [], exports: { "a.ts": ["thing"] } },
        );
        expect(findings).toContainEqual({ kind: "unit-row-file-untested", detail: "a.ts" });
    });

    test("a unit row whose file a test imports is clean", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "unit", reason: "directly asserted by a.test.ts" }],
            ["a.ts"],
            { importedByTest: ["a.ts"], exports: { "a.ts": ["thing"] } },
        );
        expect(findings).toEqual([]);
    });

    test("only the unit arm owes an import — a gap row's file needs none", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "gap", reason: "occupant: thing" }],
            ["a.ts"],
            { importedByTest: [], exports: { "a.ts": ["thing"] } },
        );
        expect(findings).toEqual([]);
    });

    test("a gap row naming no export of its own file is a finding", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "gap", reason: "occupant: theRenamedThing" }],
            ["a.ts"],
            { importedByTest: [], exports: { "a.ts": ["thing"] } },
        );
        expect(findings).toContainEqual({ kind: "gap-row-names-no-export", detail: "a.ts" });
    });

    test("a gap row's occupant matches on a word boundary, not a substring", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "gap", reason: "occupant: buildWebEjected" }],
            ["a.ts"],
            { importedByTest: [], exports: { "a.ts": ["buildWeb"] } },
        );
        expect(findings).toContainEqual({ kind: "gap-row-names-no-export", detail: "a.ts" });
    });

    test("a gap row over a file with no exports at all is a finding, not a free pass", () => {
        const findings = checkCliCoverage(
            [{ file: "a.ts", arm: "gap", reason: "occupant: thing" }],
            ["a.ts"],
            { importedByTest: [], exports: { "a.ts": [] } },
        );
        expect(findings).toContainEqual({ kind: "gap-row-names-no-export", detail: "a.ts" });
    });
});

describe("globToRegExp", () => {
    test("* matches within one path segment, not across /", () => {
        const re = globToRegExp("packages/shallot/bin/*.ts");
        expect(re.test("packages/shallot/bin/cli.ts")).toBe(true);
        expect(re.test("packages/shallot/bin/nested/cli.ts")).toBe(false);
        expect(re.test("packages/shallot/bin/cli.test.ts")).toBe(true); // filtered by cliPopulation, not the glob itself
    });
});

describe("cliPopulation suffix exclusion", () => {
    test("excludes all four test-tier suffixes testing.md names, not just .test.ts", async () => {
        const root = resolve(import.meta.dir, "../../.."); // packages/shallot/tests -> repo root
        const population = await cliPopulation(root);
        expect(population).not.toContain("packages/shallot/bin/verify.probes.ts");
        for (const path of population) {
            expect(path).not.toMatch(/\.(test|oracle|probes|lab)\.ts$/);
        }
    });
});

// the real-filesystem seam: walks the real shallot repo and proves the registry both directions against
// it. This is the check the spec calls "completeness enforced from the start, not behind a flag" — no
// on-switch, no exemption list, red the moment a walked file lacks a row, a row drifts from the walk, or
// a file carries more than one row.
describe("CLI_COVERAGE against the real repo (both directions)", () => {
    const root = resolve(import.meta.dir, "../../.."); // packages/shallot/tests -> repo root

    test("every walked file has exactly one row, every row names a walked file", async () => {
        const population = await cliPopulation(root);
        expect(population.length).toBeGreaterThan(0); // the walk itself must reach real files
        const links = await cliCoverageLinks(root, population);
        expect(links.importedByTest.length).toBeGreaterThan(0); // the import walk must resolve something
        const findings = checkCliCoverage(CLI_COVERAGE, population, links);
        expect(findings).toEqual([]);
    });

    test("every row carries a non-empty reason", () => {
        for (const row of CLI_COVERAGE) {
            expect(row.reason.trim().length).toBeGreaterThan(0);
        }
    });

    test("every extract row names the stage that discharges it", () => {
        for (const row of CLI_COVERAGE) {
            if (row.arm === "extract") expect(row.stage).toBeDefined();
        }
    });

    test("the population globs cover the three CLI/toolchain paths plus stage 6's outline straggler", () => {
        expect(CLI_POPULATION_GLOBS).toEqual([
            "packages/shallot/bin/*.ts",
            "packages/shallot/src/project/*.ts",
            "packages/create-shallot/index.ts",
            "packages/shallot/src/extras/outline/*.ts",
        ]);
    });

    // the Locked decision's "extract, never mock" is asserted once at stage 2 by a one-off grep — this
    // makes it a standing check over the same population `cliPopulation` walks (its test-tier inverse,
    // `cliTestFiles`), so the next test added to this layer can't quietly mock a module.
    test("no test in this layer mocks a module", async () => {
        const testFiles = await cliTestFiles(root);
        expect(testFiles.length).toBeGreaterThan(0); // the walk itself must reach real test files
        for (const file of testFiles) {
            const content = await Bun.file(resolve(root, file)).text();
            expect(content).not.toContain("mock.module(");
            expect(content).not.toContain("spyOn(");
        }
    });
});
