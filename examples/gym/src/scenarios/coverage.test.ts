import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import { scenarioNames } from "../gym";
import "./index";
import {
    COMPLETENESS_ENFORCED,
    checkCompleteness,
    checkGateEntries,
    GATE_EXEMPTIONS,
    globToRegExp,
    gpuModulePopulation,
    SCENARIO_GATES,
} from "./coverage";

// fixture-only red-proofs (coding.md "a check is evidence only if you've seen it fail" — each was run
// against a broken checker first and confirmed to fail before the checker was fixed to pass it):
// a glob matching nothing, a registered scenario missing its entry, a module in neither covered nor
// exempt, and an exemption that shadows an already-covered module. None of these touch the filesystem or
// the real scenario roster.
describe("gate coverage checker (fixtures)", () => {
    test("a covers glob matching no real module is a finding", () => {
        const findings = checkGateEntries(
            { foo: { covers: ["no/such/path/**/*.ts"] } },
            {},
            ["foo"],
            ["real/file.ts"],
        );
        expect(findings).toContainEqual({
            kind: "glob-matches-nothing",
            detail: "foo: no/such/path/**/*.ts",
        });
    });

    test("a table key naming an unregistered scenario is a finding", () => {
        const findings = checkGateEntries({ ghost: {} }, {}, ["real"], []);
        expect(findings).toContainEqual({ kind: "unregistered-table-key", detail: "ghost" });
    });

    test("an exemption with no reason is a finding", () => {
        const findings = checkGateEntries({}, { "dir/a.ts": "" }, [], []);
        expect(findings).toContainEqual({ kind: "missing-exemption-reason", detail: "dir/a.ts" });
    });

    test("an exemption for a module a scenario already covers is a finding", () => {
        const findings = checkGateEntries(
            { a: { covers: ["dir/**/*.ts"] } },
            { "dir/a.ts": "thought this was uncovered" },
            ["a"],
            ["dir/a.ts"],
        );
        expect(findings).toContainEqual({
            kind: "exempt-shadows-covered",
            detail: "dir/a.ts (dir/a.ts)",
        });
    });

    test("a registered scenario with no table entry is a completeness finding", () => {
        const findings = checkCompleteness({ a: {} }, {}, ["a", "b"], []);
        expect(findings).toContainEqual({ kind: "scenario-missing-entry", detail: "b" });
    });

    test("a module in neither covered nor exempt is a completeness finding", () => {
        const findings = checkCompleteness(
            { a: { covers: ["dir/a.ts"] } },
            { "dir/b.ts": "reason" },
            ["a"],
            ["dir/a.ts", "dir/b.ts", "dir/c.ts"],
        );
        expect(findings).toContainEqual({ kind: "module-not-covered", detail: "dir/c.ts" });
        // the covered and exempted modules are clean — the checker doesn't over-fire on the two it should pass.
        expect(findings.some((f) => f.detail === "dir/a.ts" || f.detail === "dir/b.ts")).toBe(
            false,
        );
    });

    test("globToRegExp matches ** across segments and * within one", () => {
        const re = globToRegExp("dir/**/*.ts");
        expect(re.test("dir/a.ts")).toBe(true);
        expect(re.test("dir/sub/a.ts")).toBe(true);
        expect(re.test("dir/sub/a.js")).toBe(false);
        expect(re.test("other/a.ts")).toBe(false);
    });
});

// live from 3a: the per-entry directions against the real table, the real registered roster, and the real
// filesystem. The two completeness directions turn on with COMPLETENESS_ENFORCED at 3b.
describe("gate coverage (real data)", () => {
    const root = resolve(import.meta.dir, "../../../..");

    test("every table key is a registered scenario, every covers glob resolves, every exemption has a reason", async () => {
        const population = await gpuModulePopulation(root);
        const findings = checkGateEntries(
            SCENARIO_GATES,
            GATE_EXEMPTIONS,
            scenarioNames(),
            population,
        );
        expect(findings).toEqual([]);
    });

    test.skipIf(!COMPLETENESS_ENFORCED)(
        "every registered scenario has a gate entry and covered ∪ exempt equals the population — 3b's done-signal",
        async () => {
            const population = await gpuModulePopulation(root);
            // both completeness directions pass vacuously over an empty population or an empty roster, so
            // a glob typo or a moved directory would read as full coverage. The floors are smoke guards at
            // roughly half the real counts (~104 modules, 56 scenarios) — they catch structural breakage
            // without churning on every added file.
            expect(population.length).toBeGreaterThan(50);
            expect(scenarioNames().length).toBeGreaterThan(25);
            const findings = checkCompleteness(
                SCENARIO_GATES,
                GATE_EXEMPTIONS,
                scenarioNames(),
                population,
            );
            expect(findings).toEqual([]);
        },
    );
});
