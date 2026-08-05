import { describe, expect, test } from "bun:test";
import { scenarioNames } from "../gym";
import {
    assertBudget,
    BUDGET_EXEMPTIONS,
    BUDGETS_ENFORCED,
    checkBudgetCompleteness,
    checkBudgetEntries,
    isDefaultParams,
    SCENARIO_BUDGETS,
} from "./budget-coverage";
import "./index";

// fixture-only red-proofs (coding.md "a check is evidence only if you've seen it fail" — each was run
// against a broken checker first and confirmed to fail before the checker was fixed to pass it): a table
// key naming an unregistered scenario, an exemption with no reason, a scenario both budgeted and exempt,
// and a registered scenario missing both. None of these touch the real scenario roster or a live GPU.
describe("budget registry checker (fixtures)", () => {
    test("a table key naming an unregistered scenario is a finding", () => {
        const findings = checkBudgetEntries({ ghost: { pipelines: 1, gpuBytes: 1 } }, {}, ["real"]);
        expect(findings).toContainEqual({ kind: "unregistered-table-key", detail: "ghost" });
    });

    test("an exemption with no reason is a finding", () => {
        const findings = checkBudgetEntries({}, { a: "" }, ["a"]);
        expect(findings).toContainEqual({ kind: "missing-exemption-reason", detail: "a" });
    });

    test("a scenario both budgeted and exempt is a finding — dropping the exemption reds this too", () => {
        // the dropped-exemption red-proof (`shallot-perf-gates` stage 3b Validation): a scenario carrying
        // both a budget AND an exemption is a contradiction (never both, `budgets.ts`'s own doc), so
        // removing an exemption while its budget entry stays is exactly the state this check exists to
        // catch — confirmed red here first, against the broken (pre-fix) checker that admitted it.
        const findings = checkBudgetEntries({ a: { pipelines: 1, gpuBytes: 1 } }, { a: "reason" }, [
            "a",
        ]);
        expect(findings).toContainEqual({ kind: "budgeted-and-exempt", detail: "a" });
    });

    test("a registered scenario with neither a budget nor an exemption is a completeness finding", () => {
        const findings = checkBudgetCompleteness({ a: { pipelines: 1, gpuBytes: 1 } }, {}, [
            "a",
            "b",
        ]);
        expect(findings).toContainEqual({ kind: "scenario-missing-budget", detail: "b" });
    });

    test("a covered-both-ways table has no completeness finding", () => {
        const findings = checkBudgetCompleteness(
            { a: { pipelines: 1, gpuBytes: 1 } },
            { b: "reason" },
            ["a", "b"],
        );
        expect(findings).toEqual([]);
    });
});

describe("isDefaultParams", () => {
    test("every declared param at its own default is true", () => {
        expect(
            isDefaultParams(
                [
                    { key: "mode", type: "select", default: "cull", options: ["cull"] },
                    { key: "count", type: "number", default: 4096 },
                ],
                { mode: "cull", count: 4096 },
            ),
        ).toBe(true);
    });

    test("a --count override is a non-default run", () => {
        expect(
            isDefaultParams([{ key: "count", type: "number", default: 4096 }], { count: 8192 }),
        ).toBe(false);
    });

    test("no declared params is vacuously default", () => {
        expect(isDefaultParams([], {})).toBe(true);
    });
});

describe("assertBudget (pure, no live Profile/GPU)", () => {
    test("an unbudgeted scenario emits no checks — nothing to compare against yet", () => {
        expect(assertBudget("fixture-scenario", true, { pipelines: 29, gpuBytes: 1 })).toEqual([]);
    });

    test("real render budget: exact match passes both checks", () => {
        const budget = SCENARIO_BUDGETS.render;
        const checks = assertBudget("render", true, {
            pipelines: budget.pipelines,
            gpuBytes: budget.gpuBytes,
        });
        expect(checks).toEqual([
            {
                name: "budget:pipelines",
                pass: true,
                detail: `measured ${budget.pipelines}, budget ${budget.pipelines}`,
            },
            {
                name: "budget:bytes",
                pass: true,
                detail: `measured ${budget.gpuBytes}, budget ${budget.gpuBytes}`,
            },
        ]);
    });

    test("a bogus allocation reds the byte budget (real render table, byte count off by one)", () => {
        const budget = SCENARIO_BUDGETS.render;
        const checks = assertBudget("render", true, {
            pipelines: budget.pipelines,
            gpuBytes: budget.gpuBytes + 1,
        });
        expect(checks.find((c) => c.name === "budget:bytes")?.pass).toBe(false);
        expect(checks.find((c) => c.name === "budget:pipelines")?.pass).toBe(true);
    });

    test("a bogus pipeline count reds the count budget, not the byte one", () => {
        const budget = SCENARIO_BUDGETS.render;
        const checks = assertBudget("render", true, {
            pipelines: budget.pipelines + 1,
            gpuBytes: budget.gpuBytes,
        });
        expect(checks.find((c) => c.name === "budget:pipelines")?.pass).toBe(false);
        expect(checks.find((c) => c.name === "budget:bytes")?.pass).toBe(true);
    });

    test("a non-default-params run reports both axes as visibly inapplicable, never silently skipped", () => {
        const checks = assertBudget("render", false, { pipelines: 1, gpuBytes: 1 });
        expect(checks).toHaveLength(2);
        for (const c of checks) {
            expect(c.pass).toBe(true);
            expect(c.detail).toMatch(/inapplicable/);
        }
    });

    test("an exempt scenario emits no checks even with a table-shaped mismatch", () => {
        // injected table+exemptions (the same fixture-injection shape checkBudgetEntries's fixtures use)
        // so this actually exercises the `name in exemptions` early return, not the `!budget` fallthrough
        // the real (currently-empty) BUDGET_EXEMPTIONS would silently hit instead.
        const checks = assertBudget(
            "outline",
            true,
            { pipelines: 0, gpuBytes: 0 },
            { outline: { pipelines: 999, gpuBytes: 999 } },
            { outline: "fixture exemption" },
        );
        expect(checks).toEqual([]);
    });
});

// live from the real registry + roster: every table key is a registered scenario, every exemption has a
// reason, no scenario is both. The completeness direction turns on with BUDGETS_ENFORCED at stage 4.
describe("budget registry (real data)", () => {
    test("every SCENARIO_BUDGETS key is a registered scenario, every exemption has a reason, none is both", () => {
        const findings = checkBudgetEntries(SCENARIO_BUDGETS, BUDGET_EXEMPTIONS, scenarioNames());
        expect(findings).toEqual([]);
    });

    test.skipIf(!BUDGETS_ENFORCED)(
        "every registered scenario has a budget or an exemption — stage 4's done-signal",
        () => {
            expect(scenarioNames().length).toBeGreaterThan(25); // smoke floor, coverage.test.ts's precedent
            const findings = checkBudgetCompleteness(
                SCENARIO_BUDGETS,
                BUDGET_EXEMPTIONS,
                scenarioNames(),
            );
            expect(findings).toEqual([]);
        },
    );

    // the reverse direction, unconditional (never skipped): nothing else forces `BUDGETS_ENFORCED` back
    // to `true` once stage 4 actually finishes the roster, so a completed table with the flag left `false`
    // would silently ship the completeness direction disabled forever. Vacuously true today — the roster
    // is still partial (only `render`), so `checkBudgetCompleteness` reports findings and the guard never
    // fires; it starts failing the moment stage 4's last entry lands, forcing the flip in the same PR.
    test("BUDGETS_ENFORCED is true whenever the roster is already complete", () => {
        const complete =
            checkBudgetCompleteness(SCENARIO_BUDGETS, BUDGET_EXEMPTIONS, scenarioNames()).length ===
            0;
        if (complete) {
            expect(BUDGETS_ENFORCED).toBe(true);
        }
    });
});
