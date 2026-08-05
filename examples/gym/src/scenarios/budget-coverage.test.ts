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
// key naming an unregistered scenario, an exemption with no reason, a (scenario, axis) pair both budgeted
// and exempt, and a registered scenario missing coverage on an axis. None of these touch the real scenario
// roster or a live GPU.
describe("budget registry checker (fixtures)", () => {
    test("a table key naming an unregistered scenario is a finding", () => {
        const findings = checkBudgetEntries({ ghost: { pipelines: 1, gpuBytes: 1 } }, {}, ["real"]);
        expect(findings).toContainEqual({ kind: "unregistered-table-key", detail: "ghost" });
    });

    test("an exemption with no reason is a finding, per axis", () => {
        const findings = checkBudgetEntries({}, { a: { pipelines: "" } }, ["a"]);
        expect(findings).toContainEqual({
            kind: "missing-exemption-reason",
            detail: "a/pipelines",
        });
    });

    test("a (scenario, axis) pair both budgeted and exempt is a finding — dropping the exemption reds this too", () => {
        // the dropped-exemption red-proof (`shallot-perf-gates` stage 3b/4b Validation): a scenario axis
        // carrying both a golden AND an exemption reason is a contradiction (never both, `budgets.ts`'s
        // own doc), so removing an exemption while its budget entry stays is exactly the state this check
        // exists to catch — confirmed red here first, against the broken (pre-fix) checker that admitted
        // it. Only the `pipelines` axis conflicts; `gpuBytes` is clean on both sides, so it must not fire.
        const findings = checkBudgetEntries(
            { a: { pipelines: 1, gpuBytes: 1 } },
            { a: { pipelines: "reason" } },
            ["a"],
        );
        expect(findings).toContainEqual({ kind: "budgeted-and-exempt", detail: "a/pipelines" });
        expect(findings).not.toContainEqual({ kind: "budgeted-and-exempt", detail: "a/gpuBytes" });
    });

    test("a registered scenario missing coverage on one axis is a completeness finding for that axis only", () => {
        const findings = checkBudgetCompleteness(
            { a: { pipelines: 1, gpuBytes: 1 }, b: { pipelines: 2 } },
            {},
            ["a", "b"],
        );
        expect(findings).toEqual([{ kind: "scenario-missing-budget", detail: "b/gpuBytes" }]);
    });

    test("a scenario covered on one axis by a budget and the other by an exemption has no completeness finding", () => {
        // this is the per-axis split itself (`shallot-perf-gates` stage 4b): `render`-shaped coverage,
        // pipelines budgeted + gpuBytes exempt, is complete — never both directions failing at once.
        const findings = checkBudgetCompleteness(
            { a: { pipelines: 1 } },
            { a: { gpuBytes: "reason" } },
            ["a"],
        );
        expect(findings).toEqual([]);
    });

    test("a covered-both-ways table has no completeness finding", () => {
        const findings = checkBudgetCompleteness(
            { a: { pipelines: 1, gpuBytes: 1 } },
            { b: { pipelines: "reason", gpuBytes: "reason" } },
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

    test("real accel budget: exact match passes both checks", () => {
        const budget = SCENARIO_BUDGETS.accel;
        const checks = assertBudget("accel", true, {
            pipelines: budget.pipelines as number,
            gpuBytes: budget.gpuBytes as number,
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

    test("a bogus allocation reds the byte budget (real accel table, byte count off by one)", () => {
        const budget = SCENARIO_BUDGETS.accel;
        const checks = assertBudget("accel", true, {
            pipelines: budget.pipelines as number,
            gpuBytes: (budget.gpuBytes as number) + 1,
        });
        expect(checks.find((c) => c.name === "budget:bytes")?.pass).toBe(false);
        expect(checks.find((c) => c.name === "budget:pipelines")?.pass).toBe(true);
    });

    test("a bogus pipeline count reds the count budget, not the byte one", () => {
        const budget = SCENARIO_BUDGETS.accel;
        const checks = assertBudget("accel", true, {
            pipelines: (budget.pipelines as number) + 1,
            gpuBytes: budget.gpuBytes as number,
        });
        expect(checks.find((c) => c.name === "budget:pipelines")?.pass).toBe(false);
        expect(checks.find((c) => c.name === "budget:bytes")?.pass).toBe(true);
    });

    test("a non-default-params run reports both axes as visibly inapplicable, never silently skipped", () => {
        const checks = assertBudget("accel", false, { pipelines: 1, gpuBytes: 1 });
        expect(checks).toHaveLength(2);
        for (const c of checks) {
            expect(c.pass).toBe(true);
            expect(c.detail).toMatch(/inapplicable/);
        }
    });

    test("a scenario exempt on both axes emits no checks even with a table-shaped mismatch", () => {
        // injected table+exemptions (the same fixture-injection shape checkBudgetEntries's fixtures use)
        // so this actually exercises the `exemption?.[axis] !== undefined` early-continue on both axes,
        // not the `!golden` fallthrough the real (currently pipelines-covered) registry would hit instead.
        const checks = assertBudget(
            "outline",
            true,
            { pipelines: 0, gpuBytes: 0 },
            { outline: { pipelines: 999, gpuBytes: 999 } },
            { outline: { pipelines: "fixture exemption", gpuBytes: "fixture exemption" } },
        );
        expect(checks).toEqual([]);
    });

    // `shallot-perf-gates` stage 4b's own Validation criterion: exempting a scenario's `gpuBytes` axis
    // must never drop its exact `pipelines` golden. Red-first, against the real registry's `render` row
    // (pipelines: 29, budgeted; gpuBytes: exempt) — this is precisely what the pre-4b per-scenario
    // exemption shape could not do (proven separately: a reconstruction of that shape returned `[]` for
    // this exact call, so no check existed to red at all).
    test("a bogus pipeline count reds on a byte-exempted scenario (render)", () => {
        expect(BUDGET_EXEMPTIONS.render?.gpuBytes).toBeDefined();
        expect(SCENARIO_BUDGETS.render?.pipelines).toBe(29);

        const checks = assertBudget("render", true, { pipelines: 999, gpuBytes: 1 });
        expect(checks).toEqual([
            { name: "budget:pipelines", pass: false, detail: "measured 999, budget 29" },
        ]);
    });

    test("the correct pipeline count is green on the same byte-exempted scenario (render)", () => {
        const checks = assertBudget("render", true, { pipelines: 29, gpuBytes: 1 });
        expect(checks).toEqual([
            { name: "budget:pipelines", pass: true, detail: "measured 29, budget 29" },
        ]);
    });

    test("a byte-exempted scenario never emits a budget:bytes check, however wrong the measured bytes are", () => {
        const checks = assertBudget("render", true, { pipelines: 29, gpuBytes: 999_999_999_999 });
        expect(checks.find((c) => c.name === "budget:bytes")).toBeUndefined();
    });
});

// live from the real registry + roster: every table key is a registered scenario, every exemption has a
// reason, no (scenario, axis) pair is both. The completeness direction turns on with BUDGETS_ENFORCED at
// stage 4.
describe("budget registry (real data)", () => {
    test("every SCENARIO_BUDGETS key is a registered scenario, every exemption has a reason, no axis is both", () => {
        const findings = checkBudgetEntries(SCENARIO_BUDGETS, BUDGET_EXEMPTIONS, scenarioNames());
        expect(findings).toEqual([]);
    });

    // stage 4b's own coverage floor: pipeline count is exact on every registered scenario, so every one of
    // the 6 byte-exempt rows must still carry a `pipelines` golden.
    test("every scenario in BUDGET_EXEMPTIONS still carries a pipelines golden", () => {
        for (const name of Object.keys(BUDGET_EXEMPTIONS)) {
            expect(SCENARIO_BUDGETS[name]?.pipelines).toBeDefined();
        }
    });

    test.skipIf(!BUDGETS_ENFORCED)(
        "every registered scenario has a budget or an exemption on both axes — stage 4's done-signal",
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
    // to `true` once the roster is complete, so a completed table with the flag left `false` would
    // silently ship the completeness direction disabled forever. Stage 4 populated the full roster and
    // flipped the flag in the same commit — the roster is complete, so this holds non-vacuously now.
    test("BUDGETS_ENFORCED is true whenever the roster is already complete", () => {
        const complete =
            checkBudgetCompleteness(SCENARIO_BUDGETS, BUDGET_EXEMPTIONS, scenarioNames()).length ===
            0;
        if (complete) {
            expect(BUDGETS_ENFORCED).toBe(true);
        }
    });
});
