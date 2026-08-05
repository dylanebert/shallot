// The both-directions budget-registry check over `SCENARIO_BUDGETS` + `BUDGET_EXEMPTIONS`, the same shape
// `coverage.ts` already runs for `SCENARIO_GATES` + `GATE_EXEMPTIONS`: `checkBudgetEntries` is pure over
// plain data, so `budget-coverage.test.ts` red-proves it against fixtures with no filesystem or scenario
// import; `scenarioNames()` (imported for its side effect via `../gym`) is the one real-registration seam.
// `assertBudget` is the runtime half — the exact-equality check `gym.ts`'s `installHarness` folds into
// every scenario's verdict, pure over already-read numbers so it needs no live `Profile` or GPU to test.
import type { Check, Param, Params } from "../gym";
import { BUDGET_EXEMPTIONS, SCENARIO_BUDGETS, type ScenarioBudget } from "./budgets";

/** `SCENARIO_BUDGETS` + `BUDGET_EXEMPTIONS` cover every registered scenario (`shallot-perf-gates` stage 4),
 *  so the completeness assertion in `budget-coverage.test.ts` runs unconditionally, the same
 *  `COMPLETENESS_ENFORCED` shape `coverage.ts` already uses. */
export const BUDGETS_ENFORCED = true;

export type FindingKind =
    | "unregistered-table-key"
    | "missing-exemption-reason"
    | "budgeted-and-exempt"
    | "scenario-missing-budget";

export interface Finding {
    kind: FindingKind;
    detail: string;
}

/** the per-entry directions: every `SCENARIO_BUDGETS` key names a registered scenario, every exemption
 *  carries a non-empty reason, and no scenario is both budgeted and exempt (`ScenarioBudget`'s own doc:
 *  "never both"). Pure over the table + exemptions + the real registered names, so a fixture proves each
 *  direction with no filesystem or scenario import. */
export function checkBudgetEntries(
    table: Record<string, ScenarioBudget>,
    exemptions: Record<string, string>,
    scenarioNames: readonly string[],
): Finding[] {
    const findings: Finding[] = [];
    const registered = new Set(scenarioNames);

    for (const name of Object.keys(table)) {
        if (!registered.has(name)) {
            findings.push({ kind: "unregistered-table-key", detail: name });
        }
        if (name in exemptions) {
            findings.push({ kind: "budgeted-and-exempt", detail: name });
        }
    }

    for (const [name, reason] of Object.entries(exemptions)) {
        if (reason.trim() === "") {
            findings.push({ kind: "missing-exemption-reason", detail: name });
        }
    }

    return findings;
}

/** the completeness direction: every registered scenario has a budget or an exemption. Gated by
 *  {@link BUDGETS_ENFORCED} (`shallot-perf-gates` stage 4 populated the full roster and turned it on). */
export function checkBudgetCompleteness(
    table: Record<string, ScenarioBudget>,
    exemptions: Record<string, string>,
    scenarioNames: readonly string[],
): Finding[] {
    const findings: Finding[] = [];
    for (const name of scenarioNames) {
        if (!(name in table) && !(name in exemptions)) {
            findings.push({ kind: "scenario-missing-budget", detail: name });
        }
    }
    return findings;
}

/** every declared param resolves to its own default — the property that makes a budget comparable at
 *  all (`budgets.ts`: goldens are declared at default params, exact equality has no tolerance to absorb
 *  a `--count`/`--param` override). Pure over the scenario's own `params` declarations, so a fixture
 *  proves it with no scenario import. */
export function isDefaultParams(decls: readonly Param[], params: Params): boolean {
    return decls.every((p) => params[p.key] === p.default);
}

/** measured counts read straight off `Profile` — kept as a plain record (not the `Profile` type itself)
 *  so this file's checker stays pure and testable with fabricated numbers, never a live GPU. */
export interface MeasuredBudget {
    pipelines: number;
    gpuBytes: number;
}

/** the runtime exact-equality check `installHarness` folds into every scenario's verdict: a budgeted
 *  scenario at default params gets one check per axis (pipeline count, GPU bytes), each exact equality
 *  against `table`; a non-default-params run reports both as visibly inapplicable rather than silently
 *  skipping them (`shallot-perf-gates` stage 3b: "a run at non-default params must report the check as
 *  inapplicable visibly, never silently skipped"); an exempt scenario emits nothing, since there is no
 *  golden to check against. `table` and
 *  `exemptions` default to the real registry — `installHarness` never passes them — and are parameters
 *  (not a module-level read) so a fixture can drive the exempt branch without mutating the real table,
 *  the same injection shape {@link checkBudgetEntries} already uses. Pure over the table + exemptions +
 *  the caller's already-read numbers. */
export function assertBudget(
    name: string,
    atDefaultParams: boolean,
    measured: MeasuredBudget,
    table: Record<string, ScenarioBudget> = SCENARIO_BUDGETS,
    exemptions: Record<string, string> = BUDGET_EXEMPTIONS,
): Check[] {
    if (name in exemptions) return [];
    const budget = table[name];
    if (!budget) return [];

    if (!atDefaultParams) {
        return [
            {
                name: "budget:pipelines",
                pass: true,
                detail: "inapplicable — non-default params, budget is declared at defaults only",
            },
            {
                name: "budget:bytes",
                pass: true,
                detail: "inapplicable — non-default params, budget is declared at defaults only",
            },
        ];
    }

    return [
        {
            name: "budget:pipelines",
            pass: measured.pipelines === budget.pipelines,
            detail: `measured ${measured.pipelines}, budget ${budget.pipelines}`,
        },
        {
            name: "budget:bytes",
            pass: measured.gpuBytes === budget.gpuBytes,
            detail: `measured ${measured.gpuBytes}, budget ${budget.gpuBytes}`,
        },
    ];
}

export { BUDGET_EXEMPTIONS, SCENARIO_BUDGETS };
