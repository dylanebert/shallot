// The both-directions budget-registry check over `SCENARIO_BUDGETS` + `BUDGET_EXEMPTIONS`, the same shape
// `coverage.ts` already runs for `SCENARIO_GATES` + `GATE_EXEMPTIONS` — but per **axis**, not per scenario:
// pipeline count is exact on every registered scenario, so every scenario
// gates on it regardless of whether its byte axis is exempt. `checkBudgetEntries` is pure over plain data,
// so `budget-coverage.test.ts` red-proves it against fixtures with no filesystem or scenario import;
// `scenarioNames()` (imported for its side effect via `../gym`) is the one real-registration seam.
// `assertBudget` is the runtime half — the exact-equality check `gym.ts`'s `installHarness` folds into
// every scenario's verdict, pure over already-read numbers so it needs no live `Profile` or GPU to test.
import type { Check, Param, Params } from "../gym";
import {
    AXES,
    type Axis,
    type AxisBudget,
    type AxisExemption,
    BUDGET_EXEMPTIONS,
    SCENARIO_BUDGETS,
} from "./budgets";

/** `SCENARIO_BUDGETS` + `BUDGET_EXEMPTIONS` cover every registered scenario, per axis, so the
 *  completeness assertion in `budget-coverage.test.ts` runs unconditionally, the same
 *  `COMPLETENESS_ENFORCED` shape `coverage.ts` already uses. */
export const BUDGETS_ENFORCED = true;

/** the quantities a budget row covers, derived from `AxisBudget`'s own keys (`budgets.ts`) rather than
 *  listed here — a third quantity added to the type reaches this list, `MeasuredBudget`, the entries and
 *  completeness checks, and `assertBudget` in one edit. Deliberately not privileged against each other
 *  anywhere in this module or `budgets.ts`'s types — every current row budgets all three and nothing here
 *  hardcodes that. The both-directions completeness assert is what enforces coverage, not the type
 *  system. */
export { AXES, type Axis };

const CHECK_NAME: Record<Axis, string> = {
    pipelines: "budget:pipelines",
    pipelineCalls: "budget:pipeline-calls",
    gpuBytes: "budget:bytes",
};

export type FindingKind =
    | "unregistered-table-key"
    | "unregistered-exemption-key"
    | "missing-exemption-reason"
    | "budgeted-and-exempt"
    | "scenario-missing-budget";

export interface Finding {
    kind: FindingKind;
    detail: string;
}

/** the per-entry directions, per axis: every `SCENARIO_BUDGETS` key AND every `BUDGET_EXEMPTIONS` key
 *  names a registered scenario — a stale key in either table silently drops the coverage it was meant to
 *  provide, so both directions are checked, not just the budget table's — every exemption reason is
 *  non-empty, and no (scenario, axis) pair is both budgeted and exempt (`AxisBudget`'s own doc: "never
 *  both"). Pure over the table + exemptions + the real registered names, so a fixture proves each
 *  direction with no filesystem or scenario import. */
export function checkBudgetEntries(
    table: Record<string, AxisBudget>,
    exemptions: Record<string, AxisExemption>,
    scenarioNames: readonly string[],
): Finding[] {
    const findings: Finding[] = [];
    const registered = new Set(scenarioNames);

    for (const name of Object.keys(table)) {
        if (!registered.has(name)) {
            findings.push({ kind: "unregistered-table-key", detail: name });
        }
    }

    for (const name of Object.keys(exemptions)) {
        if (!registered.has(name)) {
            findings.push({ kind: "unregistered-exemption-key", detail: name });
        }
    }

    const names = new Set([...Object.keys(table), ...Object.keys(exemptions)]);
    for (const name of names) {
        for (const axis of AXES) {
            if (table[name]?.[axis] !== undefined && exemptions[name]?.[axis] !== undefined) {
                findings.push({ kind: "budgeted-and-exempt", detail: `${name}/${axis}` });
            }
        }
    }

    for (const [name, reasons] of Object.entries(exemptions)) {
        for (const axis of AXES) {
            const reason = reasons[axis];
            if (reason !== undefined && reason.trim() === "") {
                findings.push({ kind: "missing-exemption-reason", detail: `${name}/${axis}` });
            }
        }
    }

    return findings;
}

/** the completeness direction, per axis: every registered scenario carries exactly one of a golden or an
 *  exemption reason on each of {@link AXES} — independently, so a scenario budgeted
 *  on one axis and exempt on another is complete. Gated by {@link BUDGETS_ENFORCED}. */
export function checkBudgetCompleteness(
    table: Record<string, AxisBudget>,
    exemptions: Record<string, AxisExemption>,
    scenarioNames: readonly string[],
): Finding[] {
    const findings: Finding[] = [];
    for (const name of scenarioNames) {
        for (const axis of AXES) {
            const budgeted = table[name]?.[axis] !== undefined;
            const exempt = exemptions[name]?.[axis] !== undefined;
            if (!budgeted && !exempt) {
                findings.push({ kind: "scenario-missing-budget", detail: `${name}/${axis}` });
            }
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
 *  so this file's checker stays pure and testable with fabricated numbers, never a live GPU. One number
 *  per {@link Axis}, mapped rather than listed, so a new axis can't be forgotten here. */
export type MeasuredBudget = { [K in Axis]: number };

/** the runtime exact-equality check `installHarness` folds into every scenario's verdict, evaluated
 *  independently per axis (three today, `budgets.ts`): a budgeted axis at default params gets one
 *  exact-equality check; a budgeted axis at non-default params reports visibly inapplicable rather than silently skipping;
 *  an **exempt** axis emits nothing — there is no golden to check against,
 *  and the exemption reason already names why. A scenario's checks are therefore three, fewer, or none,
 *  depending on which axes it budgets vs. exempts (`render` emits only
 *  `budget:pipelines`, its `budget:bytes` axis exempt). `table` and `exemptions` default to the real
 *  registry — `installHarness` never passes them — and are parameters (not a module-level read) so a
 *  fixture can drive the exempt branch without mutating the real table, the same injection shape
 *  {@link checkBudgetEntries} already uses. Pure over the table + exemptions + the caller's already-read
 *  numbers. */
export function assertBudget(
    name: string,
    atDefaultParams: boolean,
    measured: MeasuredBudget,
    table: Record<string, AxisBudget> = SCENARIO_BUDGETS,
    exemptions: Record<string, AxisExemption> = BUDGET_EXEMPTIONS,
): Check[] {
    const checks: Check[] = [];
    const budget = table[name];
    const exemption = exemptions[name];

    for (const axis of AXES) {
        if (exemption?.[axis] !== undefined) continue; // exempt axis — no golden to check, emit nothing
        const golden = budget?.[axis];
        if (golden === undefined) continue; // neither budgeted nor exempt yet — nothing to compare

        if (!atDefaultParams) {
            checks.push({
                name: CHECK_NAME[axis],
                pass: true,
                detail: "inapplicable — non-default params, budget is declared at defaults only",
            });
            continue;
        }

        checks.push({
            name: CHECK_NAME[axis],
            pass: measured[axis] === golden,
            detail: `measured ${measured[axis]}, budget ${golden}`,
        });
    }

    return checks;
}

export { BUDGET_EXEMPTIONS, SCENARIO_BUDGETS };
