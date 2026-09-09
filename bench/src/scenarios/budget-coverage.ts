// The runtime budget check `gym.ts`'s `installHarness` folds into every scenario's verdict: exact
// equality per **axis**, not per scenario, because pipeline count is exact on every registered scenario
// whether or not its byte axis is exempt. Pure over already-read numbers, so it needs no live `Profile`
// or GPU. The both-directions registry meta-check that used to sit beside it left with the gate-table
// meta-checks; the table below is data read by its consumers, not a registry asserted against itself.
import type { Check, Param, Params } from "../gym";
import {
    AXES,
    type Axis,
    type AxisBudget,
    type AxisExemption,
    BUDGET_EXEMPTIONS,
    SCENARIO_BUDGETS,
} from "./budgets";

/** the quantities a budget row covers, derived from `AxisBudget`'s own keys (`budgets.ts`) rather than
 *  listed here — a third quantity added to the type reaches this list, `MeasuredBudget` and
 *  `assertBudget` in one edit. Deliberately not privileged against each other anywhere in this module or
 *  `budgets.ts`'s types — every current row budgets all three and nothing here hardcodes that. */
export { AXES, type Axis };

const CHECK_NAME: Record<Axis, string> = {
    pipelines: "budget:pipelines",
    pipelineCalls: "budget:pipeline-calls",
    gpuBytes: "budget:bytes",
};

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
