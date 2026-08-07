import { integerDiscipline, noDivision, noIntegerDivision, pointerDiscipline } from "./wgsl";

// The declared registry + pure checker half of the TGSL-corpus meta-test (spec shallot-tgsl-standards,
// stage 2). Shape copied from `examples/gym/src/scenarios/timeouts.ts` + `coverage.ts`: plain committed
// data, a pure checker asserted both directions, red-provable against fixtures. `standards.test.ts` is
// the one real-filesystem/real-import/real-resolve seam — it walks `packages/shallot/src`, dynamic-imports
// every module, duck-detects live TGSL kernels, resolves each to WGSL, and runs the checks below against
// the real text, producing a `Population` this file's `checkStandards` consumes. This file imports
// `tests/wgsl.ts`'s existing checkers (the four discipline checks it already ships and every ported-kernel
// test file already calls) rather than reimplementing their regexes a second time — "one source of truth"
// (coding.md) outweighs a literal zero-runtime-import reading here, since the alternative is a second,
// driftable copy of the same regex logic.

export type DisciplineCheck =
    | "noIntegerDivision"
    | "integerDiscipline"
    | "pointerDiscipline"
    | "noDivision";

const CHECKERS: Record<DisciplineCheck, (src: string) => void> = {
    noIntegerDivision,
    integerDiscipline,
    pointerDiscipline,
    noDivision,
};

/** the checks every live kernel's resolved WGSL is held to by default. `noDivision` is deliberately NOT
 *  here — it is the stronger form for "a kernel whose every index is a shift, mask or multiply-add"
 *  (`wgsl.ts`'s own JSDoc); applied corpus-wide it would red every kernel doing a legitimate float
 *  division (a normalize, a reciprocal). A kernel opts into it via its registry entry's `also`. */
export const DEFAULT_CHECKS: readonly DisciplineCheck[] = [
    "noIntegerDivision",
    "integerDiscipline",
    "pointerDiscipline",
];

/** runs one discipline check against a kernel's resolved WGSL; returns the violation message, or `null`
 *  when the check holds. `tests/wgsl.ts`'s checkers assert via `expect(...)`, throwing on failure — this
 *  is the one seam that turns that throw into a value, so {@link checkStandards} stays pure over data. */
export function violation(check: DisciplineCheck, wgsl: string): string | null {
    try {
        CHECKERS[check](wgsl);
        return null;
    } catch (e) {
        return e instanceof Error ? e.message : String(e);
    }
}

export interface KernelEntry {
    /** checks beyond {@link DEFAULT_CHECKS} this kernel is also held to. */
    also?: readonly DisciplineCheck[];
    /** per-check exemptions. Each reason claims the load-bearing property that actually exempts the
     *  kernel, never a structural shape (`testing.md`'s exemption-reason law). Keyed (kernel, check) —
     *  never by kernel alone, so an exemption for one check can't shadow the three sound ones
     *  (Locked decision, "Key an exemption by (subject, quantity)"). */
    exempt?: Partial<Record<DisciplineCheck, string>>;
}

export type StandardsRegistry = Record<string, KernelEntry>;

/** stage 3 (shallot-tgsl-standards) triages the corpus's discipline-check reds and populates this —
 *  deliberately empty here: stage 2 enumerates and wires the checker, it does not fix or exempt a single
 *  violation. */
export const STANDARDS_REGISTRY: StandardsRegistry = {};

export type FindingKind =
    | "stale-exemption-key"
    | "missing-exemption-reason"
    | "exempt-shadows-passing"
    | "unexempted-violation";

export interface Finding {
    kind: FindingKind;
    detail: string;
}

/** kernel name -> the checks (from {@link DEFAULT_CHECKS} ∪ its registry entry's `also`) that actually
 *  fail against its real resolved WGSL, computed by the real-import/resolve seam (`standards.test.ts`).
 *  This file never resolves a kernel or touches the filesystem — the population is handed in. */
export type Population = Record<string, readonly DisciplineCheck[]>;

/** the both-directions check over `{population, registry}`, pure — no filesystem, no dynamic import, no
 *  WGSL resolution. Mirrors `coverage.ts`'s `checkGateEntries`/`checkCompleteness` split: a declared
 *  registry asserted against a real population the caller supplies.
 *
 *  Four finding kinds: a registry key naming a kernel no longer in the population (`stale-exemption-key`,
 *  a rename/removal orphaned the exemption); an exemption with no reason (`missing-exemption-reason`);
 *  an exemption for a (kernel, check) pair whose check doesn't actually fail — a dead exemption hiding
 *  a check that would otherwise run clean (`exempt-shadows-passing`); and a live kernel violating an
 *  applicable check with no matching exemption (`unexempted-violation`) — the corpus-wide direction stage 3
 *  triages, left as a `test.todo` in `standards.test.ts` until every current violation is fixed or exempted. */
export function checkStandards(population: Population, registry: StandardsRegistry): Finding[] {
    const findings: Finding[] = [];

    for (const [name, entry] of Object.entries(registry)) {
        if (!(name in population)) {
            findings.push({ kind: "stale-exemption-key", detail: name });
            continue;
        }
        const violations = population[name];
        for (const [check, reason] of Object.entries(entry.exempt ?? {}) as [
            DisciplineCheck,
            string,
        ][]) {
            if (!reason || reason.trim() === "") {
                findings.push({ kind: "missing-exemption-reason", detail: `${name}:${check}` });
            }
            if (!violations.includes(check)) {
                findings.push({ kind: "exempt-shadows-passing", detail: `${name}:${check}` });
            }
        }
    }

    for (const [name, violations] of Object.entries(population)) {
        const exempt = new Set(Object.keys(registry[name]?.exempt ?? {}));
        for (const check of violations) {
            if (!exempt.has(check)) {
                findings.push({ kind: "unexempted-violation", detail: `${name}:${check}` });
            }
        }
    }

    return findings;
}
