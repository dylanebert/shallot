// Pure derivation of a graded task's result kind from its three determined inputs — typecheck ok,
// build ok, gate ok. No side effects, no @playwright/test import: `grade.ts` is a top-level script
// (argv parsing, top-level await) that can never be imported by an arm, so this sibling module is
// what a test imports and calls directly instead of reading `grade.ts`'s source as an AST.
//
// INCOMPLETE is reserved for a run where NOTHING determined the outcome — the browser gate never ran
// (no display, a staging failure, no envelope) and typecheck/build both held. A determined typecheck
// or build failure outranks an unrunnable gate: it is a legitimate FAIL even when the gate never ran,
// never laundered into "could not measure" about a task this run did measure and fail.
export type ResultKind = "PASS" | "FAIL" | "INCOMPLETE";

export function deriveResultKind(
    typecheckOk: boolean,
    buildOk: boolean,
    gateOk: boolean | null,
): ResultKind {
    // A determined failure is decisive regardless of whether the gate ever ran.
    if (typecheckOk === false || buildOk === false) return "FAIL";
    // Typecheck and build both held; the gate is the only remaining input. Its null means the harness
    // never determined an outcome for it (skipped, staging failure, no envelope) — INCOMPLETE, not FAIL.
    if (gateOk === null) return "INCOMPLETE";
    return gateOk ? "PASS" : "FAIL";
}

export function resultKindToPass(kind: ResultKind): boolean | null {
    return kind === "PASS" ? true : kind === "FAIL" ? false : null;
}
