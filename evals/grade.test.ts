// A gate that could not run maps to INCOMPLETE, never to a crash and never to a pass.
//
// grade.ts catches a gate-setup throw and sets gate.ok = null; deriveResultKind (the pure
// derivation in ./result) maps gateOk=null with an ok typecheck and build to INCOMPLETE.
// This arm drives that derivation directly — grade.ts itself is a top-level script (argv parsing,
// top-level await) that can never be imported.

import { expect, test } from "bun:test";
import { deriveResultKind, resultKindToPass } from "./result";

test("deriveResultKind — staging failure (gateOk=null) with ok typecheck/build → INCOMPLETE", () => {
    // The invariant: a staging failure (gate never ran, gateOk=null) with ok typecheck and build
    // maps to INCOMPLETE, not PASS and not a crash. Before the fix, the throw from stageOnWindows
    // propagated uncaught through grade.ts's try/finally (no catch), crashing the grader.
    const kind = deriveResultKind(true, true, null);
    expect(kind).toBe("INCOMPLETE");
    expect(resultKindToPass(kind)).toBe(null);
});

test("deriveResultKind — a determined typecheck failure outranks an unrunnable gate", () => {
    // A staging failure does not launder a typecheck failure into INCOMPLETE — the determined
    // failure is decisive regardless of whether the gate ran.
    expect(deriveResultKind(false, true, null)).toBe("FAIL");
    expect(deriveResultKind(true, false, null)).toBe("FAIL");
    expect(deriveResultKind(false, false, null)).toBe("FAIL");
});

test("deriveResultKind — a staging failure with ok typecheck/build is never PASS", () => {
    // The staging failure path (gateOk=null) never reads as PASS — INCOMPLETE is the honest verdict
    // for a run where nothing determined the gate outcome.
    const kind = deriveResultKind(true, true, null);
    expect(kind).not.toBe("PASS");
});
