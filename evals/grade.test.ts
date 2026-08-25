// S3 arm — evals/grade.ts mapping of stageOnWindows throw to INCOMPLETE
//
// Invariant: a staging throw (stageOnWindows failed) maps to INCOMPLETE, not a crash.
// The S2 fix added a catch block in grade.ts that maps the throw to gate.ok = null, and
// deriveResultKind (the pure derivation in ./harness/result) maps gateOk=null with ok
// typecheck/build to INCOMPLETE. This arm drives the pure derivation directly — grade.ts
// itself is a top-level script (argv parsing, top-level await) that can never be imported.
//
// The companion arm (wsl.test.ts) pins that stageOnWindows actually throws on a failed
// staging. Together they cover the wsl.ts:49 site: the throw fires, and the grader maps it
// to INCOMPLETE rather than crashing.

import { expect, test } from "bun:test";
import { deriveResultKind, resultKindToPass } from "./harness/result";

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
