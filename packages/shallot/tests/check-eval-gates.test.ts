// A meta-test over the eval harness's structural invariants — same shape as
// `check-scripts.test.ts` (a meta-test over repo-root tooling, placed here so
// it rides the default `bun test` sweep), not a unit test of engine behaviour.
//
// Route decision (S1): the eval tree `evals/` sits outside `bunfig.toml`'s
// `root = "packages/shallot"`, so a `bun test` arm cannot live inside `evals/`
// and be discovered. But an arm under `packages/shallot/tests/` IS discovered,
// and it can reach `evals/` by reading source files as text — no imports from
// `evals/`, so no playwright/chromium dependency is pulled into the test suite.
// The rejected alternative was a `scripts/check-eval-gates.ts` script in the
// `check` chain: that would red the primary structural-check gate (`bun run
// check`) on S2/S3 properties, which is more disruptive than a red in the
// `bun test` population (which the kex-root fast suite does not reach at all).
//
// ── Properties ──
//
// S2 — no task gate under `evals/tasks/*/gate.ts` carries a hand-written
//      `setTimeout` number; every one derives from a single exported
//      worst-case boot budget in `evals/harness/lib.ts`. The check is
//      structural over ALL task gates (the class), not the five sites the
//      audit named.
//
//      Surface forms of a hand-written timeout (the defect):
//        test.setTimeout(80_000)      — numeric literal with underscore
//        test.setTimeout(120_000)      — numeric literal with underscore
//        test.setTimeout(80000)        — numeric literal without underscore
//        test.setTimeout(80 * 1000)    — arithmetic of literals
//
//      Surface form of a derived timeout (the fix):
//        test.setTimeout(BOOT_BUDGET_MS) — reference to an exported constant
//
//      The check: for each gate, every `setTimeout` call's argument must NOT
//      start with a digit (a numeric literal or arithmetic of literals), and
//      `lib.ts` must export a named boot-budget constant.
//
// S3 — a staging failure (failed `bun install` / `bunx playwright install`)
//      in `evals/grade.ts` grades as a distinct INCOMPLETE result kind, never
//      as the agent's task FAIL. The spec's fix: make `sh` throw on failure.
//
//      Surface form of the defect (current state):
//        sh(["bun","install"], runDir);  — return value discarded, sh does not throw
//
//      Surface form of the fix (after S3):
//        sh throws on non-zero exit code, caught and mapped to INCOMPLETE
//
//      The check: the `sh` function body must contain a `throw` on non-zero
//      exit code. Today it does not — it returns `{ ok: false }` silently.
//
// ── Owned red ──
//
// The S2 and S3 arms below are RED today because S2/S3 have not landed. The red
// is real (not a skip): each arm asserts the property, not the current defect,
// and fails because the property does not hold. S2 discharges the timeout
// arms; S3 discharges the staging arm. The green arms (cardinality, file
// existence, harness imports) verify the mechanism itself, which S1
// legitimately makes true.

import { describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const REPO_ROOT = resolve(import.meta.dir, "..", "..", "..");
const EVALS = join(REPO_ROOT, "evals");
const TASKS = join(EVALS, "tasks");
const HARNESS = join(EVALS, "harness");

// Enumerate the class — every task gate under `evals/tasks/*/gate.ts`. This is
// a selection (the check picks its subject out of a corpus), so the caller
// asserts cardinality to guard against a silent zero-match re-point.
function gateFiles(): string[] {
    return readdirSync(TASKS, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => join(TASKS, d.name, "gate.ts"))
        .filter((p) => existsSync(p));
}

// Extract a function body by brace counting from `function <name>(` to its
// matching closing brace. Used to read `sh`'s definition without importing
// `grade.ts` (which would pull playwright into the test graph).
function extractFunction(src: string, name: string): string {
    const needle = `function ${name}(`;
    const start = src.indexOf(needle);
    if (start === -1) throw new Error(`function ${name} not found`);
    let i = src.indexOf("{", start);
    if (i === -1) throw new Error(`function ${name}: no opening brace`);
    let depth = 0;
    const begin = i;
    for (; i < src.length; i++) {
        if (src[i] === "{") depth++;
        else if (src[i] === "}") {
            depth--;
            if (depth === 0) return src.slice(begin, i + 1);
        }
    }
    throw new Error(`function ${name}: unterminated`);
}

describe("eval gate surface — mechanism (green: S1)", () => {
    test("discovers all task gates (cardinality)", () => {
        const gates = gateFiles();
        // The class is six gates today. A gate added or removed reds this arm
        // — the sweep must see the whole population, and a silent zero-match
        // re-point (wrong dir) reads 0 here, not 6.
        expect(gates).toHaveLength(6);
        const names = gates.map((p) => p.split("/").slice(-2, -1)[0]).sort();
        expect(names).toEqual([
            "color-on-key",
            "falling-box",
            "orbit-on-drag",
            "persist-color",
            "red-box",
            "striped-material",
        ]);
    });

    test("each gate imports from the shared harness lib", () => {
        const gates = gateFiles();
        expect(gates.length).toBeGreaterThan(0); // cardinality guard
        for (const gate of gates) {
            const src = readFileSync(gate, "utf8");
            // Every gate builds on `harness/lib` — the boot driver, the pixel
            // utilities, the envelope emitter. A gate that stops importing it
            // is a gate that stopped using the shared harness.
            expect(src).toContain("harness/lib");
        }
    });

    test("each gate has at least one setTimeout call", () => {
        const gates = gateFiles();
        expect(gates.length).toBeGreaterThan(0); // cardinality guard
        for (const gate of gates) {
            const src = readFileSync(gate, "utf8");
            expect(src).toMatch(/setTimeout\s*\(/);
        }
    });
});

// S2 — derived boot budget. RED today: all six gates carry hand-written
// `setTimeout` literals (80_000 or 120_000), and `lib.ts` exports no budget
// constant. S2 discharges both arms.
describe("S2 — derived boot budget (red: owed to S2)", () => {
    test("lib.ts exports a worst-case boot budget constant", () => {
        const lib = readFileSync(join(HARNESS, "lib.ts"), "utf8");
        // After S2, `lib.ts` exports a named constant for the worst-case boot
        // budget (e.g. `export const BOOT_BUDGET_MS = …`). Today it does not.
        // The property named in words: a single exported budget that every
        // gate derives from. The surface form: `export const <…Budget…>`.
        expect(lib).toMatch(/export\s+(?:const|let)\s+\w*[Bb]udget\w*/);
    });

    test("no gate carries a hand-written setTimeout number", () => {
        const gates = gateFiles();
        expect(gates.length).toBeGreaterThan(0); // cardinality guard
        // A `setTimeout` call whose argument starts with a digit is a
        // hand-written numeric literal (80_000, 120_000, 80000, 80 * 1000).
        // After S2 every gate passes a reference to the exported budget, which
        // starts with an identifier, not a digit.
        const numericLiteral = /setTimeout\s*\(\s*\d/;
        for (const gate of gates) {
            const src = readFileSync(gate, "utf8");
            expect(src).not.toMatch(numericLiteral);
        }
    });
});

// S3 — staging failure maps to INCOMPLETE. RED today: `sh` returns
// `{ ok: false }` silently — no throw — and the callers discard the return
// value, so a failed `bun install` / `bunx playwright install` surfaces as
// "gate produced no result" and grades the agent's task FAIL. S3 discharges
// this arm by making `sh` throw on non-zero exit code.
describe("S3 — staging failure maps to INCOMPLETE (red: owed to S3)", () => {
    test("sh throws on non-zero exit code", () => {
        const grade = readFileSync(join(EVALS, "grade.ts"), "utf8");
        const shBody = extractFunction(grade, "sh");
        // Today `sh` returns `{ ok: p.exitCode === 0, out: … }` without
        // throwing — a staging failure is silently swallowed. After S3 the
        // function throws on non-zero exit code, so a staging failure
        // propagates and is caught to map INCOMPLETE (not FAIL).
        expect(shBody).toContain("throw");
    });
});
