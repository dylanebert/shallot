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
//      The check resolves the identifier each gate's `setTimeout` references
//      and asserts that same identifier is exported by `lib.ts` — one claim
//      about one owner, rather than two spellings (a name regex on the export
//      and a numeric-literal ban on the gates) that must agree. The old
//      name-regex arm is dropped: it pinned a spelling S2 had not chosen, so
//      a correct fix naming the export `BOOT_CEILING_MS` would red it, and an
//      arm fitted to a name the later stage has not chosen yet is "fixed" by
//      renaming the export to suit the test. Resolving the identifier the
//      gates reference and asserting that same identifier is exported makes
//      the "no hand-written number" leg and the "references the exported
//      budget" leg one claim about one owner.
//
//      `evals/harness/gate.config.ts`'s own `timeout: 90_000` /
//      `globalTimeout: 180_000` is out of S1/S2's scope and stays hand-written
//      by design: each gate's `test.setTimeout` override wins over the config
//      default, so the config number is inert while the per-gate override
//      stands. S2's invariant is over the per-gate override, not the config
//      default; a future stage that makes the config default derive from the
//      same exported budget would carry its own arm.
//
// S3 — a staging failure (failed `bun install` / `bunx playwright install`)
//      in `evals/grade.ts` grades as a distinct INCOMPLETE result kind, never
//      as the agent's task FAIL. The spec's fix: make `sh` throw on failure,
//      catch it at the staging call sites, and map to INCOMPLETE.
//
//      Three arms pin three legs of this property, all `test.failing`:
//        1. `sh`'s body contains a `throw` keyword (the mechanism).
//        2. `grade.ts` contains an INCOMPLETE identifier or exact `"INCOMPLETE"`
//           quoted token outside comments (the declared result kind).
//        3. A `try` block precedes and a `catch` follows the staging call
//           sites textually (the throw is caught at the call site).
//
// ── Hole records (test.failing) ──
//
// The S2 and S3 arms below are `test.failing`: they pass (green) while their
// assertions fail (the defect stands), and fail (red) when their assertions
// pass (the fix lands). checks.md's hole-record arm requires each to name the
// unit that will flip it: S2 discharges the timeout arm; S3 discharges the
// three staging arms. The label is what makes each a hole record rather than
// a defence of the defect — a permanently-red suite is not shippable, and
// `.failing` is what carries the current state without weakening any assertion.
// The green arms (cardinality, harness import, setTimeout presence) verify
// the mechanism itself, which S1 legitimately makes true.

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

// Extract a function body by brace counting from the body's opening `{` to its
// matching closing `}`. The body brace is the first `{` after the *parameter
// list's* matching close-paren — not the first `{` anywhere after the needle,
// which would mistake a return-type annotation like `{ ok: boolean; out: string }`
// for the body (that balanced brace pair closes before the body brace, so the
// helper would return the return-type fragment instead of the body).
function extractFunction(src: string, name: string): string {
    const needle = `function ${name}(`;
    const start = src.indexOf(needle);
    if (start === -1) throw new Error(`function ${name} not found`);
    // Skip the parameter list to its matching close-paren, so a return-type
    // annotation with balanced braces (e.g. `{ ok: boolean; out: string }`)
    // is not mistaken for the body.
    let i = start + needle.length;
    let parenDepth = 1;
    for (; i < src.length; i++) {
        if (src[i] === "(") parenDepth++;
        else if (src[i] === ")") {
            parenDepth--;
            if (parenDepth === 0) break;
        }
    }
    if (parenDepth !== 0) throw new Error(`function ${name}: unterminated parameter list`);
    // After the close-paren there may be a return-type annotation with
    // balanced braces (e.g. `{ ok: boolean; out: string }`) that closes
    // before the body brace. If a `:` precedes the first `{` and that
    // `{...}` is followed by another `{`, the first block is the return
    // type — the body brace is the second `{`.
    let bodyBrace = src.indexOf("{", i);
    if (bodyBrace === -1) throw new Error(`function ${name}: no opening brace`);
    const colonIdx = src.indexOf(":", i);
    if (colonIdx !== -1 && colonIdx < bodyBrace) {
        let depth = 0;
        let k = bodyBrace;
        for (; k < src.length; k++) {
            if (src[k] === "{") depth++;
            else if (src[k] === "}") {
                depth--;
                if (depth === 0) break;
            }
        }
        if (depth !== 0) throw new Error(`function ${name}: unterminated`);
        let next = k + 1;
        while (next < src.length && /\s/.test(src[next])) next++;
        if (next < src.length && src[next] === "{") {
            bodyBrace = next;
        }
    }
    // Extract the body from bodyBrace to its matching close brace.
    let depth = 0;
    const begin = bodyBrace;
    for (i = bodyBrace; i < src.length; i++) {
        if (src[i] === "{") depth++;
        else if (src[i] === "}") {
            depth--;
            if (depth === 0) return src.slice(begin, i + 1);
        }
    }
    throw new Error(`function ${name}: unterminated`);
}

// Strip string literals (double-quoted, single-quoted, template) so a bare
// identifier search does not match display strings. Used to distinguish a
// declared type member from a string literal in the output logic.
function stripStringLiterals(src: string): string {
    return src
        .replace(/"(?:[^"\\]|\\.)*"/g, '""')
        .replace(/'(?:[^'\\]|\\.)*'/g, "''")
        .replace(/`(?:[^`\\]|\\.)*`/g, "``");
}

// Strip `//` line comments and `/* */` block comments so a bare word in a
// comment does not match an identifier search. Cheapest form: a regex strip,
// not a full lexer — adequate for the INCOMPLETE arm's purpose, which only
// needs to prevent a zero-behavior-change comment from flipping the arm.
function stripComments(src: string): string {
    return src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

// Escape a string for use in a RegExp — prevents identifier characters that
// are also regex metacharacters from being interpreted.
function escapeRegex(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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
        // Cardinality guard — same count as the cardinality arm, not merely
        // > 0: a silent zero-match re-point (wrong dir) reads 0 here, not 6,
        // and a sweep that silently drops one gate reads 5, not 6.
        expect(gates).toHaveLength(6);
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
        // Cardinality guard — same count as the cardinality arm.
        expect(gates).toHaveLength(6);
        for (const gate of gates) {
            const src = readFileSync(gate, "utf8");
            expect(src).toMatch(/setTimeout\s*\(/);
        }
    });
});

// S2 — derived boot budget. `test.failing`: green while all six gates carry
// hand-written `setTimeout` literals and `lib.ts` exports no budget constant;
// red the moment S2 makes every gate derive from a single exported budget, so
// S2 must flip this arm inside its own diff.
describe("S2 — derived boot budget (failing: owed to S2)", () => {
    // Resolves the identifier each gate's `setTimeout` references and asserts
    // that same identifier is exported by `lib.ts`. This subsumes the old
    // name-regex arm (which pinned a spelling S2 had not chosen) and the old
    // no-numeric-literal arm (which could not see whether the identifier was
    // the exported budget or a local constant): one claim about one owner.
    //
    // Matcher CAN: detect that each gate's setTimeout argument is an
    // identifier (not a numeric literal) and that identifier is exported by
    // lib.ts (as const, let, function, or enum).
    // Matcher CANNOT: see whether the identifier is the *worst-case* budget
    // (it could be any exported constant); cannot verify the full expression
    // if the argument is arithmetic (e.g. `BUDGET * 2` — the regex captures
    // only the first token); cannot see whether all six gates reference the
    // *same* identifier (each gate is checked independently).
    //
    // Witnessed flip: adding `export const BOOT_CEILING_MS = 120_000;` to
    // lib.ts and changing every gate's `test.setTimeout(80_000)` /
    // `test.setTimeout(120_000)` to `test.setTimeout(BOOT_CEILING_MS)` reds
    // this arm (`^ this test is marked as failing but it passed`). Also
    // witnessed red with `export function bootCeilingMs(): number { return
    // 120_000; }` and `test.setTimeout(bootCeilingMs())` — the function-form
    // export, which the widened exportRe now admits.
    test.failing("each gate's setTimeout argument is an identifier exported by lib.ts (discharged by S2)", () => {
        const gates = gateFiles();
        expect(gates).toHaveLength(6);
        const lib = readFileSync(join(HARNESS, "lib.ts"), "utf8");
        for (const gate of gates) {
            const src = readFileSync(gate, "utf8");
            // Extract the first setTimeout call's argument.
            const m = src.match(/setTimeout\s*\(\s*([^)\s,(]+)/);
            expect(m).not.toBeNull();
            const arg = m![1];
            // The argument must be an identifier, not a numeric literal.
            // Today all six gates pass numeric literals (80_000 or 120_000).
            expect(arg).toMatch(/^[A-Za-z_$]/);
            // That identifier must be exported by lib.ts — one owner for all
            // six gates, not a local constant or an unrelated reference.
            // Widened to admit `export function NAME` and `export enum NAME`
            // alongside `export const|let NAME`, so a derived budget spelled
            // as a function (e.g. `bootCeilingMs()`) is not invisible.
            const exportRe = new RegExp(
                `export\\s+(?:const|let|function|enum)\\s+${escapeRegex(arg)}\\b`,
            );
            expect(lib).toMatch(exportRe);
        }
    });
});

// S3 — staging failure maps to INCOMPLETE. All three arms are `test.failing`:
// green while the defect stands, red the moment S3 makes the property true, so
// S3 must flip each arm inside its own diff.
describe("S3 — staging failure maps to INCOMPLETE (failing: owed to S3)", () => {
    // Leg 1: `sh`'s body contains a `throw` keyword — the mechanism that makes
    // a staging failure propagate instead of being silently swallowed.
    //
    // Matcher CAN: detect that the keyword `throw` appears somewhere in
    // `sh`'s function body (after fixing extractFunction to skip the return-
    // type annotation's balanced braces and read the real body).
    // Matcher CANNOT: distinguish a throw on non-zero exit from a throw for
    // any other reason; cannot confirm the throw is reachable from the
    // staging call sites; cannot see whether the throw is caught and mapped
    // to INCOMPLETE. Legs 2 and 3 cover those properties.
    //
    // Witnessed flip: adding `if (p.exitCode !== 0) { throw new Error(...) }`
    // to the real `sh` body reds this arm (`^ this test is marked as failing
    // but it passed`). Before the extractFunction fix the arm read the return-
    // type fragment `{ ok: boolean; out: string }` and stayed green even with
    // the throw present — the blocker.
    test.failing("sh's body contains a throw keyword (discharged by S3)", () => {
        const grade = readFileSync(join(EVALS, "grade.ts"), "utf8");
        const shBody = extractFunction(grade, "sh");
        expect(shBody).toContain("throw");
    });

    // Leg 2: `grade.ts` contains an INCOMPLETE identifier or exact `"INCOMPLETE"`
    // quoted token outside comments — the declared result kind, not merely a
    // display string. Today "INCOMPLETE" appears only inside the display string
    // `"INCOMPLETE (gate did not run)"`; after S3 it should be a declared
    // member of the result union (bare identifier in an enum/const/type alias,
    // or a string-literal union member `"INCOMPLETE"`).
    //
    // Two admissible forms:
    //   1. Bare `\bincomplete\b` identifier in the comment-stripped, string-
    //      stripped source (catches `enum ResultKind { … INCOMPLETE … }` or
    //      `const INCOMPLETE = …`).
    //   2. Exact `"INCOMPLETE"` quoted token in the comment-stripped source
    //      (catches `type ResultKind = "PASS" | "FAIL" | "INCOMPLETE"`).
    //      The display string `"INCOMPLETE (gate did not run)"` does NOT match
    //      because there is no closing `"` immediately after `INCOMPLETE`.
    //
    // Matcher CAN: detect that "INCOMPLETE" appears as a bare identifier or
    // exact quoted token outside comments — i.e., in a type declaration,
    // const, or enum, or as a string-literal union member, not just in a
    // display string.
    // Matcher CANNOT: confirm the identifier is a member of the result
    // union specifically (it could be an unrelated constant); cannot
    // distinguish a type union member from a runtime constant; cannot tell
    // union membership from an unrelated constant of the same name.
    //
    // Witnessed flip: adding `type ResultKind = "PASS" | "FAIL" | "INCOMPLETE";`
    // to grade.ts reds this arm (`^ this test is marked as failing but it
    // passed`) — the string-literal-union form that was invisible before the
    // quoted-token admissible form was added. Also reds on a bare `enum` or
    // `const INCOMPLETE` declaration. A bare comment
    // `// TODO: … incomplete …` does NOT red (comments are stripped first).
    test.failing("grade.ts contains an INCOMPLETE identifier or exact quoted token outside comments (discharged by S3)", () => {
        const grade = readFileSync(join(EVALS, "grade.ts"), "utf8");
        const noComments = stripComments(grade);
        // Form 1: bare identifier outside string literals.
        const noStrings = stripStringLiterals(noComments);
        const bareIdent = /\bincomplete\b/i.test(noStrings);
        // Form 2: exact "INCOMPLETE" quoted token (string-literal union member).
        // Does not match "INCOMPLETE (gate did not run)" — no closing `"` after.
        const quotedToken = /"INCOMPLETE"/.test(noComments);
        expect(bareIdent || quotedToken).toBe(true);
    });

    // Leg 3: a `try` block precedes and a `catch` follows the staging call
    // sites textually — a throw from `sh` is caught at the call site, not
    // left to propagate. Today the staging calls are bare `sh(...)` statements
    // with discarded return values and no error handling.
    //
    // Matcher CAN: detect that a `try` keyword appears before the staging
    // call sites and a `catch` keyword appears after them — i.e., S3 added
    // error handling somewhere around the staging calls.
    // Matcher CANNOT: confirm the try/catch wraps the staging calls
    // specifically (text matching cannot determine brace nesting); cannot
    // confirm the catch maps to INCOMPLETE rather than rethrowing.
    //
    // Witnessed flip: wrapping the `sh(["bun", "install"], runDir)` and
    // `sh(["bunx", "playwright", …], runDir)` calls in a `try { … } catch { … }`
    // reds this arm (`^ this test is marked as failing but it passed`).
    test.failing("a try block precedes and a catch follows the staging call sites textually (discharged by S3)", () => {
        const grade = readFileSync(join(EVALS, "grade.ts"), "utf8");
        const stagingIdx = grade.indexOf('sh(["bun", "install"]');
        expect(stagingIdx).not.toBe(-1);
        // There must be a 'try' before the staging calls — today there is
        // none (the only try in grade.ts is the playwright run's try/finally,
        // which is AFTER the staging calls).
        const before = grade.slice(0, stagingIdx);
        expect(before).toMatch(/\btry\s*\{/);
        // There must be a 'catch' after the staging calls — today there is
        // none (the playwright run uses try/finally, not try/catch).
        const after = grade.slice(stagingIdx);
        expect(after).toMatch(/\bcatch\b/);
    });
});
