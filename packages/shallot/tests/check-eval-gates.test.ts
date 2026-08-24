// A meta-test over the eval harness's structural invariants — same shape as
// `check-scripts.test.ts` (a meta-test over repo-root tooling, placed here so
// it rides the default `bun test` sweep), not a unit test of engine behaviour.
//
// Route decision (S1): the eval tree `evals/` sits outside `bunfig.toml`'s
// `root = "packages/shallot"`, so a `bun test` arm cannot live inside `evals/`
// and be discovered. But an arm under `packages/shallot/tests/` IS discovered,
// and it can reach `evals/` by *parsing* its sources — no imports from
// `evals/`, so no playwright/chromium dependency is pulled into the suite.
// The rejected alternative was a `scripts/check-eval-gates.ts` in the `check`
// chain: same reach, but it reds the gate every other shallot lane runs first.
//
// ── Instrument: a parser, not a pattern ──
//
// Rounds 1–3 of this file hand-rolled regex and brace-counting over `evals/`
// source. All three were green at their own gate and each was holed in the
// surface form that round's fixtures did not vary:
//   * an `extractFunction` that returned the inline return-type annotation
//     `{ ok: boolean; out: string }` instead of `sh`'s body, so a correct S3
//     fix could never flip the arm — and whose `function NAME(` needle threw
//     outright on a generic or arrow form;
//   * a `stripComments` running before `stripStringLiterals`, where the `//`
//     inside `` `http://localhost:${port}/` `` shifted backtick parity for the
//     rest of `grade.ts`;
//   * an `"INCOMPLETE"` token test that a bare `console.log("INCOMPLETE")`
//     discharges.
// `checks.md` names this class: a check that has failed N rounds each green at
// its own gate is a finding about the gate's *kind*, and a third widening
// round is the signal the shape is wrong rather than the constant. So every
// arm below reads a `@babel/parser` AST and asserts a structural property —
// a call expression's argument node kind, a function's real `body` node, a
// `TSUnionType`'s members, a `TryStatement`'s block. The strip-order
// machinery is retired outright rather than fixed: a parser tokenizes
// `` `http://localhost:${port}/` `` correctly by construction.
//
// `@babel/parser` (not `typescript`) is the parser because `typescript@^7` in
// this tree is the native port, whose npm package ships only `lib/tsc.js` —
// there is no compiler API and `ts.ScriptTarget` is `undefined` under bun.
// `@babel/parser` is a devDependency of the private workspace root, beside
// the existing `@babel/core`; the published package is `packages/shallot`,
// which ships no `tests/` entry.
//
// ── Properties ──
//
// S2 — no task gate under `evals/tasks/*/gate.ts` carries a hand-written
//      `setTimeout` number; every one derives from a single owner exported by
//      `evals/harness/lib.ts`. Structural over ALL task gates (the class),
//      not the five sites the audit named. The export's *spelling* is not
//      pinned: the arm resolves the identifier the gates reference and
//      asserts that same identifier is exported.
//
//      `evals/harness/gate.config.ts`'s own `timeout: 90_000` is out of
//      S1/S2's scope and stays hand-written by design: each gate's
//      `test.setTimeout` override wins over the config default, so the config
//      number is inert while the per-gate override stands.
//
// S3 — a staging failure (failed `bun install` / `bunx playwright install`)
//      in `evals/grade.ts` grades as a distinct INCOMPLETE result kind, never
//      as the agent's task FAIL: `sh` throws, the throw is caught at the
//      staging call sites, and the result maps to INCOMPLETE.
//
// ── Hole records (test.failing) ──
//
// The four S2/S3 arms are `test.failing`: green while their assertions fail
// (the defect stands), red the moment their assertions pass (the fix lands),
// printing `^ this test is marked as failing but it passed`. Each names the
// stage that discharges it, which is what makes it a hole record rather than
// a defence of the defect.
//
// A `test.failing` arm is *also* green when the arm itself is broken —
// mis-pathed, malformed, reading a corrupted span — so each arm carries a
// witnessed flip differential in its own docblock: a realistic correct fix
// applied to a scratch reconstruction of the eval tree, and the observed red.
// The readers are parameterized by a root directory (`CHECK_EVAL_GATES_ROOT`)
// precisely so those differentials are runnable rather than described.

import { describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parse } from "@babel/parser";

type Node = { type: string } & Record<string, unknown>;

// The repo root this arm reads. Overridable so the flip differentials in the
// docblocks below can be re-run against a scratch reconstruction under /tmp
// without editing this file: the readers take a root, they do not hard-code
// the worktree path.
function evalRoot(): string {
    return process.env.CHECK_EVAL_GATES_ROOT ?? resolve(import.meta.dir, "..", "..", "..");
}

function parseTs(src: string): Node {
    return parse(src, { sourceType: "module", plugins: ["typescript"] }) as unknown as Node;
}

function parseFile(path: string): Node {
    return parseTs(readFileSync(path, "utf8"));
}

// Generic AST walk. Comment arrays are skipped: a comment is not a node the
// structural properties below are ever about, and skipping them is what makes
// "a comment-only change flips nothing" true by construction rather than by a
// strip pass whose ordering was round 2's hole.
function walk(node: unknown, visit: (n: Node) => void): void {
    if (Array.isArray(node)) {
        for (const child of node) walk(child, visit);
        return;
    }
    if (!node || typeof node !== "object") return;
    const n = node as Node;
    if (typeof n.type === "string") visit(n);
    for (const key of Object.keys(n)) {
        if (
            key === "loc" ||
            key === "leadingComments" ||
            key === "trailingComments" ||
            key === "innerComments" ||
            key === "comments"
        ) {
            continue;
        }
        walk((n as Record<string, unknown>)[key], visit);
    }
}

function collect(node: unknown, type: string): Node[] {
    const out: Node[] = [];
    walk(node, (n) => {
        if (n.type === type) out.push(n);
    });
    return out;
}

// Enumerate the class — every task gate under `<root>/evals/tasks/*/gate.ts`.
// This is a selection (the check picks its subject out of a corpus), so every
// caller asserts cardinality: a silent zero-match re-point reads 0, not 6.
function gateFiles(root: string): string[] {
    const tasks = join(root, "evals", "tasks");
    return readdirSync(tasks, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => join(tasks, d.name, "gate.ts"))
        .filter((p) => existsSync(p))
        .sort();
}

// Every `setTimeout(...)` call in a gate, whatever its callee shape
// (`test.setTimeout(x)`, a bare `setTimeout(x)`, `page.setTimeout(x)`).
function setTimeoutCalls(ast: Node): Node[] {
    return collect(ast, "CallExpression").filter((call) => {
        const callee = call.callee as Node | undefined;
        if (!callee) return false;
        if (callee.type === "Identifier") return callee.name === "setTimeout";
        if (callee.type === "MemberExpression") {
            const prop = callee.property as Node | undefined;
            return prop?.type === "Identifier" && prop.name === "setTimeout";
        }
        return false;
    });
}

// The identifier a timeout argument ultimately names, or `null` when the
// argument names nothing (a literal). Unwraps the forms a derived budget can
// legitimately take: a bare identifier, a call of one, a namespaced member, an
// arithmetic expression over one, and TS-only wrappers.
function ownerIdentifier(node: Node | undefined): string | null {
    if (!node) return null;
    switch (node.type) {
        case "Identifier":
            return node.name as string;
        case "CallExpression":
            return ownerIdentifier(node.callee as Node);
        case "MemberExpression": {
            const prop = node.property as Node;
            return prop?.type === "Identifier" ? (prop.name as string) : null;
        }
        case "BinaryExpression":
            return ownerIdentifier(node.left as Node) ?? ownerIdentifier(node.right as Node);
        case "TSAsExpression":
        case "TSNonNullExpression":
        case "TSSatisfiesExpression":
        case "TSTypeAssertion":
            return ownerIdentifier(node.expression as Node);
        default:
            return null;
    }
}

// Every name `<root>/evals/harness/lib.ts` exports — value or type, in every
// declaration form (`export const|let|var`, `export function`, `export class`,
// `export enum`, `export type|interface`, and re-export specifiers). Read off
// the AST, so no spelling of the export is pinned.
function libExportedNames(root: string): Set<string> {
    const ast = parseFile(join(root, "evals", "harness", "lib.ts"));
    const names = new Set<string>();
    for (const decl of collect(ast, "ExportNamedDeclaration")) {
        const inner = decl.declaration as Node | null | undefined;
        if (inner) {
            if (inner.type === "VariableDeclaration") {
                for (const d of (inner.declarations as Node[]) ?? []) {
                    const id = d.id as Node;
                    if (id?.type === "Identifier") names.add(id.name as string);
                }
            } else {
                const id = inner.id as Node | undefined;
                if (id?.type === "Identifier") names.add(id.name as string);
            }
        }
        for (const spec of (decl.specifiers as Node[]) ?? []) {
            const exported = spec.exported as Node | undefined;
            if (exported?.type === "Identifier") names.add(exported.name as string);
            else if (exported?.type === "StringLiteral") names.add(exported.value as string);
        }
    }
    return names;
}

// The function body node of a top-level `NAME` in `src`, in every declaration
// form: `function NAME(...)`, `function NAME<T>(...)`, `const NAME = (...) =>`,
// `const NAME = function (...)`, and any of those behind `export`. Returns the
// *body* node — never a return-type annotation, which is a sibling of `body`
// on the AST and so cannot be confused with it.
function functionBody(ast: Node, name: string): Node {
    for (const kind of ["FunctionDeclaration", "TSDeclareFunction"]) {
        for (const fn of collect(ast, kind)) {
            const id = fn.id as Node | undefined;
            if (id?.type === "Identifier" && id.name === name && fn.body) return fn.body as Node;
        }
    }
    for (const d of collect(ast, "VariableDeclarator")) {
        const id = d.id as Node | undefined;
        if (id?.type !== "Identifier" || id.name !== name) continue;
        const init = d.init as Node | undefined;
        if (
            init &&
            (init.type === "ArrowFunctionExpression" || init.type === "FunctionExpression") &&
            init.body
        ) {
            return init.body as Node;
        }
    }
    throw new Error(`no function declaration for ${name}`);
}

// Every call site of `sh(...)` in `grade.ts` — the population leg D controls
// against, so a try/catch arm cannot go vacuously green on an empty set.
function shCallSites(ast: Node): Node[] {
    return collect(ast, "CallExpression").filter((call) => {
        const callee = call.callee as Node | undefined;
        return callee?.type === "Identifier" && callee.name === "sh";
    });
}

function gradeAst(root: string): Node {
    return parseFile(join(root, "evals", "grade.ts"));
}

describe("eval gate surface — mechanism (green: S1)", () => {
    test("discovers all task gates (cardinality)", () => {
        const gates = gateFiles(evalRoot());
        // The class is six gates today. A gate added or removed reds this arm
        // — the sweep must see the whole population, and a silent zero-match
        // re-point (wrong dir) reads 0 here, not 6.
        expect(gates).toHaveLength(6);
        expect(gates.map((p) => p.split("/").slice(-2, -1)[0])).toEqual([
            "color-on-key",
            "falling-box",
            "orbit-on-drag",
            "persist-color",
            "red-box",
            "striped-material",
        ]);
    });

    test("each gate imports from the shared harness lib", () => {
        const gates = gateFiles(evalRoot());
        expect(gates).toHaveLength(6);
        for (const gate of gates) {
            // An `ImportDeclaration`'s `source`, not a substring of the file:
            // a comment or a string mentioning the path cannot satisfy it.
            const sources = collect(parseFile(gate), "ImportDeclaration").map(
                (d) => (d.source as Node).value as string,
            );
            expect(sources.some((s) => s.endsWith("harness/lib"))).toBe(true);
        }
    });

    test("each gate carries exactly one setTimeout call", () => {
        const gates = gateFiles(evalRoot());
        expect(gates).toHaveLength(6);
        for (const gate of gates) {
            // Cardinality of the subject leg A selects: exactly one call per
            // gate, so leg A's "the first one" is the whole population and not
            // a `.find` that silently re-points once a second call appears.
            expect(setTimeoutCalls(parseFile(gate))).toHaveLength(1);
        }
    });
});

// S2 — derived boot budget. `test.failing`: green while the gates carry
// hand-written `setTimeout` literals; red the moment S2 makes every gate
// derive from a single owner exported by `lib.ts`, so S2 flips this arm
// inside its own diff.
describe("S2 — derived boot budget (failing: owed to S2)", () => {
    // Leg A. Over ALL gates under `evals/tasks/*/gate.ts` as a class, not the
    // five sites the audit named: each gate's `setTimeout` argument must be an
    // `Identifier`/`CallExpression` (never a `NumericLiteral`) whose resolved
    // name is exported by `evals/harness/lib.ts`, and all gates must name the
    // *same* owner. Pre-fix reading, structural: every argument node is a
    // `NumericLiteral` (`80_000` ×5, `120_000` for persist-color).
    //
    // The export's spelling is not pinned — the arm resolves the identifier
    // the gates reference and asserts that identifier is in `lib.ts`'s export
    // set, so a fix naming it `BOOT_CEILING_MS` or `bootCeilingMs()` passes
    // equally. Nothing here is fitted to a name S2 has not chosen.
    //
    // Matcher CAN: distinguish a literal argument from an identifier one by
    // AST node kind; resolve the identifier through call, member, arithmetic
    // and TS-wrapper forms; assert that name is in `lib.ts`'s exported set;
    // assert the owner is single across the whole class.
    // Matcher CANNOT: see whether the exported owner is the *worst-case* boot
    // budget rather than some other exported number, nor whether its value
    // covers the ≈101 s retry path — a value claim needs a runtime reading,
    // not a parse. It also cannot see a second timeout expressed some way
    // other than a `setTimeout` call (the green arm above pins the count at
    // one per gate, which is what bounds this).
    //
    // Witnessed flip differential (2026-08-25). Method for every arm below:
    // `evals/` copied verbatim into a scratch dir under /tmp, one realistic
    // fix applied there, then
    // `CHECK_EVAL_GATES_ROOT=<scratch> bun test packages/shallot/tests/check-eval-gates.test.ts`.
    // Control first: the *unmodified* copy reads 7 pass / 0 fail at the
    // scratch root, so the readers are not silently reading the worktree.
    //
    // Leg A input — `export function bootCeilingMs(): number { return
    // 130_000; }` appended to `evals/harness/lib.ts`; each of the six gates'
    // `test.setTimeout(80_000)` / `test.setTimeout(120_000)` rewritten to
    // `test.setTimeout(bootCeilingMs())` with `bootCeilingMs` added to that
    // gate's `../../harness/lib` import list. Observed: 6 pass / 1 fail, this
    // arm the fail, `^ this test is marked as failing but it passed`; legs
    // B/C/D unmoved.
    // Negative, witnessed in the same shape: a comment-only edit (`//
    // derived budget: owed to S2` inserted above each gate's `setTimeout`
    // line, plus one comment line atop `lib.ts` and one atop `grade.ts`)
    // reads 7 pass / 0 fail — nothing flips.
    //
    // Placement note, not overstated: the reader finds the owner wherever the
    // `setTimeout` call sits inside the gate, but it does require the argument
    // to be *that call's* argument and the name to be exported from
    // `evals/harness/lib.ts` specifically — a budget owned by another harness
    // file would leave this arm green.
    test.failing("every task gate's setTimeout argument resolves to one owner exported by harness/lib.ts (discharged by S2)", () => {
        const root = evalRoot();
        const gates = gateFiles(root);
        // Population control: the class must be non-empty, or every
        // per-gate assertion below is vacuous.
        expect(gates.length).toBeGreaterThan(0);
        const exported = libExportedNames(root);
        const owners = new Set<string>();
        for (const gate of gates) {
            const calls = setTimeoutCalls(parseFile(gate));
            expect(calls.length).toBeGreaterThan(0);
            for (const call of calls) {
                const arg = ((call.arguments as Node[]) ?? [])[0];
                // Today a `NumericLiteral`, so `ownerIdentifier` is null.
                const owner = ownerIdentifier(arg);
                expect(owner).not.toBeNull();
                expect([...exported]).toContain(owner as string);
                owners.add(owner as string);
            }
        }
        // One owner for the whole class — the locked structural fix is a
        // single derived budget, not six identifiers that happen to be
        // exported.
        expect(owners.size).toBe(1);
    });
});

// S3 — staging failure maps to INCOMPLETE. Three `test.failing` arms, one per
// leg of the mechanism; S3 flips each inside its own diff.
describe("S3 — staging failure maps to INCOMPLETE (failing: owed to S3)", () => {
    // Leg B. `sh`'s *body span* in `evals/grade.ts` contains a
    // `ThrowStatement`. Read off the `body` node of `sh`'s declaration, which
    // on the AST is a sibling of `returnType` — so the inline return-type
    // annotation `{ ok: boolean; out: string }` that round 1 matched instead of
    // the body is not reachable from here by construction. `functionBody`
    // accepts the declaration, generic-declaration, arrow and function-
    // expression forms, so a fix that changes `sh`'s shape does not throw the
    // reader (round 3's latent `function NAME(` needle).
    //
    // Pre-fix reading, structural: `sh` is a `FunctionDeclaration` whose body
    // holds 0 `ThrowStatement` nodes.
    //
    // Matcher CAN: assert a `throw` exists inside `sh`'s real body span, in
    // any nesting.
    // Matcher CANNOT: distinguish a throw on non-zero exit from a throw for
    // some other reason; confirm the throw is reachable from the staging call
    // sites; see whether it is caught and mapped to INCOMPLETE (legs C and D
    // cover those). A throw inside a callback nested in `sh` also satisfies it.
    //
    // Witnessed flip differential (2026-08-25, same scratch method and
    // command as leg A): input — `if (p.exitCode !== 0) throw new
    // Error(`command failed: ...`);` inserted into `sh`'s real body,
    // immediately before its `return { ok: p.exitCode === 0, ... }`.
    // Observed: 6 pass / 1 fail, this arm the fail, `^ this test is marked as
    // failing but it passed`; legs A/C/D unmoved. The comment-only negative
    // above leaves it green.
    test.failing("sh's body span in grade.ts contains a ThrowStatement (discharged by S3)", () => {
        const body = functionBody(gradeAst(evalRoot()), "sh");
        expect(collect(body, "ThrowStatement").length).toBeGreaterThan(0);
    });

    // Leg C. A type-level union in `evals/grade.ts` carries an `INCOMPLETE`
    // string-literal member — i.e. INCOMPLETE is a declared *result kind*, not
    // a display string. Round 3 holed here because a token test over stripped
    // source is discharged by a bare `console.log("INCOMPLETE")`; a
    // `TSUnionType` member cannot be forged that way, because a string in
    // expression position is never a `TSLiteralType`.
    //
    // Pre-fix reading, structural: 8 `TSUnionType` nodes in `grade.ts`, none
    // carrying any string-literal member; `INCOMPLETE` appears only inside the
    // display string `"INCOMPLETE (gate did not run)"`, which is a
    // `StringLiteral` in expression position and so invisible here.
    //
    // Matcher CAN: assert a type-level union carries the exact string-literal
    // member `INCOMPLETE`, and ignore every expression-position occurrence of
    // the same text (comment, log line, template).
    // Matcher CANNOT: confirm the union is the *result* kind rather than some
    // other union, nor that any code path assigns INCOMPLETE on a staging
    // failure. It also does not admit an `enum`/`const` spelling of the kind:
    // this arm is deliberately narrower than "INCOMPLETE exists somewhere",
    // and a fix spelling the kind as a TS enum would leave it green — see the
    // residue note in this unit's fold rather than widening it to a token
    // test, which is exactly the hole this round retires.
    //
    // Witnessed flip differential (2026-08-25, same scratch method and
    // command as leg A): input — the single line `type ResultKind = "PASS" |
    // "FAIL" | "INCOMPLETE";` added to `grade.ts` above `interface
    // Assertion` (the type alias alone; no other edit). Observed: 6 pass /
    // 1 fail, this arm the fail, `^ this test is marked as failing but it
    // passed`; legs A/B/D unmoved.
    // Negative witnessed directly, and it is round 3's own hole: a
    // `console.log("INCOMPLETE");` statement added to `grade.ts` reads 7 pass
    // / 0 fail — this arm stays green, where round 3's token test flipped.
    // The comment-only negative also leaves it green.
    test.failing("a type-level union in grade.ts carries an INCOMPLETE string-literal member (discharged by S3)", () => {
        const unions = collect(gradeAst(evalRoot()), "TSUnionType");
        // Population control: `grade.ts` must actually contain unions, or
        // the `some` below is vacuously false for the wrong reason and the
        // arm's green would be about a mis-parse rather than the defect.
        expect(unions.length).toBeGreaterThan(0);
        const carriesIncomplete = unions.some((u) =>
            ((u.types as Node[]) ?? []).some((member) => {
                if (member.type !== "TSLiteralType") return false;
                const lit = member.literal as Node | undefined;
                return lit?.type === "StringLiteral" && lit.value === "INCOMPLETE";
            }),
        );
        expect(carriesIncomplete).toBe(true);
    });

    // Leg D. A `try` block with a `catch` handler wraps at least one `sh()`
    // call site in `evals/grade.ts` — the throw leg B pins is caught where the
    // staging calls are made, rather than crashing the run. Asserted against
    // the `sh()` call-site count as a population control, so the arm cannot go
    // vacuously green if `sh` is renamed or the reader is mis-pathed.
    //
    // Pre-fix reading, structural: 4 `sh()` call sites, 0 of them inside any
    // `TryStatement` block. `grade.ts` does contain one `TryStatement` today —
    // the playwright run — but it has a `finalizer` and no `handler`, and it
    // contains no `sh()` call, so both legs of this property are false for
    // independent reasons.
    //
    // Matcher CAN: assert some `sh()` call site is lexically inside a
    // `TryStatement`'s `block` whose `handler` (catch clause) is present, and
    // that the `sh()` population is non-empty.
    // Matcher CANNOT: tell *which* `sh()` call is wrapped — a fix wrapping
    // only the typecheck call would satisfy it — nor that the catch maps to
    // INCOMPLETE rather than rethrowing or swallowing. Leg C pins the kind's
    // existence; nothing here pins the assignment, which is the residue this
    // arm set carries into S3's own review.
    //
    // Witnessed flip differential (2026-08-25, same scratch method and
    // command as leg A): input — the two staging calls `sh(["bun",
    // "install"], runDir)` and `sh(["bunx", "playwright", "install",
    // "chromium"], runDir)` wrapped in `try { … } catch (e) { result.pass =
    // null; throw e; }`. Observed: 6 pass / 1 fail, this arm the fail, `^
    // this test is marked as failing but it passed`; legs A/B/C unmoved. The
    // pre-existing `try`/`finally` around the playwright run is present in
    // that same file and does not satisfy the arm (no `handler`, no `sh()`
    // inside its block), which is what the pre-fix green reads. The
    // comment-only negative leaves it green.
    test.failing("a try/catch wraps an sh() staging call site in grade.ts (discharged by S3)", () => {
        const ast = gradeAst(evalRoot());
        const calls = shCallSites(ast);
        // Population control — the arm is about wrapping these calls, so
        // an empty set means the reader lost its subject, not that the
        // property holds.
        expect(calls.length).toBeGreaterThan(0);
        const wrapped = collect(ast, "TryStatement").some(
            (t) => t.handler != null && shCallSites(t.block as Node).length > 0,
        );
        expect(wrapped).toBe(true);
    });
});
