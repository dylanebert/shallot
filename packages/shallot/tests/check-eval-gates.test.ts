// A meta-test over the eval harness's structural invariants — same shape as
// `check-scripts.test.ts` (a meta-test over repo-root tooling, placed here so
// it rides the default `bun test` sweep), not a unit test of engine behaviour.
//
// Route (S1): the eval tree `evals/` sits outside `bunfig.toml`'s
// `root = "packages/shallot"`, so a `bun test` arm cannot live inside `evals/`
// and be discovered. An arm under `packages/shallot/tests/` IS discovered and
// reaches `evals/` by parsing its sources — no imports from `evals/`, so no
// playwright/chromium dependency enters the suite. The rejected alternative
// was a `scripts/check-eval-gates.ts` in the `check` chain: same reach, but
// it reds the gate every other shallot lane runs first.
//
// ── Instrument: a parser, not a pattern ──
//
// Rounds 1–3 hand-rolled regex and brace-counting over `evals/` source. All
// three were green at their own gate and each was holed in the surface form
// that round's fixtures did not vary. `checks.md` names this class: a check
// that has failed N rounds each green at its own gate is a finding about the
// gate's *kind*. So every arm below reads a `@babel/parser` AST and asserts a
// structural property — a call expression's argument node kind, a function's
// real `body` node, a type-level union's members, a `TryStatement`'s block.
//
// `@babel/parser` (not `typescript`) is the parser because `typescript@^7` in
// this tree is the native port, whose npm package ships only `lib/tsc.js` —
// no compiler API, `ts.ScriptTarget` is `undefined` under bun.
// `@babel/parser` is a devDependency of the private workspace root, beside
// the existing `@babel/core`; the published package is `packages/shallot`,
// which ships no `tests/` entry.
//
// ── Pins (green pre-fix, red on fix) ──
//
// The four S2/S3 arms are ordinary green tests asserting the PRE-FIX
// structure. Each names the stage that replaces it and states that the pin
// expires with that stage: S2/S3 replaces its pin with the post-fix
// assertion in the same diff. A green arm that cannot read its subject
// fails, so the green-because-broken class is eliminated by construction.
// The readers are parameterized by `CHECK_EVAL_GATES_ROOT`; an absent or
// empty root reads red (every arm throws or fails its population control),
// which is the discriminating control — a byte-identical copy is not.
//
// No spelling pins: the function `sh` is located by its call structure (its
// body wraps a `Bun.spawnSync` call), never by the literal name; call sites
// are resolved from that declaration's name, not hardcoded. The owner
// identifier leg A will assert post-fix is resolved through its
// `ImportSpecifier` binding from `harness/lib`, not compared as a spelling
// against an export set — that is S2's arm, not this pin.
//
// `evals/harness/gate.config.ts`'s own `timeout: 90_000` is out of S1/S2
// scope and stays hand-written by design: each gate's `test.setTimeout`
// override wins over the config default, so the config number is inert
// while the per-gate override stands.

import { describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parse } from "@babel/parser";

type Node = { type: string } & Record<string, unknown>;

// The repo root this arm reads. Overridable so the control — an absent or
// empty root — is runnable without editing this file: the readers take a
// root, they do not hard-code the worktree path.
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
// "a comment-only change flips nothing" true by construction rather than by
// a strip pass whose ordering was round 2's hole.
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

// Does a function body contain a `Bun.spawnSync` call — the structural
// signature of `sh`, resolved by call structure rather than by a literal name.
function bodyWrapsBunSpawnSync(body: Node): boolean {
    return collect(body, "CallExpression").some((call) => {
        const callee = call.callee as Node | undefined;
        if (callee?.type !== "MemberExpression") return false;
        const obj = callee.object as Node | undefined;
        const prop = callee.property as Node | undefined;
        return (
            obj?.type === "Identifier" &&
            obj.name === "Bun" &&
            prop?.type === "Identifier" &&
            prop.name === "spawnSync"
        );
    });
}

// Locate the function whose body contains a `Bun.spawnSync` call — the
// structural signature of `sh`, resolved by call structure rather than by a
// literal name. Collects ALL declaration forms (FunctionDeclaration,
// VariableDeclarator whose init is an ArrowFunctionExpression or
// FunctionExpression) and asserts exactly one match — a decoy helper wrapping
// `Bun.spawnSync` above the real subject would otherwise be read instead.
// Returns a normalized { body, id } so callers read both the body span
// (leg B) and the declared name (leg D's call-site search) regardless of form.
function shDeclaration(ast: Node): { body: Node; id: Node | undefined } {
    const matches: { body: Node; id: Node | undefined }[] = [];
    for (const fn of collect(ast, "FunctionDeclaration")) {
        const body = fn.body as Node;
        if (bodyWrapsBunSpawnSync(body)) {
            matches.push({ body, id: fn.id as Node | undefined });
        }
    }
    for (const decl of collect(ast, "VariableDeclarator")) {
        const init = decl.init as Node | undefined;
        if (!init) continue;
        if (init.type !== "ArrowFunctionExpression" && init.type !== "FunctionExpression") continue;
        const body = init.body as Node;
        if (bodyWrapsBunSpawnSync(body)) {
            matches.push({ body, id: decl.id as Node | undefined });
        }
    }
    if (matches.length === 0) throw new Error("no function wrapping Bun.spawnSync");
    if (matches.length !== 1)
        throw new Error(`expected exactly one function wrapping Bun.spawnSync, found ${matches.length}`);
    return matches[0];
}

// Call sites of a function, resolved from the declaration's name rather than
// from a hardcoded string — so renaming `sh` does not break the reader.
function callSitesOf(ast: Node, decl: { id: Node | undefined }): Node[] {
    const id = decl.id;
    if (id?.type !== "Identifier") return [];
    const name = id.name as string;
    return collect(ast, "CallExpression").filter((call) => {
        const callee = call.callee as Node | undefined;
        return callee?.type === "Identifier" && callee.name === name;
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

    // The shared harness lib is S2's owner file. The import arm above asserts
    // an import source STRING, which passes with the target absent. This arm
    // parses `lib.ts` and asserts a structural fact — it exports the `boot`
    // driver every gate calls — so deleting `lib.ts` reds this file.
    test("harness/lib.ts exports the shared boot driver", () => {
        const ast = parseFile(join(evalRoot(), "evals", "harness", "lib.ts"));
        const exports = collect(ast, "ExportNamedDeclaration");
        expect(exports.length).toBeGreaterThan(0);
        const hasBoot = exports.some((e) => {
            const d = e.declaration as Node | undefined;
            if (d?.type === "FunctionDeclaration") {
                const id = d.id as Node | undefined;
                return id?.type === "Identifier" && id.name === "boot";
            }
            return false;
        });
        expect(hasBoot).toBe(true);
    });
});

// S2 — derived boot budget. Pre-fix pin: every gate's `setTimeout` argument
// is a `NumericLiteral`. S2 replaces this pin with the assertion that every
// argument resolves to a single owner exported by `harness/lib.ts`.
describe("S2 — derived boot budget (pre-fix pin)", () => {
    // Leg A — pin (S2 replaces). Over ALL gates under `evals/tasks/*/gate.ts`
    // as a class, not the five sites the audit named: each gate's `setTimeout`
    // argument must be a `NumericLiteral`. This pin expires with S2 — S2
    // replaces it with the post-fix assertion in the same diff.
    //
    // The arm reds the moment S2 makes any argument an `Identifier` or
    // `CallExpression`. A green arm that cannot read its subject (mis-pathed
    // gate, empty root) fails — the population control asserts the class is
    // non-empty and each gate has at least one `setTimeout` call.
    //
    // Pre-fix reading (this round): 6 gates, every `setTimeout` argument a
    // `NumericLiteral` (`80_000` ×5, `120_000` persist-color).
    test("every task gate's setTimeout argument is a NumericLiteral (pin; S2 replaces)", () => {
        const root = evalRoot();
        const gates = gateFiles(root);
        // The class is six gates — the arm carries the class claim itself
        // rather than leaning on a sibling cardinality arm, so a root with
        // five of six gates removed reds here, not just in the discovery arm.
        expect(gates).toHaveLength(6);
        for (const gate of gates) {
            const calls = setTimeoutCalls(parseFile(gate));
            expect(calls.length).toBeGreaterThan(0);
            for (const call of calls) {
                const arg = ((call.arguments as Node[]) ?? [])[0];
                expect(arg.type).toBe("NumericLiteral");
            }
        }
    });
});

// S3 — staging failure maps to INCOMPLETE. Three pre-fix pins, one per leg of
// the mechanism; S3 replaces each with the post-fix assertion in its own diff.
describe("S3 — staging failure maps to INCOMPLETE (pre-fix pin)", () => {
    // Leg B — pin (S3 replaces). `sh`'s body span in `evals/grade.ts` holds no
    // `ThrowStatement`. This pin expires with S3 — S3 replaces it with the
    // assertion that a throw exists, in the same diff.
    //
    // `sh` is located by call structure — the function whose body wraps a
    // `Bun.spawnSync` call — not by the literal name. The reader collects
    // every declaration form (FunctionDeclaration, ArrowFunctionExpression,
    // FunctionExpression) and asserts exactly one match, so a decoy helper
    // above the real subject cannot be read instead. The body node is a
    // sibling of `returnType` on the AST, so the inline return-type
    // annotation is not reachable from here by construction.
    //
    // Pre-fix reading (this round): the function wrapping `Bun.spawnSync` is a
    // `FunctionDeclaration` whose body holds 0 `ThrowStatement` nodes.
    test("sh's body span in grade.ts holds no ThrowStatement (pin; S3 replaces)", () => {
        const decl = shDeclaration(gradeAst(evalRoot()));
        const body = decl.body as Node;
        expect(collect(body, "ThrowStatement").length).toBe(0);
    });

    // Leg C — pin (S3 replaces). No type-level declaration in `evals/grade.ts`
    // carries an `INCOMPLETE` member. The reader is exhaustive: every
    // `TSLiteralType` anywhere in the AST whose literal is the string
    // `INCOMPLETE`, plus every `TSEnumMember` whose id is `INCOMPLETE`
    // (identifier or string-literal). Both sets must be empty. This pin
    // expires with S3 — S3 replaces it with the assertion that INCOMPLETE
    // exists as a declared result kind, in the same diff.
    //
    // A `console.log("INCOMPLETE")` statement cannot forge a `TSLiteralType`
    // — a string in expression position is never a type-level literal. The
    // exhaustive reader catches every S3 landing shape: a string-union alias,
    // a TS enum, a discriminated union, an interface literal field, and a
    // bare literal alias, because each produces a `TSLiteralType` or
    // `TSEnumMember` node carrying the string.
    //
    // Population control: the arm anchors on `grade.ts`'s own subject — the
    // function wrapping `Bun.spawnSync` must be present, so a decoy root with
    // a union but no `sh` reds here, not green.
    //
    // Pre-fix reading (this round): 0 `TSLiteralType` nodes whose literal is
    // `INCOMPLETE`; 0 `TSEnumMember` nodes whose id is `INCOMPLETE`.
    test("no type-level declaration in grade.ts carries an INCOMPLETE member (pin; S3 replaces)", () => {
        const ast = gradeAst(evalRoot());
        // Population control — anchor on grade.ts's own subject.
        shDeclaration(ast);
        const incompleteLiterals = collect(ast, "TSLiteralType").filter((lit) => {
            const literal = lit.literal as Node | undefined;
            return literal?.type === "StringLiteral" && literal.value === "INCOMPLETE";
        });
        expect(incompleteLiterals).toHaveLength(0);
        const incompleteEnumMembers = collect(ast, "TSEnumMember").filter((m) => {
            const id = m.id as Node | undefined;
            if (id?.type === "StringLiteral") return id.value === "INCOMPLETE";
            if (id?.type === "Identifier") return id.name === "INCOMPLETE";
            return false;
        });
        expect(incompleteEnumMembers).toHaveLength(0);
    });

    // Leg D — pin (S3 replaces). No `sh()` call site sits inside a `try` with a
    // `handler`. This pin expires with S3 — S3 replaces it with the assertion
    // that a try/catch wraps a staging call, in the same diff.
    //
    // `sh` is located by call structure (the reader above), and call sites are
    // resolved from that declaration's name — not hardcoded. The population
    // control asserts the call-site set is non-empty, so the arm fails if the
    // reader loses its subject.
    //
    // Pre-fix reading (this round): 4 call sites of the function wrapping
    // `Bun.spawnSync`, 0 inside a `TryStatement` with a `handler`; 1
    // `TryStatement` total (handler absent, no `sh()` call in its block).
    test("no sh() call site sits inside a try with a handler (pin; S3 replaces)", () => {
        const ast = gradeAst(evalRoot());
        const decl = shDeclaration(ast);
        const calls = callSitesOf(ast, decl);
        // Population control — the arm is about wrapping these calls, so an
        // empty set means the reader lost its subject, not that the property
        // holds.
        expect(calls.length).toBeGreaterThan(0);
        const wrapped = collect(ast, "TryStatement").some(
            (t) => t.handler != null && callSitesOf(t.block as Node, decl).length > 0,
        );
        expect(wrapped).toBe(false);
    });
});
