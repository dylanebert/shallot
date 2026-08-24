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
// gate's *kind*. So the arms that parse read a `@babel/parser` AST and assert
// a structural property — a call expression's argument node kind, a function's
// real `body` node, a `TryStatement`'s block, an `ImportSpecifier` binding
// resolved against an export set. The cardinality arm alone never parses: it
// reaches the tree through `readdirSync`/`existsSync` in `gateFiles` and
// asserts the gate population.
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
// The three S2/S3 arms are ordinary green tests asserting the PRE-FIX
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
// are resolved from that declaration's name, not hardcoded. The harness/lib
// arm resolves each gate's `ImportSpecifier` bindings from `harness/lib`
// against `lib.ts`'s export set, never comparing a name as a spelling.
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

// Collect `lib.ts`'s export set: every name reachable from an
// `ExportNamedDeclaration` — its `specifiers` (for `export { foo }`),
// declarator ids (for `export const foo = …`), `FunctionDeclaration.id`
// (for `export function foo() {}`), and type-declaration ids (for
// `export interface Foo`, `export type Foo = …`). Resolving the gates'
// `ImportSpecifier` bindings against this set — rather than comparing a
// name as a spelling — is what makes the harness/lib arm free of name
// and form pins.
function libExportedNames(ast: Node): Set<string> {
    const names = new Set<string>();
    for (const e of collect(ast, "ExportNamedDeclaration")) {
        const specifiers = e.specifiers as Node[] | undefined;
        if (specifiers) {
            for (const spec of specifiers) {
                const exported = spec.exported as Node | undefined;
                if (exported?.type === "Identifier") names.add(exported.name as string);
                else if (exported?.type === "StringLiteral") names.add(exported.value as string);
            }
        }
        const decl = e.declaration as Node | undefined;
        if (!decl) continue;
        // export function foo() {}, export interface Foo, export type Foo = …
        const id = decl.id as Node | undefined;
        if (id?.type === "Identifier") names.add(id.name as string);
        // export const foo = …, bar = …
        if (decl.type === "VariableDeclaration") {
            const declarations = decl.declarations as Node[] | undefined;
            if (declarations) {
                for (const d of declarations) {
                    const did = d.id as Node | undefined;
                    if (did?.type === "Identifier") names.add(did.name as string);
                }
            }
        }
    }
    return names;
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

    // The shared harness lib is S2's owner file. This arm resolves each gate's
    // `ImportSpecifier` bindings from `harness/lib` against `lib.ts`'s export
    // set — `ExportNamedDeclaration.specifiers`, declarator ids,
    // `FunctionDeclaration.id`, and type-declaration ids — never comparing a
    // name as a spelling. So `export const boot = async function bootImpl(…)`
    // and `async function boot(…)` + `export { boot }` are green when the
    // imported name is in the set. Deleting `lib.ts` reds here because
    // `parseFile(libPath)` throws ENOENT; an empty-but-present `lib.ts` is a
    // separate scenario that reds on the export-set population control. The
    // arm carries the import-source check too: each gate must import from
    // `harness/lib`, and the imported names must resolve, so a source-string-
    // only check that stays green with the target deleted is not what this arm
    // is.
    //
    // Pre-fix reading (this round): 6 gates, each importing from
    // `harness/lib`; `lib.ts` exports 11 names; every imported name resolves.
    test("each gate's harness/lib imports resolve against lib.ts's export set", () => {
        const root = evalRoot();
        const gates = gateFiles(root);
        expect(gates).toHaveLength(6);

        // Build lib.ts's export set.
        const libPath = join(root, "evals", "harness", "lib.ts");
        const exportedNames = libExportedNames(parseFile(libPath));
        // Population control — lib.ts must export something.
        expect(exportedNames.size).toBeGreaterThan(0);

        for (const gate of gates) {
            const ast = parseFile(gate);
            // Find imports from harness/lib and collect ImportSpecifier names.
            const importNames: string[] = [];
            for (const imp of collect(ast, "ImportDeclaration")) {
                const source = (imp.source as Node).value as string;
                if (!source.endsWith("harness/lib")) continue;
                const specs = imp.specifiers as Node[] | undefined;
                if (!specs) continue;
                for (const spec of specs) {
                    if (spec.type !== "ImportSpecifier") continue;
                    const imported = spec.imported as Node | undefined;
                    if (imported?.type === "Identifier") importNames.push(imported.name as string);
                    else if (imported?.type === "StringLiteral") importNames.push(imported.value as string);
                }
            }
            // Population control — each gate must import from harness/lib.
            expect(importNames.length).toBeGreaterThan(0);
            // Every imported name must resolve against lib.ts's export set.
            for (const name of importNames) {
                expect(exportedNames.has(name)).toBe(true);
            }
        }
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
    // gate, empty root) fails — the population control asserts the class has
    // exactly six gates and each gate has at least one `setTimeout` call.
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

// S3 — staging failure maps to INCOMPLETE. Two pre-fix pins, one per leg of
// the mechanism; S3 replaces each with the post-fix assertion in its own diff.
describe("S3 — staging failure maps to INCOMPLETE (pre-fix pin)", () => {
    // Leg B — pin (S3 replaces). `sh`'s body span in `evals/grade.ts` holds no
    // `ThrowStatement`. This pin expires with S3 — S3 replaces it with the
    // assertion that a throw exists, in the same diff.
    //
    // `sh` is located by call structure — the function whose body wraps a
    // `Bun.spawnSync` call — not by the literal name. The reader collects
    // ALL declaration forms (FunctionDeclaration, VariableDeclarator whose
    // init is an ArrowFunctionExpression or FunctionExpression) and asserts
    // exactly one match, so a decoy helper above the real subject cannot be
    // read instead. The body node is a sibling of `returnType` on the AST,
    // so the inline return-type annotation is not reachable from here by
    // construction.
    //
    // Pre-fix reading (this round): the function wrapping `Bun.spawnSync` is a
    // `FunctionDeclaration` whose body holds 0 `ThrowStatement` nodes.
    test("sh's body span in grade.ts holds no ThrowStatement (pin; S3 replaces)", () => {
        const decl = shDeclaration(gradeAst(evalRoot()));
        const body = decl.body as Node;
        expect(collect(body, "ThrowStatement").length).toBe(0);
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
