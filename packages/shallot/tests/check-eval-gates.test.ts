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
// S2 and S3 have both landed and replaced their pins with post-fix
// assertions in their own diffs — every task gate's setTimeout argument
// resolves, via its ImportSpecifier binding from harness/lib, to the same
// single name in lib.ts's export set (leg A); `sh`'s body now holds a
// ThrowStatement (leg B).
//
// Leg D repaired: a review disproved the landed "a try with a handler wraps
// ≥2 sh() call sites together" count with two runnable mutants (false
// positive: group the two graded sites into one try, leave both staging
// sites unwrapped, still reads ≥2 while an install failure now throws
// uncaught; false negative: wrap each of the four calls in its own
// single-call try/catch, a fully correct fix that reads <2 everywhere). The
// repair is exhaustive, never counted: every `sh()` call site sits inside a
// try with a handler, full stop, and no argv-literal (`"install"` vs
// `"tsc"`) discriminates a site.
//
// Leg C repaired: the verdict derivation (and `ResultKind`) is extracted to
// the pure, import-safe `evals/harness/result.ts`; `grade.ts` imports both.
// Leg C now resolves `grade.ts`'s `harness/result` import binding to a real
// declaration in `result.ts` — never a spelling, never a filename guess —
// admissible as a presence check only because this diff creates the file.
//
// `result.ts` also gets this unit's first BEHAVIORAL arm (below the S3
// describe block): `grade.ts` is a top-level script that can never be
// imported by a test, but its pure sibling can be, so an arm imports
// `deriveResultKind` and calls it across the exhaustive truth table of its
// three inputs — replacing a structural proxy with a real call, and pinning
// Law 3 (a determined typecheck/build failure outranks an unrunnable gate:
// INCOMPLETE only when NOTHING determined the outcome).
//
// No pins remain green pre-fix in this file — every leg asserts the
// POST-FIX structure. A green arm that cannot read its subject fails, so the
// green-because-broken class is eliminated by construction. The readers are
// parameterized by `CHECK_EVAL_GATES_ROOT`; an absent or empty root reads red
// (every arm throws or fails its population control), which is the
// discriminating control — a byte-identical copy is not.
//
// No spelling pins: the function `sh` is located by its call structure (its
// body wraps a `Bun.spawnSync` call), never by the literal name; call sites
// are resolved from that declaration's name, not hardcoded. The harness/lib
// arm resolves each gate's `ImportSpecifier` bindings from `harness/lib`
// against `lib.ts`'s export set, never comparing a name as a spelling — the
// same resolution, applied to `result.ts`, is what makes leg C's repair a
// binding check rather than a spelling one.
//
// `evals/harness/gate.config.ts`'s own `timeout` default was out of S1/S2
// scope (each gate's `test.setTimeout` override wins over it, so it stays
// inert while the per-gate override stands) but is in S4's: both `timeout`
// and `globalTimeout` now derive from `harness/lib` too (S4, below), even
// though `timeout` remains inert for the same reason.

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
        throw new Error(
            `expected exactly one function wrapping Bun.spawnSync, found ${matches.length}`,
        );
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

// Collect a module's export set: every name reachable from an
// `ExportNamedDeclaration` — its `specifiers` (for `export { foo }`),
// declarator ids (for `export const foo = …`), `FunctionDeclaration.id`
// (for `export function foo() {}`), and type-declaration ids (for
// `export interface Foo`, `export type Foo = …`). Resolving an
// `ImportSpecifier` binding against this set — rather than comparing a
// name as a spelling — is what makes the harness/lib arm (and the
// harness/result arm below it) free of name and form pins. Generic over
// its subject module: called on `lib.ts` for the harness/lib arm and on
// `result.ts` for leg C.
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

// Resolve an `Identifier` used as a config value (a `setTimeout` argument, a
// `timeout`/`globalTimeout` property, a `timeoutMs` property) to the name it
// was imported as, by matching its local name against the AST's own
// `ImportSpecifier`s from a source ending `harness/lib` — never by comparing
// the identifier's name against a hardcoded string. Returns undefined if the
// identifier isn't bound to such an import (e.g. it's a literal, a local
// const, or imported from elsewhere). Module-scope (not S2-local) so S4's
// legs over `gate.config.ts` and `grade.ts` share leg A's exact binding
// resolution rather than re-deriving it — moving this function is not a
// change to what leg A asserts.
//
// The source suffix is `/lib` rather than `harness/lib`: `gate.config.ts`
// lives INSIDE `harness/`, so its own import of the same module is `./lib`
// (no `harness` segment), where every other reader of this module
// (`grade.ts`, the six task gates) sits one level out and writes
// `harness/lib`. Both are the same file; the suffix check names the file,
// not a spelling of the identifier under test.
function resolveLibImportName(ast: Node, identifierName: string): string | undefined {
    for (const imp of collect(ast, "ImportDeclaration")) {
        const source = (imp.source as Node).value as string;
        if (!source.endsWith("/lib") && source !== "./lib") continue;
        const specs = imp.specifiers as Node[] | undefined;
        if (!specs) continue;
        for (const spec of specs) {
            if (spec.type !== "ImportSpecifier") continue;
            const local = spec.local as Node | undefined;
            if (local?.type !== "Identifier" || local.name !== identifierName) continue;
            const imported = spec.imported as Node | undefined;
            if (imported?.type === "Identifier") return imported.name as string;
            if (imported?.type === "StringLiteral") return imported.value as string;
        }
    }
    return undefined;
}

// Extract an `ObjectExpression`'s property value by key name — a schema
// field on the object being read (`gate.config.ts`'s own `timeout` /
// `globalTimeout`, `RunArgs`' own `timeoutMs`), never the subject under
// test itself. The subject under test is always the property's VALUE,
// resolved below via `resolveLibImportName`, never the key.
function objectProperty(obj: Node, key: string): Node | undefined {
    const props = (obj.properties as Node[] | undefined) ?? [];
    for (const p of props) {
        if (p.type !== "ObjectProperty") continue;
        const k = p.key as Node;
        if (k.type === "Identifier" && k.name === key) return p.value as Node;
        if (k.type === "StringLiteral" && k.value === key) return p.value as Node;
    }
    return undefined;
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
    // Reading at S2 (the count moves with each stage that adds an owner, so
    // it is a convenience, not a claim this arm makes): 6 gates, each
    // importing from `harness/lib`; `lib.ts` exports 12 names — 11 at S1 plus
    // the boot budget S2 added; every imported name resolves.
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
                    else if (imported?.type === "StringLiteral")
                        importNames.push(imported.value as string);
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

// S2 — derived boot budget (landed). Leg A's post-fix assertion: every
// gate's `setTimeout` argument is an `Identifier` whose binding resolves —
// via that file's own `ImportSpecifier` from `harness/lib` — to the same
// single name in `lib.ts`'s export set. "Same single name" is the one-owner
// property: the arithmetic lives once in `lib.ts`, and every gate derives
// from it rather than restating a number.
describe("S2 — derived boot budget", () => {
    // Leg A — post-fix (S2 landed; replaces the pre-fix "every argument is a
    // NumericLiteral" pin). Over ALL gates under `evals/tasks/*/gate.ts` as a
    // class, not the five sites the audit named: each gate's `setTimeout`
    // argument must resolve to an import from `harness/lib` whose imported
    // name is (a) present in `lib.ts`'s real export set and (b) identical
    // across every gate — one owner, not six hand-written numbers that
    // happen to agree. No spelling pin: the owner's name is never written
    // down here, only derived by resolving each usage's own binding and
    // comparing the six resolved names to each other and to the export set
    // `libExportedNames` reads off `lib.ts`.
    //
    // The arm reds the moment any gate reverts to a `NumericLiteral` or
    // literal-timeout argument, imports the budget from somewhere other than
    // `harness/lib`, or two gates derive from different exported names. A
    // green arm that cannot read its subject (mis-pathed gate, empty root)
    // fails — the population control asserts the class has exactly six gates,
    // each with at least one `setTimeout` call, each argument resolving.
    //
    // Post-fix reading (this round): 6 gates, every `setTimeout` argument an
    // `Identifier` bound to the same `harness/lib` import, resolving to the
    // same name in `lib.ts`'s export set.
    test("every task gate's setTimeout argument derives from one harness/lib owner (post-fix; S2)", () => {
        const root = evalRoot();
        const gates = gateFiles(root);
        // The class is six gates — the arm carries the class claim itself
        // rather than leaning on a sibling cardinality arm, so a root with
        // five of six gates removed reds here, not just in the discovery arm.
        expect(gates).toHaveLength(6);

        const libPath = join(root, "evals", "harness", "lib.ts");
        const exportedNames = libExportedNames(parseFile(libPath));
        // Population control — lib.ts must export something.
        expect(exportedNames.size).toBeGreaterThan(0);

        let ownerName: string | undefined;
        for (const gate of gates) {
            const ast = parseFile(gate);
            const calls = setTimeoutCalls(ast);
            expect(calls.length).toBeGreaterThan(0);
            for (const call of calls) {
                const arg = ((call.arguments as Node[]) ?? [])[0];
                expect(arg.type).toBe("Identifier");
                const resolved = resolveLibImportName(ast, arg.name as string);
                // Population control — the argument must actually bind to a
                // harness/lib import, or the reader lost its subject.
                expect(resolved).toBeDefined();
                const name = resolved as string;
                expect(exportedNames.has(name)).toBe(true);
                if (ownerName === undefined) ownerName = name;
                // One owner: every gate resolves to the same exported name.
                expect(name).toBe(ownerName);
            }
        }
    });
});

// S3 — staging failure maps to INCOMPLETE (landed, then repaired). Leg B
// replaced its pre-fix pin with the post-fix assertion in S3's own diff. Legs
// D and C are this repair round's: leg D drops the count an adversarial
// review disproved for an exhaustive assertion over every sh() call site;
// leg C (transferred here from S1 per Approach S1's fourth-hole verdict) now
// resolves grade.ts's import binding into the newly-extracted result.ts
// rather than reading a union type declared in grade.ts itself.
describe("S3 — staging failure maps to INCOMPLETE (post-fix)", () => {
    // Leg B — post-fix (S3 landed; replaces the pre-fix "holds no
    // ThrowStatement" pin). `sh`'s body span in `evals/grade.ts` now holds a
    // `ThrowStatement` — `sh` throws on a nonzero exit instead of returning
    // `{ ok: false }` silently.
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
    // Post-fix reading (this round): the function wrapping `Bun.spawnSync` is
    // a `FunctionDeclaration` whose body holds 1 `ThrowStatement` node.
    test("sh's body span in grade.ts holds a ThrowStatement (post-fix; S3)", () => {
        const decl = shDeclaration(gradeAst(evalRoot()));
        const body = decl.body as Node;
        expect(collect(body, "ThrowStatement").length).toBeGreaterThan(0);
    });

    // Leg D — repaired (a review disproved the landed count pin with two
    // runnable mutants). The landed form asserted "a try with a handler
    // wraps ≥2 sh() call sites together", which is satisfiable by the wrong
    // try: grouping the two *graded* sites (`tsc`, the build) into one
    // shared try/catch while leaving both *staging* sites unwrapped reads
    // ≥2 and stays green even though an install failure now throws
    // uncaught — strictly worse than the pre-fix defect. And it is holed
    // the other way too: wrapping each of the four `sh()` calls in its own
    // single-call try/catch (each staging catch still routing to
    // INCOMPLETE) is a fully correct alternative fix that reads <2
    // everywhere and goes red.
    //
    // The repair drops the count and goes exhaustive: EVERY `sh()` call
    // site — not some subset, not a particular pair — must sit inside a
    // try with a handler. Safety is exhaustiveness over the file's own
    // call-site set (a source of truth), not a fixture list, and no argv
    // literal (`"install"` vs `"tsc"`) discriminates a site — that would be
    // a spelling pin the next refactor moves, the exact class this repairs.
    //
    // `sh` is located by call structure (the reader above), and call sites
    // are resolved from that declaration's name — not hardcoded. A call
    // site counts as wrapped if it is contained (by node identity, so
    // arbitrarily nested inside the try's block still counts) in some
    // `TryStatement` whose `handler` is non-null.
    //
    // Post-fix reading (this round): 4 call sites of the function wrapping
    // `Bun.spawnSync`; every one of the 4 sits inside a try/catch, whatever
    // the grouping (today: 3 try/catches, one wrapping the staging pair,
    // two wrapping one graded call site each).
    test("every sh() call site sits inside a try with a handler (post-fix; S3)", () => {
        const ast = gradeAst(evalRoot());
        const decl = shDeclaration(ast);
        const calls = callSitesOf(ast, decl);
        // Population control — the arm is about wrapping these calls, so an
        // empty set means the reader lost its subject, not that the property
        // holds.
        expect(calls.length).toBeGreaterThan(0);

        // Every sh() call site reachable inside some try/catch's block —
        // collected by node identity, so a call nested arbitrarily deep
        // inside the block (an if, another call's argument) still counts,
        // and a call outside every try/catch's block never does.
        const wrappedCalls = new Set<Node>();
        for (const t of collect(ast, "TryStatement")) {
            if (t.handler == null) continue;
            for (const call of callSitesOf(t.block as Node, decl)) wrappedCalls.add(call);
        }
        // Population control — at least one try/catch actually wraps a real
        // sh() call site; an empty set means the reader lost its subject
        // rather than the property being false.
        expect(wrappedCalls.size).toBeGreaterThan(0);

        for (const call of calls) {
            expect(wrappedCalls.has(call)).toBe(true);
        }
    });

    // Leg C — repaired (Law 2). `grade.ts` no longer declares `ResultKind`
    // itself: the verdict derivation (and its `ResultKind`) is extracted to
    // the pure, import-safe sibling `evals/harness/result.ts`, and `grade.ts`
    // imports both the derivation function and the type. Leg C's obligation
    // — resolve the subject to a real declaration, never a spelling or a
    // filename guess — now reads as: `grade.ts`'s `ImportDeclaration` from a
    // source ending `harness/result` resolves (via the same binding-
    // resolution the harness/lib arm uses, `libExportedNames`, applied here
    // to `result.ts`) against a real export of that module, and that module
    // itself declares a union type carrying an INCOMPLETE member — a
    // presence check admissible only because this diff is the one that
    // creates `result.ts`.
    //
    // Post-fix reading (this round): `grade.ts` imports from a source ending
    // `harness/result`; every imported name resolves against `result.ts`'s
    // export set; `result.ts` declares 1 `TSTypeAliasDeclaration` with a
    // `TSUnionType` annotation, 3 `TSLiteralType` members — `"PASS"`,
    // `"FAIL"`, `"INCOMPLETE"`.
    test("grade.ts's harness/result import resolves to a union type carrying an INCOMPLETE member (post-fix; S3)", () => {
        const root = evalRoot();
        const ast = gradeAst(root);

        const importedNames: string[] = [];
        for (const imp of collect(ast, "ImportDeclaration")) {
            const source = (imp.source as Node).value as string;
            if (!source.endsWith("harness/result")) continue;
            for (const spec of (imp.specifiers as Node[] | undefined) ?? []) {
                if (spec.type !== "ImportSpecifier") continue;
                const imported = spec.imported as Node | undefined;
                if (imported?.type === "Identifier") importedNames.push(imported.name as string);
                else if (imported?.type === "StringLiteral") importedNames.push(imported.value as string);
            }
        }
        // Population control — grade.ts must import from harness/result, or
        // the reader lost its subject rather than the property being false.
        expect(importedNames.length).toBeGreaterThan(0);

        const resultPath = join(root, "evals", "harness", "result.ts");
        const resultAst = parseFile(resultPath);
        const exportedNames = libExportedNames(resultAst);
        // Population control — result.ts must export something.
        expect(exportedNames.size).toBeGreaterThan(0);
        for (const name of importedNames) {
            expect(exportedNames.has(name)).toBe(true);
        }

        // Presence: result.ts itself declares a union type carrying the
        // INCOMPLETE literal — never resolved by a bare string anywhere in
        // the file (a display string, a comment) since only a
        // `TSUnionType` member produces this node shape.
        const aliases = collect(resultAst, "TSTypeAliasDeclaration");
        expect(aliases.length).toBeGreaterThan(0);
        const literalValues: string[] = [];
        for (const alias of aliases) {
            const ann = alias.typeAnnotation as Node | undefined;
            if (ann?.type !== "TSUnionType") continue;
            for (const member of (ann.types as Node[] | undefined) ?? []) {
                if (member.type !== "TSLiteralType") continue;
                const literal = member.literal as Node | undefined;
                if (literal?.type === "StringLiteral") literalValues.push(literal.value as string);
            }
        }
        // Population control — at least one union-type alias with string-
        // literal members exists; an empty list means the reader found the
        // wrong declaration shape, not that the property is false.
        expect(literalValues.length).toBeGreaterThan(0);
        expect(literalValues).toContain("INCOMPLETE");
    });
});

// S3 repair, Law 2 & 3 — the unit's first behavioral oracle. `grade.ts` is a
// top-level script (argv parsing, top-level await) that can never be
// imported by an arm — every leg above reads its AST instead of running it.
// `evals/harness/result.ts` is a pure sibling with no @playwright/test
// import, so it CAN be imported and called directly: this arm imports it and
// drives `deriveResultKind` across the exhaustive truth table of its three
// determined inputs, replacing a structural proxy with a real call.
//
// This is also Law 3's witness: a determined typecheck or build failure
// outranks an unrunnable gate — INCOMPLETE is reserved for a run where
// NOTHING determined the outcome, so `deriveResultKind(false, true, null)`
// is FAIL, never INCOMPLETE, even though the gate never ran.
//
// Red at the parent ref: `harness/result.ts` does not exist there, so the
// dynamic import throws and the test fails outright — never a green arm
// that cannot read its subject.
describe("S3 — result kind derivation (behavioral)", () => {
    test("deriveResultKind covers the exhaustive truth table of its three determined inputs", async () => {
        const root = evalRoot();
        const resultPath = join(root, "evals", "harness", "result.ts");
        const mod = (await import(resultPath)) as {
            deriveResultKind?: (a: boolean, b: boolean, c: boolean | null) => unknown;
        };
        // Population control — the module must actually export the function
        // under test, or the reader lost its subject rather than the
        // property being false.
        expect(typeof mod.deriveResultKind).toBe("function");
        const deriveResultKind = mod.deriveResultKind as (
            a: boolean,
            b: boolean,
            c: boolean | null,
        ) => unknown;

        const bools = [true, false];
        const gateVals: (boolean | null)[] = [true, false, null];
        let cases = 0;
        for (const typecheckOk of bools) {
            for (const buildOk of bools) {
                for (const gateOk of gateVals) {
                    cases++;
                    const expected =
                        typecheckOk === false || buildOk === false
                            ? "FAIL"
                            : gateOk === null
                              ? "INCOMPLETE"
                              : gateOk
                                ? "PASS"
                                : "FAIL";
                    expect(deriveResultKind(typecheckOk, buildOk, gateOk)).toBe(expected);
                }
            }
        }
        // Population control — the truth table actually iterated all 12
        // combinations (2 × 2 × 3); a broken loop reads as a pass on zero
        // cases.
        expect(cases).toBe(12);
    });
});

// S4 — the timeout ladder derives from the one owner. S2's `BOOT_BUDGET_MS`
// covers one `boot()` call; the review that closed S2 found two gates whose
// real worst path is worse than that — `persist-color` calls `boot()` TWICE
// (top, then after the reload its own positive claim requires), and
// `falling-box` adds an un-retried post-reload selector wait plus an
// unconditional sampling loop on top of one boot — and found two more
// ceilings above the per-test budget, neither derived from anything:
// `gate.config.ts`'s `globalTimeout` (a whole-run wall clock no per-test
// call can override) and `grade.ts`'s spawn `timeoutMs` backstop. Both sat
// BELOW the real worst-case per-test path, so `persist-color`'s documented
// worst path was never reachable — the run dies with no envelope before its
// own `test.setTimeout` would ever fire.
//
// This stage's shape: ONE OWNER SIZED TO THE WORST GATE, not per-gate
// expressions. `harness/lib` now exports `MAX_GATE_BUDGET_MS` (2 *
// `BOOT_BUDGET_MS` — covers `persist-color`'s two-boot path, and covers
// `falling-box`'s smaller-magnitude shape too: BOOT_BUDGET_MS + a 20s
// selector wait + a 6s sample loop is well under 2 * BOOT_BUDGET_MS). All
// six gates derive their `test.setTimeout` from this SAME name — the
// withdrawn design lock's differentiator (Approach S4) no longer decides
// between one owner and per-gate expressions, because a per-gate budget at
// `persist-color` alone would still have to clear the same two ceilings
// above it that a shared worst-gate owner does, so per-gate arithmetic
// buys nothing here at the cost of restating the same 2×boot expression at
// two sites (`persist-color`'s own gate, and whatever else composes
// `falling-box`'s shape) instead of one. `GLOBAL_TIMEOUT_MS` and
// `SPAWN_BACKSTOP_MS` (each one `BOOT_BUDGET_MS` of headroom above the rung
// below) are the other two owned names — every rung in the ladder is now a
// bare identifier imported from `harness/lib`, never a literal.
//
// Consequence for leg A (S2's, above): UNCHANGED. Leg A's property is that
// every gate resolves to the SAME single exported name, and this stage kept
// that true (the name is now `MAX_GATE_BUDGET_MS` instead of
// `BOOT_BUDGET_MS`, but leg A never pins the name itself — it compares the
// six resolved names to each other and to `lib.ts`'s real export set, so it
// reads the new owner with no edit to its own assertions). Only
// `resolveLibImportName` moved to module scope, so this describe block's
// two structural legs can reuse leg A's exact binding resolution rather
// than re-deriving it — a refactor, not a behavior change (verified: leg A
// above still passes unedited).
describe("S4 — the timeout ladder derives from the one owner", () => {
    // Leg E — `gate.config.ts`'s own `timeout` and `globalTimeout` each
    // resolve, via THIS file's `ImportSpecifier` binding from `harness/lib`,
    // to a name in `lib.ts`'s real export set — leg A's exact shape, applied
    // to a different file and a different call: never comparing the
    // resolved name, or the wrapping call's own callee (`defineConfig`), as
    // a spelling. Located structurally: the config module's ONE
    // `export default <call>`, whose sole argument is the config object
    // literal — a second default export, or a default export that isn't a
    // call, fails the population control below rather than silently reading
    // the wrong node. `timeout`/`globalTimeout` are the object's own schema
    // field names (Playwright's `TestConfig`), not the subject under test —
    // the subject is each field's VALUE, resolved by binding.
    //
    // Reds at the parent ref (`be9a52e`): both values are `NumericLiteral`s
    // there, so `expect(v.type).toBe("Identifier")` fails before binding
    // resolution is even reached.
    test("gate.config.ts's timeout and globalTimeout resolve against harness/lib's export set", () => {
        const root = evalRoot();
        const configPath = join(root, "evals", "harness", "gate.config.ts");
        const ast = parseFile(configPath);

        const defaults = collect(ast, "ExportDefaultDeclaration");
        // Population control — exactly one default export, or the reader
        // lost its subject rather than the property being false.
        expect(defaults.length).toBe(1);
        const decl = defaults[0].declaration as Node;
        expect(decl.type).toBe("CallExpression");
        const configArg = ((decl.arguments as Node[]) ?? [])[0];
        expect(configArg?.type).toBe("ObjectExpression");

        const libPath = join(root, "evals", "harness", "lib.ts");
        const exportedNames = libExportedNames(parseFile(libPath));
        // Population control — lib.ts must export something.
        expect(exportedNames.size).toBeGreaterThan(0);

        for (const key of ["timeout", "globalTimeout"]) {
            const value = objectProperty(configArg as Node, key);
            // Population control — both ceiling properties must be present in
            // the config object, or the reader lost its subject rather than
            // the property being false.
            expect(value).toBeDefined();
            const v = value as Node;
            expect(v.type).toBe("Identifier");
            const resolved = resolveLibImportName(ast, v.name as string);
            // Population control — the value must actually bind to a
            // harness/lib import, or the reader lost its subject.
            expect(resolved).toBeDefined();
            expect(exportedNames.has(resolved as string)).toBe(true);
        }
    });

    // Leg F — `grade.ts`'s spawn backstop (`runPlaywright({ …, timeoutMs })`)
    // resolves the same way. Located structurally: the one call site
    // anywhere in `grade.ts` whose first argument is an object literal
    // carrying a `timeoutMs` property — never by comparing the call's own
    // callee (`runPlaywright`) as a spelling. A second such call site (or
    // none) fails the population control rather than silently reading the
    // wrong one.
    //
    // Reds at the parent ref: the property's value is a `NumericLiteral`
    // (`240_000`) there.
    test("grade.ts's spawn backstop (timeoutMs) resolves against harness/lib's export set", () => {
        const root = evalRoot();
        const ast = gradeAst(root);

        const candidates = collect(ast, "CallExpression").filter((call) => {
            const arg = ((call.arguments as Node[]) ?? [])[0];
            return (
                arg?.type === "ObjectExpression" && objectProperty(arg, "timeoutMs") !== undefined
            );
        });
        // Population control — exactly one call site in grade.ts carries a
        // timeoutMs property, or the reader lost its subject rather than the
        // property being false.
        expect(candidates.length).toBe(1);
        const arg = ((candidates[0].arguments as Node[]) ?? [])[0] as Node;
        const value = objectProperty(arg, "timeoutMs") as Node;
        expect(value.type).toBe("Identifier");

        const libPath = join(root, "evals", "harness", "lib.ts");
        const exportedNames = libExportedNames(parseFile(libPath));
        expect(exportedNames.size).toBeGreaterThan(0);
        const resolved = resolveLibImportName(ast, value.name as string);
        expect(resolved).toBeDefined();
        expect(exportedNames.has(resolved as string)).toBe(true);
    });
});

// S4 — ladder ordering (behavioral). The two structural legs above (and leg
// A, S2's) each show a rung resolves to SOME name in `lib.ts`'s export set;
// none of them can see whether the resulting NUMBERS are actually ordered
// per-test <= config ceiling <= spawn backstop — the ordering invariant the
// whole stage exists to hold. That is a numeric property over three plain
// exported constants, cheaply expressible as a pure read rather than an AST
// walk, so per Validation's preference for the behavioral route over a
// structural proxy wherever the property is a pure function, this arm
// imports `harness/lib` directly and reads its three numbers — priced over
// a structural alternative (parsing three `BinaryExpression` trees and
// evaluating them by hand, which would just re-derive `lib.ts`'s own
// arithmetic a second time, the exact re-derivation class this spec exists
// to close) and chosen because it is both cheaper and immune to that class.
//
// No `@playwright/test` import: `lib.ts`'s own `@playwright/test` import is
// `import type { Page }`, erased before this module ever executes (verified
// — this dynamic import runs with no `@playwright/test` or `pngjs`
// resolution failure even where neither is installed for `Page`, since only
// the type import is erased; `pngjs`'s value import IS real, but it is a
// declared dependency already, same as `harness/result.ts`'s import-safety
// note above).
//
// Red at the parent ref: `lib.ts` there exports none of these three names,
// so each is `undefined` and the `typeof … === "number"` population control
// fails outright — never a green arm that cannot read its subject.
describe("S4 — ladder ordering (behavioral)", () => {
    test("MAX_GATE_BUDGET_MS <= GLOBAL_TIMEOUT_MS <= SPAWN_BACKSTOP_MS, all derived from harness/lib", async () => {
        const root = evalRoot();
        const libPath = join(root, "evals", "harness", "lib.ts");
        const mod = (await import(libPath)) as {
            // biome-ignore lint/style/useNamingConvention: lib.ts's own real export names.
            MAX_GATE_BUDGET_MS?: number;
            // biome-ignore lint/style/useNamingConvention: lib.ts's own real export names.
            GLOBAL_TIMEOUT_MS?: number;
            // biome-ignore lint/style/useNamingConvention: lib.ts's own real export names.
            SPAWN_BACKSTOP_MS?: number;
        };
        // Population control — the module must actually export all three
        // names as numbers, or the reader lost its subject rather than the
        // property being false.
        expect(typeof mod.MAX_GATE_BUDGET_MS).toBe("number");
        expect(typeof mod.GLOBAL_TIMEOUT_MS).toBe("number");
        expect(typeof mod.SPAWN_BACKSTOP_MS).toBe("number");

        const perTest = mod.MAX_GATE_BUDGET_MS as number;
        const configCeiling = mod.GLOBAL_TIMEOUT_MS as number;
        const spawnBackstop = mod.SPAWN_BACKSTOP_MS as number;
        expect(perTest).toBeLessThanOrEqual(configCeiling);
        expect(configCeiling).toBeLessThanOrEqual(spawnBackstop);
    });
});
