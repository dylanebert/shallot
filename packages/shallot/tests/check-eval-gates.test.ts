// A meta-test over the eval harness's structural invariants — same shape as
// `check-scripts.test.ts` (a meta-test over repo-root tooling, placed here so
// it rides the default `bun test` sweep), not a unit test of engine behaviour.
//
// Route (S1): the eval tree `evals/` sits outside `bunfig.toml`'s
// `root = "."` (widened from `"packages/shallot"` in S3), so a `bun test` arm CAN now live
// inside `evals/` and be discovered — the S3 arms under `evals/` are reached this way.
// This file was written before the widening, when `evals/` sat outside the `root` scope; it
// remains a parser-based arm (no imports from `evals/`, so no playwright/chromium dependency
// enters the suite) and is now joined by the behavioral arms the widening enabled. The rejected
// alternative was a `scripts/check-eval-gates.ts` in the `check` chain: same reach, but
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
// Leg C RETIRED (round 2 of this repair): the type-literal scan — resolve
// `grade.ts`'s `harness/result` import binding to the specific declaration
// it names in `result.ts`, then read only that declaration's own type
// members for the INCOMPLETE literal — was itself disproved by a runnable
// mutant (a decoy alias `grade.ts` ALSO imports discharges it while the real
// `ResultKind` carries something else: 1 pass / 0 fail with the property
// false), plus two further demonstrated false-negative shapes (`export {
// ResultKind }` by specifier; a template-literal union member) — the signal
// that this is the wrong KIND of arm, not the wrong reader, per this file's
// N-rounds clause. Dominance measurement, reproduced as this round's
// acceptance evidence rather than re-derived: `bunx tsc` typechecks
// `evals/` (`tsconfig.json`'s `include` covers it). Mutating ONLY the type
// (`ResultKind = "PASS" | "FAIL" | "STOPPED"`, implementation untouched)
// took `bunx tsc` from its 1-error floor to 4 errors, 2 inside `grade.ts`
// (`grade.ts(118,5)` TS2322, `grade.ts(262,9)` TS2367). Mutating type AND
// implementation consistently still reds `bunx tsc` (grade.ts's own
// literals) AND reds this arm file 10 pass / 2 fail (this leg plus the
// behavioral truth-table arm below, `S3 — result kind derivation
// (behavioral)`, whose `toBe(expected)` carries "INCOMPLETE" as an external
// expectation). So the INCOMPLETE-membership property is already carried by
// the TYPECHECKER (`bunx tsc`) UNION the BEHAVIORAL TRUTH-TABLE ARM — leg
// C's type scan was a strictly dominated proxy, and a fifth widening of its
// reader is not authorized. A later stage deleting either of those two
// enforcers voids the INCOMPLETE-membership property this file no longer
// polices itself.
//
// What survives as leg C: the one fact neither enforcer above sees —
// that `grade.ts` CONSUMES the derivation from the pure sibling module
// rather than re-declaring it inline (`result.ts` could survive untouched
// while `grade.ts` grew its own copy; the behavioral arm would stay green
// regardless, since it imports `result.ts` directly). So leg C keeps only
// the import-binding half: `grade.ts` imports from a source ending
// `harness/result`, and every name it imports resolves to a real export of
// that module. Renamed for what it now asserts, not for the property this
// round retired.
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
// same resolution, applied to `result.ts`, is what makes leg C's retained
// import-binding half a binding check rather than a spelling one.
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
    // importing from `harness/lib`; every imported name resolves against
    // whatever `lib.ts` currently exports — a count is not written here
    // because `libExportedNames` re-derives the set from the file on every
    // run, and a hand-written number beside it would rot the moment a later
    // stage adds or removes an owner.
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
// replaced its pre-fix pin with the post-fix assertion in S3's own diff. Leg
// D drops the count an adversarial review disproved for an exhaustive
// assertion over every sh() call site. Leg C (transferred here from S1 per
// Approach S1's fourth-hole verdict, then RETIRED down to an import-binding
// check in this round's repair — see the docblock above) no longer scans
// result.ts's type-level members at all; the INCOMPLETE-membership property
// is carried by bunx tsc union the behavioral truth-table arm below.
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

    // Leg C — RETIRED down to an import-binding check (round 2 of this
    // repair; see the file's top docblock for the full record). The prior
    // shape resolved `grade.ts`'s `harness/result` import binding to the
    // specific `TSTypeAliasDeclaration` it names in `result.ts` and read
    // that declaration's own union members for the INCOMPLETE literal —
    // never a file-wide scan. An adversarial pass holed even THAT with a
    // runnable mutant: `importedAliases` is computed as the UNION over
    // every imported name that resolves to a type alias, then the literal
    // scan runs over the union of their members — so a decoy alias
    // `grade.ts` ALSO imports (e.g. a second type import from the same
    // `harness/result` specifier list) discharges the check while the real
    // `ResultKind` carries something else entirely: 1 pass / 0 fail with
    // the property false. Two further false-negative shapes were
    // demonstrated on top of that (`export { ResultKind }` reached by
    // specifier rather than declaration, and a template-literal union
    // member `TSLiteralType` never matches) — the signal that an absence-
    // over-an-open-shape-space arm cannot be made exhaustive by widening
    // its reader (`checks.md`'s N-rounds clause), so this leg retires the
    // scan rather than authoring a fifth reader.
    //
    // Dominance measurement (this round's acceptance evidence, reproduced
    // rather than re-derived — see the report for the actual run):
    // `shallot/tsconfig.json`'s `include` covers `evals`, so `bunx tsc`
    // typechecks it. Mutating ONLY the type (`ResultKind = "PASS" | "FAIL"
    // | "STOPPED"`, implementation untouched) took `bunx tsc` from its
    // 1-error floor to 4 errors, 2 inside `grade.ts` (`grade.ts(118,5)`
    // TS2322, `grade.ts(262,9)` TS2367). Mutating type AND implementation
    // consistently still reds `bunx tsc` (grade.ts's own literals) AND
    // reds this arm file 10 pass / 2 fail — this leg plus the behavioral
    // truth-table arm below (`S3 — result kind derivation (behavioral)`),
    // whose `toBe(expected)` carries the string "INCOMPLETE" as an external
    // expectation. So the INCOMPLETE-membership property is carried by the
    // TYPECHECKER (`bunx tsc`) UNION the BEHAVIORAL TRUTH-TABLE ARM, and
    // the type-literal scan was a strictly dominated proxy over both. A
    // later stage that deletes either of those two enforcers is what voids
    // the property this leg used to police directly.
    //
    // What this leg still uniquely covers: neither enforcer above sees
    // whether `grade.ts` CONSUMES the derivation from the pure sibling
    // module rather than re-declaring it inline — `result.ts` could survive
    // byte-for-byte untouched while `grade.ts` grew its own copy of the
    // union and the derivation, and the behavioral arm would stay green
    // regardless, since it imports `result.ts` directly rather than through
    // `grade.ts`. So the retained assertion is import-binding only: every
    // name `grade.ts` imports from a source ending `harness/result` must
    // resolve to a real export of that module (never a spelling, never a
    // filename guess — the same binding-resolution shape as the
    // harness/lib arm above). Renamed for exactly that assertion, per
    // `checks.md`'s "name the arm for what it asserts, not for the
    // property you wish it had".
    test("grade.ts's harness/result import resolves to a real export of that module", () => {
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
                else if (imported?.type === "StringLiteral")
                    importedNames.push(imported.value as string);
            }
        }
        // Population control — grade.ts must import from harness/result, or
        // the reader lost its subject rather than the property being false.
        expect(importedNames.length).toBeGreaterThan(0);

        const resultPath = join(root, "evals", "harness", "result.ts");
        const resultAst = parseFile(resultPath);

        // Every imported name must resolve to a real export in result.ts —
        // binding resolution against the module's real export set, never a
        // spelling. This is the whole assertion: it says nothing about what
        // any resolved declaration's own members are (that property is
        // carried by bunx tsc union the behavioral truth-table arm, per the
        // docblock above).
        const exportedNames = libExportedNames(resultAst);
        // Population control — result.ts must export something.
        expect(exportedNames.size).toBeGreaterThan(0);
        for (const name of importedNames) {
            expect(exportedNames.has(name)).toBe(true);
        }
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

// Leg G — S4 repair — ladder ordering, VALUE-resolved, not membership. The review
// demonstrated the prior form of this block (and legs E/F standing alone)
// was satisfiable by a real defect: `gate.config.ts` binding `globalTimeout`
// to `BOOT_BUDGET_MS` (121_000, a real, valid `harness/lib` export, still an
// `Identifier`) — smaller than the per-test ceiling it must sit above —
// passed leg A's identity check, leg E's `exportedNames.has(resolved)`
// membership check, AND the prior form of this test, because that prior
// form imported `lib.ts`'s three canonical names DIRECTLY BY SPELLING
// (`mod.MAX_GATE_BUDGET_MS`, `mod.GLOBAL_TIMEOUT_MS`, `mod.SPAWN_BACKSTOP_MS`)
// rather than reading what each SITE is actually bound to — so it verified
// `lib.ts`'s own internal arithmetic, which the mutant never touches, and
// said nothing about the wiring. Recorded reading under that mutant: 10
// pass / 0 fail, both for `gate.config.ts`'s `globalTimeout` and for
// `grade.ts`'s `timeoutMs` bound the same way.
//
// The repair: resolve every rung's bound identifier through its OWN
// `harness/lib` `ImportSpecifier` — leg A/E/F's exact binding-resolution
// shape, no name ever written down here as a literal — then dynamically
// import `lib.ts` FROM THE ROOT UNDER TEST and read THAT RESOLVED NAME's
// own value via computed property access (`libModule[name]`, never
// `libModule.GLOBAL_TIMEOUT_MS`). A membership check only asks "is this
// name somewhere in the export set"; this asks "what is the actual runtime
// number this site is wired to", which is what the ordering invariant is
// actually about.
//
// No `@playwright/test` import: `lib.ts`'s own `@playwright/test` import is
// `import type { Page }`, erased before this module ever executes (verified
// — this dynamic import runs with no `@playwright/test` or `pngjs`
// resolution failure even where neither is installed for `Page`, since only
// the type import is erased; `pngjs`'s value import IS real, but it is a
// declared dependency already, same as `harness/result.ts`'s import-safety
// note above).
//
// Docblock's own limit: this leg can distinguish "the value bound at this
// site is smaller than the value bound at the rung above it" — nothing
// more. It cannot tell WHICH constant a site SHOULD reference (any two
// exports landing in the right relative order still pass), only that
// whatever each site is actually wired to keeps the documented ceilings
// intact end to end.
//
// Red at the parent ref (`be9a52e`): `gate.config.ts`'s `timeout` /
// `globalTimeout` and `grade.ts`'s `timeoutMs` are `NumericLiteral`s there,
// so `resolveLibImportName` returns `undefined` for all of them and the
// population controls below fail before any value is read.
describe("S4 — ladder ordering, resolved by binding (behavioral)", () => {
    test("gate <= config.timeout <= config.globalTimeout <= spawn backstop, every value read off its OWN resolved binding", async () => {
        const root = evalRoot();
        const libPath = join(root, "evals", "harness", "lib.ts");
        const libModule = (await import(libPath)) as Record<string, unknown>;

        // Resolve `identifierName` (as used at `ast`'s own site) to its
        // harness/lib binding, then read THAT NAME's own value off the
        // dynamically-imported module — never a name chosen by this file.
        function resolvedValue(ast: Node, identifierName: string): number {
            const name = resolveLibImportName(ast, identifierName);
            // Population control — the identifier must actually bind to a
            // harness/lib import, or the reader lost its subject rather
            // than the property being false.
            expect(name).toBeDefined();
            const value = libModule[name as string];
            expect(typeof value).toBe("number");
            return value as number;
        }

        // Six gates' own per-test values.
        const gates = gateFiles(root);
        expect(gates).toHaveLength(6);
        const gateValues = gates.map((gate) => {
            const ast = parseFile(gate);
            const calls = setTimeoutCalls(ast);
            expect(calls.length).toBeGreaterThan(0);
            // Cardinality — the locator below reads `[0]`; assert exactly one
            // setTimeout call in this gate file so the subject cannot silently
            // re-point at whichever member sorts first.
            expect(calls.length, `gate ${gate} must have exactly one setTimeout call`).toBe(1);
            const arg = ((calls[0].arguments as Node[]) ?? [])[0];
            expect(arg.type).toBe("Identifier");
            return resolvedValue(ast, arg.name as string);
        });

        // gate.config.ts's own `timeout` and `globalTimeout`.
        const configAst = parseFile(join(root, "evals", "harness", "gate.config.ts"));
        const defaults = collect(configAst, "ExportDefaultDeclaration");
        expect(defaults.length).toBe(1);
        const configArg = ((defaults[0].declaration as Node).arguments as Node[])[0] as Node;
        const timeoutIdent = objectProperty(configArg, "timeout") as Node | undefined;
        const globalTimeoutIdent = objectProperty(configArg, "globalTimeout") as Node | undefined;
        expect(timeoutIdent).toBeDefined();
        expect(globalTimeoutIdent).toBeDefined();
        expect((timeoutIdent as Node).type).toBe("Identifier");
        expect((globalTimeoutIdent as Node).type).toBe("Identifier");
        const configTimeoutValue = resolvedValue(configAst, (timeoutIdent as Node).name as string);
        const configGlobalValue = resolvedValue(
            configAst,
            (globalTimeoutIdent as Node).name as string,
        );

        // grade.ts's spawn backstop.
        const gAst = gradeAst(root);
        const candidates = collect(gAst, "CallExpression").filter((call) => {
            const arg = ((call.arguments as Node[]) ?? [])[0];
            return (
                arg?.type === "ObjectExpression" && objectProperty(arg, "timeoutMs") !== undefined
            );
        });
        expect(candidates.length).toBe(1);
        const spawnArg = ((candidates[0].arguments as Node[]) ?? [])[0] as Node;
        const spawnIdent = objectProperty(spawnArg, "timeoutMs") as Node;
        expect(spawnIdent.type).toBe("Identifier");
        const spawnValue = resolvedValue(gAst, spawnIdent.name as string);

        // The ordering the whole ladder exists to hold — over the values each
        // site is ACTUALLY wired to, not over names known to be somewhere in
        // lib.ts's export set.
        for (const gateValue of gateValues) {
            expect(gateValue).toBeLessThanOrEqual(configTimeoutValue);
        }
        expect(configTimeoutValue).toBeLessThanOrEqual(configGlobalValue);
        expect(configGlobalValue).toBeLessThanOrEqual(spawnValue);
    });
});

// S4 repair — blocker 2's floor. The ordering leg above shows every gate's
// resolved value sits below the ceilings above it, but that alone is silent
// on whether the value is BIG ENOUGH for what the gate itself does:
// reverting all six gates' `test.setTimeout` to `BOOT_BUDGET_MS` uniformly
// (one boot's worth) still clears every ceiling above it unchanged — the
// ordering leg stays green — while `persist-color`'s own two-`boot()` worst
// path (S2's review finding) is fully unreachable again under a nominally
// green suite. Recorded reading under that mutant: 10 pass / 0 fail.
//
// So each gate's requirement is derived from a fact only that gate's own
// source can state: how many times IT calls `boot()` — exhaustive over the
// file's own `CallExpression` population (never a fixture list; the same
// escape S3's leg D lock names, where the count is the property of the
// subject rather than a coincidental correlate), resolved through the same
// `harness/lib` binding leg A/E/F/G use (an aliased `import { boot as b }`
// still counts) rather than a bare spelling of the LOCAL identifier at each
// call site. Requirement = that count * `BOOT_BUDGET_MS`'s own resolved
// value — the harness's one owner for a single boot's worst case.
//
// Docblock's own limit: this leg can distinguish a gate whose per-test value
// has fallen below its OWN boot-count-derived floor, whatever that count is.
// It cannot distinguish where a shortfall comes from (a wrong per-test value
// vs. a wrong `BOOT_BUDGET_MS`), and it cannot tell whether a gate's boot()
// call count is itself the right number for what the gate does — only that
// the two numbers, multiplied, are covered.
//
// Reds under blocker 2's mutant: `persist-color`'s own file still calls
// `boot()` twice, so its requirement is unchanged at 2 * 121_000 = 242_000,
// but its resolved per-test value is now 121_000 (`BOOT_BUDGET_MS`) —
// 121_000 >= 242_000 is false, and the arm reds. Reads on the real tree:
// `persist-color` requires 2 * 121_000 = 242_000 and resolves to 242_000
// (`MAX_GATE_BUDGET_MS`); the other five require 1 * 121_000 = 121_000 and
// resolve to the same 242_000 — over-provisioned by the one-owner design
// (Approach S4), never under.
describe("S4 — blocker 2's floor: per-test value covers the gate's OWN boot() call count", () => {
    test("each gate's resolved per-test value is >= its own boot() call-site count * BOOT_BUDGET_MS", async () => {
        const root = evalRoot();
        const libPath = join(root, "evals", "harness", "lib.ts");
        const libModule = (await import(libPath)) as Record<string, unknown>;
        const bootBudget = libModule.BOOT_BUDGET_MS;
        // Population control — the owner's per-boot budget must resolve to a
        // real number, or the reader lost its subject rather than the
        // property being false.
        expect(typeof bootBudget).toBe("number");

        const gates = gateFiles(root);
        expect(gates).toHaveLength(6);

        for (const gate of gates) {
            const ast = parseFile(gate);

            // Exhaustive over this file's own CallExpression population:
            // every call whose callee resolves — via ImportSpecifier binding
            // from harness/lib, never a spelling of the local name — to the
            // exported name `boot`.
            const bootCalls = collect(ast, "CallExpression").filter((call) => {
                const callee = call.callee as Node | undefined;
                if (callee?.type !== "Identifier") return false;
                return resolveLibImportName(ast, callee.name as string) === "boot";
            });
            // Population control — every gate calls boot() at least once, or
            // the reader lost its subject rather than the property being
            // false.
            expect(bootCalls.length).toBeGreaterThan(0);

            const requirement = bootCalls.length * (bootBudget as number);

            const calls = setTimeoutCalls(ast);
            expect(calls.length).toBeGreaterThan(0);
            // Cardinality — the locator below reads `[0]`; assert exactly one
            // setTimeout call in this gate file so the subject cannot silently
            // re-point at whichever member sorts first.
            expect(calls.length, `gate ${gate} must have exactly one setTimeout call`).toBe(1);
            const arg = ((calls[0].arguments as Node[]) ?? [])[0];
            expect(arg.type).toBe("Identifier");
            const name = resolveLibImportName(ast, arg.name as string);
            // Population control — the setTimeout argument must actually bind
            // to a harness/lib import, or the reader lost its subject.
            expect(name).toBeDefined();
            const gateValue = libModule[name as string];
            expect(typeof gateValue).toBe("number");

            expect(gateValue as number).toBeGreaterThanOrEqual(requirement);
        }
    });
});

// Leg H, round 2 — FAIL-CLOSED, and scoped to what it can actually see. The
// round-1 shape scanned every `ObjectProperty` node in a gate file's own AST
// for a `timeout` key and required a bare `Identifier` value — blind to an
// options object built in ANOTHER file and passed by identifier (demonstrated
// live: `export const SNEAKY_OPTS = { timeout: 999_999 }` in `harness/lib.ts`,
// imported into `red-box/gate.ts` and passed as
// `page.waitForSelector("canvas", SNEAKY_OPTS)`, left round 1's leg at 4 pass /
// 0 fail with a 999_999 ms hand-written timeout live) and, in the safe
// direction, over-narrow: `timeout: 2 * FALLING_BOX_RELOAD_SELECTOR_TIMEOUT_MS`
// reds identically to a hand literal, because round 1 required the whole value
// to be a bare `Identifier`.
//
// New shape. For every gate file, take each Playwright CALL rooted at the
// test's own `page` fixture (`page.foo(...)`, `page.foo.bar(...)`,
// `page.foo(...).bar(...)` — any callee chain that bottoms out at the
// `page` identifier, walked structurally rather than by a method-name
// spelling), and read its LAST argument — the options slot in every
// Playwright method that accepts one:
//   (a) an `ObjectExpression` carrying a `timeout:` property: the value
//       expression must carry at least one `Identifier` leaf, and EVERY
//       `Identifier` leaf in it must resolve — via this gate's own
//       `ImportSpecifier` binding from `harness/lib`, `resolveLibImportName`,
//       the exact shape leg A/E/F/G already use, reused rather than
//       reinvented — to a name whose runtime value in the root under test is
//       a number. A value with zero `Identifier` leaves (a bare
//       `NumericLiteral`, or an expression built entirely from literals, e.g.
//       `2 * 20_000`) has nothing tying it to an owner and is inadmissible; a
//       literal coefficient beside a resolved leaf (`2 *
//       FALLING_BOX_RELOAD_SELECTOR_TIMEOUT_MS`) is admissible, because the
//       quantity is still traceable to a real owner, only scaled. An
//       `ObjectExpression` carrying no `timeout:` key at all (`{ steps: 12 }`)
//       is not applicable and is left alone.
//   (b) NOT an `ObjectExpression` — a bare `Identifier`, a `SpreadElement`, or
//       a `CallExpression` — resolved to its declaration: an `Identifier`
//       bound to a `harness/lib` import is read as that module's own export
//       (rule (a) applied to ITS `timeout:`, if it carries one); an
//       `Identifier` bound to a declaration in the SAME gate file is read the
//       same way if that declaration is itself an `ObjectExpression`, and
//       left alone (not applicable) if it resolves to anything else — a
//       coordinate, a computed member read — since that is structurally not
//       an options object. An `Identifier` with NO findable declaration, or a
//       `SpreadElement`/`CallExpression` argument (this leg does not evaluate
//       spreads or call results), is UNREADABLE and reds the leg — fail
//       closed, so an indirection this leg cannot see through is a red rather
//       than a silent pass.
// Any other last-argument shape (a `StringLiteral` selector, a
// `TemplateLiteral`, an event-handler `ArrowFunctionExpression`, a
// `MemberExpression` like `target.key`, a plain coordinate expression) is
// none of the forms above and is not itself carrying a `timeout:` — left
// alone, not applicable.
//
// Boundary, stated rather than claimed away: this leg reads exactly the
// shapes above — a direct options object, or ONE hop through an identifier
// bound to a `harness/lib` export or a same-file declaration. A `timeout:`
// reachable only through a DEEPER indirection (an identifier bound to a
// function call's return value, a value assembled two modules away and
// re-exported, an options object spread together at the call site) evades
// it; that shape reds this leg instead of passing silently, but it is not
// something this leg positively verifies as safe. This is a match-shape
// limit, not a promise that every restatement is caught — `checks.md`'s own
// clause on why a docblock claiming "any restatement" would be a false
// statement in a permanent file.
//
// Population control across the whole class: at least one `timeout:` is
// actually evaluated (checked, pass or fail) somewhere in the six gates
// today (falling-box's) — a root where every gate's last-argument scan finds
// nothing to evaluate would otherwise pass vacuously.
//
// Two-sided witness (reproduced this round, see the report for the actual
// readings): reverting falling-box's `waitForSelector` back to the bare
// literal `{ timeout: 20_000 }` reds this leg (zero Identifier leaves); the
// SNEAKY_OPTS mutant above reds it (the resolved harness/lib export's own
// `timeout:` is a bare `999_999`, zero Identifier leaves); the real tree is
// green; `2 * FALLING_BOX_RELOAD_SELECTOR_TIMEOUT_MS` is green (one Identifier
// leaf, owner-resolved); `2 * 20_000` reds (zero Identifier leaves).
//
// `while (Date.now() - t0 < 6_000)` on falling-box:37 is a sampling window
// (how long to keep drawing centroid samples), never an argument passed to a
// `page.*` call at all — this leg's `playwrightPageCalls` walk never reaches
// it, and it is left alone.

// Does this call's callee chain bottom out at the test's own `page` fixture?
// Walked structurally (MemberExpression.object, CallExpression.callee)
// rather than by a method-name spelling, so `page.foo(...)`,
// `page.foo.bar(...)` and `page.foo(...).bar(...)` (chained locator calls)
// are all reached the same way.
function calleeRootsAtPage(node: Node): boolean {
    if (node.type === "Identifier") return node.name === "page";
    if (node.type === "MemberExpression") return calleeRootsAtPage(node.object as Node);
    if (node.type === "CallExpression") return calleeRootsAtPage(node.callee as Node);
    return false;
}

function playwrightPageCalls(ast: Node): Node[] {
    return collect(ast, "CallExpression").filter((call) => calleeRootsAtPage(call.callee as Node));
}

// `export const NAME = <init>` in a module — generic over the module, used
// both for `harness/lib`'s exports (rule (b)'s harness/lib branch) and
// reusable for any sibling module of the same export shape.
function exportedDeclarationInit(ast: Node, name: string): Node | undefined {
    for (const e of collect(ast, "ExportNamedDeclaration")) {
        const decl = e.declaration as Node | undefined;
        if (decl?.type !== "VariableDeclaration") continue;
        for (const d of (decl.declarations as Node[] | undefined) ?? []) {
            const id = d.id as Node | undefined;
            if (id?.type === "Identifier" && id.name === name) return d.init as Node | undefined;
        }
    }
    return undefined;
}

// A same-file `const NAME = <init>` (or `let`/`var`) — file-wide match, the
// same simplification `resolveLibImportName` and friends already make
// throughout this file (no real scope analysis), used for rule (b)'s
// non-import branch (e.g. `cy` in orbit-on-drag, a plain coordinate).
function localDeclarationInit(ast: Node, name: string): Node | undefined {
    for (const d of collect(ast, "VariableDeclarator")) {
        const id = d.id as Node | undefined;
        if (id?.type === "Identifier" && id.name === name) return d.init as Node | undefined;
    }
    return undefined;
}

// Does `name`, as used at a site in `ast`, resolve through `ast`'s own
// `harness/lib` `ImportSpecifier` binding to an export whose runtime value in
// the root under test is a number? Reuses `resolveLibImportName` rather than
// re-deriving the binding resolution S4's lock already demands elsewhere in
// this file.
function resolvesToNumericOwner(
    ast: Node,
    name: string,
    libModule: Record<string, unknown>,
): boolean {
    const resolved = resolveLibImportName(ast, name);
    return resolved !== undefined && typeof libModule[resolved] === "number";
}

// Rule (a)'s admissibility test for a `timeout:` value expression: at least
// one `Identifier` leaf, and every `Identifier` leaf anywhere in the
// expression resolves to a numeric harness/lib owner. A value built entirely
// from literals (zero Identifier leaves) has nothing tying it to an owner.
// `collect` also walks a non-computed MemberExpression's own `.property`
// Identifier as a "leaf" — none of today's six gates' timeout values contain
// a MemberExpression, so this is a disclosed gap rather than a witnessed one.
function timeoutValueAdmissible(
    ast: Node,
    valueExpr: Node,
    libModule: Record<string, unknown>,
): boolean {
    const leaves = collect(valueExpr, "Identifier");
    if (leaves.length === 0) return false;
    return leaves.every((leaf) => resolvesToNumericOwner(ast, leaf.name as string, libModule));
}

type OptionsOutcome = "timeout-checked" | "unresolved" | "not-applicable";

// Rules (a) and (b) for one options argument (a Playwright call's last
// argument). Pushes a message to `violations` for every inadmissible or
// unresolvable shape; returns which of the three outcomes applied, so the
// caller's population control can tell "nothing to check here" apart from
// "checked and it held".
function evaluateOptionsArgument(
    ast: Node,
    arg: Node,
    libAst: Node,
    libModule: Record<string, unknown>,
    violations: string[],
    label: string,
): OptionsOutcome {
    if (arg.type === "ObjectExpression") {
        const tval = objectProperty(arg, "timeout");
        if (!tval) return "not-applicable";
        if (!timeoutValueAdmissible(ast, tval, libModule)) {
            violations.push(
                `${label}: a timeout: value must have >=1 Identifier leaf, every leaf resolving to a harness/lib numeric owner (got ${tval.type})`,
            );
        }
        return "timeout-checked";
    }
    if (arg.type === "Identifier") {
        const libName = resolveLibImportName(ast, arg.name as string);
        if (libName !== undefined) {
            const init = exportedDeclarationInit(libAst, libName);
            if (init === undefined) {
                violations.push(
                    `${label}: options identifier ${arg.name as string} resolves to harness/lib export ${libName}, whose own declaration this leg could not find — fail closed`,
                );
                return "unresolved";
            }
            if (init.type !== "ObjectExpression") return "not-applicable";
            const tval = objectProperty(init, "timeout");
            if (!tval) return "not-applicable";
            if (!timeoutValueAdmissible(libAst, tval, libModule)) {
                violations.push(
                    `${label}: options identifier ${arg.name as string} (harness/lib export ${libName}) carries a timeout: that does not resolve to a numeric owner (got ${tval.type})`,
                );
            }
            return "timeout-checked";
        }
        const localInit = localDeclarationInit(ast, arg.name as string);
        if (localInit === undefined) {
            violations.push(
                `${label}: options identifier ${arg.name as string} has no declaration this leg can find — fail closed`,
            );
            return "unresolved";
        }
        if (localInit.type !== "ObjectExpression") return "not-applicable";
        const tval = objectProperty(localInit, "timeout");
        if (!tval) return "not-applicable";
        if (!timeoutValueAdmissible(ast, tval, libModule)) {
            violations.push(
                `${label}: options identifier ${arg.name as string} (local declaration) carries a timeout: that does not resolve to a numeric owner (got ${tval.type})`,
            );
        }
        return "timeout-checked";
    }
    if (arg.type === "SpreadElement" || arg.type === "CallExpression") {
        violations.push(
            `${label}: options argument is a ${arg.type}, which this leg does not resolve — fail closed`,
        );
        return "unresolved";
    }
    return "not-applicable";
}

describe("timeout options — every options argument a gate passes to a page call, hand-written or one hop away", () => {
    test("every reachable timeout: resolves to a harness/lib numeric owner; an unresolvable indirection fails closed", async () => {
        const root = evalRoot();
        const gates = gateFiles(root);
        expect(gates).toHaveLength(6);

        const libPath = join(root, "evals", "harness", "lib.ts");
        const libAst = parseFile(libPath);
        const libModule = (await import(libPath)) as Record<string, unknown>;
        const exportedNames = libExportedNames(libAst);
        // Population control — lib.ts must export something.
        expect(exportedNames.size).toBeGreaterThan(0);

        const violations: string[] = [];
        let timeoutHits = 0;
        for (const gate of gates) {
            const ast = parseFile(gate);
            for (const call of playwrightPageCalls(ast)) {
                const args = (call.arguments as Node[] | undefined) ?? [];
                if (args.length === 0) continue;
                const outcome = evaluateOptionsArgument(
                    ast,
                    args[args.length - 1],
                    libAst,
                    libModule,
                    violations,
                    gate,
                );
                if (outcome === "timeout-checked") timeoutHits++;
            }
        }
        // Population control — at least one timeout: is actually evaluated
        // somewhere in the class today (falling-box's), or the reader lost its
        // subject rather than the property being universally absent.
        expect(timeoutHits).toBeGreaterThan(0);
        expect(violations).toEqual([]);
    });
});
