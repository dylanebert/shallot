import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
// A meta-test over repo-root tooling, same placement pattern as check-scripts.test.ts /
// check-exports.test.ts — `scripts/check-tumble-fp.ts` stays at `scripts/`, beside its siblings;
// only the test moves, so it rides the default `bun test` sweep instead of running by hand.
import {
    maskLiterals,
    splitTopLevel,
    stripComments,
    sweep,
    sweepLiterals,
    sweepTrig,
    sweepUnparsed,
    TRIG_ALLOWLIST,
} from "../../../scripts/check-tumble-fp";

// The real tumble engine source — S1 (audit-tumble-engine-bitexact-literals) pins the check's
// red count against this tree, so the check itself can't drift silently. The pinned floor is a
// measurement re-read at claim (spec Validation), not a target: 11, matching the spec's recorded
// sweep, and including the three live findings named in the spec's Locked decision
// (core.ts:23 OVERLAP_SLOP, hull.ts:1410 minH, distance.ts:1154 kToleranceSquared) plus eight
// sites that are equal to the C reference only by coincidence (the ratio the spec cites as the
// reason this is a standing check, not three edits).
const REAL_ROOT = resolve(import.meta.dir, "../src/standard/tumble");

// Fixture trees live under the OS tmpdir, never the repo — same `--root`-style isolation as
// check-scripts.test.ts / check-exports.test.ts, so a planted violation never touches a tracked
// file.
const roots: string[] = [];

function fixture(files: Record<string, string>): string {
    const dir = mkdtempSync(join(tmpdir(), "check-tumble-fp-"));
    roots.push(dir);
    for (const [rel, content] of Object.entries(files)) {
        const full = join(dir, rel);
        mkdirSync(join(full, ".."), { recursive: true });
        writeFileSync(full, content);
    }
    return dir;
}

afterEach(() => {
    while (roots.length > 0) {
        const dir = roots.pop();
        if (dir) rmSync(dir, { recursive: true, force: true });
    }
});

describe("splitTopLevel", () => {
    test("splits a single binary op", () => {
        expect(splitTopLevel("0.1 * LINEAR_SLOP")).toEqual(["0.1", "LINEAR_SLOP"]);
    });

    test("treats a leading unary minus as part of the operand, not a split", () => {
        expect(splitTopLevel("-0.5 * xWidth")).toEqual(["-0.5", "xWidth"]);
    });

    test("does not split inside nested parens", () => {
        expect(splitTopLevel("f32(a - b) * c")).toEqual(["f32(a - b)", "c"]);
    });

    test("does not split an exponent suffix's sign", () => {
        expect(splitTopLevel("1.1920928955078125e-7")).toEqual(["1.1920928955078125e-7"]);
    });

    test("splits a three-operand chain", () => {
        expect(splitTopLevel("2.0 * Math.PI * rowFrequency")).toEqual([
            "2.0",
            "Math.PI",
            "rowFrequency",
        ]);
    });
});

// `maskLiterals` is the S1b rebuild of the S1 `stripComments`. It masks string and comment
// content (→ spaces, newlines preserved) so a `)` or `{` inside a string never confuses the
// depth counters in `findCalls` / `splitTopLevelTokens`, and it handles template-literal
// interpolations (`${...}`) by preserving the interpolation expression as real code while masking
// the template text. Any construct it cannot close is reported as an `UnparsedSite`.
describe("maskLiterals", () => {
    test("masks a line comment, preserving the newline", () => {
        const { masked } = maskLiterals("const x = 1; // f32(0.1 * mass)\nconst y = 2;");
        expect(masked).not.toContain("f32(0.1 * mass)");
        expect(masked).toContain("\n");
        expect(masked).toContain("const y = 2;");
    });

    test("masks string content so comment-like text inside strings is invisible", () => {
        const { masked } = maskLiterals('const s = "// not a comment";');
        expect(masked).not.toContain("// not a comment");
        expect(masked).toContain("const s = ");
        expect(masked).toContain(";");
    });

    test("masks template-literal text but preserves interpolation expressions", () => {
        // biome-ignore lint/suspicious/noTemplateCurlyInString: test fixture intentionally contains template syntax
        const { masked } = maskLiterals("const msg = `hello ${f32(0.1 * mass)}`;");
        // The template text "hello " is masked, but the interpolation expression is preserved.
        expect(masked).toContain("f32(0.1 * mass)");
        expect(masked).not.toContain("hello");
    });

    test("masks nested template literals inside interpolations", () => {
        // biome-ignore lint/suspicious/noTemplateCurlyInString: test fixture intentionally contains template syntax
        const { masked } = maskLiterals("const msg = `outer ${`inner ${f32(0.1)}`}`;");
        expect(masked).toContain("f32(0.1)");
        expect(masked).not.toContain("outer");
        expect(masked).not.toContain("inner");
    });

    test("reports an unterminated string literal as an unparsed site", () => {
        const { unparsed } = maskLiterals('const s = "unterminated');
        expect(unparsed).toHaveLength(1);
        expect(unparsed[0].reason).toBe("unterminated string literal");
    });

    test("reports an unterminated template literal as an unparsed site", () => {
        const { unparsed } = maskLiterals("const msg = `unterminated");
        expect(unparsed).toHaveLength(1);
        expect(unparsed[0].reason).toBe("unterminated template literal");
    });

    test("reports an unterminated block comment as an unparsed site", () => {
        const { unparsed } = maskLiterals("/* unterminated");
        expect(unparsed).toHaveLength(1);
        expect(unparsed[0].reason).toBe("unterminated block comment");
    });

    test("reports a template literal with an unbalanced interpolation brace as an unparsed site", () => {
        // `${ {a: 1}` — the `}` closes the object literal but there is no closing `}` for the
        // interpolation itself. The backtick after `}` starts a nested template (we are still
        // inside the interpolation), which consumes the rest of the file. The template is
        // reported as unterminated — not silently swallowed.
        const { unparsed } = maskLiterals(
            // biome-ignore lint/suspicious/noTemplateCurlyInString: test fixture intentionally contains template syntax
            "const msg = `hello ${ {a: 1}`;\nconst bad = f32(0.1 * mass);\n",
        );
        expect(unparsed).toHaveLength(1);
        expect(unparsed[0].reason).toBe("unterminated template literal");
    });
});

// `stripComments` is a backward-compatible wrapper around `maskLiterals` — the masked text only,
// without the unparsed-site list. The S1 tests checked that comment content is removed and string
// content is preserved; the S1b rebuild masks string content too (so a `)` inside a string never
// confuses the depth counters), so the string test now checks masking rather than preservation.
describe("stripComments", () => {
    test("masks a line comment but preserves the newline", () => {
        const out = stripComments("const x = 1; // f32(0.1 * mass)\nconst y = 2;");
        expect(out).not.toContain("f32(0.1 * mass)");
        expect(out).toContain("\n");
        expect(out).toContain("const y = 2;");
    });

    test("masks string content so comment-like text inside strings is invisible", () => {
        const out = stripComments('const s = "// not a comment";');
        expect(out).not.toContain("// not a comment");
        expect(out).toContain("const s = ");
        expect(out).toContain(";");
    });
});

describe("TRIG_ALLOWLIST — exactly two entries", () => {
    test("granularity is (file, documented deviation), not call line — 4 call lines, 2 entries", () => {
        expect(TRIG_ALLOWLIST).toHaveLength(2);
        const totalLines = TRIG_ALLOWLIST.reduce((n, e) => n + e.lines.length, 0);
        expect(totalLines).toBe(4);
    });
});

// S1b demonstrated members — the three shapes the rebuilt predicate must catch, each with a real
// f32 divergence measured by the S1 review pass. Each arm selects its subject (a literal finding
// or an unparsed finding) out of a corpus, so cardinality is asserted in the same breath as
// content.
describe("sweepLiterals — S1b demonstrated members (structural traversal)", () => {
    // Member 1: a non-exact literal inside a sanctioned call argument. The old predicate only
    // inspected the top-level split's own operands, so `0.1` hiding inside `Math.sqrt(...)`'s
    // argument list was invisible. The rebuilt predicate recurses into every parenthesized group
    // (call-argument parens included), so `0.1` is found. The matcher distinguishes this from a
    // bare `f32(0.1 * a)` (same literal, no call wrapper) and from a properly wrapped
    // `f32(Math.sqrt(f32(0.1) * a))` (nested f32 call skipped by the recursion).
    test("flags a non-exact literal inside a sanctioned call argument: f32(Math.sqrt(0.1 * a))", async () => {
        const root = fixture({
            "fake/call-arg.ts": "export const bad = f32(Math.sqrt(0.1 * a));\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(1);
        expect(findings[0].kind).toBe("literal");
        if (findings[0].kind === "literal") {
            expect(findings[0].literals).toEqual(["0.1"]);
        }
    });

    // Control for member 1: the same shape but the literal is properly pre-wrapped in a nested
    // f32() call. The recursion skips nested f32() calls (findCalls handles them separately), so
    // this stays silent — a control green at both ends.
    test("does not flag a properly pre-wrapped literal inside a call argument", async () => {
        const root = fixture({
            "fake/wrapped-call-arg.ts": "export const good = f32(Math.sqrt(f32(0.1) * a));\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    // Member 2: a whole-level literal-only chain of 2+ ops. `f32(1 - 1/3)` has 3 literals and 2
    // operators (`-` and `/`). The intermediate `1/3 = 0.333…` is non-exact and is not rounded
    // to f32 before the subtraction, exactly the double-rounding gap rule 1a names. The old
    // predicate's `reduceLiteralCluster` was single-pass left-to-right (`(1-1)/3 = 0`, which IS
    // exact in f32), silently hiding the divergence. The rebuilt evaluator is two-pass
    // (precedence-aware), so `1 - 1/3 = 0.666…` is correctly non-exact. The matcher distinguishes
    // this from a bare `f32(1 / 3)` (1 operator, double-rounding theorem covers it, stays silent).
    test("flags a whole-level literal-only chain of 2+ ops: f32(1 - 1/3)", async () => {
        const root = fixture({
            "fake/chain.ts": "export const bad = f32(1 - 1/3);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(1);
        expect(findings[0].kind).toBe("literal");
        if (findings[0].kind === "literal") {
            expect(findings[0].literals).toEqual(["1", "1", "3"]);
        }
    });

    // Control for member 2: a bare fraction of individually-exact literals with no further fused
    // operator — `f32(1 / 3)`. This is exactly the correct place to round it (the double-rounding
    // theorem covers a two-operand op whose operands are already f32-valued), so it stays silent.
    // The matcher distinguishes this from `f32(1 - 1/3)` (2+ ops, intermediate not rounded).
    test("does not flag a bare fraction of exact literals: f32(1 / 3)", async () => {
        const root = fixture({
            "fake/bare-fraction.ts": "export const ok = f32(1 / 3);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    // Control for member 2: a whole-level chain of 2+ ops where all literals and all intermediates
    // are exactly representable — `f32(2 * 3 + 1)` = 7, exact. Must stay silent.
    test("does not flag a whole-level chain whose result is exactly representable", async () => {
        const root = fixture({
            "fake/exact-chain.ts": "export const ok = f32(2 * 3 + 1);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });
});

describe("sweepLiterals — red-first proof on a fixture tree (S1 carried forward)", () => {
    test("flags a bare non-exact literal inside an f32(...) arithmetic wrap", async () => {
        const root = fixture({
            "fake/unwrapped.ts": "export const bad = f32(0.1 * mass);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(1);
        expect(findings[0].kind).toBe("literal");
        expect(findings[0].file).toBe("fake/unwrapped.ts");
        expect(findings[0].line).toBe(1);
        if (findings[0].kind === "literal") {
            expect(findings[0].literals).toEqual(["0.1"]);
        }
    });

    test("does not flag a properly pre-wrapped literal", async () => {
        const root = fixture({
            "fake/wrapped.ts": "export const good = f32(f32(0.1) * mass);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    test("does not flag an exactly-representable literal (0.5, 2.0, 0.25, k/2^n)", async () => {
        const root = fixture({
            "fake/exact.ts":
                "export const ok = f32(0.5 * mass);\nexport const ok2 = f32(2.0 * ok);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    test("does not flag a single-literal fround with no top-level op", async () => {
        const root = fixture({
            "fake/rounded.ts": "export const eps = f32(0.1);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    test("a comment quoting the anti-pattern as documentation is not a finding", async () => {
        const root = fixture({
            "fake/documented.ts":
                "// Never write f32(0.1 * mass) — fround the literal first.\nexport const ok = f32(f32(0.1) * mass);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    test("two bad literals in one call is one finding, not two (site = call, not literal)", async () => {
        const root = fixture({
            "fake/double.ts": "export const bad = f32(0.05 * 0.05);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(1);
        if (findings[0].kind === "literal") {
            expect(findings[0].literals).toEqual(["0.05", "0.05"]);
        }
    });

    test("skips a *.test.ts file — the sweep scope is production source, not tests", async () => {
        const root = fixture({
            "fake/unwrapped.test.ts": "export const bad = f32(0.1 * mass);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    test("flags a non-exact fraction of exact literals fused with a further operator", async () => {
        const root = fixture({
            "fake/fraction.ts": "export const bad = f32(1 / 3 * mass);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(1);
        if (findings[0].kind === "literal") {
            expect(findings[0].literals).toEqual(["1", "3"]);
        }
    });

    test("does not flag an exact fraction fused with a further operator", async () => {
        const root = fixture({
            "fake/fraction.ts": "export const ok = f32(1 / 4 * mass);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    test("flags a non-exact literal nested one paren-level deeper than the wrap's top level", async () => {
        const root = fixture({
            "fake/nested.ts": "export const bad = f32((0.4 * mass) + 1);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(1);
        if (findings[0].kind === "literal") {
            expect(findings[0].literals).toEqual(["0.4"]);
        }
    });

    test("does not flag an exact literal nested one paren-level deeper than the wrap's top level", async () => {
        const root = fixture({
            "fake/nested.ts": "export const ok = f32((0.5 * mass) + 1);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });
});

describe("sweepTrig — the allowlist arm", () => {
    test("a synthetic Math.sin site outside the allowlist reds", async () => {
        const root = fixture({
            "fake/other.ts": "export const h = f32(Math.sin(x));\n",
        });
        const findings = await sweepTrig(root);
        expect(findings).toHaveLength(1);
        expect(findings[0].kind).toBe("trig");
        expect(findings[0].file).toBe("fake/other.ts");
    });

    test("Math.cos and Math.atan2 are covered by the same arm", async () => {
        const root = fixture({
            "fake/other.ts":
                "export const a = f32(Math.cos(x));\nexport const b = f32(Math.atan2(y, x));\n",
        });
        const findings = await sweepTrig(root);
        expect(findings).toHaveLength(2);
    });

    test("an allowlisted (file, line) pair on a matching relative path is not flagged", async () => {
        // Mirrors the real allowlist shape at a synthetic path so the allowlist match itself is
        // exercised without touching real engine source.
        const root = fixture({
            "engine/mesh.ts": `${"\n".repeat(863)}const rowHeight = f32(Math.sin(x));\n${"\n".repeat(2)}const columnHeight = f32(Math.sin(x));\n`,
        });
        const findings = await sweepTrig(root);
        expect(findings).toHaveLength(0);
    });
});

// S1b fail-loud invariant: any `f32(` region the lexer cannot fully decompose, or a depth counter
// that does not return to zero, is reported as an unparsed site and exits non-zero. The arm
// selects unparsed findings out of the sweep corpus, so cardinality is asserted.
describe("sweepUnparsed — the fail-loud arm", () => {
    // Member 3: a template literal carrying an unbalanced brace. The S1 hand-rolled tokenizer
    // silently swallowed the rest of the file and returned zero findings in a net — the worst
    // failure shape for a standing check. The rebuilt lexer reports it as an unparsed site.
    test("a template literal carrying an unbalanced brace is reported, not silently swallowed", async () => {
        const root = fixture({
            // biome-ignore lint/suspicious/noTemplateCurlyInString: test fixture intentionally contains template syntax
            "fake/template.ts": "const msg = `hello ${ {a: 1}`;\nconst bad = f32(0.1 * mass);\n",
        });
        const findings = await sweepUnparsed(root);
        expect(findings).toHaveLength(1);
        expect(findings[0].kind).toBe("unparsed");
        expect(findings[0].file).toBe("fake/template.ts");
    });

    // A deliberately undecomposable region — an unterminated string — is reported rather than
    // skipped. The matcher distinguishes "reported as unparsed" from "silently returns zero
    // findings": the sweep exits non-zero with at least one `unparsed` finding.
    test("a deliberately undecomposable region is reported rather than skipped", async () => {
        const root = fixture({
            "fake/undecomposable.ts":
                'const s = "unterminated string\nconst bad = f32(0.1 * mass);\n',
        });
        const findings = await sweepUnparsed(root);
        expect(findings).toHaveLength(1);
        expect(findings[0].kind).toBe("unparsed");
    });

    // An unbalanced f32() call (depth counter does not return to zero) is reported as an unparsed
    // site. The matcher distinguishes this from a balanced call with a real finding.
    test("an unbalanced f32() call is reported as an unparsed site", async () => {
        const root = fixture({
            "fake/unbalanced.ts": "export const bad = f32(0.1 * mass;\n",
        });
        const findings = await sweepUnparsed(root);
        expect(findings).toHaveLength(1);
        expect(findings[0].kind).toBe("unparsed");
        if (findings[0].kind === "unparsed") {
            expect(findings[0].reason).toBe("unbalanced parens in f32() call");
        }
    });

    // Control: a clean file with no unparsable constructs produces zero unparsed findings.
    test("a clean file produces zero unparsed findings", async () => {
        const root = fixture({
            "fake/clean.ts": "export const ok = f32(f32(0.1) * mass);\n",
        });
        const findings = await sweepUnparsed(root);
        expect(findings).toHaveLength(0);
    });
});

// The full sweep over a template-literal-with-unbalanced-brace fixture must NOT return zero
// findings — the S1 tokenizer's worst failure shape was returning zero findings in a net. The
// rebuilt sweep reports the unparsed site, so the net is non-zero.
describe("sweep — template literal with unbalanced brace is not zero findings in a net", () => {
    test("the sweep returns non-zero findings (the unparsed site), not zero", async () => {
        const root = fixture({
            // biome-ignore lint/suspicious/noTemplateCurlyInString: test fixture intentionally contains template syntax
            "fake/template.ts": "const msg = `hello ${ {a: 1}`;\nconst bad = f32(0.1 * mass);\n",
        });
        const findings = await sweep(root);
        expect(findings.length).toBeGreaterThan(0);
        const unparsed = findings.filter((f) => f.kind === "unparsed");
        expect(unparsed).toHaveLength(1);
    });
});

describe("sweep — the real engine tree, pinned floor", () => {
    test("measures exactly 11 live findings against the current tumble engine source", async () => {
        const findings = await sweep(REAL_ROOT);
        // Full site list pinned (not just the count) so a predicate drift shows exactly which
        // site moved, not just that the number did.
        const sites = findings.map((f) => `${f.file}:${f.line}`).sort();
        expect(sites).toEqual(
            [
                "engine/api.ts:1716",
                "engine/api.ts:2178",
                "engine/core.ts:16",
                "engine/core.ts:23",
                "engine/core.ts:29",
                "engine/distance.ts:1154",
                "engine/distance.ts:1249",
                "engine/geometry.ts:334",
                "engine/hull.ts:1410",
                "engine/manifold.ts:1049",
                "engine/types.ts:244",
            ].sort(),
        );
        expect(findings).toHaveLength(11);
    });

    test("the three spec-named live findings are among them", async () => {
        const findings = await sweep(REAL_ROOT);
        const sites = new Set(findings.map((f) => `${f.file}:${f.line}`));
        expect(sites.has("engine/core.ts:23")).toBe(true); // OVERLAP_SLOP
        expect(sites.has("engine/hull.ts:1410")).toBe(true); // minH
        expect(sites.has("engine/distance.ts:1154")).toBe(true); // kToleranceSquared
    });

    test("the trig arm is clean against the real tree — only the two allowlisted deviations exist", async () => {
        const findings = await sweepTrig(REAL_ROOT);
        expect(findings).toHaveLength(0);
    });

    test("the unparsed arm is clean against the real tree — no undecomposable regions", async () => {
        const findings = await sweepUnparsed(REAL_ROOT);
        expect(findings).toHaveLength(0);
    });
});
