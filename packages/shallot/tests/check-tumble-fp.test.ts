import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
// A meta-test over repo-root tooling, same placement pattern as check-scripts.test.ts /
// check-exports.test.ts — `scripts/check-tumble-fp.ts` stays at `scripts/`, beside its siblings;
// only the test moves, so it rides the default `bun test` sweep instead of running by hand.
import {
    splitTopLevel,
    stripComments,
    sweep,
    sweepLiterals,
    sweepTrig,
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

describe("stripComments", () => {
    test("blanks a line comment but preserves the newline", () => {
        const out = stripComments("const x = 1; // f32(0.1 * mass)\nconst y = 2;");
        expect(out).toBe("const x = 1; \nconst y = 2;");
    });

    test("preserves string literals containing comment-like text", () => {
        const out = stripComments('const s = "// not a comment";');
        expect(out).toBe('const s = "// not a comment";');
    });
});

describe("TRIG_ALLOWLIST — exactly two entries", () => {
    test("granularity is (file, documented deviation), not call line — 4 call lines, 2 entries", () => {
        expect(TRIG_ALLOWLIST).toHaveLength(2);
        const totalLines = TRIG_ALLOWLIST.reduce((n, e) => n + e.lines.length, 0);
        expect(totalLines).toBe(4);
    });
});

describe("sweepLiterals — red-first proof on a fixture tree", () => {
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

    // Rule 1a's own named example: a non-exact fraction built from individually-exact literals
    // (`1`, `3`), fused with a further operator in the same wrap. Neither `1` nor `3` alone fails
    // the old per-operand exactness check, so a predicate that only tests each split operand's own
    // value misses this shape entirely.
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

    // Two-sided control: the same shape, but the fraction (1/4 = 0.25) is exactly representable —
    // must stay silent.
    test("does not flag an exact fraction fused with a further operator", async () => {
        const root = fixture({
            "fake/fraction.ts": "export const ok = f32(1 / 4 * mass);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    // A bare fraction of individually-exact literals, with nothing further fused in the same wrap,
    // is exactly the correct place to round it (the double-rounding theorem covers a two-operand op
    // whose operands are already f32-valued) — must stay silent, unlike the fused case above.
    test("does not flag a bare fraction of exact literals with no further fused operator", async () => {
        const root = fixture({
            "fake/fraction.ts": "export const ok = f32(1 / 3);\n",
        });
        const findings = await sweepLiterals(root);
        expect(findings).toHaveLength(0);
    });

    // A literal nested one paren-level deeper than the f32(...) call's own top level — the old
    // predicate only inspected the top-level split's own operands, so a literal hiding inside a
    // parenthesized sub-expression at that level was invisible.
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

    // Two-sided control: the same nested shape, but the nested literal (0.5) is exact — must stay
    // silent.
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
});
