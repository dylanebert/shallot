import { join, resolve } from "node:path";
import { Glob } from "bun";

// The tumble engine's rule 1a (`.claude/rules/tumble.md` § "The contract: bit-exact f32
// parity") says to `fround` a non-exact float literal before it enters `f32(...)` arithmetic:
// `f32(f32(0.4) * mass)`, never `f32(0.4 * mass)` — JS `0.4` is an f64 operand until rounded,
// and it multiplies/adds differently than C's `f`-suffixed `0.4f` for some right-hand values.
// This sweeps every `f32(...)` call under the tumble engine source for that exact shape: a
// bare (unwrapped) literal, at the call's own top level, whose `Math.fround` differs from
// itself. It carries a second, independent arm: every `Math.sin`/`Math.cos`/`Math.atan2` call
// site outside the two documented trig deviations (below) is also a finding — rule 1's "port
// Box3D's own portable trig, never JS transcendentals" clause.
//
// `bun run scripts/check-tumble-fp.ts` is EXPECTED to exit 1 at S1 (audit-tumble-engine-
// bitexact-literals) — the check is built and pinned red before the three live divergences
// move (S2). Wiring it into `bun run check` is S2's job, not this one's: wiring a red check in
// would ship a red default gate.
//
// `--root <dir>` points the sweep at an alternate tree (fixture-driven proof — same shape as
// `check-boundary.ts` / `check-scripts.ts`'s own `--root`), so the arm can drive real assertions
// over a synthetic tree without touching tumble engine source.

export type Finding =
    | {
          kind: "literal";
          file: string;
          line: number;
          expression: string;
          literals: string[];
      }
    | {
          kind: "trig";
          file: string;
          line: number;
          expression: string;
          fn: string;
      };

// The two documented trig deviations (tumble.md "One documented trig deviation" +
// heightfield.ts's own self-documented second site — see spec Validation, "entry granularity
// is (file, documented deviation), not call line"): `createWaveMesh`'s two `Math.sin` calls are
// one entry, `createWave`'s two calls are the other. A third `Math.sin`/`cos`/`atan2` site,
// anywhere, is a finding rather than silently joining this list.
export const TRIG_ALLOWLIST: { file: string; deviation: string; lines: number[] }[] = [
    { file: "engine/mesh.ts", deviation: "createWaveMesh", lines: [864, 867] },
    { file: "engine/heightfield.ts", deviation: "createWave", lines: [1059, 1062] },
];

const TEST_SUFFIX = /\.(test|fixture|lab|oracle|probes|tier)\.ts$/;

function isExactF32(n: number): boolean {
    return Math.fround(n) === n;
}

// Same shape as check-exports.ts's `stripComments` (block comments blanked but newline-
// preserving, so line numbers stay correct; string literals copied through untouched) — a
// documentation comment quoting the anti-pattern (`f32(0.4 * mass)`) as a bad example must
// never read as a real finding.
export function stripComments(content: string): string {
    let result = "";
    let i = 0;
    const len = content.length;

    while (i < len) {
        if (content[i] === '"' || content[i] === "'" || content[i] === "`") {
            const quote = content[i];
            result += content[i];
            i++;
            while (i < len) {
                if (content[i] === "\\" && i + 1 < len) {
                    result += content[i] + content[i + 1];
                    i += 2;
                    continue;
                }
                result += content[i];
                if (content[i] === quote) {
                    i++;
                    break;
                }
                i++;
            }
            continue;
        }

        if (content[i] === "/" && i + 1 < len && content[i + 1] === "/") {
            i += 2;
            while (i < len && content[i] !== "\n") i++;
            continue;
        }

        if (content[i] === "/" && i + 1 < len && content[i + 1] === "*") {
            i += 2;
            while (i < len) {
                if (content[i] === "*" && i + 1 < len && content[i + 1] === "/") {
                    i += 2;
                    break;
                }
                if (content[i] === "\n") result += "\n";
                i++;
            }
            continue;
        }

        result += content[i];
        i++;
    }

    return result;
}

// Split `expr` on its top-level (paren/bracket depth 0) binary operators. A leading `+`/`-`, or
// one immediately after another operator/open-paren/comma, is a unary sign (part of the
// operand, not a split point); an `e`/`E`-adjacent `+`/`-` is an exponent suffix, never a split
// point either. Box3D's own `-ffp-contract=off` + this port's "one op per f32 wrap" convention
// means most calls carry exactly one top-level operator, but a chain (`2.0 * Math.PI * f`)
// splits into every operand so each is checked independently.
export function splitTopLevel(expr: string): string[] {
    const operands: string[] = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i < expr.length; i++) {
        const c = expr[i];
        if (c === "(" || c === "[" || c === "{") {
            depth++;
        } else if (c === ")" || c === "]" || c === "}") {
            depth--;
        } else if (depth === 0 && (c === "+" || c === "-" || c === "*" || c === "/")) {
            let j = i - 1;
            while (j >= start && /\s/.test(expr[j])) j--;
            const prevChar = j >= start ? expr[j] : "";
            const isExponent = (c === "+" || c === "-") && (prevChar === "e" || prevChar === "E");
            const isUnary = prevChar === "" || "+-*/(,".includes(prevChar);
            if (isExponent || isUnary) continue;
            operands.push(expr.slice(start, i).trim());
            start = i + 1;
        }
    }
    operands.push(expr.slice(start).trim());
    return operands;
}

// A bare numeric-literal token — the exact shape rule 1a cares about. A wrapped operand
// (`f32(0.4)`) or an identifier/member/call expression never matches this, so "bare" falls out
// of the regex itself rather than needing a separate wrapped-vs-bare check.
const NUMERIC_LITERAL = /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;

type Call = { start: number; argStart: number; argEnd: number };

// Every call to `name(` in `text` (not preceded by an identifier char, so `isF32(` doesn't
// match `f32(`), paired with the byte offsets of its balanced-paren argument span.
function findCalls(text: string, name: string): Call[] {
    const calls: Call[] = [];
    const marker = `${name}(`;
    let idx = 0;
    while ((idx = text.indexOf(marker, idx)) !== -1) {
        const before = idx > 0 ? text[idx - 1] : "";
        if (before !== "" && /[\w$]/.test(before)) {
            idx += marker.length;
            continue;
        }
        const argStart = idx + marker.length;
        let depth = 1;
        let i = argStart;
        for (; i < text.length && depth > 0; i++) {
            if (text[i] === "(") depth++;
            else if (text[i] === ")") depth--;
        }
        calls.push({ start: idx, argStart, argEnd: i - 1 });
        idx = argStart;
    }
    return calls;
}

function lineOf(text: string, pos: number): number {
    let line = 1;
    for (let i = 0; i < pos && i < text.length; i++) {
        if (text[i] === "\n") line++;
    }
    return line;
}

async function* sourceFiles(root: string): AsyncGenerator<{ rel: string; text: string }> {
    const glob = new Glob("**/*.ts");
    for await (const rel of glob.scan({ cwd: root })) {
        if (TEST_SUFFIX.test(rel)) continue;
        const full = join(root, rel);
        yield { rel, text: stripComments(await Bun.file(full).text()) };
    }
}

// The literal arm: one finding per `f32(...)` call whose top-level argument carries at least
// one bare non-exact literal operand — a "site" is the call, not the literal (kToleranceSquared's
// `f32(0.05 * 0.05)` is one site with two offending literals, matching the spec's "11 sites").
export async function sweepLiterals(root: string): Promise<Finding[]> {
    const findings: Finding[] = [];
    for await (const { rel, text } of sourceFiles(root)) {
        for (const call of findCalls(text, "f32")) {
            const arg = text.slice(call.argStart, call.argEnd);
            const operands = splitTopLevel(arg);
            if (operands.length < 2) continue; // no top-level binary op — a bare fround, not this class
            const literals: string[] = [];
            for (const operand of operands) {
                if (!NUMERIC_LITERAL.test(operand)) continue;
                const value = Number(operand);
                if (isExactF32(value)) continue;
                literals.push(operand);
            }
            if (literals.length === 0) continue;
            findings.push({
                kind: "literal",
                file: rel,
                line: lineOf(text, call.start),
                expression: `f32(${arg.trim()})`,
                literals,
            });
        }
    }
    return findings;
}

// The trig arm: any `Math.sin`/`Math.cos`/`Math.atan2` call outside the allowlist above.
export async function sweepTrig(root: string): Promise<Finding[]> {
    const findings: Finding[] = [];
    for await (const { rel, text } of sourceFiles(root)) {
        for (const fn of ["Math.sin", "Math.cos", "Math.atan2"]) {
            for (const call of findCalls(text, fn)) {
                const line = lineOf(text, call.start);
                const allowed = TRIG_ALLOWLIST.some(
                    (a) => a.file === rel && a.lines.includes(line),
                );
                if (allowed) continue;
                const arg = text.slice(call.argStart, call.argEnd);
                findings.push({
                    kind: "trig",
                    file: rel,
                    line,
                    expression: `${fn}(${arg.trim()})`,
                    fn,
                });
            }
        }
    }
    return findings;
}

export async function sweep(root: string): Promise<Finding[]> {
    const [literals, trig] = await Promise.all([sweepLiterals(root), sweepTrig(root)]);
    return [...literals, ...trig].sort((a, b) =>
        a.file === b.file ? a.line - b.line : a.file < b.file ? -1 : 1,
    );
}

if (import.meta.main) {
    const rootArgIdx = process.argv.indexOf("--root");
    const root = resolve(
        rootArgIdx >= 0
            ? process.argv[rootArgIdx + 1]
            : resolve(import.meta.dir, "../packages/shallot/src/standard/tumble"),
    );

    const findings = await sweep(root);

    if (findings.length > 0) {
        console.error(`✗ ${findings.length} bit-exact-literal finding(s):\n`);
        for (const f of findings) {
            if (f.kind === "literal") {
                console.error(
                    `  ${f.file}:${f.line}  ${f.expression}  (bare non-exact literal: ${f.literals.join(", ")})`,
                );
            } else {
                console.error(`  ${f.file}:${f.line}  ${f.expression}  (undocumented trig call)`);
            }
        }
        console.error(
            "\nRule 1a: fround a non-exact literal before it enters f32 arithmetic —\n" +
                "f32(f32(0.4) * mass), never f32(0.4 * mass). A Math.sin/cos/atan2 call outside the\n" +
                "two documented trig deviations must port Box3D's own portable trig instead.\n" +
                '(.claude/rules/tumble.md § "The contract: bit-exact f32 parity")',
        );
        process.exit(1);
    }

    console.log("✓ tumble bit-exact literal sweep clean (0 findings)");
}
