import { join, resolve } from "node:path";
import { Glob } from "bun";
import { TEST_TIER_SUFFIXES } from "../packages/shallot/tests/test-tiers";

// The tumble engine's rule 1a (`.claude/rules/tumble.md` § "The contract: bit-exact f32
// parity") says to `fround` a non-exact float literal before it enters `f32(...)` arithmetic:
// `f32(f32(0.4) * mass)`, never `f32(0.4 * mass)` — JS `0.4` is an f64 operand until rounded,
// and it multiplies/adds differently than C's `f`-suffixed `0.4f` for some right-hand values.
// This sweeps every `f32(...)` call under the tumble engine source for that exact shape: a
// bare (unwrapped) literal, anywhere in the call's argument tree, whose `Math.fround` differs
// from itself. It carries a second, independent arm: every `Math.sin`/`Math.cos`/`Math.atan2`
// call site outside the two documented trig deviations (below) is also a finding — rule 1's
// "port Box3D's own portable trig, never JS transcendentals" clause.
//
// **S1b — structural safety.** The predicate's soundness no longer rests on per-sample
// demonstration. The sweep lexes once with quote / template-literal / comment awareness
// (`maskLiterals`), then traverses *every* nested position exhaustively — call-argument parens
// included, not just bare paren groups (`collectOffendingLiterals` recurses into any
// parenthesized group that is not itself a nested `f32(...)` call, which `findCalls` handles
// separately). Any `f32(` region the lexer cannot fully decompose, or a depth counter that does
// not return to zero, is *reported as an unparsed site* and exits non-zero. Safety rests on
// exhaustiveness over the region set plus a loud inconclusive — never on a sample.
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
      }
    | {
          kind: "unparsed";
          file: string;
          line: number;
          reason: string;
          snippet: string;
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

// The shared roster (`test-tiers.ts`) plus this sweep's own tumble-local `.fixture.ts` exclusion —
// `fixture` isn't one of the five roster suffix names (test-tiers.ts's own module list excludes it),
// so it rides beside the derived roster as its own single-suffix check rather than folding into a
// second hand-written list (`check-docs.ts`'s tier-suffix roster arm forbids restating 3+ of the 5
// names in any shape — regex alternation or array literal).
const FIXTURE_SUFFIX = /\.fixture\.ts$/;

function isTestFile(rel: string): boolean {
    return TEST_TIER_SUFFIXES.test(rel) || FIXTURE_SUFFIX.test(rel);
}

function isExactF32(n: number): boolean {
    return Math.fround(n) === n;
}

function lineOf(text: string, pos: number): number {
    let line = 1;
    for (let i = 0; i < pos && i < text.length; i++) {
        if (text[i] === "\n") line++;
    }
    return line;
}

// ---------------------------------------------------------------------------
// Lexer: mask strings, comments, and template-literal text (preserve interpolations)
// ---------------------------------------------------------------------------

// A region the lexer could not fully decompose — an unterminated string, template literal, or
// block comment, or an unbalanced template interpolation. The sweep reports these as `unparsed`
// findings and exits non-zero, so safety rests on exhaustiveness plus a loud inconclusive.
export type UnparsedSite = { line: number; reason: string; snippet: string };

// Single-pass lexer producing a "masked" copy of the text where:
//   - line-comment and block-comment content → spaces (newlines preserved, so line numbers stay
//     correct);
//   - single- and double-quoted string content → spaces (quote delimiters → spaces too, so a `)`
//     inside a string never confuses the paren depth counter in `findCalls`);
//   - template-literal text portions → spaces (backtick delimiters → spaces);
//   - template-literal interpolation expressions (`${...}`) → **preserved as real code**, so a
//     `f32(...)` inside an interpolation IS found by `findCalls` (it is a real call);
//   - everything else → preserved verbatim.
//
// Any construct the lexer cannot close (unterminated string, unterminated template, unterminated
// block comment, unbalanced interpolation brace) is recorded as an `UnparsedSite` rather than
// silently swallowing the rest of the file — the worst failure shape for a standing check, which
// the S1 hand-rolled tokenizer exhibited on a template literal carrying an unbalanced brace.
export function maskLiterals(content: string): { masked: string; unparsed: UnparsedSite[] } {
    const unparsed: UnparsedSite[] = [];
    const out: string[] = [];
    let i = 0;
    const len = content.length;

    // Mask a single- or double-quoted string. Returns true if the closing quote was found.
    function maskString(quote: string): boolean {
        out.push(" "); // opening quote → space
        i++;
        while (i < len) {
            if (content[i] === "\\" && i + 1 < len) {
                out.push(content[i + 1] === "\n" ? "\n" : " ", " ");
                i += 2;
                continue;
            }
            if (content[i] === quote) {
                out.push(" "); // closing quote → space
                i++;
                return true;
            }
            out.push(content[i] === "\n" ? "\n" : " ");
            i++;
        }
        return false;
    }

    // Mask a template literal. Text portions and backtick delimiters → spaces; `${...}`
    // interpolation expressions are preserved as real code (the `${` and closing `}` → spaces,
    // but the expression between them is output verbatim). Returns true if the closing backtick
    // was found. Handles nested templates, strings, and comments inside interpolations. An
    // unbalanced interpolation brace (depth never returns to 0) makes the template unterminated.
    function maskTemplate(): boolean {
        out.push(" "); // opening backtick → space
        i++;
        while (i < len) {
            if (content[i] === "\\" && i + 1 < len) {
                out.push(content[i + 1] === "\n" ? "\n" : " ", " ");
                i += 2;
                continue;
            }
            if (content[i] === "`") {
                out.push(" "); // closing backtick → space
                i++;
                return true;
            }
            if (content[i] === "$" && i + 1 < len && content[i + 1] === "{") {
                out.push(" ", " "); // ${ → spaces
                i += 2;
                // Interpolation body: track brace depth, handle nested constructs.
                let depth = 1;
                while (i < len && depth > 0) {
                    if (content[i] === "/" && i + 1 < len && content[i + 1] === "/") {
                        while (i < len && content[i] !== "\n") {
                            out.push(" ");
                            i++;
                        }
                        continue;
                    }
                    if (content[i] === "/" && i + 1 < len && content[i + 1] === "*") {
                        out.push(" ", " ");
                        i += 2;
                        while (i < len) {
                            if (content[i] === "*" && i + 1 < len && content[i + 1] === "/") {
                                out.push(" ", " ");
                                i += 2;
                                break;
                            }
                            out.push(content[i] === "\n" ? "\n" : " ");
                            i++;
                        }
                        continue;
                    }
                    if (content[i] === '"' || content[i] === "'") {
                        maskString(content[i]);
                        continue;
                    }
                    if (content[i] === "`") {
                        maskTemplate(); // nested template — return value unchecked (outer reports)
                        continue;
                    }
                    if (content[i] === "{") {
                        depth++;
                        out.push("{");
                        i++;
                        continue;
                    }
                    if (content[i] === "}") {
                        depth--;
                        if (depth === 0) {
                            out.push(" "); // closing } of interpolation → space
                            i++;
                            break;
                        }
                        out.push("}");
                        i++;
                        continue;
                    }
                    out.push(content[i]);
                    i++;
                }
                if (depth > 0) return false; // unbalanced interpolation → unterminated template
                continue;
            }
            // Template text content → mask
            out.push(content[i] === "\n" ? "\n" : " ");
            i++;
        }
        return false; // unterminated template
    }

    while (i < len) {
        const c = content[i];

        // Line comment
        if (c === "/" && i + 1 < len && content[i + 1] === "/") {
            while (i < len && content[i] !== "\n") {
                out.push(" ");
                i++;
            }
            continue;
        }

        // Block comment
        if (c === "/" && i + 1 < len && content[i + 1] === "*") {
            const start = i;
            out.push(" ", " ");
            i += 2;
            let closed = false;
            while (i < len) {
                if (content[i] === "*" && i + 1 < len && content[i + 1] === "/") {
                    out.push(" ", " ");
                    i += 2;
                    closed = true;
                    break;
                }
                out.push(content[i] === "\n" ? "\n" : " ");
                i++;
            }
            if (!closed) {
                unparsed.push({
                    line: lineOf(content, start),
                    reason: "unterminated block comment",
                    snippet: content.slice(start, Math.min(start + 40, len)).trim(),
                });
            }
            continue;
        }

        // Single/double-quoted string
        if (c === '"' || c === "'") {
            const start = i;
            if (!maskString(c)) {
                unparsed.push({
                    line: lineOf(content, start),
                    reason: "unterminated string literal",
                    snippet: content.slice(start, Math.min(start + 40, len)).trim(),
                });
            }
            continue;
        }

        // Template literal
        if (c === "`") {
            const start = i;
            if (!maskTemplate()) {
                unparsed.push({
                    line: lineOf(content, start),
                    reason: "unterminated template literal",
                    snippet: content.slice(start, Math.min(start + 40, len)).trim(),
                });
            }
            continue;
        }

        out.push(c);
        i++;
    }

    return { masked: out.join(""), unparsed };
}

// Backward-compatible wrapper: returns just the masked text. Replaces the S1 `stripComments`
// (which preserved string content and did not handle template-literal interpolations). String
// and comment content is now masked so a `)` or `{` inside a string never confuses the depth
// counters in `findCalls` / `splitTopLevelTokens`.
export function stripComments(content: string): string {
    return maskLiterals(content).masked;
}

// ---------------------------------------------------------------------------
// Token splitting
// ---------------------------------------------------------------------------

// A top-level (paren/bracket/brace depth 0) operand, paired with the operator immediately
// preceding it (`null` for the first operand in the expression).
type Token = { text: string; opBefore: "+" | "-" | "*" | "/" | null };

// Split `expr` into its top-level (paren/bracket/brace depth 0) operands, each paired with the
// operator that split it off the previous one. A leading `+`/`-`, or one immediately after another
// operator/open-paren/comma, is a unary sign (part of the operand, not a split point); an `e`/`E`-
// adjacent `+`/`-` is an exponent suffix, never a split point either. Box3D's own
// `-ffp-contract=off` + this port's "one op per f32 wrap" convention means most calls carry
// exactly one top-level operator, but a chain (`2.0 * Math.PI * f`) splits into every operand so
// each is checked independently.
function splitTopLevelTokens(expr: string): Token[] {
    const tokens: Token[] = [];
    let depth = 0;
    let start = 0;
    let pendingOp: Token["opBefore"] = null;
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
            // Discriminate a real exponent suffix (`1e+5`, `1.5E-3`) from an identifier ending
            // in `e`/`E` (`distance + 0.1`) by what precedes the `e`/`E` — a digit or `.` —
            // rather than by the `e`/`E` alone. Without this, `f32(distance + 0.1)` collapses
            // into one non-literal token and the bare `0.1` is silently missed.
            const isExponent =
                (c === "+" || c === "-") &&
                (prevChar === "e" || prevChar === "E") &&
                j > start &&
                /[0-9.]/.test(expr[j - 1]);
            const isUnary = prevChar === "" || "+-*/(,".includes(prevChar);
            if (isExponent || isUnary) continue;
            tokens.push({ text: expr.slice(start, i).trim(), opBefore: pendingOp });
            pendingOp = c;
            start = i + 1;
        }
    }
    tokens.push({ text: expr.slice(start).trim(), opBefore: pendingOp });
    return tokens;
}

// Public operand-only view of `splitTopLevelTokens`, kept for the existing call sites/tests that
// only care about the split, not the operators between splits.
export function splitTopLevel(expr: string): string[] {
    return splitTopLevelTokens(expr).map((t) => t.text);
}

// Split a call's argument list on top-level commas (depth 0, tracking parens/brackets/braces),
// so each argument is recursed into independently by `collectOffendingLiterals`.
function splitArgs(args: string): string[] {
    const parts: string[] = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i < args.length; i++) {
        const c = args[i];
        if (c === "(" || c === "[" || c === "{") depth++;
        else if (c === ")" || c === "]" || c === "}") depth--;
        else if (c === "," && depth === 0) {
            parts.push(args.slice(start, i).trim());
            start = i + 1;
        }
    }
    const last = args.slice(start).trim();
    if (last.length > 0) parts.push(last);
    return parts;
}

// ---------------------------------------------------------------------------
// Literal analysis
// ---------------------------------------------------------------------------

// A bare numeric-literal token — the exact shape rule 1a cares about. A wrapped operand
// (`f32(0.4)`) or an identifier/member/call expression never matches this, so "bare" falls out
// of the regex itself rather than needing a separate wrapped-vs-bare check.
const NUMERIC_LITERAL = /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;

// Left-to-right reduction of a maximal run of adjacent bare-literal top-level tokens (a "literal
// cluster", e.g. the `1`, `3` of `1 / 3 * mass`), using the operators `splitTopLevelTokens`
// recorded between them. **Two-pass** (first `*`/`/`, then `+`/`-`) to respect JS operator
// precedence — `1 - 1/3` must evaluate to `0.666…` (not `(1-1)/3 = 0`), and the old single-pass
// left-to-right evaluator silently returned `0`, which IS exactly representable in f32, hiding
// the divergence. Sound for the chains rule 1a's "one op per f32 wrap" convention actually
// produces — not a general expression evaluator, and never needs to be one under that convention.
function reduceLiteralCluster(cluster: Token[]): number {
    // Pass 1: fold * and / left-to-right into the preceding value.
    const pass1: { value: number; op: "+" | "-" | null }[] = [
        { value: Number(cluster[0].text), op: null },
    ];
    for (let k = 1; k < cluster.length; k++) {
        const rhs = Number(cluster[k].text);
        const op = cluster[k].opBefore;
        if (op === "*" || op === "/") {
            const prev = pass1[pass1.length - 1];
            prev.value = op === "*" ? prev.value * rhs : prev.value / rhs;
        } else {
            pass1.push({ value: rhs, op });
        }
    }
    // Pass 2: fold + and - left-to-right.
    let result = pass1[0].value;
    for (let k = 1; k < pass1.length; k++) {
        result = pass1[k].op === "+" ? result + pass1[k].value : result - pass1[k].value;
    }
    return result;
}

// Every bare literal (recursively, through *every* nested position — call-argument parens
// included, not just bare paren groups) that contributes to a non-exact f32 value somewhere inside
// `expr` — the argument of an `f32(...)` call, or any operand nested one or more levels inside it.
//
// Three shapes, all real misses of a naive "is this one split-off operand a bare non-exact
// literal" check:
//
//   1. A non-exact literal inside a *call argument* (`f32(Math.sqrt(0.1 * a))` — `0.1` never
//      appears as a top-level operand of the outer split, only inside `Math.sqrt(...)`'s
//      argument list). Handled by recursing into any parenthesized group that is not itself a
//      nested `f32(...)` call (which `findCalls` handles separately, so a properly pre-wrapped
//      `f32(f32(0.4) * mass)` does not double-count).
//   2. A non-exact literal one paren-level deeper than the enclosing split (`f32((0.4 * mass) +
//      1)` — `0.4` never appears as a top-level operand of the outer split, only inside the
//      parenthesized sub-expression). Handled by the same recursion — a bare `(…)` group is just
//      a call argument with no callee.
//   3. A non-exact *fraction* built from individually-exact literals, then fused with a further
//      operator in the same wrap (`f32(1 / 3 * mass)` — neither `1` nor `3` is individually
//      non-exact, but `1 / 3` is, and unlike a bare `f32(1 / 3)` — correct by the double-rounding
//      theorem, since both operands are already f32-valued — fusing it with `* mass` inside the
//      same wrap means the fraction's own value never gets rounded to f32 before the further
//      multiply, exactly the double-rounding gap rule 1a names for a bare non-exact literal).
//      Handled by clustering adjacent bare-literal tokens and evaluating the cluster's own value
//      whenever the cluster is a strict subset of the level's tokens (i.e. it IS fused with
//      something else at that level). A cluster spanning the whole level with 2+ operators
//      (3+ literals) is checked on the same principle: the first op's intermediate isn't rounded
//      before the second. A cluster spanning the whole level with exactly 1 operator (2 literals)
//      is the double-rounding theorem's covered case and stays silent.
function collectOffendingLiterals(expr: string): string[] {
    const tokens = splitTopLevelTokens(expr);
    const offending: string[] = [];
    let cluster: Token[] = [];

    const flushCluster = () => {
        if (cluster.length === 0) return;
        const individuallyNonExact = cluster.some((t) => !isExactF32(Number(t.text)));
        const isWholeLevel = cluster.length === tokens.length;
        const combinedNonExact =
            cluster.length >= 2 &&
            !isExactF32(reduceLiteralCluster(cluster)) &&
            (!isWholeLevel || cluster.length >= 3);
        if (individuallyNonExact || combinedNonExact) {
            offending.push(...cluster.map((t) => t.text));
        }
        cluster = [];
    };

    for (const tok of tokens) {
        if (NUMERIC_LITERAL.test(tok.text)) {
            cluster.push(tok);
            continue;
        }
        flushCluster();
        // Recurse into any parenthesized group — call-argument parens included, not just bare
        // paren groups. A nested f32() call is skipped (findCalls handles it separately, so a
        // properly pre-wrapped literal does not double-count).
        const openParen = tok.text.indexOf("(");
        if (openParen >= 0) {
            const beforeParen = tok.text.slice(0, openParen).trim();
            if (beforeParen === "f32") continue;
            // Find the matching close paren for the first open paren.
            let depth = 0;
            let closeParen = -1;
            for (let j = openParen; j < tok.text.length; j++) {
                if (tok.text[j] === "(") depth++;
                else if (tok.text[j] === ")") {
                    depth--;
                    if (depth === 0) {
                        closeParen = j;
                        break;
                    }
                }
            }
            if (closeParen < 0) continue;
            const argContent = tok.text.slice(openParen + 1, closeParen);
            for (const arg of splitArgs(argContent)) {
                offending.push(...collectOffendingLiterals(arg));
            }
        }
    }
    flushCluster();
    return offending;
}

// ---------------------------------------------------------------------------
// Call finding
// ---------------------------------------------------------------------------

type Call = { start: number; argStart: number; argEnd: number; balanced: boolean };

// Every call to `name(` in `text` (not preceded by an identifier char, so `isF32(` doesn't
// match `f32(`), paired with the byte offsets of its balanced-paren argument span. A call whose
// parens don't balance (depth never returns to zero before end of text) is marked `balanced:
// false` so `sweepUnparsed` can report it as an unparsed site and `sweepLiterals` can skip it.
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
        calls.push({ start: idx, argStart, argEnd: i - 1, balanced: depth === 0 });
        idx = argStart;
    }
    return calls;
}

// ---------------------------------------------------------------------------
// Source file generator
// ---------------------------------------------------------------------------

async function* sourceFiles(
    root: string,
): AsyncGenerator<{ rel: string; text: string; unparsed: UnparsedSite[] }> {
    const glob = new Glob("**/*.ts");
    for await (const rel of glob.scan({ cwd: root })) {
        if (isTestFile(rel)) continue;
        const full = join(root, rel);
        const content = await Bun.file(full).text();
        const { masked, unparsed } = maskLiterals(content);
        yield { rel, text: masked, unparsed };
    }
}

// ---------------------------------------------------------------------------
// Sweep arms
// ---------------------------------------------------------------------------

// The literal arm: one finding per `f32(...)` call whose argument carries at least one bare
// non-exact literal contributing to its value — a "site" is the call, not the literal
// (kToleranceSquared's `f32(0.05 * 0.05)` is one site with two offending literals, matching the
// spec's "11 sites"). `collectOffendingLiterals` handles a literal inside a call argument, a
// literal nested one paren-level deeper, and a non-exact fraction/cluster of individually-exact
// literals fused with a further operator — including a whole-level 2+ op chain (`f32(1 - 1/3)`).
export async function sweepLiterals(root: string): Promise<Finding[]> {
    const findings: Finding[] = [];
    for await (const { rel, text } of sourceFiles(root)) {
        for (const call of findCalls(text, "f32")) {
            if (!call.balanced) continue; // reported by sweepUnparsed
            const arg = text.slice(call.argStart, call.argEnd);
            // Skip a bare fround — a single numeric literal with no operator (`f32(0.1)`). This
            // is the correct way to round a literal, not a finding. But do NOT skip a single
            // non-literal operand (`f32(Math.sqrt(0.1 * a))`) — it may carry a non-exact literal
            // inside a call argument that the recursion must reach.
            const tokens = splitTopLevel(arg);
            if (tokens.length === 1 && NUMERIC_LITERAL.test(tokens[0].trim())) continue;
            const literals = collectOffendingLiterals(arg);
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
                if (!call.balanced) continue;
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

// The unparsed arm: any region the lexer could not fully decompose — an unterminated string,
// template literal, or block comment from `maskLiterals`, or an `f32(` call whose parens don't
// balance. Safety rests on exhaustiveness over the region set plus a loud inconclusive: the sweep
// exits non-zero rather than silently swallowing the rest of the file and returning zero findings.
export async function sweepUnparsed(root: string): Promise<Finding[]> {
    const findings: Finding[] = [];
    for await (const { rel, text, unparsed } of sourceFiles(root)) {
        for (const u of unparsed) {
            findings.push({
                kind: "unparsed",
                file: rel,
                line: u.line,
                reason: u.reason,
                snippet: u.snippet,
            });
        }
        for (const call of findCalls(text, "f32")) {
            if (!call.balanced) {
                findings.push({
                    kind: "unparsed",
                    file: rel,
                    line: lineOf(text, call.start),
                    reason: "unbalanced parens in f32() call",
                    snippet: text.slice(call.start, Math.min(call.start + 40, text.length)).trim(),
                });
            }
        }
    }
    return findings;
}

export async function sweep(root: string): Promise<Finding[]> {
    const [literals, trig, unparsed] = await Promise.all([
        sweepLiterals(root),
        sweepTrig(root),
        sweepUnparsed(root),
    ]);
    return [...literals, ...trig, ...unparsed].sort((a, b) =>
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
        console.error(`✗ ${findings.length} finding(s):\n`);
        for (const f of findings) {
            if (f.kind === "literal") {
                console.error(
                    `  ${f.file}:${f.line}  ${f.expression}  (bare non-exact literal: ${f.literals.join(", ")})`,
                );
            } else if (f.kind === "trig") {
                console.error(`  ${f.file}:${f.line}  ${f.expression}  (undocumented trig call)`);
            } else {
                console.error(`  ${f.file}:${f.line}  UNPARSED: ${f.reason}  ${f.snippet}`);
            }
        }
        console.error(
            "\nRule 1a: fround a non-exact literal before it enters f32 arithmetic —\n" +
                "f32(f32(0.4) * mass), never f32(0.4 * mass). A Math.sin/cos/atan2 call outside the\n" +
                "two documented trig deviations must port Box3D's own portable trig instead.\n" +
                "An unparsed site is a region the lexer could not fully decompose — safety rests\n" +
                "on exhaustiveness over the region set plus a loud inconclusive, never a sample.\n" +
                '(.claude/rules/tumble.md § "The contract: bit-exact f32 parity")',
        );
        process.exit(1);
    }

    console.log("✓ tumble bit-exact literal sweep clean (0 findings)");
}
