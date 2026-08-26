// Shared predicate module for the check-docs citation-resolution arm.
// Shape predicates and candidate extraction for the check-docs citation-resolution
// arm. The population predicate is formatting-invariant: every token in
// `.claude/rules/**` outside a fenced code block matching an identifier *shape*
// — camelCase, PascalCase, SCREAMING_SNAKE, snake_case, lowercase-with-digits,
// or a backticked `*.ts` path — **backticked or bare** — must resolve against
// the tree or a committed roster.
//
// Predicate design (round 7):
//  - All identifier shapes (camelCase, PascalCase, SCREAMING_SNAKE, snake_case,
//    lowercase-with-digits) are caught bare or backticked, both in solo spans
//    and in multi-token spans. The population is formatting-invariant: a token
//    is caught whether it's in backticks or bare in prose, so removing
//    backticks no longer removes a citation from the arm's population.
//  - Shape false positives (prose terms that match an identifier shape but are
//    not code citations) are excluded by predicate or fixed in the prose, never
//    by a per-entry allowlist. The round-6 `SHAPE_FALSE_POSITIVES` set is
//    deleted — it was an unpinned per-entry allowlist that could green a dead
//    symbol in one line (measured: adding a string to the set took a seeded
//    dead symbol from exit 1 to exit 0). The three former entries are handled
//    per site: `AoSoA` and `iGPUs` are prose terms fixed in the rule prose
//    (gpu.md:110, gpu.md:242); `EndFrame` is a shorthand for the live symbol
//    `EndFrameSystem` and is re-spelled onto the live member (render.md:68,
//    render.md:161), with its sentence fixed in round 7 (the claim "sole
//    `last: true` system" was false — the scheduler kahn-sorts a bucket of
//    several `last: true` systems).
//  - Bare weak shapes (snake_case, lowercase-with-digits) are re-admitted into
//    the population, including in-span (round 7). The 11 distinct bare
//    weak-shape tokens that did not resolve at 6ee6ed6 are adjudicated per
//    site: `webgpu_inspector` is a genuine foreign tool name added to the Tools
//    roster; `memory64` is a Wasm feature (not a tool) filed under
//    WasmFeatures; 9 prose terms (bench metrics, formula variables, hardware
//    terms, data formats, benchmark labels) are fixed in the rule prose so
//    the sentence no longer carries a bare identifier-shaped token for a thing
//    that is not a code symbol. Round 7 admits weak shapes in-span too,//    yielding 5 new sites: `gain_effect`/`direct_effect` (Steam Audio upstream
//    filenames → SteamAudio roster) and `working_set`/`L2_size` (formula
//    variables → ARITH_RE widened by comparison operators).
//  - A bare `*`-prefix drop is inadmissible (`*foo` also spells a mis-bulleted
//    dead symbol). Only `*`-prefixed tokens starting with `_` are skipped —
//    these are glob suffixes (e.g. `*_WGSL`, `*_REQUIRED`). A `*`-prefixed
//    token starting with a letter is caught.
//  - Do not re-tokenize the interior of a span already extracted as a `.ts`
//    path (kills `_measure`).
//  - Split in-span tokens by predicate: a token followed by `(` is a call
//    citation and must resolve; one in arithmetic context (including
//    comparison operators) is a formula variable and is excluded.

import { resolve } from "path";

// ── Shape predicates ───────────────────────────────────────────────────────────────────────

export function isCamelCase(w: string): boolean {
    return /^[a-z]/.test(w) && /[A-Z]/.test(w) && !w.includes("_");
}
export function isPascalCase(w: string): boolean {
    return /^[A-Z]/.test(w) && /[a-z][A-Z]/.test(w) && !w.includes("_");
}
export function isSnakeOrScreaming(w: string): boolean {
    return (
        w.includes("_") &&
        /^[A-Za-z_][A-Za-z0-9_]*$/.test(w) &&
        /[A-Za-z0-9]/.test(w) &&
        w.length >= 2
    );
}

/** SCREAMING_SNAKE — all uppercase letters, digits, and underscores. */
export function isScreamingSnake(w: string): boolean {
    return w.includes("_") && /^[A-Z0-9_]+$/.test(w) && /[A-Z]/.test(w) && w.length >= 2;
}

/** snake_case — underscored but not all-uppercase (excludes SCREAMING_SNAKE). */
export function isSnakeCase(w: string): boolean {
    return isSnakeOrScreaming(w) && !isScreamingSnake(w);
}
export function isLowercaseWithDigits(w: string): boolean {
    return /^[a-z][a-z0-9]*$/.test(w) && /[0-9]/.test(w) && !w.includes("_");
}
export function isHex(w: string): boolean {
    return /^[0-9a-f]+$/.test(w) && /[0-9]/.test(w) && /[a-f]/.test(w);
}

/** Strong shapes (camelCase, PascalCase) — caught bare or backticked. */
export function matchesStrongShape(w: string): boolean {
    if (w.length < 2) return false;
    if (isHex(w)) return false;
    return isCamelCase(w) || isPascalCase(w);
}

/** Weak shapes (snake_case, lowercase-with-digits) — caught bare or backticked. */
export function matchesWeakShape(w: string): boolean {
    if (w.length < 2) return false;
    if (isHex(w)) return false;
    return isSnakeCase(w) || isLowercaseWithDigits(w);
}

/** SCREAMING_SNAKE — caught bare or backticked. */
export function matchesScreamingSnake(w: string): boolean {
    if (w.length < 2) return false;
    if (isHex(w)) return false;
    return isScreamingSnake(w);
}

/**
 * Full shape match — all identifier shapes (camelCase, PascalCase, SCREAMING_SNAKE,
 * snake_case, lowercase-with-digits) are caught bare or backticked. The population
 * is formatting-invariant: a token is caught whether it's in backticks or bare in
 * prose. Shape false positives (prose terms that match an identifier shape but
 * are not code citations) are excluded by predicate or fixed in the prose, never by
 * a per-entry allowlist.
 */
export function matchesShape(w: string): boolean {
    if (matchesStrongShape(w)) return true;
    if (matchesScreamingSnake(w)) return true;
    if (matchesWeakShape(w)) return true;
    return false;
}

// ── Candidate types ─────────────────────────────────────────────────────────────────────────

export type CitationCandidate = {
    file: string;
    line: number;
    ref: string;
    kind: "ts-path" | "identifier";
    backticked: boolean;
};

// ── Marker vocabulary ──────────────────────────────────────────────────────────────────────

export const MARKER_VOCABULARY = ["(retired)", "(gone)", "(anti-pattern)"] as const;
export type Marker = (typeof MARKER_VOCABULARY)[number];

export function lineHasMarker(line: string): boolean {
    return MARKER_VOCABULARY.some((m) => line.includes(m));
}

// ── Candidate extraction ───────────────────────────────────────────────────────────────────

const TS_PATH_RE = /`([^`]*\.ts)`/g;
const IDENTIFIER_RE = /`([A-Za-z_][A-Za-z0-9_]*(?:\(\))?)`/g;
const BARE_TOKEN_RE = /[A-Za-z_][A-Za-z0-9_]*/g;
// A multi-token backtick span: backtick content that is not a single identifier or .ts path
const MULTI_TOKEN_SPAN_RE = /`([^`]+)`/g;
// Arithmetic context: preceded or followed by +, -, *, /, =, ^, ·, ×, ÷, ≤, ≥
// (comparison operators ≤/≥ only — not </> which are TypeScript angle brackets in spans)
const ARITH_RE = /[-+*/=^·×÷−≤≥]/;

/**
 * Extract citation candidates from rule files.
 *
 * Returns candidates keyed by {file, line, ref} — deduplicated.
 *
 * Predicate fixes applied:
 *  1. Bare tokens preceded by `*` are glob fragments → skipped.
 *  2. `.ts` path interiors are not re-tokenized → stripped before bare extraction.
 *  3. All identifier shapes (including weak shapes) are caught bare or backticked.
 *  4. In-span tokens: a token followed by `(` is a call citation (must resolve);
 *     one in arithmetic context is a formula variable (excluded).
 */
export async function extractCandidates(
    ruleFiles: string[],
    root: string,
): Promise<{ candidates: CitationCandidate[]; markerExempted: Map<string, Set<string>> }> {
    const candidates: CitationCandidate[] = [];
    const seen = new Set<string>();
    // markerExempted: file → set of refs that are on a marker-carrying line
    const markerExempted = new Map<string, Set<string>>();

    function addCandidate(
        file: string,
        line: number,
        ref: string,
        kind: "ts-path" | "identifier",
        backticked: boolean,
    ) {
        const key = `${file}:${line}:${ref}`;
        if (seen.has(key)) return;
        seen.add(key);
        candidates.push({ file, line, ref, kind, backticked });
    }

    for (const file of ruleFiles) {
        const fullPath = resolve(root, file);
        const text = await Bun.file(fullPath).text();
        const lines = text.split("\n");
        let inFence = false;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.trim().startsWith("```")) {
                inFence = !inFence;
                continue;
            }
            if (inFence) continue;

            const lineHasMarkerFlag = lineHasMarker(line);

            // Track which refs are on marker-carrying lines (for the marker exemption)
            if (lineHasMarkerFlag) {
                if (!markerExempted.has(file)) markerExempted.set(file, new Set());
            }

            // 1. Backtick-cited .ts paths
            const tsPathSpans: string[] = [];
            for (const m of line.matchAll(TS_PATH_RE)) {
                const ref = m[1];
                if (
                    ref.startsWith(".") ||
                    ref.includes("*") ||
                    ref.includes(" ") ||
                    ref.includes("{")
                )
                    continue;
                addCandidate(file, i + 1, ref, "ts-path", true);
                tsPathSpans.push(m[0]);
                if (lineHasMarkerFlag) {
                    markerExempted.get(file)!.add(ref);
                }
            }

            // 2. Backtick-cited identifiers (solo-backtick spans)
            for (const m of line.matchAll(IDENTIFIER_RE)) {
                const ref = m[1].replace(/\(\)$/, "");
                if (ref.endsWith(".ts")) continue;
                if (matchesShape(ref)) {
                    addCandidate(file, i + 1, ref, "identifier", true);
                    if (lineHasMarkerFlag) {
                        markerExempted.get(file)!.add(ref);
                    }
                }
            }

            // 3. Bare identifier-shaped tokens
            // Strip URLs (tokens inside URLs are not citations)
            let stripped = line.replace(/https?:\/\/[^\s)]*/g, " ");
            // Strip ALL backtick spans — in-span tokens are handled by the in-span logic below,
            // which applies the arithmetic-context and call-citation predicates. The bare
            // extraction catches only tokens outside any backtick span.
            stripped = stripped.replace(/`[^`]+`/g, " ");

            for (const m of stripped.matchAll(BARE_TOKEN_RE)) {
                const ref = m[0];
                const index = m.index ?? 0;
                // A bare token preceded by `*` and starting with `_` is a glob suffix
                // (e.g. `*_WGSL`, `*_REQUIRED`). A bare `*`-prefix drop is inadmissible —
                // `*foo` also spells a mis-bulleted dead symbol — so `*`-prefixed tokens
                // starting with a letter are caught.
                if (index > 0 && stripped[index - 1] === "*" && ref.startsWith("_")) continue;
                // All identifier shapes are caught bare (re-admitted, round 6b): camelCase,
                // PascalCase, SCREAMING_SNAKE, snake_case, and lowercase-with-digits.
                if (matchesShape(ref)) {
                    addCandidate(file, i + 1, ref, "identifier", false);
                }
            }

            // 4. In-span tokens from multi-token backtick spans
            // Re-scan the original line for multi-token backtick spans
            for (const m of line.matchAll(MULTI_TOKEN_SPAN_RE)) {
                const spanContent = m[1];
                // Skip if this is a .ts path (already extracted) or a solo identifier (already extracted)
                if (spanContent.endsWith(".ts")) continue;
                if (/^[A-Za-z_][A-Za-z0-9_]*(?:\(\))?$/.test(spanContent)) continue;

                // Tokenize the span content and apply in-span predicate
                for (const tm of spanContent.matchAll(BARE_TOKEN_RE)) {
                    const ref = tm[0];
                    const tIndex = tm.index ?? 0;
                    // Find the next non-whitespace char after the token
                    let afterIdx = tIndex + ref.length;
                    while (afterIdx < spanContent.length && spanContent[afterIdx] === " ")
                        afterIdx++;
                    const afterChar = afterIdx < spanContent.length ? spanContent[afterIdx] : "";
                    // Find the previous non-whitespace char before the token
                    let beforeIdx = tIndex - 1;
                    while (beforeIdx >= 0 && spanContent[beforeIdx] === " ") beforeIdx--;
                    const beforeChar = beforeIdx >= 0 ? spanContent[beforeIdx] : "";

                    // Fix 4a: a token followed by `(` is a call citation — must resolve
                    if (afterChar === "(") {
                        if (matchesShape(ref)) {
                            addCandidate(file, i + 1, ref, "identifier", true);
                            if (lineHasMarkerFlag) {
                                markerExempted.get(file)!.add(ref);
                            }
                        }
                        continue;
                    }
                    // Fix 4b: a token in arithmetic context is a formula variable — excluded
                    if (ARITH_RE.test(beforeChar) || ARITH_RE.test(afterChar)) {
                        continue;
                    }
                    // Otherwise: a token inside a multi-token span — all identifier
                    // shapes (weak shapes admitted in-span, round 7). Formula variables
                    // in arithmetic context (including comparison operators) are excluded
                    // above.
                    if (matchesShape(ref)) {
                        addCandidate(file, i + 1, ref, "identifier", true);
                        if (lineHasMarkerFlag) {
                            markerExempted.get(file)!.add(ref);
                        }
                    }
                }
            }
        }
    }

    return { candidates, markerExempted };
}

// ── Token index ────────────────────────────────────────────────────────────────────────────

const INDEX_TOKEN_RE = /[A-Za-z_][A-Za-z0-9_]*/g;

/**
 * Build a one-pass token index over `*.ts`/`*.rs`/`*.wgsl` files.
 * Excludes `node_modules`, `scripts/check-docs.ts`, `scripts/rosters.ts`,
 * and `scripts/stale-claim-predicates.ts` (their comments mention the symbols
 * they check, which would false-resolve dead citations).
 */
export async function buildTokenIndex(trackedFiles: string[], root: string): Promise<Set<string>> {
    const index = new Set<string>();
    const sourceFiles = trackedFiles.filter(
        (f) =>
            (f.endsWith(".ts") || f.endsWith(".rs") || f.endsWith(".wgsl")) &&
            !f.includes("node_modules") &&
            f !== "scripts/check-docs.ts" &&
            f !== "scripts/rosters.ts" &&
            f !== "scripts/stale-claim-predicates.ts",
    );
    for (const f of sourceFiles) {
        const text = await Bun.file(resolve(root, f)).text();
        for (const m of text.matchAll(INDEX_TOKEN_RE)) {
            index.add(m[0]);
        }
    }
    return index;
}

// ── Resolution ──────────────────────────────────────────────────────────────────────────────

export function tsPathResolves(path: string, trackedSet: Set<string>): boolean {
    const tries = [path, `packages/shallot/src/${path}`, `packages/shallot/${path}`];
    for (const t of tries) {
        if (trackedSet.has(t)) return true;
    }
    const suffix = `/${path}`;
    for (const f of trackedSet) {
        if (f.endsWith(suffix)) return true;
    }
    return false;
}

export function resolvesAnywhere(
    ref: string,
    kind: string,
    tokenIndex: Set<string>,
    trackedSet: Set<string>,
    combinedRoster: Set<string>,
): boolean {
    if (kind === "ts-path") {
        if (tsPathResolves(ref, trackedSet)) return true;
        return combinedRoster.has(ref);
    }
    if (tokenIndex.has(ref)) return true;
    return combinedRoster.has(ref);
}
