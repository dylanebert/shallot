// Shared predicate module for the check-docs citation-resolution arm.
// One copy of the shape functions — two copies is two detectors that disagree
// the moment either moves.
//
// The population predicate is formatting-invariant: every token in
// `.claude/rules/**` outside a fenced code block matching an identifier *shape*
// — camelCase, PascalCase, SCREAMING_SNAKE, snake_case, lowercase-with-digits,
// or a backticked `*.ts` path — **backticked or bare** — must resolve against
// the tree or a committed roster.
//
// Predicate fixes (round 6):
//  - SCREAMING_SNAKE is re-admitted bare (caught with or without backticks).
//    This is the shape of this spec's first verified member `ENTITY_COLS_WGSL`,
//    which was invisible to the arm when written bare. False-positive population
//    when bare: 0 (measured at 5b48a22 — no bare SCREAMING_SNAKE token in the
//    rules fails to resolve against the tree or rosters).
//  - snake_case and lowercase-with-digits remain backticked-only — excluded
//    from bare extraction by the backtick context predicate. False-positive
//    population when bare: 11 distinct tokens (5 snake_case: `bytes_moved`,
//    `peak_BW`, `theoretical_min`, `warp_size`, `webgpu_inspector`; 6
//    lowercase-with-digits: `f16x2`, `l12`, `l4`, `memory64`, `snorm8`,
//    `wg32`) — prose terms (bench metrics, formula variables, GPU hardware
//    terms, tool names, benchmark labels), not code citations.
//  - A bare `*`-prefix drop is inadmissible (`*foo` also spells a mis-bulleted
//    dead symbol). Only `*`-prefixed tokens starting with `_` are skipped —
//    these are glob suffixes (e.g. `*_WGSL`, `*_REQUIRED`). A `*`-prefixed
//    token starting with a letter is caught.
//  - Do not re-tokenize the interior of a span already extracted as a `.ts`
//    path (kills `_measure`).
//  - Split in-span tokens by predicate: a token followed by `(` is a call
//    citation and must resolve; one in arithmetic context is a formula variable
//    and is excluded.

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
    return (
        w.includes("_") &&
        /^[A-Z0-9_]+$/.test(w) &&
        /[A-Z]/.test(w) &&
        w.length >= 2
    );
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

export const SHAPE_FALSE_POSITIVES = new Set(["AoSoA", "iGPUs", "EndFrame"]);

/** Strong shapes (camelCase, PascalCase) — caught bare or backticked. */
export function matchesStrongShape(w: string): boolean {
    if (w.length < 2) return false;
    if (SHAPE_FALSE_POSITIVES.has(w)) return false;
    if (isHex(w)) return false;
    return isCamelCase(w) || isPascalCase(w);
}

/** Weak shapes (snake_case, lowercase-with-digits) — caught only when backticked. */
export function matchesWeakShape(w: string): boolean {
    if (w.length < 2) return false;
    if (SHAPE_FALSE_POSITIVES.has(w)) return false;
    if (isHex(w)) return false;
    return isSnakeCase(w) || isLowercaseWithDigits(w);
}

/** SCREAMING_SNAKE — caught bare or backticked (re-admitted bare, round 6). */
export function matchesScreamingSnake(w: string): boolean {
    if (w.length < 2) return false;
    if (SHAPE_FALSE_POSITIVES.has(w)) return false;
    if (isHex(w)) return false;
    return isScreamingSnake(w);
}

/**
 * Full shape match — strong shapes (camelCase, PascalCase) and SCREAMING_SNAKE
 * always caught (bare or backticked); snake_case and lowercase-with-digits only
 * when backticked (excluded from bare by the backtick context predicate).
 */
export function matchesShape(w: string, backticked: boolean): boolean {
    if (matchesStrongShape(w)) return true;
    if (matchesScreamingSnake(w)) return true;
    if (backticked && matchesWeakShape(w)) return true;
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
// Arithmetic context: preceded or followed by +, -, *, /, =, ^, ·, ×, ÷
const ARITH_RE = /[-+*/=^·×÷−]/;

/**
 * Extract citation candidates from rule files.
 *
 * Returns candidates keyed by {file, line, ref} — deduplicated.
 *
 * Predicate fixes applied:
 *  1. Bare tokens preceded by `*` are glob fragments → skipped.
 *  2. `.ts` path interiors are not re-tokenized → stripped before bare extraction.
 *  3. Weak shapes (snake, lowercase-with-digits) only match when backticked.
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
                if (matchesShape(ref, true)) {
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
                // SCREAMING_SNAKE is caught bare (re-admitted, round 6); snake_case and
                // lowercase-with-digits are backticked-only (excluded by the backtick context predicate).
                if (matchesShape(ref, false)) {
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
                        if (matchesShape(ref, true)) {
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
                    // Otherwise: a token inside a multi-token span — strong shapes only
                    if (matchesShape(ref, false)) {
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
