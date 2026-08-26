#!/usr/bin/env bun

// Stale-claim detector: scans `.claude/rules/**` for identifier-shaped tokens
// (backticked or bare, outside fenced code blocks) and backtick-cited `*.ts`
// paths, then resolves each against a one-pass token index over `*.ts`/`*.rs`/`*.wgsl`
// and committed rosters. This is a one-off audit tool, NOT a gate — the gate
// is the check-docs citation-resolution arm.
//
// Run: `bun run scripts/detect-stale-claims.ts` from shallot root.

import { resolve } from "path";
import { FOREIGN_NAMESPACES, WEBGPU_IDL, WGSL_BUILTINS, X86_ISA } from "./rosters";

const root = resolve(import.meta.dir, "..");

// ── 1. Collect all rule files (git-tracked) ─────────────────────────────────
const rulesTracked = Bun.spawnSync(["git", "-C", root, "ls-files", "-z", "*.md"], {
    cwd: root,
});
const ruleFiles = rulesTracked.stdout
    .toString()
    .split("\0")
    .filter(Boolean)
    .filter((f) => f.startsWith(".claude/rules/"));

// ── 2. Build one-pass token index over *.ts / *.rs / *.wgsl ──────────────────
const allFiles = Bun.spawnSync(["git", "-C", root, "ls-files", "-z"], { cwd: root });
const trackedFiles = allFiles.stdout.toString().split("\0").filter(Boolean);
const trackedSet = new Set(trackedFiles);

const sourceFiles = trackedFiles.filter(
    (f) =>
        (f.endsWith(".ts") || f.endsWith(".rs") || f.endsWith(".wgsl")) &&
        !f.includes("node_modules") &&
        f !== "scripts/check-docs.ts" &&
        f !== "scripts/detect-stale-claims.ts",
);

const tokenIndex = new Set<string>();
const TOKEN_RE = /[A-Za-z_][A-Za-z0-9_]*/g;
for (const f of sourceFiles) {
    const text = await Bun.file(resolve(root, f)).text();
    for (const m of text.matchAll(TOKEN_RE)) {
        tokenIndex.add(m[0]);
    }
}

// ── 3. Combined roster set ───────────────────────────────────────────────────
const combinedRoster = new Set<string>();
for (const s of WGSL_BUILTINS) combinedRoster.add(s);
for (const s of WEBGPU_IDL) combinedRoster.add(s);
for (const s of X86_ISA) combinedRoster.add(s);
for (const [, roster] of Object.entries(FOREIGN_NAMESPACES)) {
    for (const s of roster) combinedRoster.add(s);
}

// ── 4. Identifier shape predicates (same as check-docs arm) ─────────────────
function isCamelCase(w: string): boolean {
    return /^[a-z]/.test(w) && /[A-Z]/.test(w) && !w.includes("_");
}
function isPascalCase(w: string): boolean {
    return /^[A-Z]/.test(w) && /[a-z][A-Z]/.test(w) && !w.includes("_");
}
function isSnakeOrScreaming(w: string): boolean {
    return (
        w.includes("_") &&
        /^[A-Za-z_][A-Za-z0-9_]*$/.test(w) &&
        /[A-Za-z0-9]/.test(w) &&
        w.length >= 2
    );
}
function isLowercaseWithDigits(w: string): boolean {
    return /^[a-z][a-z0-9]*$/.test(w) && /[0-9]/.test(w) && !w.includes("_");
}
function isHex(w: string): boolean {
    return /^[0-9a-f]+$/.test(w) && /[0-9]/.test(w) && /[a-f]/.test(w);
}

const SHAPE_FALSE_POSITIVES = new Set(["AoSoA", "iGPUs", "EndFrame"]);

function matchesShape(w: string): boolean {
    if (w.length < 2) return false;
    if (SHAPE_FALSE_POSITIVES.has(w)) return false;
    if (isHex(w)) return false;
    return isCamelCase(w) || isPascalCase(w) || isSnakeOrScreaming(w) || isLowercaseWithDigits(w);
}

// ── 5. Extract candidates ───────────────────────────────────────────────────
type Candidate = {
    file: string;
    line: number;
    text: string;
    kind: "ts-path" | "identifier";
    ref: string;
    backticked: boolean;
};

const candidates: Candidate[] = [];
const seen = new Set<string>();

const TS_PATH_RE = /`([^`]*\.ts)`/g;
const IDENTIFIER_RE = /`([A-Za-z_][A-Za-z0-9_]*(?:\(\))?)`/g;

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
    candidates.push({ file, line, kind, ref, backticked, text: "" });
}

for (const file of ruleFiles) {
    const text = await Bun.file(resolve(root, file)).text();
    const lines = text.split("\n");
    let inFence = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim().startsWith("```")) {
            inFence = !inFence;
            continue;
        }
        if (inFence) continue;

        // Backtick-cited .ts paths
        for (const m of line.matchAll(TS_PATH_RE)) {
            const ref = m[1];
            if (ref.startsWith(".") || ref.includes("*") || ref.includes(" ") || ref.includes("{"))
                continue;
            addCandidate(file, i + 1, ref, "ts-path", true);
        }

        // Backtick-cited identifiers
        for (const m of line.matchAll(IDENTIFIER_RE)) {
            const ref = m[1].replace(/\(\)$/, "");
            if (ref.endsWith(".ts")) continue;
            if (matchesShape(ref)) addCandidate(file, i + 1, ref, "identifier", true);
        }

        // Bare identifier-shaped tokens (strip URLs, not backtick spans)
        const stripped = line.replace(/https?:\/\/[^\s)]*/g, " ");
        for (const m of stripped.matchAll(TOKEN_RE)) {
            const ref = m[0];
            if (matchesShape(ref)) addCandidate(file, i + 1, ref, "identifier", false);
        }
    }
}

// ── 6. Resolve each candidate ──────────────────────────────────────────────
function tsPathResolves(path: string): boolean {
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

function resolvesAnywhere(ref: string, kind: string): boolean {
    if (kind === "ts-path") {
        if (tsPathResolves(ref)) return true;
        return combinedRoster.has(ref);
    }
    if (tokenIndex.has(ref)) return true;
    return combinedRoster.has(ref);
}

// ── 7. Report ──────────────────────────────────────────────────────────────
console.log("=== Stale-claim detector ===\n");
console.log(
    `Scanned ${ruleFiles.length} rule files, found ${candidates.length} candidates (${new Set(candidates.map((c) => c.ref)).size} distinct).`,
);
console.log(
    `Token index: ${tokenIndex.size} unique tokens from ${sourceFiles.length} source files.`,
);

const stale: Candidate[] = [];
const live: Candidate[] = [];

for (const c of candidates) {
    if (resolvesAnywhere(c.ref, c.kind)) {
        live.push(c);
    } else {
        stale.push(c);
    }
}

console.log(`\n--- LIVE (${live.length}) ---`);
for (const c of live) {
    console.log(`  [${c.kind}] ${c.file}:${c.line} → ${c.ref} ${c.backticked ? "(bt)" : "(bare)"}`);
}

console.log(`\n--- STALE (${stale.length}) ---`);
for (const c of stale) {
    console.log(`  [${c.kind}] ${c.file}:${c.line} → ${c.ref} ${c.backticked ? "(bt)" : "(bare)"}`);
}
