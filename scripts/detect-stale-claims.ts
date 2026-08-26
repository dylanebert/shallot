#!/usr/bin/env bun

// Stale-claim detector: scans `.claude/rules/**` for backtick-cited `*.ts` paths
// and `*_WGSL`/`*Wgsl`-shaped symbols, then resolves each against the tree.
// Also scans for prose-name references to removed concepts.
//
// Run: `bun run scripts/detect-stale-claims.ts` from shallot root.
// This is a one-off audit tool, NOT a gate — the gate is the check-docs arm.

import { existsSync } from "node:fs";
import { Glob } from "bun";
import { resolve } from "path";

const root = resolve(import.meta.dir, "..");

// ── 1. Collect all rule files ──────────────────────────────────────────────
const ruleFiles: string[] = [];
for await (const match of new Glob("**/*.md").scan({ cwd: resolve(root, ".claude/rules") })) {
    ruleFiles.push(`.claude/rules/${match}`);
}

// ── 2. Extract backtick-cited candidates ───────────────────────────────────
// A backtick-cited `*.ts` path: `something/foo.ts` or `foo.ts`
// A `*_WGSL` or `*Wgsl`-shaped symbol: `ENTITY_COLS_WGSL`, `xformWgsl`, etc.

type Candidate = {
    file: string;
    line: number;
    text: string;
    kind: "ts-path" | "wgsl-symbol";
    ref: string;
};

const candidates: Candidate[] = [];

// Regex for backtick-cited .ts paths
const tsPathRe = /`([^`]*\.ts)`/g;
// Regex for *_WGSL or *Wgsl symbols (inside backticks or bare)
const wgslSymbolRe = /`([A-Za-z_][A-Za-z0-9_]*_WGSL)`|`([A-Za-z_][A-Za-z0-9_]*Wgsl(?:\(\))?)`/g;

for (const file of ruleFiles) {
    const text = await Bun.file(resolve(root, file)).text();
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        // .ts paths
        for (const m of line.matchAll(tsPathRe)) {
            candidates.push({ file, line: i + 1, text: line.trim(), kind: "ts-path", ref: m[1] });
        }
        // WGSL symbols
        for (const m of line.matchAll(wgslSymbolRe)) {
            const ref = m[1] || m[2];
            candidates.push({ file, line: i + 1, text: line.trim(), kind: "wgsl-symbol", ref });
        }
    }
}

// ── 3. Resolve each candidate against the tree ──────────────────────────────

function tsPathExists(path: string): boolean {
    // The path could be relative to src/ or packages/shallot/src/ etc.
    // Try various resolutions
    const tries = [
        resolve(root, path),
        resolve(root, "packages/shallot/src", path),
        resolve(root, "packages/shallot", path),
        resolve(root, "packages/shallot/src", path.replace(/^packages\/shallot\/src\//, "")),
    ];
    for (const t of tries) {
        if (existsSync(t)) return true;
    }
    // Also check via git ls-files for tracked paths
    const git = Bun.spawnSync(["git", "-C", root, "ls-files", "--error-unmatch", path], {
        stdout: "pipe",
        stderr: "pipe",
    });
    if (git.success) return true;
    // Try with packages/shallot/src prefix
    const git2 = Bun.spawnSync(
        ["git", "-C", root, "ls-files", "--error-unmatch", `packages/shallot/src/${path}`],
        { stdout: "pipe", stderr: "pipe" },
    );
    if (git2.success) return true;
    return false;
}

function symbolExists(symbol: string): boolean {
    // Strip trailing () for function-call forms
    const name = symbol.replace(/\(\)$/, "");
    // Grep for the symbol in .ts and .rs files (not node_modules)
    const git = Bun.spawnSync(
        ["git", "-C", root, "grep", "-l", "--fixed-strings", name, "--", "*.ts", "*.rs"],
        { stdout: "pipe", stderr: "pipe" },
    );
    if (!git.success) return false;
    const files = git.stdout.toString().trim().split("\n").filter(Boolean);
    // Exclude node_modules and the arm/detector's own source (their comments mention the symbols they check)
    return files.some(
        (f) =>
            !f.includes("node_modules") &&
            f !== "scripts/check-docs.ts" &&
            f !== "scripts/detect-stale-claims.ts",
    );
}

// ── 4. Report ──────────────────────────────────────────────────────────────

console.log("=== Stale-claim detector ===\n");
console.log(`Scanned ${ruleFiles.length} rule files, found ${candidates.length} candidates.\n`);

const stale: Candidate[] = [];
const live: Candidate[] = [];

for (const c of candidates) {
    const exists = c.kind === "ts-path" ? tsPathExists(c.ref) : symbolExists(c.ref);
    if (exists) {
        live.push(c);
    } else {
        stale.push(c);
    }
}

console.log(`--- LIVE (${live.length}) ---`);
for (const c of live) {
    console.log(`  [${c.kind}] ${c.file}:${c.line} → ${c.ref}`);
}

console.log(`\n--- STALE (${stale.length}) ---`);
for (const c of stale) {
    console.log(`  [${c.kind}] ${c.file}:${c.line} → ${c.ref}`);
    console.log(`    ${c.text.substring(0, 200)}`);
}

// ── 5. Prose-name queries ──────────────────────────────────────────────────
// Key on prose names, not just identifiers — a removed field's paraphrase
// survives an identifier grep and leaves the doc contradicting itself.

console.log("\n=== Prose-name queries ===\n");

const proseQueries: { query: string; pattern: RegExp; desc: string }[] = [
    {
        query: "submodule",
        pattern: /submodule/gi,
        desc: "references to git submodules (repo has no .gitmodules)",
    },
    {
        query: "git submodule update",
        pattern: /git submodule update/gi,
        desc: "git submodule update commands (unrunnable without .gitmodules)",
    },
];

for (const { query, pattern, desc } of proseQueries) {
    console.log(`--- Query: "${query}" (${desc}) ---`);
    for (const file of ruleFiles) {
        const text = await Bun.file(resolve(root, file)).text();
        const lines = text.split("\n");
        for (let i = 0; i < lines.length; i++) {
            if (pattern.test(lines[i])) {
                console.log(`  ${file}:${i + 1}: ${lines[i].trim().substring(0, 200)}`);
            }
        }
    }
    console.log();
}
