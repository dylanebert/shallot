import { existsSync, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";
import { CHECK_CLASSES, CHECK_TIERS, validateDeclaration } from "../src/harness/declaration";

// The check population, derived by discovery and never hand-written: every declared check under
// `src/` and `scripts/`, plus every recipe check declared in an `examples/*/shallot.json`.
// `--list` prints it; `scripts/check-surface.ts` reads the same collection.

/** one row of the population. */
export interface SurfaceRow {
    claim: string;
    class: string;
    tier: string;
    premises: string[];
    budget: number;
    /** path relative to the tree root. */
    file: string;
}

/** a test-suffix file that declared nothing, or reached for the runner directly. */
export interface UndeclaredFile {
    file: string;
    reason: string;
}

/** the whole discovered surface of one tree. */
export interface Population {
    root: string;
    rows: SurfaceRow[];
    undeclared: UndeclaredFile[];
    invalid: string[];
}

const SUFFIX = /\.(test|tier|oracle)\.ts$/;
const SKIP = new Set(["node_modules", "fixtures", "target", "dist", "generate"]);
const BUN_TEST_IMPORT = /import\s*(?:type\s*)?\{([^}]*)\}\s*from\s*["']bun:test["']/g;
const REGISTRARS = new Set(["test", "it", "describe"]);

function literal(node: unknown): unknown {
    const n = node as { type?: string; value?: unknown; elements?: unknown[] };
    if (
        n?.type === "StringLiteral" ||
        n?.type === "NumericLiteral" ||
        n?.type === "BooleanLiteral"
    ) {
        return n.value;
    }
    if (n?.type === "ArrayExpression") return (n.elements ?? []).map(literal);
    return undefined;
}

function objectOf(node: unknown): Record<string, unknown> | undefined {
    const n = node as { type?: string; properties?: unknown[] };
    if (n?.type !== "ObjectExpression") return undefined;
    const out: Record<string, unknown> = {};
    for (const raw of n.properties ?? []) {
        const prop = raw as {
            type?: string;
            key?: { name?: string; value?: string };
            value?: unknown;
        };
        if (prop.type !== "ObjectProperty") continue;
        const key = prop.key?.name ?? prop.key?.value;
        if (typeof key !== "string") continue;
        out[key] = literal(prop.value);
    }
    return out;
}

function walk(node: unknown, visit: (call: Record<string, unknown>) => void): void {
    if (node === null || typeof node !== "object") return;
    if (Array.isArray(node)) {
        for (const item of node) walk(item, visit);
        return;
    }
    const n = node as Record<string, unknown>;
    if (n.type === "CallExpression") {
        const callee = n.callee as { type?: string; name?: string } | undefined;
        if (callee?.type === "Identifier" && callee.name === "check") visit(n);
    }
    for (const [key, value] of Object.entries(n)) {
        if (key === "loc" || key === "leadingComments" || key === "trailingComments") continue;
        walk(value, visit);
    }
}

function readFileDeclarations(root: string, path: string, population: Population): void {
    const file = relative(root, path);
    const source = readFileSync(path, "utf8");
    for (const match of source.matchAll(BUN_TEST_IMPORT)) {
        const names = match[1]
            .split(",")
            .map((part) =>
                part
                    .trim()
                    .split(/\s+as\s+/)[0]
                    .trim(),
            )
            .filter((name) => REGISTRARS.has(name));
        if (names.length > 0) {
            population.undeclared.push({
                file,
                reason: `imports ${names.join(", ")} from bun:test instead of declaring through check()`,
            });
            return;
        }
    }
    const ast = parse(source, { sourceType: "module", plugins: ["typescript"] });
    let found = 0;
    walk(ast.program, (call) => {
        found += 1;
        const args = call.arguments as unknown[];
        const name = literal(args?.[0]);
        const where = `${file} check(${typeof name === "string" ? JSON.stringify(name) : "?"})`;
        const decl = objectOf(args?.[1]);
        try {
            const valid = validateDeclaration(where, decl);
            population.rows.push({
                claim: valid.claim,
                class: valid.class,
                tier: valid.tier,
                premises: [...valid.premises],
                budget: valid.budget,
                file,
            });
        } catch (error) {
            population.invalid.push((error as Error).message);
        }
    });
    if (found === 0)
        population.undeclared.push({ file, reason: "registers no check() declaration" });
}

function readManifest(root: string, path: string, population: Population): void {
    const manifest = JSON.parse(readFileSync(path, "utf8")) as { check?: unknown };
    if (manifest.check === undefined) return;
    const dir = relative(root, resolve(path, ".."));
    const entries = Array.isArray(manifest.check) ? manifest.check : [manifest.check];
    for (const entry of entries) {
        const decl = entry as Record<string, unknown> | null;
        const file = typeof decl?.file === "string" ? `${dir}/${decl.file}` : `${dir}/shallot.json`;
        try {
            const valid = validateDeclaration(`${dir}/shallot.json check`, decl);
            population.rows.push({
                claim: valid.claim,
                class: valid.class,
                tier: valid.tier,
                premises: [...valid.premises],
                budget: valid.budget,
                file,
            });
        } catch (error) {
            population.invalid.push((error as Error).message);
        }
    }
}

/** Walk one tree and return every declared check, plus the files that declared nothing. */
export function collectPopulation(root: string): Population {
    const population: Population = { root, rows: [], undeclared: [], invalid: [] };
    for (const dir of ["src", "scripts"]) {
        const base = resolve(root, dir);
        if (!existsSync(base)) continue;
        for (const match of new Glob("**/*.ts").scanSync({ cwd: base, dot: false })) {
            if (!SUFFIX.test(match)) continue;
            if (match.split("/").some((part) => SKIP.has(part))) continue;
            readFileDeclarations(root, resolve(base, match), population);
        }
    }
    const examples = resolve(root, "examples");
    if (!existsSync(examples)) return finish(population);
    for (const match of new Glob("*/shallot.json").scanSync({ cwd: examples })) {
        readManifest(root, resolve(examples, match), population);
    }
    return finish(population);
}

function finish(population: Population): Population {
    population.rows.sort((a, b) => a.claim.localeCompare(b.claim));
    population.undeclared.sort((a, b) => a.file.localeCompare(b.file));
    return population;
}

const COLUMNS = ["claim", "class", "tier", "premises", "budget", "file"] as const;

/** Render the population as the table `--list` prints. */
export function formatPopulation(population: Population): string {
    const cells = population.rows.map((row) => [
        row.claim,
        row.class,
        row.tier,
        row.premises.join(" ") || "-",
        `${row.budget}ms`,
        row.file,
    ]);
    const widths = COLUMNS.map((name, index) =>
        Math.max(name.length, ...cells.map((cell) => cell[index].length), 0),
    );
    const line = (cell: readonly string[]) =>
        cell
            .map((value, index) => value.padEnd(widths[index]))
            .join("  ")
            .trimEnd();
    return [line(COLUMNS), ...cells.map(line), `${population.rows.length} checks`].join("\n");
}

if (import.meta.main) {
    const args = Bun.argv.slice(2);
    const rootIndex = args.indexOf("--root");
    const root =
        rootIndex === -1 ? resolve(import.meta.dir, "..") : resolve(args[rootIndex + 1] ?? ".");
    if (!args.includes("--list")) {
        console.error("usage: bun scripts/surface.ts --list [--root <dir>]");
        process.exit(1);
    }
    console.log(formatPopulation(collectPopulation(root)));
}

export { CHECK_CLASSES, CHECK_TIERS };
