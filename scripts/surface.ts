import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";
import { CHECK_CLASSES, CHECK_TIERS, validateDeclaration } from "../src/harness/declaration";

// The check population, derived by discovery and never hand-written: every declared check under
// `src/` and `scripts/`, plus every recipe check declared in an `examples/*/shallot.json`.
// `--list` prints it; `scripts/check-surface.ts` reads the same collection.

/** one row of the population. */
export interface SurfaceRow {
    /** the name passed to Bun's test runner. */
    name: string;
    claim: string;
    class: string;
    tier: string;
    premises: string[];
    budget?: number;
    /** path relative to the tree root. */
    file: string;
}

export interface QuarantineRow {
    file: string;
    claim: string;
    reason: string;
    expires: string;
    spec: string;
}

export interface QuarantineFile {
    rows: QuarantineRow[];
    errors: string[];
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

interface StaticValue {
    ok: boolean;
    value?: unknown;
}

function literal(node: unknown): StaticValue {
    const n = node as {
        type?: string;
        value?: unknown;
        elements?: unknown[];
    };
    if (
        n?.type === "StringLiteral" ||
        n?.type === "NumericLiteral" ||
        n?.type === "BooleanLiteral"
    ) {
        return { ok: true, value: n.value };
    }
    if (n?.type === "ArrayExpression") {
        const values: unknown[] = [];
        for (const element of n.elements ?? []) {
            const parsed = literal(element);
            if (!parsed.ok) return { ok: false };
            values.push(parsed.value);
        }
        return { ok: true, value: values };
    }
    return { ok: false };
}

function objectOf(node: unknown): { value?: Record<string, unknown>; error?: string } {
    const n = node as { type?: string; properties?: unknown[] };
    if (n?.type !== "ObjectExpression") return { error: "options is not an object literal" };
    const out: Record<string, unknown> = {};
    for (const raw of n.properties ?? []) {
        const prop = raw as {
            type?: string;
            computed?: boolean;
            key?: { name?: string; value?: string };
            value?: unknown;
        };
        if (prop.type === "SpreadElement") return { error: "options contains a spread" };
        if (prop.type !== "ObjectProperty")
            return { error: "options contains a non-literal property" };
        if (prop.computed) return { error: "options contains a computed property" };
        const key = prop.key?.name ?? prop.key?.value;
        if (typeof key !== "string") return { error: "options contains a computed property" };
        const value = literal(prop.value);
        if (!value.ok) return { error: `options field \`${key}\` is not a literal` };
        out[key] = value.value;
    }
    return { value: out };
}

function quarantinePath(root: string): string {
    return resolve(root, "quarantine.json");
}

function isIsoDate(value: unknown): value is string {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const parsed = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

/** Read and shape the root quarantine file without making an orphan claim disappear. */
export function readQuarantine(root: string): QuarantineFile {
    const path = quarantinePath(root);
    if (!existsSync(path)) return { rows: [], errors: [] };
    let parsed: unknown;
    try {
        parsed = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
        return { rows: [], errors: [`invalid quarantine.json: ${(error as Error).message}`] };
    }
    if (!Array.isArray(parsed))
        return { rows: [], errors: ["invalid quarantine.json: expected an array"] };
    const rows: QuarantineRow[] = [];
    const errors: string[] = [];
    for (const [index, raw] of parsed.entries()) {
        const row = raw as Record<string, unknown> | null;
        const fields = ["file", "claim", "reason", "expires", "spec"];
        const missing = fields.filter(
            (field) => typeof row?.[field] !== "string" || (row[field] as string).trim() === "",
        );
        if (missing.length > 0) {
            errors.push(
                `invalid quarantine row ${index + 1}: fields must be strings: ${missing.join(", ")}`,
            );
            continue;
        }
        if (!isIsoDate(row?.expires)) {
            errors.push(`invalid quarantine row ${index + 1}: expires must be an ISO date`);
            continue;
        }
        rows.push(row as unknown as QuarantineRow);
    }
    return { rows, errors };
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
        const parsedName = literal(args?.[0]);
        const name = typeof parsedName.value === "string" ? parsedName.value : "?";
        const where = `${file} check(${JSON.stringify(name)})`;
        const parsedDeclaration = objectOf(args?.[1]);
        if (parsedDeclaration.error !== undefined) {
            population.invalid.push(
                `non-literal declaration: ${where} ${parsedDeclaration.error}; expected an object literal of literal fields`,
            );
            return;
        }
        try {
            const valid = validateDeclaration(where, parsedDeclaration.value);
            population.rows.push({
                name,
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
                name: valid.claim,
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

const COLUMNS = ["claim", "class", "tier", "premises", "budget", "file", "status"] as const;

function quarantineKey(file: string, claim: string): string {
    return `${file}\u0000${claim}`;
}

/** Render the population as the table `--list` prints. */
export function formatPopulation(
    population: Population,
    quarantines: readonly QuarantineRow[] = readQuarantine(population.root).rows,
): string {
    const marked = new Map(
        quarantines.map((row) => [quarantineKey(row.file, row.claim), row.reason]),
    );
    const cells = population.rows.map((row) => [
        row.claim,
        row.class,
        row.tier,
        row.premises.join(" ") || "-",
        row.budget === undefined ? "-" : `${row.budget}ms`,
        row.file,
        marked.has(quarantineKey(row.file, row.claim))
            ? `quarantined: ${marked.get(quarantineKey(row.file, row.claim))}`
            : "-",
    ]);
    const widths = COLUMNS.map((name, index) =>
        Math.max(name.length, ...cells.map((cell) => cell[index].length), 0),
    );
    const line = (cell: readonly string[]) =>
        cell
            .map((value, index) => value.padEnd(widths[index]))
            .join("  ")
            .trimEnd();
    return [
        line(COLUMNS),
        ...cells.map(line),
        `${population.rows.length} checks (parsed ${population.rows.length}; ${quarantines.length} quarantined)`,
    ].join("\n");
}

interface BuildPins {
    stable: string;
    nightly: string;
    target: string;
    hasBinaryen: boolean;
}

function readBuildPins(root: string): BuildPins {
    const toolchain = readFileSync(resolve(root, "rust-toolchain.toml"), "utf8");
    const stable = toolchain.match(/^channel\s*=\s*["']([^"']+)["']/m)?.[1];
    const target = toolchain.match(/^targets\s*=\s*\[\s*["']([^"']+)["']/m)?.[1];
    const kernel = readFileSync(resolve(root, "crates/physics/scripts/build-kernel.ts"), "utf8");
    const nightly = kernel.match(/const NIGHTLY = ["']([^"']+)["']/)?.[1];
    const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8")) as {
        dependencies?: Record<string, unknown>;
        devDependencies?: Record<string, unknown>;
        optionalDependencies?: Record<string, unknown>;
    };
    const hasBinaryen = [
        packageJson.dependencies,
        packageJson.devDependencies,
        packageJson.optionalDependencies,
    ].some((group) => group?.binaryen !== undefined);
    if (!stable || !target || !nightly) {
        throw new Error(
            "hosted workflow pins missing: rust-toolchain.toml needs channel and target, and build-kernel.ts needs NIGHTLY",
        );
    }
    return { stable, nightly, target, hasBinaryen };
}

/** Emit the hosted cadence from the discovered population and project build pins. */
export function renderWorkflow(population: Population): string {
    const pins = readBuildPins(population.root);
    const needsChromium = population.rows.some((row) => row.tier === "browser");
    const common = [
        "      - uses: actions/checkout@v4",
        "      - uses: oven-sh/setup-bun@v2",
        "      - run: bun install --frozen-lockfile",
        ...(needsChromium ? ["      - run: bunx playwright install --with-deps chromium"] : []),
        `      - run: rustup toolchain install ${pins.stable} --target ${pins.target}`,
        `      - run: rustup toolchain install ${pins.nightly} --component rust-src --target ${pins.target}`,
        ...(pins.hasBinaryen ? [] : ["      - run: bun add --global binaryen"]),
        "      - run: bun run build",
        "      - run: bun run check",
        "      - run: bun run test",
    ];
    const job = (id: string, runner: string, condition: string): string[] => [
        `  ${id}:`,
        `    if: ${condition}`,
        `    runs-on: ${runner}`,
        "    steps:",
        ...common,
    ];
    return [
        "name: test-surface",
        "",
        "on:",
        "  push:",
        "    branches: [main]",
        '    tags: ["v*"]',
        "  pull_request:",
        "",
        "jobs:",
        ...job(
            "ubuntu",
            "ubuntu-latest",
            "github.event_name == 'pull_request' || (github.event_name == 'push' && github.ref == 'refs/heads/main')",
        ),
        ...job("macos", "macos-latest", "startsWith(github.ref, 'refs/tags/v')"),
        ...job("windows", "windows-latest", "startsWith(github.ref, 'refs/tags/v')"),
        "",
    ].join("\n");
}

if (import.meta.main) {
    const args = Bun.argv.slice(2);
    const rootIndex = args.indexOf("--root");
    const root =
        rootIndex === -1 ? resolve(import.meta.dir, "..") : resolve(args[rootIndex + 1] ?? ".");
    const population = collectPopulation(root);
    if (args.includes("--workflow")) {
        const workflow = resolve(root, ".github/workflows/test-surface.yml");
        mkdirSync(resolve(root, ".github/workflows"), { recursive: true });
        writeFileSync(workflow, renderWorkflow(population));
        console.log(`wrote ${relative(root, workflow)}`);
        process.exit(0);
    }

    if (!args.includes("--list")) {
        console.error("usage: bun scripts/surface.ts --list|--workflow [--root <dir]");
        process.exit(1);
    }
    const quarantine = readQuarantine(root);
    console.log(formatPopulation(population, quarantine.rows));
    if (quarantine.errors.length > 0) process.exit(1);
}

export { CHECK_CLASSES, CHECK_TIERS };
