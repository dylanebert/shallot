import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { relative, resolve, sep } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";
import {
    CHECK_REQUIREMENTS,
    CHECK_SIZES,
    INTEGRATION_BUDGET_MS,
    UNIT_BUDGET_MS,
    validateDeclaration,
} from "./declaration";

/** A discovered check, independent of the project that owns it. */
export interface SurfaceRow {
    name: string;
    claim: string;
    size: (typeof CHECK_SIZES)[number];
    requires: string[];
    budget: number;
    file: string;
    /** Source file(s) whose token changes select this integration row. */
    subjects: string[];
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

export interface UndeclaredFile {
    file: string;
    reason: string;
}

export interface Population {
    root: string;
    rows: SurfaceRow[];
    undeclared: UndeclaredFile[];
    invalid: string[];
    /** Entrypoints discovered by the carrier, not a hand-maintained registry. */
    files: string[];
}

const SUFFIX = /\.(test|tier|oracle)\.ts$/;
const ORACLE_SUFFIX = /\.oracle\.ts$/;
const SKIP = new Set([".git", ".cache", "node_modules", "fixtures", "target", "dist", "coverage"]);
const BUN_TEST_IMPORT = /import\s*(?:type\s*)?\{([^}]*)\}\s*from\s*["']bun:test["']/g;
const REGISTRARS = new Set(["test", "it", "describe"]);
const COLUMNS = ["claim", "size", "requires", "budget", "file"] as const;

interface StaticValue {
    ok: boolean;
    value?: unknown;
}

function literal(node: unknown): StaticValue {
    const n = node as { type?: string; value?: unknown; elements?: unknown[] };
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

function relativeFile(root: string, path: string): string {
    return relative(root, path).split(sep).join("/");
}

function checkImportNames(source: string): string[] {
    const names: string[] = [];
    for (const match of source.matchAll(BUN_TEST_IMPORT)) {
        names.push(
            ...match[1]
                .split(",")
                .map(
                    (part) =>
                        part
                            .trim()
                            .split(/\s+as\s+/)[0]
                            ?.trim() ?? "",
                )
                .filter((name) => REGISTRARS.has(name)),
        );
    }
    return names;
}

function subjectList(value: unknown): string[] {
    if (value === undefined) return [];
    const values = Array.isArray(value) ? value : [value];
    return values
        .filter((entry): entry is string => typeof entry === "string" && entry.trim() !== "")
        .map((entry) => resolveSubject(entry));
}

function resolveSubject(subject: string): string {
    if (subject.startsWith("/")) return subject;
    // A relative subject is rooted at the project, not at the declaration file. `./` is accepted
    // because manifests and member projects commonly use it when spelling a source path.
    return subject.replace(/^\.\//, "");
}

export function readFileDeclarations(root: string, path: string, population: Population): void {
    const file = relativeFile(root, path);
    population.files.push(file);
    let source: string;
    try {
        source = readFileSync(path, "utf8");
    } catch (error) {
        population.invalid.push(`cannot read ${file}: ${(error as Error).message}`);
        return;
    }
    const imported = checkImportNames(source);
    if (imported.length > 0) {
        population.undeclared.push({
            file,
            reason: `imports ${imported.join(", ")} from bun:test instead of declaring through check()`,
        });
        return;
    }
    let ast: any;
    try {
        ast = parse(source, { sourceType: "module", plugins: ["typescript"] });
    } catch (error) {
        population.invalid.push(`cannot parse ${file}: ${(error as Error).message}`);
        return;
    }
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
                size: valid.size,
                requires: [...valid.requires],
                budget: valid.budget,
                file,
                subjects: subjectList(parsedDeclaration.value?.subject),
            });
        } catch (error) {
            population.invalid.push((error as Error).message);
        }
    });
    if (found === 0)
        population.undeclared.push({ file, reason: "registers no check() declaration" });
}

interface ManifestEntry {
    file: string;
}

interface ManifestRead {
    entries: ManifestEntry[];
    hasCheck: boolean;
}

function manifestEntries(root: string, path: string, population: Population): ManifestRead {
    const label = relativeFile(root, path);
    let parsed: unknown;
    try {
        parsed = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
        population.invalid.push(`invalid manifest: ${label}: ${(error as Error).message}`);
        return { entries: [], hasCheck: false };
    }
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        population.invalid.push(`invalid manifest: ${label}: expected an object`);
        return { entries: [], hasCheck: false };
    }
    const manifest = parsed as Record<string, unknown>;
    if (!Object.hasOwn(manifest, "check")) return { entries: [], hasCheck: false };
    const check = manifest.check;
    const entries = Array.isArray(check) ? check : [check];
    if (entries.length === 0) {
        population.invalid.push(`invalid manifest: ${label}: check array must not be empty`);
        return { entries: [], hasCheck: true };
    }
    const seen = new Set<string>();
    const result: ManifestEntry[] = [];
    for (const [index, raw] of entries.entries()) {
        if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
            population.invalid.push(
                `invalid manifest: ${label} check[${index + 1}] must be an object with file`,
            );
            continue;
        }
        const entry = raw as Record<string, unknown>;
        const extras = Object.keys(entry).filter((key) => key !== "file");
        if (extras.length > 0)
            population.invalid.push(
                `invalid manifest: ${label} check[${index + 1}] has unknown fields ${extras.join(", ")}`,
            );
        if (typeof entry.file !== "string" || entry.file.trim() === "") {
            population.invalid.push(`invalid manifest: ${label} check[${index + 1}] needs a file`);
            continue;
        }
        const file = resolve(path, "..", entry.file);
        const relativePath = relativeFile(root, file);
        if (relativePath === ".." || relativePath.startsWith("../")) {
            population.invalid.push(`manifest entry escapes project root: ${relativePath}`);
            continue;
        }
        if (seen.has(relativePath)) {
            population.invalid.push(`duplicate manifest entry: ${relativePath}`);
            continue;
        }
        seen.add(relativePath);
        if (!existsSync(file)) {
            population.invalid.push(`manifest entry does not exist: ${relativePath}`);
            continue;
        }
        if (!SUFFIX.test(file))
            population.invalid.push(`manifest entry is not a check file: ${relativePath}`);
        result.push({ file });
    }
    return { entries: result, hasCheck: true };
}

/** Read one manifest-selected check file through the same carrier reader. */
export function readCheckDeclarations(
    root: string,
    path: string,
): { rows: SurfaceRow[]; errors: string[] } {
    const population: Population = {
        root: resolve(root),
        rows: [],
        undeclared: [],
        invalid: [],
        files: [],
    };
    if (!existsSync(path))
        return {
            rows: [],
            errors: [`recipe check file does not exist: ${relativeFile(root, path)}`],
        };
    readFileDeclarations(root, path, population);
    return {
        rows: population.rows,
        errors: [
            ...population.invalid,
            ...population.undeclared.map(
                (file) => `undeclared recipe check file: ${file.file} ${file.reason}`,
            ),
        ],
    };
}

function discoveredFiles(root: string): string[] {
    const files: string[] = [];
    for (const match of new Glob("**/*.{test,tier,oracle}.ts").scanSync({
        cwd: root,
        dot: false,
    })) {
        if (!SUFFIX.test(match) || match.split("/").some((part) => SKIP.has(part))) continue;
        files.push(resolve(root, match));
    }
    return files.sort();
}

/** Discover all check files from the project root; no engine folders or member names are assumed. */
export function collectPopulation(root: string): Population {
    const population: Population = {
        root: resolve(root),
        rows: [],
        undeclared: [],
        invalid: [],
        files: [],
    };
    const files = discoveredFiles(population.root);
    const manifests = new Glob("**/shallot.json").scanSync({ cwd: population.root, dot: false });
    const manifestFiles = [...manifests]
        .filter((match) => !match.split("/").some((part) => SKIP.has(part)))
        .sort();
    const manifestResults = new Map<string, ManifestRead>();
    for (const match of manifestFiles)
        manifestResults.set(
            match,
            manifestEntries(population.root, resolve(population.root, match), population),
        );
    const rootResult = manifestResults.get("shallot.json");
    const rootIsAuthoritative = rootResult?.hasCheck === true;
    const admitted = rootIsAuthoritative
        ? new Set((rootResult?.entries ?? []).map((entry) => resolve(entry.file)))
        : new Set<string>(files);
    if (rootIsAuthoritative) {
        const listed = new Set(admitted);
        for (const file of files) {
            if (!listed.has(file))
                population.invalid.push(
                    `unlisted check file: ${relativeFile(population.root, file)}; root shallot.json check is authoritative`,
                );
        }
    } else {
        for (const result of manifestResults.values())
            for (const entry of result.entries) admitted.add(resolve(entry.file));
    }
    for (const path of [...admitted].sort()) {
        if (!SUFFIX.test(path) || path.split(sep).some((part) => SKIP.has(part))) continue;
        readFileDeclarations(population.root, path, population);
    }
    population.rows.sort((a, b) => a.claim.localeCompare(b.claim));
    population.undeclared.sort((a, b) => a.file.localeCompare(b.file));
    population.files = [...new Set(population.files)].sort();
    return population;
}

function isIsoDate(value: unknown): value is string {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const date = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(date.valueOf()) && date.toISOString().slice(0, 10) === value;
}

export function readQuarantine(root: string): QuarantineFile {
    const path = resolve(root, "quarantine.json");
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

export function formatPopulation(
    population: Population,
    quarantines: readonly QuarantineRow[] = readQuarantine(population.root).rows,
): string {
    const cells = population.rows.map((row) => [
        row.claim,
        row.size,
        row.requires.join(" ") || "-",
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
    return [
        line(COLUMNS),
        ...cells.map(line),
        `${population.rows.length} checks (parsed ${population.rows.length}; ${quarantines.length} quarantined)`,
    ].join("\n");
}

function projectPackage(root: string): boolean {
    return existsSync(resolve(root, "package.json"));
}

/** Strip comments while preserving strings, then return the complete lexical token stream. */
export function subjectTokens(source: string): string[] {
    const withoutComments = source.replace(
        /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\/\/[^\n]*|\/\*[\s\S]*?\*\//g,
        (_match, string) => string ?? "",
    );
    return (
        withoutComments.match(
            /(?:[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|===|!==|=>|==|!=|<=|>=|&&|\|\||\+\+|--|\*\*|[^\s])/g,
        ) ?? []
    );
}

function readAt(root: string, ref: string, path: string): string {
    const proc = Bun.spawnSync(["git", "show", `${ref}:${path}`], {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    return proc.exitCode === 0 ? proc.stdout.toString() : "";
}

/** Root-law selector: comments do not select; complete pre/post subject token streams do. */
export function subjectChanged(
    root: string,
    subjects: readonly string[],
    base: string,
    diff: string,
): boolean {
    if (subjects.length === 0) return true;
    return subjects.some(
        (subject) =>
            JSON.stringify(subjectTokens(readAt(root, base, subject))) !==
            JSON.stringify(subjectTokens(readAt(root, diff, subject))),
    );
}

export function selectIntegrationRows(
    population: Population,
    base: string,
    diff: string,
): SurfaceRow[] {
    return population.rows.filter(
        (row) =>
            row.size === "integration" &&
            !ORACLE_SUFFIX.test(row.file) &&
            subjectChanged(population.root, row.subjects, base, diff),
    );
}

function workflowRows(population: Population): SurfaceRow[] {
    return population.rows.filter((row) => !ORACLE_SUFFIX.test(row.file));
}

function workflowNeedsChromium(population: Population): boolean {
    return workflowRows(population).some((row) => row.requires.includes("chromium"));
}

function workflowPath(root: string): string {
    return resolve(root, ".github/workflows/test-surface.yml");
}

/** Render only portable Bun/project steps. The carrier never names a member, branch or engine. */
export function renderWorkflow(population: Population): string {
    if (workflowRows(population).length === 0) return "";
    const steps = [
        "      - uses: actions/checkout@v4",
        "        with:",
        "          fetch-depth: 0",
        "      - uses: oven-sh/setup-bun@v2",
        "      - name: resolve surface refs",
        "        env:",
        "          SURFACE_EVENT: $" + "{{ github.event_name }}",
        "          SURFACE_PR_BASE: $" + "{{ github.event.pull_request.base.sha }}",
        "          SURFACE_BEFORE: $" + "{{ github.event.before }}",
        "          SURFACE_DIFF: $" + "{{ github.sha }}",
        "          SURFACE_REF: $" + "{{ github.ref }}",
        "          SURFACE_DEFAULT_BRANCH: $" + "{{ github.event.repository.default_branch }}",
        "        run: |",
        "          set -euo pipefail",
        "          zero=0000000000000000000000000000000000000000",
        '          if [ "$SURFACE_EVENT" = "pull_request" ]; then',
        '            base="$SURFACE_PR_BASE"',
        '          elif [ -n "$SURFACE_BEFORE" ] && [ "$SURFACE_BEFORE" != "$zero" ]; then',
        '            base="$SURFACE_BEFORE"',
        '          elif [ "$SURFACE_REF" = "refs/heads/$SURFACE_DEFAULT_BRANCH" ]; then',
        '            base="$(git rev-parse --verify --quiet --end-of-options "$SURFACE_DIFF^")" || {',
        '              echo "surface refused: the initial default-branch push has no parent commit" >&2',
        "              exit 1",
        "            }",
        "          else",
        '            base="$(git merge-base "$SURFACE_DIFF" "origin/$SURFACE_DEFAULT_BRANCH")" || {',
        '              echo "surface refused: could not derive a merge base for the new branch push" >&2',
        "              exit 1",
        "            }",
        "          fi",
        '          git rev-parse --verify --quiet --end-of-options "$base^{commit}" >/dev/null || {',
        '            echo "surface refused: base is not an existing commit object: $base" >&2',
        "            exit 1",
        "          }",
        '          git rev-parse --verify --quiet --end-of-options "$SURFACE_DIFF^{commit}" >/dev/null || {',
        '            echo "surface refused: diff is not an existing commit object: $SURFACE_DIFF" >&2',
        "            exit 1",
        "          }",
        '          echo "surface refs: base=$base diff=$SURFACE_DIFF"',
        '          echo "SHALLOT_SURFACE_BASE=$base" >> "$GITHUB_ENV"',
        '          echo "SHALLOT_SURFACE_DIFF=$SURFACE_DIFF" >> "$GITHUB_ENV"',
        "      - run: bun install --frozen-lockfile",
    ];
    if (workflowNeedsChromium(population))
        steps.push("      - run: bunx playwright install --with-deps chromium");
    steps.push(
        "      - run: bun run check",
        "      - run: bun run test",
        "      - run: bun run test:integration -- --base $SHALLOT_SURFACE_BASE --diff $SHALLOT_SURFACE_DIFF",
        "",
    );
    return [
        "name: test-surface",
        "",
        "on: [push, pull_request]",
        "",
        "jobs:",
        "  surface:",
        "    runs-on: ubuntu-latest",
        "    steps:",
        ...steps,
    ].join("\n");
}

export function readSurface(root: string): string[] {
    const population = collectPopulation(root);
    const violations = [
        ...population.invalid,
        ...population.undeclared.map(
            (file) => `undeclared check file: ${file.file} ${file.reason}`,
        ),
    ];
    const seen = new Map<string, string>();
    for (const row of population.rows) {
        const first = seen.get(row.claim);
        if (first !== undefined)
            violations.push(`duplicate claim: "${row.claim}" declared in ${first} and ${row.file}`);
        else seen.set(row.claim, row.file);
        const ceiling = row.size === "unit" ? UNIT_BUDGET_MS : INTEGRATION_BUDGET_MS;
        if (row.budget > ceiling)
            violations.push(
                `over-budget declaration: "${row.claim}" in ${row.file} declares ${row.budget}ms above the ${row.size} ceiling of ${ceiling}ms`,
            );
        for (const subject of row.subjects) {
            if (subject.startsWith("/") || subject.includes(".."))
                violations.push(
                    `invalid subject: "${subject}" in ${row.file}; subjects must be project-rooted paths`,
                );
        }
    }
    const quarantine = readQuarantine(root);
    violations.push(...quarantine.errors);
    const files = new Set(population.rows.map((row) => row.file));
    const claims = new Set(population.rows.map((row) => row.claim));
    const exactRows = new Set(population.rows.map((row) => `${row.file}\u0000${row.claim}`));
    const today = new Date().toISOString().slice(0, 10);
    for (const row of quarantine.rows) {
        if (row.expires < today)
            violations.push(`expired quarantine row: "${row.claim}" expired ${row.expires}`);
        if (!files.has(row.file))
            violations.push(
                `orphan quarantine row: file "${row.file}" names no check in the population`,
            );
        if (!claims.has(row.claim))
            violations.push(
                `orphan quarantine row: claim "${row.claim}" names no check in the population`,
            );
        if (
            files.has(row.file) &&
            claims.has(row.claim) &&
            !exactRows.has(`${row.file}\u0000${row.claim}`)
        )
            violations.push(
                `orphan quarantine row: file "${row.file}" and claim "${row.claim}" do not identify the same check`,
            );
    }
    if (projectPackage(root)) {
        const path = workflowPath(root);
        if (workflowRows(population).length === 0) {
            if (existsSync(path))
                violations.push("empty population must not have a generated workflow");
        } else if (!existsSync(path)) {
            violations.push("missing generated workflow: .github/workflows/test-surface.yml");
        } else if (readFileSync(path, "utf8") !== renderWorkflow(population)) {
            violations.push(
                "generated workflow drift: .github/workflows/test-surface.yml differs from workflow rendering",
            );
        }
    }
    return violations;
}

export function writeWorkflow(root: string): "written" | "removed" | "empty" {
    const population = collectPopulation(root);
    const path = workflowPath(root);
    if (workflowRows(population).length === 0) {
        if (existsSync(path)) rmSync(path);
        return "empty";
    }
    mkdirSync(resolve(root, ".github/workflows"), { recursive: true });
    writeFileSync(path, renderWorkflow(population));
    return "written";
}

export function discoverTestFiles(root: string, includeOracles = false): string[] {
    const population = collectPopulation(root);
    return population.files
        .filter((path) => includeOracles || !ORACLE_SUFFIX.test(path))
        .map((path) => relativeFile(root, resolve(root, path)));
}

export { CHECK_REQUIREMENTS, CHECK_SIZES };
