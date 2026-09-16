import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
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
    /** the one host that can hold this row's premise, when it declares one. */
    host?: string;
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

/**
 * one declared per-frame allocation the platform forces, read from `sanctions.json` beside
 * `quarantine.json`. `site` is the owning function site as `<file>:<line>`, the function's definition
 * line, which is the half of an allocation profile's site name a minified production build keeps;
 * `reason` names the function and the platform reason; `count` is the per-frame count derived
 * from the frame's structure (one per frame, one per pass, one per submitted buffer), never a byte
 * figure; `spec` owns it; `approved` carries the person's `(user, YYYY-MM-DD)` and is empty until they
 * approve the row. No agent approves a sanction.
 */
export interface SanctionRow {
    site: string;
    reason: string;
    count: number;
    spec: string;
    approved: string;
}

export interface SanctionFile {
    rows: SanctionRow[];
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

export interface IntegrationSelection {
    all?: boolean;
    requires?: string;
    subject?: string;
    base?: string;
    diff?: string;
}

const SUFFIX = /\.(test|oracle)\.ts$/;
const ORACLE_SUFFIX = /\.oracle\.ts$/;
const SKIP = new Set([".git", ".cache", "node_modules", "fixtures", "target", "dist", "coverage"]);
const BUN_TEST_IMPORT = /import\s*(?:type\s*)?\{([^}]*)\}\s*from\s*["']bun:test["']/g;
const REGISTRARS = new Set(["test", "it", "describe"]);
const COLUMNS = ["claim", "size", "requires", "subject", "budget", "file"] as const;
const DEEP_RECIPE_IMPORT = /(?:from\s+|import\s*\(\s*)["'][^"']*\/src(?:\/|["'])/;
const PHYSICS_WORLD_ESCAPE = /\bPhysics\.world\b|\bphysicsWorld\s*\(/;

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
                ...(valid.host === undefined ? {} : { host: valid.host }),
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

function recipeSourceViolations(root: string, manifestPath: string, population: Population): void {
    let manifest: unknown;
    try {
        manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    } catch {
        return;
    }
    if (
        !manifest ||
        typeof manifest !== "object" ||
        (manifest as { kind?: unknown }).kind !== "recipe"
    )
        return;

    const sourceRoot = resolve(dirname(manifestPath), "src");
    if (!existsSync(sourceRoot)) return;
    for (const match of new Glob("**/*.ts").scanSync({ cwd: sourceRoot, dot: false })) {
        const path = resolve(sourceRoot, match);
        const source = readFileSync(path, "utf8");
        const file = relativeFile(root, path);
        if (DEEP_RECIPE_IMPORT.test(source))
            population.invalid.push(
                `recipe source uses a deep engine import: ${file}; import only from package exports`,
            );
        if (PHYSICS_WORLD_ESCAPE.test(source))
            population.invalid.push(
                `recipe source uses Physics.world/physicsWorld: ${file}; use the State-scoped public seam`,
            );
    }
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
    for (const match of new Glob("**/*.{test,oracle}.ts").scanSync({
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
    for (const match of manifestFiles) {
        const path = resolve(population.root, match);
        manifestResults.set(match, manifestEntries(population.root, path, population));
        recipeSourceViolations(population.root, path, population);
    }
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

const APPROVAL = /^\(user, \d{4}-\d{2}-\d{2}\)$/;

/**
 * the declared per-frame allocation sanctions, read by the same reader as {@link readQuarantine} so
 * `list` and every verdict count them. A malformed row is an error, never a silently dropped sanction.
 */
export function readSanctions(root: string): SanctionFile {
    const path = resolve(root, "sanctions.json");
    if (!existsSync(path)) return { rows: [], errors: [] };
    let parsed: unknown;
    try {
        parsed = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
        return { rows: [], errors: [`invalid sanctions.json: ${(error as Error).message}`] };
    }
    if (!Array.isArray(parsed))
        return { rows: [], errors: ["invalid sanctions.json: expected an array"] };
    const rows: SanctionRow[] = [];
    const errors: string[] = [];
    for (const [index, raw] of parsed.entries()) {
        const row = raw as Record<string, unknown> | null;
        const fields = ["site", "reason", "spec"];
        const missing = fields.filter(
            (field) => typeof row?.[field] !== "string" || (row[field] as string).trim() === "",
        );
        if (missing.length > 0) {
            errors.push(
                `invalid sanction row ${index + 1}: fields must be strings: ${missing.join(", ")}`,
            );
            continue;
        }
        if (typeof row?.count !== "number" || !Number.isInteger(row.count) || row.count < 1) {
            errors.push(`invalid sanction row ${index + 1}: count must be a positive integer`);
            continue;
        }
        const approved = row?.approved;
        if (typeof approved !== "string" || (approved !== "" && !APPROVAL.test(approved))) {
            errors.push(
                `invalid sanction row ${index + 1}: approved must be "" or "(user, YYYY-MM-DD)"`,
            );
            continue;
        }
        rows.push(row as unknown as SanctionRow);
    }
    return { rows, errors };
}

export function formatPopulation(
    population: Population,
    quarantines: readonly QuarantineRow[] = readQuarantine(population.root).rows,
    rows: readonly SurfaceRow[] = population.rows,
    sanctions: readonly SanctionRow[] = readSanctions(population.root).rows,
): string {
    const cells = rows.map((row) => [
        row.claim,
        row.size,
        row.requires.join(" ") || "-",
        row.subjects.join(" ") || "-",
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
        `${rows.length} checks (parsed ${population.rows.length}; ${quarantines.length} quarantined; ${sanctions.length} sanctioned, ${sanctions.filter((row) => row.approved === "").length} unapproved)`,
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

function typeAt(root: string, ref: string, path: string): string {
    const proc = Bun.spawnSync(["git", "cat-file", "-t", `${ref}:${path}`], {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    return proc.exitCode === 0 ? proc.stdout.toString().trim() : "";
}

type TreeSubject = readonly (readonly [string, readonly string[]])[];

function treeSubject(root: string, ref: string, path: string): TreeSubject {
    const proc = Bun.spawnSync(["git", "ls-tree", "-r", "-z", "--full-tree", ref, "--", path], {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    if (proc.exitCode !== 0) return [];
    const entries: Array<readonly [string, readonly string[]]> = [];
    for (const record of proc.stdout.toString().split("\0")) {
        if (record === "") continue;
        const tab = record.indexOf("\t");
        if (tab === -1) continue;
        const fields = record.slice(0, tab).split(" ");
        if (fields[1] !== "blob") continue;
        const file = record.slice(tab + 1);
        entries.push([file, subjectTokens(readAt(root, ref, file))]);
    }
    return entries.sort(([left], [right]) => left.localeCompare(right));
}

function subjectValue(root: string, ref: string, path: string): readonly string[] | TreeSubject {
    const type = typeAt(root, ref, path);
    return type === "tree" ? treeSubject(root, ref, path) : subjectTokens(readAt(root, ref, path));
}

export function selectOracleRows(population: Population, claim: string): SurfaceRow[] {
    return population.rows.filter((row) => ORACLE_SUFFIX.test(row.file) && row.claim === claim);
}

/** Root-law selector: comments do not select; complete pre/post subject token streams do. */
export function subjectChanged(
    root: string,
    subjects: readonly string[],
    base: string,
    diff: string,
): boolean {
    if (subjects.length === 0) return true;
    return subjects.some((subject) => {
        const baseType = typeAt(root, base, subject);
        const diffType = typeAt(root, diff, subject);
        if (baseType !== diffType) return true;
        return (
            JSON.stringify(subjectValue(root, base, subject)) !==
            JSON.stringify(subjectValue(root, diff, subject))
        );
    });
}

export function selectIntegrationRows(
    population: Population,
    selection: IntegrationSelection,
): SurfaceRow[];
export function selectIntegrationRows(
    population: Population,
    base: string,
    diff: string,
): SurfaceRow[];
export function selectIntegrationRows(
    population: Population,
    selectionOrBase: IntegrationSelection | string,
    legacyDiff?: string,
): SurfaceRow[] {
    const selection: IntegrationSelection =
        typeof selectionOrBase === "string"
            ? { base: selectionOrBase, diff: legacyDiff }
            : selectionOrBase;
    return population.rows.filter((row) => {
        if (row.size !== "integration" || ORACLE_SUFFIX.test(row.file)) return false;
        if (selection.requires !== undefined && !row.requires.includes(selection.requires))
            return false;
        if (
            selection.subject !== undefined &&
            !row.subjects.some((subject) => subject.startsWith(selection.subject as string))
        )
            return false;
        if (selection.base !== undefined && selection.diff !== undefined)
            return subjectChanged(population.root, row.subjects, selection.base, selection.diff);
        return true;
    });
}

function workflowRows(population: Population): SurfaceRow[] {
    return population.rows.filter((row) => !ORACLE_SUFFIX.test(row.file));
}

function workflowNeedsChromium(population: Population): boolean {
    // A row declared for one host is skipped on the hosted runner, so installing a browser for it would
    // provision a premise nothing there uses.
    return workflowRows(population).some(
        (row) => row.requires.includes("chromium") && row.host === undefined,
    );
}

function workflowNeedsCargo(population: Population): boolean {
    return workflowRows(population).some((row) => row.requires.includes("cargo"));
}

function workflowNeedsNode(population: Population): boolean {
    return workflowRows(population).some((row) => row.requires.includes("node"));
}

function workflowPath(root: string): string {
    return resolve(root, ".github/workflows/test-surface.yml");
}

/** Render only portable Bun/project steps. The carrier never names a member, branch or engine. */
export function renderWorkflow(population: Population): string {
    if (workflowRows(population).length === 0) return "";
    const steps = [
        "      - uses: actions/checkout@v7",
        "        with:",
        "          fetch-depth: 0",
        "      - uses: oven-sh/setup-bun@v2",
    ];
    if (workflowNeedsCargo(population))
        steps.push(
            "      - uses: dtolnay/rust-toolchain@stable",
            "      - uses: actions/cache@v6",
            "        with:",
            "          path: target",
            "          key: $" +
                "{{ runner.os }}-cargo-$" +
                "{{ hashFiles('**/Cargo.lock', 'rust-toolchain.toml') }}",
            "          restore-keys: |",
            "            $" + "{{ runner.os }}-cargo-",
        );
    if (workflowNeedsNode(population))
        steps.push(
            "      - uses: actions/setup-node@v6",
            "        with:",
            "          node-version-file: .node-version",
        );
    steps.push(
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
    );
    if (workflowNeedsChromium(population))
        steps.push("      - run: bunx playwright install --with-deps chromium");
    steps.push(
        "      - run: bun run check",
        "      - run: bun run test",
        "      - run: bun run test -- --integration --base $SHALLOT_SURFACE_BASE --diff $SHALLOT_SURFACE_DIFF",
        "",
    );
    return [
        "name: test-surface",
        "",
        "on:",
        "  push:",
        "    branches:",
        "      - main",
        "  pull_request:",
        "    branches:",
        "      - main",
        "",
        "concurrency:",
        "  group: test-surface-${{ github.event.pull_request.number || github.sha }}",
        "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
        "",
        "jobs:",
        "  surface:",
        "    runs-on: ubuntu-latest",
        "    steps:",
        ...steps,
    ].join("\n");
}

function shallotPackage(name: string): boolean {
    return name === "@dylanebert/shallot" || name.startsWith("@dylanebert/shallot-");
}

type ForbiddenSpecifier = "link" | "file" | "git" | "github" | "URL" | "workspace" | "portal";
const FULL_COMMIT = /^[0-9a-f]{40}$/i;
const PUBLISHED_RANGE = /^(?:[vV]?\d|[~^<>=*|])/;
const REMOTE_TARBALL = /^https?:\/\/[^\s]+\.(?:tgz|tar\.gz)(?:[?#].*)?$/i;
const LOCAL_TARBALL = /\.(?:tgz|tar\.gz)$/i;

function forbiddenSpecifier(spec: string): ForbiddenSpecifier | null {
    if (spec.startsWith("link:")) return "link";
    if (spec.startsWith("file:")) return "file";
    if (spec.startsWith("workspace:")) return "workspace";
    if (spec.startsWith("portal:")) return "portal";
    if (spec.startsWith("github:")) return "github";
    if (spec.startsWith("git")) return "git";
    try {
        if (new URL(spec).protocol) return "URL";
    } catch {}
    return null;
}

function gitCommit(spec: string): string | null {
    const hash = spec.slice(spec.lastIndexOf("#") + 1);
    return spec.includes("#") && FULL_COMMIT.test(hash) ? hash : null;
}

function lockText(root: string): string | null {
    const path = resolve(root, "bun.lock");
    return existsSync(path) ? readFileSync(path, "utf8") : null;
}

function lockHas(lock: string | null, spec: string, integrity: boolean): boolean {
    if (lock === null) return false;
    const start = lock.indexOf(spec);
    if (start < 0) return false;
    if (!integrity) return true;
    return /sha(?:256|512)-[A-Za-z0-9+/=_-]+/.test(lock.slice(start, start + 2048));
}

function checkedTarballViolation(root: string, file: string, spec: string): string | null {
    const relativePath = spec.slice("file:".length);
    if (relativePath.startsWith("/") || relativePath.split("/").includes(".."))
        return `${file}: local Shallot tarball path escapes the project: ${JSON.stringify(spec)}`;
    if (!LOCAL_TARBALL.test(relativePath))
        return `${file}: local Shallot source directory is not a sanctioned artifact: ${JSON.stringify(spec)}`;
    const tarball = resolve(root, relativePath);
    if (!existsSync(tarball))
        return `${file}: checked-in Shallot tarball is missing: ${relativePath}`;
    const digest = ["sha256", "sha512"]
        .map((algorithm) => `${tarball}.${algorithm}`)
        .find((sidecar) => existsSync(sidecar));
    if (
        digest === undefined ||
        !new RegExp("^[0-9a-f]{" + (digest.endsWith("sha256") ? 64 : 128) + "}", "im").test(
            readFileSync(digest, "utf8"),
        )
    )
        return `${file}: checked-in Shallot tarball needs a hexadecimal digest sidecar: ${relativePath}`;
    const provenance = `${tarball}.source-commit`;
    if (!existsSync(provenance) || !FULL_COMMIT.test(readFileSync(provenance, "utf8").trim()))
        return `${file}: checked-in Shallot tarball needs a full source-commit sidecar: ${relativePath}`;
    return null;
}

function dependencyViolations(root: string): string[] {
    const violations: string[] = [];
    const lock = lockText(root);
    const files = [...new Glob("**/package.json").scanSync({ cwd: root, dot: false })]
        .filter((file) => !file.split("/").some((part) => SKIP.has(part)))
        .sort();
    for (const file of files) {
        const path = resolve(root, file);
        let manifest: Record<string, unknown>;
        try {
            const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
            if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) continue;
            manifest = parsed as Record<string, unknown>;
        } catch (error) {
            violations.push(`invalid package manifest: ${file}: ${(error as Error).message}`);
            continue;
        }
        const packageName = typeof manifest.name === "string" ? manifest.name : null;
        for (const [table, raw] of Object.entries(manifest)) {
            if (!/dependencies$/i.test(table) || raw === null || typeof raw !== "object") continue;
            for (const [name, value] of Object.entries(raw as Record<string, unknown>)) {
                if (!shallotPackage(name) || typeof value !== "string") continue;
                if (name === packageName && value === "link:.") continue;
                const kind = forbiddenSpecifier(value);
                if (kind === "link" || kind === "workspace" || kind === "portal") {
                    violations.push(
                        `${file}: ${table}.${name} uses forbidden ${kind} specifier ${JSON.stringify(value)}`,
                    );
                    continue;
                }
                if (kind === "file") {
                    const violation = checkedTarballViolation(root, file, value);
                    if (violation !== null) violations.push(violation);
                    continue;
                }
                if (kind === "git" || kind === "github") {
                    const commit = gitCommit(value);
                    if (commit === null)
                        violations.push(
                            `${file}: ${table}.${name} requires a full 40-hex Git commit, got ${JSON.stringify(value)}`,
                        );
                    else if (!lockHas(lock, value, false))
                        violations.push(
                            `${file}: ${table}.${name} Git identity is not recorded in bun.lock: ${JSON.stringify(value)}`,
                        );
                    continue;
                }
                if (kind === "URL") {
                    if (!REMOTE_TARBALL.test(value))
                        violations.push(
                            `${file}: ${table}.${name} uses an unqualified Shallot URL: ${JSON.stringify(value)}`,
                        );
                    else if (!lockHas(lock, value, true))
                        violations.push(
                            `${file}: ${table}.${name} remote tarball needs matching lock integrity: ${JSON.stringify(value)}`,
                        );
                    continue;
                }
                if (!PUBLISHED_RANGE.test(value))
                    violations.push(
                        `${file}: ${table}.${name} uses mutable dist-tag ${JSON.stringify(value)}`,
                    );
            }
        }
    }
    return violations;
}

export function readSurface(root: string): string[] {
    const population = collectPopulation(root);
    const violations = [
        ...population.invalid,
        ...population.undeclared.map(
            (file) => `undeclared check file: ${file.file} ${file.reason}`,
        ),
        ...dependencyViolations(root),
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
    const sanctions = readSanctions(root);
    violations.push(...sanctions.errors);
    const sanctioned = new Set<string>();
    for (const row of sanctions.rows) {
        if (sanctioned.has(row.site))
            violations.push(`duplicate sanction row: site "${row.site}" is declared twice`);
        sanctioned.add(row.site);
    }
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
