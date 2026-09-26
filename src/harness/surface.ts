import { existsSync, readFileSync } from "node:fs";
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
}

export interface QuarantineRow {
    file: string;
    claim: string;
    reason: string;
    expires: string;
    spec: string;
}

/** the rows of one root declaration and the errors its malformed rows raised */
export interface DeclarationFile<Row> {
    rows: Row[];
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
// A temp dir a check creates must live under the host temp root; one rooted in the repository leaks into
// discovery, `git status` and sibling sweeps when a row dies before its cleanup.
const MKDTEMP_CALL = /\bmkdtempSync\s*\(([^;]*)/g;
const HOST_TEMP_ROOT = /\btmpdir\s*\(\s*\)|\bTMPDIR\b/;
const FUNCTION_NODES = new Set([
    "FunctionDeclaration",
    "FunctionExpression",
    "ArrowFunctionExpression",
    "ObjectMethod",
    "ClassMethod",
    "ClassPrivateMethod",
]);

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

/**
 * Process spawns evaluated at import, outside every function body, so discovery itself runs them.
 * Bound: only a callee named `spawnSync` (bare or member) in a `.test.ts`; spawns through an
 * imported helper, other spawn names, or an `.oracle.ts` are not seen here.
 */
function moduleScopeSpawns(node: unknown): number {
    if (node === null || typeof node !== "object") return 0;
    if (Array.isArray(node)) return node.reduce((sum, item) => sum + moduleScopeSpawns(item), 0);
    const n = node as Record<string, unknown>;
    if (typeof n.type === "string" && FUNCTION_NODES.has(n.type)) return 0;
    let count = 0;
    if (n.type === "CallExpression") {
        const callee = n.callee as
            | { type?: string; name?: string; property?: { name?: string } }
            | undefined;
        const name = callee?.type === "Identifier" ? callee.name : callee?.property?.name;
        if (name === "spawnSync") count += 1;
    }
    for (const [key, value] of Object.entries(n)) {
        if (key === "loc" || key === "leadingComments" || key === "trailingComments") continue;
        count += moduleScopeSpawns(value);
    }
    return count;
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

export function readFileDeclarations(
    root: string,
    path: string,
    population: Population,
): string | undefined {
    const file = relativeFile(root, path);
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
        return source;
    }
    let ast: any;
    try {
        ast = parse(source, { sourceType: "module", plugins: ["typescript"] });
    } catch (error) {
        population.invalid.push(`cannot parse ${file}: ${(error as Error).message}`);
        return source;
    }
    if (file.endsWith(".test.ts") && moduleScopeSpawns(ast.program) > 0)
        population.invalid.push(
            `import-time spawn: ${file} spawns a process at module scope; spawn inside a check() body so discovery runs nothing`,
        );
    for (const match of source.matchAll(MKDTEMP_CALL)) {
        if (!HOST_TEMP_ROOT.test(match[1]))
            population.invalid.push(
                `in-repository temp dir: ${file} calls mkdtempSync outside tmpdir(); root temp dirs at the host temp directory`,
            );
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
        const subjects = subjectList(parsedDeclaration.value?.subject);
        if (subjects.includes(file)) {
            population.invalid.push(
                `self subject: ${where} names its own file as subject; name the source it checks`,
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
                subjects,
            });
        } catch (error) {
            population.invalid.push((error as Error).message);
        }
    });
    if (found === 0)
        population.undeclared.push({ file, reason: "registers no check() declaration" });
    return source;
}

function recipeSourceViolations(
    root: string,
    manifestPath: string,
    population: Population,
    conventionSources: ReadonlyMap<string, string | undefined>,
): void {
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
        const file = relativeFile(root, path);
        // Reuse the declaration reader's source for convention-named files. This keeps the source
        // rules and the population reader on one physical read.
        const source = conventionSources.has(path)
            ? conventionSources.get(path)
            : readFileSync(path, "utf8");
        if (source === undefined) continue;
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
    const conventionSources = new Map<string, string | undefined>();
    // The naming convention is the complete population. Read each discovered path once, regardless of
    // what any project or example manifest happens to contain.
    for (const path of files) {
        const source = readFileDeclarations(population.root, path, population);
        conventionSources.set(path, source);
    }
    const manifests = new Glob("**/shallot.json").scanSync({ cwd: population.root, dot: false });
    const manifestFiles = [...manifests]
        .filter((match) => !match.split("/").some((part) => SKIP.has(part)))
        .sort();
    for (const match of manifestFiles)
        recipeSourceViolations(
            population.root,
            resolve(population.root, match),
            population,
            conventionSources,
        );
    population.files = files.map((path) => relativeFile(population.root, path));
    population.rows.sort((a, b) => a.claim.localeCompare(b.claim));
    population.undeclared.sort((a, b) => a.file.localeCompare(b.file));
    return population;
}

function isIsoDate(value: unknown): value is string {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    const date = new Date(`${value}T00:00:00Z`);
    return !Number.isNaN(date.valueOf()) && date.toISOString().slice(0, 10) === value;
}

/**
 * One root declaration file: which file it is, what one row is called in an error, the fields every row
 * must carry as non-empty strings, and any further per-row rules.
 */
interface Declaration {
    file: string;
    noun: string;
    strings: readonly string[];
    rules?: readonly ((row: Record<string, unknown>) => string)[];
}

const QUARANTINE: Declaration = {
    file: "quarantine.json",
    noun: "quarantine",
    strings: ["file", "claim", "reason", "expires", "spec"],
    rules: [(row) => (isIsoDate(row.expires) ? "" : "expires must be an ISO date")],
};

/**
 * Read one root declaration. An absent file is zero rows and no error, so a declaration is never
 * satisfied by deleting it; a malformed row is an error, never a silently dropped row.
 */
function readDeclaration<Row>(root: string, declaration: Declaration): DeclarationFile<Row> {
    const { file, noun, strings, rules = [] } = declaration;
    const path = resolve(root, file);
    if (!existsSync(path)) return { rows: [], errors: [] };
    let parsed: unknown;
    try {
        parsed = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
        return { rows: [], errors: [`invalid ${file}: ${(error as Error).message}`] };
    }
    if (!Array.isArray(parsed)) return { rows: [], errors: [`invalid ${file}: expected an array`] };
    const rows: Row[] = [];
    const errors: string[] = [];
    for (const [index, raw] of parsed.entries()) {
        const row = (raw ?? {}) as Record<string, unknown>;
        const at = `invalid ${noun} row ${index + 1}`;
        const missing = strings.filter(
            (field) => typeof row[field] !== "string" || (row[field] as string).trim() === "",
        );
        if (missing.length > 0) {
            errors.push(`${at}: fields must be strings: ${missing.join(", ")}`);
            continue;
        }
        const broken = rules.map((rule) => rule(row)).find((reason) => reason !== "");
        if (broken !== undefined) {
            errors.push(`${at}: ${broken}`);
            continue;
        }
        rows.push(row as Row);
    }
    return { rows, errors };
}

export const readQuarantine = (root: string): DeclarationFile<QuarantineRow> =>
    readDeclaration(root, QUARANTINE);

export function formatPopulation(
    population: Population,
    quarantines: readonly QuarantineRow[] = readQuarantine(population.root).rows,
    rows: readonly SurfaceRow[] = population.rows,
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
        `${rows.length} checks (parsed ${population.rows.length}; ${quarantines.length} quarantined)`,
    ].join("\n");
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

export function readSurface(root: string, population = collectPopulation(root)): string[] {
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
    return violations;
}

export function discoverTestFiles(root: string, includeOracles = false): string[] {
    const population = collectPopulation(root);
    return population.files.filter((path) => includeOracles || !ORACLE_SUFFIX.test(path));
}

export { CHECK_REQUIREMENTS, CHECK_SIZES };
