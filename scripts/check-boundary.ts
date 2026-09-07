import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import { COMPUTED_LOADERS, NON_WORKSPACE_PACKAGES, TOOLING_SEAMS } from "./boundary-seams";

// Distribution boundary: a consumer of the published @dylanebert/shallot surface reaches the engine only
// through the package name, and only through its declared `exports` — never repo-only directories
// (scripts/, tests/, sibling examples) and never an unpublished internal. Two ways a consumer could break
// out:
//
//   1. a relative import that climbs out of its own project root, and
//   2. a package import that reaches a subpath outside the published `exports` (the `./src/*` wildcard is
//      the escape hatch — it resolves any internal file by package name, so a consumer using it is
//      reaching an unexposed internal; the root tsconfig's `@dylanebert/shallot/src/*` path maps the same
//      specifier onto the same files, so aliasing does not launder it).
//
// Both are violations. Cross-project access goes through the package name, into a published subpath only.
//
// The one narrow allowance: a relative import that escapes the project is permitted iff it resolves inside
// `packages/shallot/tests/` — the CPU-oracle cross-check seam (the f64 avbd solver/joint + the bvh
// fixtures/oracle a gym scenario diffs the GPU against). Those f64 references are the executable spec,
// load-bearing and unpublished by design; killing the share would force duplicating them. Engine *access*
// still must use the published exports — the allowance is the tests/ oracle only, not `src/`.
//
// The population is every declared workspace except the engine package itself, so a new workspace is
// governed by construction rather than by remembering to list it. The engine package's own tooling
// (`bin/**`) is scanned under a second rule: it may reach engine source only through a module the export
// map publishes, or through a `TOOLING_SEAMS` entry that says why. A specifier built at runtime is a hole
// in the cone rather than a detail, so a computed `import()`/`require()` needs a `COMPUTED_LOADERS` entry.
//
// Default scans this repo. `--root <dir>` scans an external consumer tree, where every project is a
// consumer and neither the tooling nor the completeness rule applies.

const PKG = "@dylanebert/shallot";
const ENGINE_PACKAGE = "packages/shallot";
const SOURCE_EXTENSIONS = [".ts", ".tsx", ".js", ".mjs", ".cjs", ".svelte"];

export interface Violation {
    file: string;
    line: number;
    import: string;
    reason: string;
}

/** the published surface, read from the engine's own `exports` map (one source of truth): the specifiers
 *  a consumer may name, and the source files those specifiers land on. The `./src/*` wildcard is
 *  deliberately dropped from both — it exists so repo tooling can deep-import by path, and a consumer
 *  reaching it is the "unexposed internal" this guards against. Any other wildcard stays a legitimate
 *  prefix. */
export function publishedSurface(exports: Record<string, unknown>): {
    exact: Set<string>;
    prefixes: string[];
    targets: Set<string>;
} {
    const exact = new Set<string>();
    const prefixes: string[] = [];
    const targets = new Set<string>();
    for (const [key, value] of Object.entries(exports)) {
        if (key === "./src/*") continue;
        const spec = key === "." ? PKG : `${PKG}/${key.slice(2)}`;
        if (spec.endsWith("/*")) prefixes.push(spec.slice(0, -1));
        else exact.add(spec);
        // a conditional target publishes its `types` source: that file is the published module, while
        // `default` is its compiled projection.
        const source =
            typeof value === "string"
                ? value
                : ((value as Record<string, string> | null)?.types ?? null);
        if (source) targets.add(source.replace(/^\.\//, ""));
    }
    return { exact, prefixes, targets };
}

/** true if a bare `@dylanebert/shallot...` specifier lands on a published subpath. */
function isPublished(spec: string, surface: { exact: Set<string>; prefixes: string[] }): boolean {
    return surface.exact.has(spec) || surface.prefixes.some((p) => spec.startsWith(p));
}

/** Blank out comment text so a specifier quoted in prose is not read as an import. Only block comments
 *  and whole-line `//` comments are removed: a trailing `//` inside a string literal (a URL) is far more
 *  common than a trailing comment naming a specifier, and a wrong strip there would hide a real import. */
export function stripComments(source: string): string {
    const withoutBlocks = source.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " "));
    return withoutBlocks
        .split("\n")
        .map((line) => (line.trimStart().startsWith("//") ? "" : line))
        .join("\n");
}

/** One module specifier a file names, with the line it sits on. `computed` marks a dynamic
 *  `import()`/`require()` whose argument is not a literal — the specifier is unknown, so the reader
 *  reports the call site instead of a target. */
export interface Reference {
    spec: string | null;
    line: number;
    computed: boolean;
}

// The surface forms, enumerated rather than approximated by one pattern (a regex is a lexical-class
// claim): static `import`/`export … from`, side-effect `import "x"`, and literal `import("x")` /
// `require("x")`. `export … from` is what carries a re-export escape, and the dynamic forms are what the
// previous single `(?:from|import)\s+["']` pattern could not see at all.
const FROM = /(?:^|[\s;}])(?:import|export)\b[\s\S]*?\bfrom\s*["']([^"']+)["']/gm;
const SIDE_EFFECT = /(?:^|[\s;}])import\s*["']([^"']+)["']/g;
const CALL_LITERAL = /\b(?:import|require)\s*\(\s*["']([^"']+)["']\s*\)/g;
const CALL_COMPUTED = /\b(?:import|require)\s*\(\s*(?!["'])/g;

/** Every module specifier a source file names, plus its computed-loader call sites. Pure over text so
 *  each surface form is testable without a tree. */
export function references(source: string): Reference[] {
    const text = stripComments(source);
    const lineOf = (index: number): number => text.slice(0, index).split("\n").length;
    const found: Reference[] = [];
    for (const re of [FROM, SIDE_EFFECT, CALL_LITERAL]) {
        re.lastIndex = 0;
        for (const m of text.matchAll(re))
            found.push({ spec: m[1], line: lineOf(m.index ?? 0), computed: false });
    }
    CALL_COMPUTED.lastIndex = 0;
    for (const m of text.matchAll(CALL_COMPUTED))
        found.push({ spec: null, line: lineOf(m.index ?? 0), computed: true });
    return found.sort((a, b) => a.line - b.line);
}

/** every source file under `dir`, skipping node_modules and build output. */
function sourceFiles(dir: string): string[] {
    if (!existsSync(dir)) return [];
    return readdirSync(dir, { recursive: true, withFileTypes: true })
        .filter(
            (e) =>
                e.isFile() &&
                SOURCE_EXTENSIONS.some((ext) => e.name.endsWith(ext)) &&
                !e.parentPath.includes(`${sep}node_modules${sep}`) &&
                !e.parentPath.includes(`${sep}dist${sep}`),
        )
        .map((e) => resolve(e.parentPath, e.name));
}

/** Every workspace directory the root manifest declares, repo-root-relative. A directory that looks like
 *  a project but no pattern reaches is not governed, so the completeness rule below names it. */
export function workspaceRoots(root: string, patterns: string[]): string[] {
    const roots: string[] = [];
    for (const pattern of patterns) {
        if (!pattern.includes("*")) {
            roots.push(pattern);
            continue;
        }
        const parent = pattern.slice(0, pattern.lastIndexOf("/"));
        const dir = resolve(root, parent);
        if (!existsSync(dir)) continue;
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            if (entry.isDirectory()) roots.push(`${parent}/${entry.name}`);
        }
    }
    return roots.sort();
}

/** every package.json directory under `dir` (skipping node_modules) is a project. */
function projectRoots(dir: string): string[] {
    const roots: string[] = [];
    for (const entry of readdirSync(dir, { recursive: true, withFileTypes: true })) {
        if (entry.name !== "package.json") continue;
        if (entry.parentPath.includes(`${sep}node_modules`)) continue;
        roots.push(resolve(entry.parentPath));
    }
    return roots;
}

/** the deepest project root containing `file`, or null if none. */
function ownerOf(file: string, roots: string[]): string | null {
    let best: string | null = null;
    for (const root of roots) {
        if (
            (file === root || file.startsWith(root + sep)) &&
            (!best || root.length > best.length)
        ) {
            best = root;
        }
    }
    return best;
}

function scanConsumers(
    repoRoot: string,
    roots: string[],
    surface: ReturnType<typeof publishedSurface>,
    oracleSeam: string,
): Violation[] {
    const violations: Violation[] = [];
    for (const root of roots) {
        for (const full of sourceFiles(root)) {
            // process each file once, under its deepest owning project (roots can nest)
            if (ownerOf(full, roots) !== root) continue;
            const at = (r: Reference, spec: string) => ({
                file: relative(repoRoot, full),
                line: r.line,
                import: spec,
            });
            for (const r of references(readFileSync(full, "utf8"))) {
                if (r.computed || !r.spec) continue;
                const spec = r.spec;
                if (spec.startsWith(".")) {
                    const resolved = resolve(dirname(full), spec);
                    if (resolved === root || resolved.startsWith(root + sep)) continue;
                    if (resolved === oracleSeam || resolved.startsWith(oracleSeam + sep)) continue;
                    violations.push({
                        ...at(r, spec),
                        reason: `escapes the project → ${relative(repoRoot, resolved)}`,
                    });
                } else if (
                    (spec === PKG || spec.startsWith(`${PKG}/`)) &&
                    !isPublished(spec, surface)
                ) {
                    violations.push({
                        ...at(r, spec),
                        reason: "reaches an unpublished @dylanebert/shallot internal (not in exports)",
                    });
                }
            }
        }
    }
    return violations;
}

/** the engine package's own tooling: it lives inside the package, so a relative reach is not a project
 *  escape — but a reach into engine source the export map does not publish is a private coupling that a
 *  later extraction has to carry, so each one is declared or refused. */
function scanTooling(
    repoRoot: string,
    surface: ReturnType<typeof publishedSurface>,
    ledger: Ledger,
): { violations: Violation[]; usedSeams: Set<string>; usedLoaders: Set<string> } {
    const violations: Violation[] = [];
    const usedSeams = new Set<string>();
    const usedLoaders = new Set<string>();
    const binDir = resolve(repoRoot, ENGINE_PACKAGE, "bin");
    for (const full of sourceFiles(binDir)) {
        const file = relative(repoRoot, full).split(sep).join("/");
        for (const r of references(readFileSync(full, "utf8"))) {
            if (r.computed) {
                if (ledger.computedLoaders[file]) usedLoaders.add(file);
                else
                    violations.push({
                        file,
                        line: r.line,
                        import: "import(<computed>)",
                        reason: "builds a module specifier at runtime with no declared bound (scripts/boundary-seams.ts)",
                    });
                continue;
            }
            const spec = r.spec as string;
            if (!spec.startsWith("../src")) continue;
            const target = relative(resolve(repoRoot, ENGINE_PACKAGE), resolve(dirname(full), spec))
                .split(sep)
                .join("/");
            const published = [target, `${target}.ts`, `${target}/index.ts`].some((candidate) =>
                surface.targets.has(candidate),
            );
            if (published) continue;
            const key = `${file} "${spec}"`;
            if (ledger.toolingSeams[key]) {
                usedSeams.add(key);
                continue;
            }
            violations.push({
                file,
                line: r.line,
                import: spec,
                reason: `reaches unpublished engine source ${target} with no declared seam (scripts/boundary-seams.ts)`,
            });
        }
    }
    return { violations, usedSeams, usedLoaders };
}

/** The declared escapes this run reads. Injectable so a fixture tree can carry its own ledger: the real
 *  tables name real repo files, and a fixture cannot satisfy them. */
export interface Ledger {
    toolingSeams: Record<string, string>;
    computedLoaders: Record<string, string>;
    nonWorkspacePackages: Record<string, string>;
}

export const REPO_LEDGER: Ledger = {
    toolingSeams: TOOLING_SEAMS,
    computedLoaders: COMPUTED_LOADERS,
    nonWorkspacePackages: NON_WORKSPACE_PACKAGES,
};

export interface BoundaryResult {
    violations: Violation[];
    /** free-standing refusals with no single import to blame: an orphaned declaration or an ungoverned
     *  project directory. */
    errors: string[];
    consumers: number;
}

/** The whole in-repo check. Pure over the tree so every clause is fixture-testable. */
export function checkBoundary(repoRoot: string, ledger: Ledger = REPO_LEDGER): BoundaryResult {
    const rootPkg = JSON.parse(readFileSync(resolve(repoRoot, "package.json"), "utf8"));
    const enginePkg = JSON.parse(
        readFileSync(resolve(repoRoot, ENGINE_PACKAGE, "package.json"), "utf8"),
    );
    const surface = publishedSurface(enginePkg.exports as Record<string, unknown>);
    const declared = workspaceRoots(repoRoot, rootPkg.workspaces as string[]);
    const consumerDirs = declared.filter((dir) => dir !== ENGINE_PACKAGE);
    const consumerRoots = consumerDirs
        .map((dir) => resolve(repoRoot, dir))
        .filter((dir) => existsSync(dir) && statSync(dir).isDirectory());
    const oracleSeam = resolve(repoRoot, ENGINE_PACKAGE, "tests");

    const violations = scanConsumers(repoRoot, consumerRoots, surface, oracleSeam);
    const tooling = scanTooling(repoRoot, surface, ledger);
    violations.push(...tooling.violations);

    // Two-way completeness. A source cone proves nothing if a project can sit outside it, or if a
    // declared escape outlives the code it excused.
    const errors: string[] = [];
    const governed = new Set(declared);
    const seenFixtures = new Set<string>();
    for (const project of projectRoots(repoRoot)) {
        const rel = relative(repoRoot, project).split(sep).join("/");
        if (rel === "" || rel.startsWith("..")) continue;
        if (governed.has(rel)) continue;
        if (ledger.nonWorkspacePackages[rel]) {
            seenFixtures.add(rel);
            continue;
        }
        errors.push(
            `package directory is neither a declared workspace nor a declared fixture: ${rel}`,
        );
    }
    for (const dir of Object.keys(ledger.nonWorkspacePackages)) {
        if (!seenFixtures.has(dir))
            errors.push(`declared non-workspace package names no directory: ${dir}`);
    }
    for (const dir of declared) {
        if (!existsSync(resolve(repoRoot, dir, "package.json")))
            errors.push(`declared workspace has no package.json: ${dir}`);
    }
    // A published package cannot carry a local production dependency: `workspace:`/`file:`/`link:` resolves
    // inside this repo and is unresolvable from the registry, so the tarball installs broken. Dev
    // dependencies are fine — they never ship.
    const local = /^(?:workspace:|file:|link:|portal:|\.{1,2}\/)/;
    for (const [name, range] of Object.entries(
        (enginePkg.dependencies ?? {}) as Record<string, string>,
    )) {
        if (local.test(range))
            errors.push(
                `published package declares a local production dependency: ${name}@${range}`,
            );
    }

    for (const key of Object.keys(ledger.toolingSeams)) {
        if (!tooling.usedSeams.has(key))
            errors.push(`declared tooling seam names no live import: ${key}`);
    }
    for (const file of Object.keys(ledger.computedLoaders)) {
        if (!tooling.usedLoaders.has(file))
            errors.push(`declared computed loader names no live call site: ${file}`);
    }
    return { violations, errors, consumers: consumerRoots.length };
}

function report(result: BoundaryResult): boolean {
    if (result.violations.length > 0) {
        console.error(`✗ ${result.violations.length} distribution-boundary violation(s):\n`);
        for (const v of result.violations) {
            console.error(`  ${v.file}:${v.line}`);
            console.error(`    import "${v.import}" ${v.reason}`);
        }
        console.error(
            "\nA consumer must reach the engine through the published @dylanebert/shallot\n" +
                "surface — a declared `exports` subpath — never a relative path into repo-only\n" +
                "code (scripts/, tests/) nor an unpublished internal (src/...).",
        );
    }
    if (result.errors.length > 0) {
        console.error(`\n✗ ${result.errors.length} boundary-completeness error(s):\n`);
        for (const e of result.errors) console.error(`  ${e}`);
    }
    return result.violations.length === 0 && result.errors.length === 0;
}

if (import.meta.main) {
    const repoRoot = resolve(import.meta.dir, "..");
    const rootArg = process.argv.indexOf("--root");
    if (rootArg >= 0) {
        // an external consumer tree: every project is a consumer, and neither the tooling nor the
        // completeness rule applies (it owns no workspace manifest of ours).
        const external = resolve(process.argv[rootArg + 1]);
        const enginePkg = JSON.parse(
            readFileSync(resolve(repoRoot, ENGINE_PACKAGE, "package.json"), "utf8"),
        );
        const surface = publishedSurface(enginePkg.exports as Record<string, unknown>);
        const roots = projectRoots(external);
        const violations = scanConsumers(
            external,
            roots,
            surface,
            resolve(repoRoot, ENGINE_PACKAGE, "tests"),
        );
        if (!report({ violations, errors: [], consumers: roots.length })) process.exit(1);
        console.log(`✓ distribution boundary clean (${roots.length} consumer project(s))`);
    } else {
        const result = checkBoundary(repoRoot);
        if (!report(result)) process.exit(1);
        console.log(
            `✓ distribution boundary clean (${result.consumers} consumer project(s), tooling seams and workspace cone complete)`,
        );
    }
}
