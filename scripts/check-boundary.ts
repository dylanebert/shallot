import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { parse, parseExpression } from "@babel/parser";

/** The declared escapes from the distribution boundary, and their bounded reasons. this file
 *  reads this table both ways: an undeclared escape refuses, and a declaration naming a specifier no
 *  live file imports refuses too, so the ledger cannot outlive what it excuses. */

/** Files allowed to build a module specifier at runtime rather than name it literally, each with the
 *  bound that keeps it readable. Every other computed `import()`/`require()` refuses: a specifier this
 *  reader cannot resolve is a hole in the source cone, not a detail. */
const COMPUTED_LOADERS: Record<string, string> = {
    "src/standard/physics/engine/pool.ts":
        "the Node-only branch loads the fixed node:worker_threads specifier with vite-ignore; the browser branch creates an embedded Blob worker",
    "src/project/command.ts":
        "eagerly resolves every enabled entry from the project root, then imports those resolved identities",
    "src/project/command.test.ts":
        "the bare-process isolation fixture imports the named command entry under test",
    "bin/features.ts":
        "preflights all enabled project-root entry identities before engine/local evaluation, then reads required features from those identities",
    "bin/bun-native.ts":
        "loads the downloaded native projection, after its sha256 matches the pinned hash",
    "bin/verify.ts":
        "loads the consumer project's own installed playwright, resolved from its package root",
};

/** Directories that carry a `package.json` but are deliberately not workspaces: install-time fixtures and
 *  eval harnesses a workspace install must not hoist. Listed so the completeness rule can tell a fixture
 *  apart from a workspace someone forgot to declare; an entry naming a directory that no longer exists
 *  refuses. */
const NON_WORKSPACE_PACKAGES: Record<string, string> = {
    "evals/harness": "eval harness installed per run, never part of the repo workspace graph",
    "scripts/install-test/widget":
        "the synthetic malformed-install fixture the packed install gate publishes into a temp tree",
};

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
// The population is every declared workspace except the engine package itself, so a new workspace is
// governed by construction rather than by remembering to list it. The package's own `src/` and `bin/`
// are one owner and may reach each other freely. A specifier built at runtime is a hole in the cone
// rather than a detail, so a computed `import()`/`require()` anywhere needs a `COMPUTED_LOADERS` entry.
//
// Limits: plugin-driven resolution and svelte.config aliases need an omission review; aliases also
// apply to node-side tests (deny-direction over-inclusion). Only literal Vite object aliases and
// tsconfigs extending the walked ancestor chain are resolvable; other declared forms refuse.
//
// Default scans this repo. `--root <dir>` scans an external consumer tree, where every project is a
// consumer and neither the loader nor the completeness rule applies.

const PKG = "@dylanebert/shallot";
const ENGINE_PACKAGE = ".";
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
const CALL_COMPUTED = /\b(?:import|require)\s*\((?!\s*["'][^"']*["']\s*\))/g;

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

const SKIP = new Set(["node_modules", ".git", "dist", "target"]);

/** every source file under `dir`. The walk never follows installs or build output: a workspace's
 *  `node_modules` links back to the root, which a recursive read loops on. */
function sourceFiles(dir: string): string[] {
    if (!existsSync(dir)) return [];
    return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
        const path = resolve(dir, e.name);
        if (e.isDirectory()) return SKIP.has(e.name) ? [] : sourceFiles(path);
        return e.isFile() && SOURCE_EXTENSIONS.some((ext) => e.name.endsWith(ext)) ? [path] : [];
    });
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

/** every package.json directory under `dir` is a project. The walk never follows installs or build
 *  output: a workspace's `node_modules` links back to the root, which a recursive read loops on. */
function projectRoots(dir: string): string[] {
    const roots: string[] = [];
    const entries = readdirSync(dir, { withFileTypes: true });
    if (entries.some((e) => e.isFile() && e.name === "package.json")) roots.push(resolve(dir));
    for (const e of entries)
        if (e.isDirectory() && !SKIP.has(e.name)) roots.push(...projectRoots(resolve(dir, e.name)));
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

type Syntax = { type?: string; name?: string; value?: unknown; [key: string]: unknown };

function walk(node: unknown, visit: (node: Syntax) => void): void {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) {
        for (const child of node) walk(child, visit);
        return;
    }
    const syntax = node as Syntax;
    visit(syntax);
    for (const [key, child] of Object.entries(syntax)) {
        if (!["loc", "start", "end", "comments", "tokens"].includes(key)) walk(child, visit);
    }
}

function objectValue(node: Syntax): unknown {
    if (node.type === "ObjectExpression")
        return Object.fromEntries(
            (node.properties as Syntax[]).map((p) => [
                (p.key as Syntax).name ?? (p.key as Syntax).value,
                objectValue(p.value as Syntax),
            ]),
        );
    if (node.type === "ArrayExpression") return (node.elements as Syntax[]).map(objectValue);
    return node.value;
}

function aliasReader(repoRoot: string) {
    type Alias = { key: string; value: string; wildcard: boolean };
    const cache = new Map<string, { aliases: Alias[]; errors: string[] }>();
    return (full: string, spec: string): { targets: string[]; errors: string[] } => {
        const directory = dirname(full);
        let result = cache.get(directory);
        if (!result) {
            const aliases: Alias[] = [];
            const errors: string[] = [];
            const chain: string[] = [];
            for (
                let dir = directory;
                dir === repoRoot || dir.startsWith(repoRoot + sep);
                dir = dirname(dir)
            ) {
                chain.push(dir);
                if (dir === repoRoot) break;
            }
            for (const dir of chain) {
                const config = resolve(dir, "tsconfig.json");
                if (existsSync(config)) {
                    const json = objectValue(
                        parseExpression(readFileSync(config, "utf8")) as unknown as Syntax,
                    ) as {
                        extends?: unknown;
                        compilerOptions?: { baseUrl?: string; paths?: Record<string, unknown> };
                    };
                    const ancestors = chain
                        .slice(chain.indexOf(dir) + 1)
                        .map((ancestor) => resolve(ancestor, "tsconfig.json"));
                    if (
                        json.extends !== undefined &&
                        (typeof json.extends !== "string" ||
                            !ancestors.includes(resolve(dir, json.extends)) ||
                            !existsSync(resolve(dir, json.extends)))
                    )
                        errors.push(
                            `${relative(repoRoot, config)}: extends must name a tsconfig on the walked ancestor chain`,
                        );
                    for (const [key, values] of Object.entries(json.compilerOptions?.paths ?? {})) {
                        if (
                            !Array.isArray(values) ||
                            values.some((value) => typeof value !== "string")
                        ) {
                            errors.push(
                                `${relative(repoRoot, config)}: paths ${key} must be an array of strings`,
                            );
                            continue;
                        }
                        for (const value of values)
                            aliases.push({
                                key,
                                value: resolve(dir, json.compilerOptions?.baseUrl ?? ".", value),
                                wildcard: true,
                            });
                    }
                }
                for (const name of [
                    "vite.config.ts",
                    "vite.config.js",
                    "vite.config.mjs",
                    "vite.config.mts",
                ]) {
                    const path = resolve(dir, name);
                    if (!existsSync(path)) continue;
                    walk(
                        parse(readFileSync(path, "utf8"), {
                            sourceType: "module",
                            plugins: ["typescript"],
                        }),
                        (node) => {
                            if (
                                node.type !== "ObjectProperty" ||
                                ((node.key as Syntax).name ?? (node.key as Syntax).value) !==
                                    "alias"
                            )
                                return;
                            const value = node.value as Syntax;
                            const refuse = (key: string) =>
                                errors.push(
                                    `${relative(repoRoot, path)}: alias ${key} must be a literal string-to-string object mapping`,
                                );
                            if (value.type !== "ObjectExpression") {
                                refuse("<form>");
                                return;
                            }
                            for (const property of value.properties as Syntax[]) {
                                const key = property.key as Syntax | undefined;
                                const replacement = property.value as Syntax | undefined;
                                const name = key?.name ?? key?.value;
                                if (
                                    property.type !== "ObjectProperty" ||
                                    property.computed ||
                                    typeof name !== "string" ||
                                    replacement?.type !== "StringLiteral"
                                ) {
                                    refuse(String(name ?? "<key>"));
                                    continue;
                                }
                                const target = replacement.value as string;
                                if (!isAbsolute(target) && !target.startsWith(".")) {
                                    refuse(name);
                                    continue;
                                }
                                aliases.push({
                                    key: name,
                                    value: resolve(dir, target),
                                    wildcard: false,
                                });
                            }
                        },
                    );
                }
            }
            result = { aliases, errors };
            cache.set(directory, result);
        }
        const targets: string[] = [];
        for (const { key, value, wildcard } of result.aliases) {
            if (wildcard) {
                const [prefix, suffix] = key.split("*");
                if (
                    key.includes("*")
                        ? !spec.startsWith(prefix) || !spec.endsWith(suffix)
                        : spec !== key
                )
                    continue;
                const middle = key.includes("*")
                    ? spec.slice(prefix.length, spec.length - suffix.length)
                    : "";
                targets.push(value.replace("*", middle));
            } else if (spec === key || spec.startsWith(`${key}/`))
                targets.push(value + spec.slice(key.length));
        }
        return { targets, errors: result.errors };
    };
}

function scanConsumers(
    repoRoot: string,
    roots: string[],
    surface: ReturnType<typeof publishedSurface>,
    sampleSeam: string,
    ledger: Ledger = { computedLoaders: {}, nonWorkspacePackages: {} },
    usedLoaders = new Set<string>(),
    errors: string[] = [],
): Violation[] {
    const violations: Violation[] = [];
    const aliasTargets = aliasReader(repoRoot);
    for (const root of roots) {
        const manifest = resolve(root, "package.json");
        if (
            existsSync(manifest) &&
            Object.hasOwn(JSON.parse(readFileSync(manifest, "utf8")), "imports")
        )
            errors.push(`${relative(repoRoot, manifest)}: imports aliases are not supported`);
        for (const full of sourceFiles(root)) {
            // process each file once, under its deepest owning project (roots can nest)
            if (ownerOf(full, roots) !== root) continue;
            const at = (r: Reference, spec: string) => ({
                file: relative(repoRoot, full),
                line: r.line,
                import: spec,
            });
            errors.push(...aliasTargets(full, "").errors);
            for (const r of references(readFileSync(full, "utf8"))) {
                if (r.computed) {
                    const file = relative(repoRoot, full).split(sep).join("/");
                    if (ledger.computedLoaders[file]?.trim()) usedLoaders.add(file);
                    else
                        violations.push({
                            ...at(r, "import(<computed>)"),
                            reason: "builds a module specifier at runtime with no declared bound (COMPUTED_LOADERS)",
                        });
                    continue;
                }
                if (!r.spec) continue;
                const spec = r.spec;
                const targets = spec.startsWith(".")
                    ? [resolve(dirname(full), spec)]
                    : aliasTargets(full, spec).targets;
                for (const resolved of targets) {
                    if (
                        isPublished(spec, surface) &&
                        [...surface.targets].some((target) => {
                            const published = resolve(repoRoot, ENGINE_PACKAGE, target);
                            return [resolved, `${resolved}.ts`, `${resolved}/index.ts`].includes(
                                published,
                            );
                        })
                    )
                        continue;
                    if (resolved === root || resolved.startsWith(root + sep)) continue;
                    if (resolved === sampleSeam || resolved.startsWith(sampleSeam + sep)) continue;
                    violations.push({
                        ...at(r, spec),
                        reason: `escapes the project → ${relative(repoRoot, resolved)}`,
                    });
                }
                if (
                    targets.length === 0 &&
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

/** Computed loaders inside the package itself (`src/`, `bin/`): a specifier built at runtime is a hole in
 *  every source reader, so each one carries a declared bound. */
function scanLoaders(repoRoot: string, ledger: Ledger, used: Set<string>): Violation[] {
    const violations: Violation[] = [];
    for (const full of ["src", "bin"].flatMap((dir) => sourceFiles(resolve(repoRoot, dir)))) {
        const file = relative(repoRoot, full).split(sep).join("/");
        for (const r of references(readFileSync(full, "utf8"))) {
            if (!r.computed) continue;
            if (ledger.computedLoaders[file]?.trim()) used.add(file);
            else
                violations.push({
                    file,
                    line: r.line,
                    import: "import(<computed>)",
                    reason: "builds a module specifier at runtime with no declared bound (COMPUTED_LOADERS)",
                });
        }
    }
    return violations;
}

/** The declared escapes this run reads. Injectable so a fixture tree can carry its own ledger: the real
 *  tables name real repo files, and a fixture cannot satisfy them. */
export interface Ledger {
    computedLoaders: Record<string, string>;
    nonWorkspacePackages: Record<string, string>;
}

export const REPO_LEDGER: Ledger = {
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
    // the committed physics sample scenes and golds consumers replay
    const sampleSeam = resolve(repoRoot, ENGINE_PACKAGE, "src/standard/physics/samples");

    const usedLoaders = new Set<string>();
    const errors: string[] = [];
    const violations = scanConsumers(
        repoRoot,
        consumerRoots,
        surface,
        sampleSeam,
        ledger,
        usedLoaders,
        errors,
    );
    violations.push(...scanLoaders(repoRoot, ledger, usedLoaders));

    // Two-way completeness. A source cone proves nothing if a project can sit outside it, or if a
    // declared escape outlives the code it excused.
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

    // Members reach the engine through the root's `link:.` self-link, never a path of their own: a local
    // engine spec in a member manifest ships in the tarball (recipes) and names a path outside any copy.
    if (rootPkg.devDependencies?.[PKG] !== "link:.")
        errors.push(`root package.json lacks the self-link devDependencies["${PKG}"]: "link:."`);
    for (const dir of consumerDirs) {
        const manifest = resolve(repoRoot, dir, "package.json");
        if (!existsSync(manifest)) continue;
        const pkg = JSON.parse(readFileSync(manifest, "utf8"));
        for (const field of ["dependencies", "devDependencies", "optionalDependencies"]) {
            const range = pkg[field]?.[PKG];
            if (typeof range === "string" && local.test(range))
                errors.push(
                    `${dir}/package.json: ${field} links the engine locally (${range}); the root self-link supplies it`,
                );
        }
    }

    for (const file of Object.keys(ledger.computedLoaders)) {
        if (!usedLoaders.has(file))
            errors.push(`declared computed loader names no live call site: ${file}`);
    }
    return { violations, errors: [...new Set(errors)], consumers: consumerRoots.length };
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
                "code (scripts/) nor an unpublished internal (src/...).",
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
        const errors: string[] = [];
        const violations = scanConsumers(
            external,
            roots,
            surface,
            // an external tree never reaches the repo's sample seam
            resolve(repoRoot, ENGINE_PACKAGE, "src/standard/physics/samples"),
            undefined,
            undefined,
            errors,
        );
        if (!report({ violations, errors: [...new Set(errors)], consumers: roots.length }))
            process.exit(1);
        console.log(`✓ distribution boundary clean (${roots.length} consumer project(s))`);
    } else {
        const result = checkBoundary(repoRoot);
        if (!report(result)) process.exit(1);
        console.log(
            `✓ distribution boundary clean (${result.consumers} consumer project(s), computed loaders and workspace cone complete)`,
        );
    }
}
