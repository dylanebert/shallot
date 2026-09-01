import { Glob } from "bun";
import { dirname, relative, resolve, sep } from "path";

// packages/shallot-tui is engine-agnostic by design: cells to bytes, no GPU, no bun-webgpu,
// testable on any seat with no adapter. This is the mechanism that carries that packaging
// decision — a check, not a repo location or a convention anybody can quietly cross. It is a
// stricter property than `check-boundary.ts`'s consumer boundary: shallot-tui may not import the
// engine at all, published surface or not, so this is its own script rather than a mode of that
// one.
//
// Five routes are closed, because "not a convention anybody can quietly cross" means all of them,
// not just the two syntactic shapes that happen to be easiest to grep:
//   1. a relative import (static or dynamic) that climbs out of `packages/shallot-tui/`;
//   2. a bare `@dylanebert/shallot` import (static or dynamic), published subpath or not;
//   3. the engine reached through a call-style specifier — `require(...)`, `createRequire(...)(...)`,
//      or any loader idiom whose sole argument is a specifier string — rather than only
//      `import`/`from`;
//   4. a workspace member that itself depends on the engine, transitively (`@dylanebert/shallot-ocean`
//      does, via its own `package.json`) — importing it is a route to the engine even though the
//      member's own package name isn't a subpath of `@dylanebert/shallot`, so this walks the
//      workspace dependency graph rather than pattern-matching the specifier string. This is not
//      restricted to `@dylanebert/`-scoped names (B3): the workspace's `examples/*` members
//      (`orbit-camera`, `gym`, …) declare unscoped names and most declare
//      `"@dylanebert/shallot": "workspace:*"` directly, so `import { x } from "orbit-camera"`
//      inside `packages/shallot-tui/src/` is exactly as much a one-hop route to the engine as a
//      scoped sibling is, and gets the same check;
//   5. a `dependencies`/`peerDependencies`/`devDependencies`/`optionalDependencies` entry in
//      shallot-tui's own `package.json` naming the engine, or a workspace member that itself
//      reaches the engine — unenforced by a source scan alone, and for a published package this
//      declared-dependency edge is the boundary a consumer actually experiences (an installed
//      dependency ships and resolves regardless of whether any source file imports it).
//
// The scan covers the whole `packages/shallot-tui/` tree (`node_modules` excluded), not just
// `src/` and `tests/` — the package's testability claim ("no GPU, no @dylanebert/shallot import
// ... testable on any seat with no adapter") covers the whole package, not just its shipped
// surface, and a hardcoded `["src", "tests"]` subdirectory list silently stopped covering a future
// sibling directory the moment one existed (N4). Multiple source extensions are scanned
// (`SOURCE_GLOB`), not `.ts` alone — the package's `files` field ships all of `src`, so a `.mts`,
// `.tsx`, or plain `.js` sibling would ship unscanned under a `.ts`-only glob.
//
// `--root <dir>` points the source scan at an alternate tree (fixture-driven proof; check-imports.ts
// and check-boundary.ts carry the same flag for the same reason). The workspace dependency graph
// (route 4) is always read from this repo's own root `package.json` `workspaces` globs (`packages/*`,
// `examples/gym`, `examples/flows/*`, `examples/recipes/*`, `examples/showcase/*`) rather than a
// hardcoded `packages/*/package.json` glob (B3) — the workspace membership is itself a property of
// this repo's `package.json`, so deriving it from anywhere else can silently drift from what the
// package manager actually resolves, and a hardcoded `packages/*` misses every `examples/*` member
// entirely, which is exactly the live evasion B3 found.

const repoRoot = resolve(import.meta.dir, "..");
const PKG = "@dylanebert/shallot";

// Static `import ... from "spec"` / `import "spec"` / `export ... from "spec"`. `\s*` (not `\s+`)
// between the keyword and the quote — `import{State}from"@dylanebert/shallot"` is valid,
// minifier-shaped JS that a `\s+`-only pattern silently missed (N4).
const STATIC_IMPORT_RE = /(?:from|import)\s*["']([^"']+)["']/g;
// Dynamic `import("spec")` / `await import("spec")` — no `from`, and `import` is followed by `(`
// rather than whitespace-then-quote, so `STATIC_IMPORT_RE` never matches this shape.
const DYNAMIC_IMPORT_RE = /\bimport\s*\(\s*["']([^"']+)["']/g;
// Any call whose sole argument is a bare string specifier — covers `require("spec")` and
// `createRequire(...)("spec")` alike, whatever the callee is named. Deliberately not restricted to
// the engine's own literal (N4): `require("@dylanebert/shallot-ocean")` is call-style *and*
// transitive, and a regex that only matched the engine's own specifier missed it. The specifier
// captured here is run through the same `classify()` every static/dynamic import goes through, so
// a non-specifier string argument to an unrelated call (e.g. `foo("hi")`) never becomes a false
// positive — `classify` only flags a relative escape, the engine, or a workspace sibling that
// reaches it.
const CALL_ARG_RE = /\(\s*["']([^"']+)["']\s*\)/g;

// File extensions the scan reads specifiers from. The package's `files` field ships all of `src`
// (`packages/shallot-tui/package.json`), not just `.ts` — a `.mts`, `.tsx`, or plain `.js` sibling
// would ship unscanned under a `.ts`-only glob (N4).
const SOURCE_GLOB = "**/*.{ts,tsx,mts,cts,js,jsx,mjs,cjs}";

type Violation = { file: string; line: number; import: string; reason: string };

/** The package-name prefix of a bare specifier, dropping any deeper subpath — `@scope/name` for a
 * scoped package, the first path segment for an unscoped one (B3: a workspace member's name is
 * not always `@dylanebert/`-scoped, e.g. `orbit-camera`). */
function packageNameFromSpecifier(spec: string): string {
    if (spec.startsWith("@")) {
        const parts = spec.split("/");
        return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : spec;
    }
    return spec.split("/")[0];
}

/** True if `pkgJson` declares `dep` under any dependency field. */
function declaresDependency(pkgJson: Record<string, unknown>, dep: string): boolean {
    for (const field of [
        "dependencies",
        "peerDependencies",
        "devDependencies",
        "optionalDependencies",
    ]) {
        const deps = pkgJson[field] as Record<string, unknown> | undefined;
        if (deps && Object.hasOwn(deps, dep)) return true;
    }
    return false;
}

/** Every dependency name (any field) declared in `pkgJson`. */
function declaredDependencies(pkgJson: Record<string, unknown>): string[] {
    const names = new Set<string>();
    for (const field of [
        "dependencies",
        "peerDependencies",
        "devDependencies",
        "optionalDependencies",
    ]) {
        const deps = pkgJson[field] as Record<string, unknown> | undefined;
        if (deps) for (const name of Object.keys(deps)) names.add(name);
    }
    return [...names];
}

/** Maps every workspace member's declared `name` to its directory, derived from this repo's own
 * root `package.json` `workspaces` globs (never a hardcoded `packages/*`, and never the scanned
 * `root` — the workspace graph is a property of the repo, not of a fixture tree). Covers
 * `examples/*` members (unscoped names like `orbit-camera`) exactly as it covers `packages/*`
 * (B3) — the workspace member set is whatever the package manager itself would resolve. */
async function packageDirs(): Promise<Map<string, string>> {
    const map = new Map<string, string>();
    const rootPkg = (await Bun.file(resolve(repoRoot, "package.json")).json()) as {
        workspaces?: unknown;
    };
    const patterns = Array.isArray(rootPkg.workspaces)
        ? (rootPkg.workspaces as string[])
        : ["packages/*"];
    for (const pattern of patterns) {
        const glob = new Glob(`${pattern}/package.json`);
        for await (const rel of glob.scan({ cwd: repoRoot })) {
            const full = resolve(repoRoot, rel);
            const pkg = (await Bun.file(full).json()) as { name?: unknown };
            if (typeof pkg.name === "string") map.set(pkg.name, dirname(full));
        }
    }
    return map;
}

/**
 * Builds a memoized `name -> path-to-engine-or-null` resolver over the workspace dependency
 * graph. `name` itself reaching the engine returns `PKG`; a sibling reaching it through its own
 * declared dependencies returns `"<name> → ...  → @dylanebert/shallot"`. A name with no
 * workspace `package.json` (an external npm package) is a dead end for this check — it covers the
 * workspace graph, not npm's.
 */
function makeReachesEngine(dirs: Map<string, string>): (name: string) => Promise<string | null> {
    const cache = new Map<string, Promise<string | null>>();
    async function reaches(name: string, visited: Set<string>): Promise<string | null> {
        if (name === PKG) return PKG;
        if (visited.has(name)) return null;
        visited.add(name);
        const dir = dirs.get(name);
        if (!dir) return null;
        const pkgJson = (await Bun.file(resolve(dir, "package.json")).json()) as Record<
            string,
            unknown
        >;
        if (declaresDependency(pkgJson, PKG)) return `${name} → ${PKG}`;
        for (const dep of declaredDependencies(pkgJson)) {
            const via = await reaches(dep, visited);
            if (via) return `${name} → ${via}`;
        }
        return null;
    }
    return (name: string) => {
        let cached = cache.get(name);
        if (!cached) {
            cached = reaches(name, new Set());
            cache.set(name, cached);
        }
        return cached;
    };
}

/** Classifies one specifier found in `full` — relative escape, direct engine import, or (via
 * `reaches`) a workspace member that transitively depends on the engine. Every bare specifier is
 * checked against the workspace graph (B3), not only `@dylanebert/`-scoped ones — `reaches` is a
 * no-op for any name outside `packageDirs()`'s workspace map (an external npm package, a Node
 * builtin, or an arbitrary string), so this never over-triggers. `null` when the specifier is none
 * of those. */
async function classify(
    spec: string,
    full: string,
    root: string,
    reaches: (name: string) => Promise<string | null>,
): Promise<string | null> {
    if (spec.startsWith(".")) {
        const resolved = resolve(dirname(full), spec);
        if (resolved === root || resolved.startsWith(root + sep)) return null;
        return `escapes packages/shallot-tui → ${relative(repoRoot, resolved)}`;
    }
    if (spec === PKG || spec.startsWith(`${PKG}/`)) {
        return "imports the engine (@dylanebert/shallot) — shallot-tui must be engine-agnostic";
    }
    const via = await reaches(packageNameFromSpecifier(spec));
    if (via) {
        return `imports workspace sibling "${spec}", which reaches the engine via ${via} — the boundary is transitive`;
    }
    return null;
}

async function scanSpecifiers(
    full: string,
    root: string,
    reaches: (name: string) => Promise<string | null>,
): Promise<Violation[]> {
    const violations: Violation[] = [];
    const lines = (await Bun.file(full).text()).split("\n");
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const seen = new Set<string>();
        for (const re of [STATIC_IMPORT_RE, DYNAMIC_IMPORT_RE, CALL_ARG_RE]) {
            for (const match of line.matchAll(re)) {
                const spec = match[1];
                const key = `${spec}`;
                if (seen.has(key)) continue;
                seen.add(key);
                const reason = await classify(spec, full, root, reaches);
                if (reason) {
                    violations.push({
                        file: relative(repoRoot, full),
                        line: i + 1,
                        import: spec,
                        reason,
                    });
                }
            }
        }
    }
    return violations;
}

async function scanPackageJson(
    root: string,
    reaches: (name: string) => Promise<string | null>,
): Promise<Violation[]> {
    const path = resolve(root, "package.json");
    const file = Bun.file(path);
    if (!(await file.exists())) return [];
    const pkgJson = (await file.json()) as Record<string, unknown>;
    const violations: Violation[] = [];
    for (const name of declaredDependencies(pkgJson)) {
        if (name === PKG) {
            violations.push({
                file: relative(repoRoot, path),
                line: 0,
                import: name,
                reason: "declares a package.json dependency on the engine (@dylanebert/shallot) — shallot-tui must be engine-agnostic",
            });
            continue;
        }
        const via = await reaches(name);
        if (via) {
            violations.push({
                file: relative(repoRoot, path),
                line: 0,
                import: name,
                reason: `declares a package.json dependency on "${name}", which reaches the engine via ${via} — the boundary is transitive`,
            });
        }
    }
    return violations;
}

async function scan(root: string): Promise<Violation[]> {
    const dirs = await packageDirs();
    const reaches = makeReachesEngine(dirs);
    const violations: Violation[] = [];
    // The whole package tree, not a hardcoded ["src", "tests"] subdirectory list (N4) — a future
    // sibling directory (`bin/`, say) is covered the moment it exists, with no gate edit owed.
    // `node_modules` is excluded explicitly since a fixture `--root` tree may carry one.
    const glob = new Glob(SOURCE_GLOB);
    for await (const path of glob.scan({ cwd: root })) {
        if (path.split(sep).includes("node_modules")) continue;
        violations.push(...(await scanSpecifiers(resolve(root, path), root, reaches)));
    }
    violations.push(...(await scanPackageJson(root, reaches)));
    return violations;
}

const rootArg = process.argv.indexOf("--root");
const root =
    rootArg >= 0 ? resolve(process.argv[rootArg + 1]) : resolve(repoRoot, "packages/shallot-tui");

const violations = await scan(root);

if (violations.length > 0) {
    console.error(`✗ ${violations.length} shallot-tui engine-boundary violation(s):\n`);
    for (const v of violations) {
        console.error(`  ${v.file}${v.line > 0 ? `:${v.line}` : ""}`);
        console.error(`    import "${v.import}" ${v.reason}`);
    }
    console.error(
        "\npackages/shallot-tui must import nothing from the engine — no relative escape into\n" +
            "packages/shallot/, no @dylanebert/shallot import at any subpath (published or not, static\n" +
            "or dynamic, or reached through require/createRequire), no workspace sibling that itself\n" +
            "depends on the engine, and no package.json dependency on the engine or on such a sibling.",
    );
    process.exit(1);
}

console.log("✓ shallot-tui engine boundary clean");
