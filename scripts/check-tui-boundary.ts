import { existsSync } from "node:fs";
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
//      or any loader idiom whose sole argument is the engine's specifier string — rather than only
//      `import`/`from`;
//   4. a workspace sibling that itself depends on the engine (`@dylanebert/shallot-ocean` does, via
//      its own `package.json`). Importing the sibling is a transitive route to the engine even
//      though the sibling's own package name isn't a subpath of `@dylanebert/shallot`, so this
//      walks the workspace dependency graph rather than pattern-matching the specifier string;
//   5. a `dependencies`/`peerDependencies`/`devDependencies`/`optionalDependencies` entry in
//      shallot-tui's own `package.json` naming the engine, or a workspace sibling that itself
//      reaches the engine — unenforced by a source scan alone, and for a published package this
//      declared-dependency edge is the boundary a consumer actually experiences (an installed
//      dependency ships and resolves regardless of whether any source file imports it).
//
// `tests/` is scanned alongside `src/` — the package's testability claim ("no GPU, no
// @dylanebert/shallot import ... testable on any seat with no adapter") covers the whole package,
// not just its shipped surface.
//
// `--root <dir>` points the source scan at an alternate tree (fixture-driven proof; check-imports.ts
// and check-boundary.ts carry the same flag for the same reason). The workspace dependency graph
// (route 4) is always read from this repo's own `packages/*/package.json`, since that graph is a
// property of this repo's workspace, not of whatever fixture tree is being scanned.

const repoRoot = resolve(import.meta.dir, "..");
const PKG = "@dylanebert/shallot";

// Static `import ... from "spec"` / `import "spec"` / `export ... from "spec"`.
const STATIC_IMPORT_RE = /(?:from|import)\s+["']([^"']+)["']/g;
// Dynamic `import("spec")` / `await import("spec")` — no `from`, and `import` is followed by `(`
// rather than whitespace-then-quote, so `STATIC_IMPORT_RE` never matches this shape.
const DYNAMIC_IMPORT_RE = /\bimport\s*\(\s*["']([^"']+)["']/g;
// Any call whose sole argument is exactly the engine's specifier (or a subpath) — covers
// `require("@dylanebert/shallot")` and `createRequire(...)("@dylanebert/shallot")` alike, since
// both are just "a call, one string argument, the engine's name" regardless of what the callee is
// named.
const CALL_ARG_ENGINE_RE = /\(\s*["'](@dylanebert\/shallot(?:\/[^"']*)?)["']\s*\)/g;

type Violation = { file: string; line: number; import: string; reason: string };

/** The scoped `@scope/name` prefix of a specifier, dropping any deeper subpath. */
function scopedPackageName(spec: string): string {
    const parts = spec.split("/");
    return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : spec;
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

/** Maps a workspace package's declared `name` to its directory, read from every
 * `packages/*\/package.json` in this repo (never the scanned `root` — the workspace graph is a
 * property of the repo, not of a fixture tree). */
async function packageDirs(): Promise<Map<string, string>> {
    const map = new Map<string, string>();
    const glob = new Glob("packages/*/package.json");
    for await (const rel of glob.scan({ cwd: repoRoot })) {
        const full = resolve(repoRoot, rel);
        const pkg = (await Bun.file(full).json()) as { name?: unknown };
        if (typeof pkg.name === "string") map.set(pkg.name, dirname(full));
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
 * `reaches`) a workspace sibling that transitively depends on the engine. `null` when the
 * specifier is none of those. */
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
    if (spec.startsWith("@dylanebert/")) {
        const via = await reaches(scopedPackageName(spec));
        if (via) {
            return `imports workspace sibling "${spec}", which reaches the engine via ${via} — the boundary is transitive`;
        }
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
        for (const re of [STATIC_IMPORT_RE, DYNAMIC_IMPORT_RE, CALL_ARG_ENGINE_RE]) {
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
    for (const sub of ["src", "tests"]) {
        const dir = resolve(root, sub);
        if (!existsSync(dir)) continue;
        const glob = new Glob("**/*.ts");
        for await (const path of glob.scan({ cwd: dir })) {
            violations.push(...(await scanSpecifiers(resolve(dir, path), root, reaches)));
        }
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
