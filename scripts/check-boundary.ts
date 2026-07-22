import { Glob } from "bun";
import { dirname, join, relative, resolve, sep } from "path";

// Distribution boundary: a consumer of the published @dylanebert/shallot surface
// reaches the engine only through the package name, and only through its declared
// `exports` — never repo-only directories (harness/, scripts/, tests/, sibling
// examples) and never an unpublished internal. Two ways a consumer could break out:
//
//   1. a relative import that climbs out of its own project root, and
//   2. a package import that reaches a subpath outside the published `exports`
//      (the `./src/*` wildcard is the escape hatch — it resolves any internal file
//      by package name, so a consumer using it is reaching an unexposed internal).
//
// Both are violations here. Cross-project access goes through the package name, into
// a published subpath only. (coding.md onion layers; AGENTS.md "Don't deep-import
// from src/".) Every example tier — recipes, gym, showcase — and external consumers
// are held to this; they exist to use the engine as a user would.
//
// The one narrow allowance: a relative import that escapes the project is permitted
// iff it resolves inside `packages/shallot/tests/` — the CPU-oracle cross-check seam
// (the f64 avbd solver/joint + the bvh fixtures/oracle a gym scenario diffs the GPU
// against). Those f64 references are the executable spec, load-bearing and unpublished
// by design; killing the share would force duplicating them. Engine *access* still must
// use the published exports — the allowance is the tests/ oracle only, not `src/`.
//
// Default scans this repo's consumer examples. `--root <dir>` scans an external
// consumer tree (e.g. orrstead), where every project is a consumer.

const repoRoot = resolve(import.meta.dir, "..");

// the one relative-escape allowance: the f64 CPU-oracle cross-check seam.
const ORACLE_SEAM = resolve(repoRoot, "packages/shallot/tests");

const PKG = "@dylanebert/shallot";

// The published surface, read from the engine's own `exports` map (one source of truth). An export
// key maps to its package specifier (`.` → `@dylanebert/shallot`, `./extras` → `@dylanebert/shallot/
// extras`). The `./src/*` wildcard is deliberately dropped: it exists so repo tooling can deep-import
// by path, but a consumer reaching it is the "unexposed internal" this guards against. Any other
// wildcard export stays a legitimate prefix.
async function publishedSurface(): Promise<{ exact: Set<string>; prefixes: string[] }> {
    const pkg = await Bun.file(resolve(repoRoot, "packages/shallot/package.json")).json();
    const exact = new Set<string>();
    const prefixes: string[] = [];
    for (const key of Object.keys(pkg.exports as Record<string, unknown>)) {
        if (key === "./src/*") continue;
        const spec = key === "." ? PKG : `${PKG}/${key.slice(2)}`;
        if (spec.endsWith("/*")) prefixes.push(spec.slice(0, -1));
        else exact.add(spec);
    }
    return { exact, prefixes };
}

const surface = await publishedSurface();

// True if a bare `@dylanebert/shallot...` import lands on a published subpath.
function isPublished(spec: string): boolean {
    return surface.exact.has(spec) || surface.prefixes.some((p) => spec.startsWith(p));
}

// Consumer project roots = every example workspace (gym included — it's a consumer of the published
// surface now, held to the boundary like the rest, save the one tests/-oracle allowance in scan()).
async function consumerRoots(): Promise<string[]> {
    const pkg = await Bun.file(resolve(repoRoot, "package.json")).json();
    const roots: string[] = [];
    for (const pattern of pkg.workspaces as string[]) {
        if (!pattern.startsWith("examples/")) continue;
        if (pattern.includes("*")) {
            const glob = new Glob(pattern.slice("examples/".length));
            for await (const match of glob.scan({
                cwd: resolve(repoRoot, "examples"),
                onlyFiles: false,
            })) {
                roots.push(resolve(repoRoot, `examples/${match}`));
            }
        } else {
            roots.push(resolve(repoRoot, pattern));
        }
    }
    return roots;
}

// Every package.json directory under `dir` (skipping node_modules) is a project.
async function projectRoots(dir: string): Promise<string[]> {
    const roots: string[] = [];
    const glob = new Glob("**/package.json");
    for await (const match of glob.scan({ cwd: dir })) {
        if (match.includes("node_modules")) continue;
        roots.push(resolve(dir, dirname(match)));
    }
    return roots;
}

// The deepest project root containing `file`, or null if none.
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

const importRe = /(?:from|import)\s+["']([^"']+)["']/g;

type Violation = { file: string; line: number; import: string; reason: string };

async function scan(roots: string[]): Promise<Violation[]> {
    const violations: Violation[] = [];
    for (const root of roots) {
        const glob = new Glob("**/*.{ts,svelte}");
        for await (const path of glob.scan({ cwd: root })) {
            if (path.includes("node_modules")) continue;
            const full = join(root, path);
            const owner = ownerOf(full, roots);
            // Process each file once, under its deepest owning project (roots can nest).
            if (owner !== root) continue;
            const lines = (await Bun.file(full).text()).split("\n");
            for (let i = 0; i < lines.length; i++) {
                for (const match of lines[i].matchAll(importRe)) {
                    const spec = match[1];
                    const at = { file: relative(repoRoot, full), line: i + 1, import: spec };
                    if (spec.startsWith(".")) {
                        const resolved = resolve(dirname(full), spec);
                        if (resolved === owner || resolved.startsWith(owner + sep)) continue;
                        // the one allowance: the f64 CPU-oracle cross-check seam under packages/shallot/tests/
                        if (resolved === ORACLE_SEAM || resolved.startsWith(ORACLE_SEAM + sep))
                            continue;
                        violations.push({
                            ...at,
                            reason: `escapes the project → ${relative(repoRoot, resolved)}`,
                        });
                    } else if ((spec === PKG || spec.startsWith(PKG + "/")) && !isPublished(spec)) {
                        violations.push({
                            ...at,
                            reason: "reaches an unpublished @dylanebert/shallot internal (not in exports)",
                        });
                    }
                }
            }
        }
    }
    return violations;
}

const rootArg = process.argv.indexOf("--root");
const roots =
    rootArg >= 0 ? await projectRoots(resolve(process.argv[rootArg + 1])) : await consumerRoots();

const violations = await scan(roots);

if (violations.length > 0) {
    console.error(`✗ ${violations.length} distribution-boundary violation(s):\n`);
    for (const v of violations) {
        console.error(`  ${v.file}:${v.line}`);
        console.error(`    import "${v.import}" ${v.reason}`);
    }
    console.error(
        "\nA consumer must reach the engine through the published @dylanebert/shallot\n" +
            "surface — a declared `exports` subpath — never a relative path into repo-only\n" +
            "code (harness/, scripts/, tests/) nor an unpublished internal (src/...).",
    );
    process.exit(1);
}

console.log(`✓ distribution boundary clean (${roots.length} consumer project(s))`);
