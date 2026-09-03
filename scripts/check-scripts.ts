import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Glob } from "bun";

// A script name is sanctioned routine (suite-speed — a
// sanctioned script is one the suite can run fast), so `package.json`'s script surface is asserted three ways rather than left to accrete
// silently — the class of drift that shipped `test:full` and then had to delete it.
//
//   1. every declared script (root + every workspace package.json) resolves to an existing
//      file/directory, or delegates (`bun run --cwd <dir> <name>` / `bun run <name>`) to a
//      script actually declared at that target;
//   2. every root script name is cited as `bun <name>` / `bun run <name>` in AGENTS.md or
//      .claude/rules/*.md — an undocumented script is the next accretion. Root-scoped only:
//      workspace scripts (gym's vite trio, the showcase test/gate pairs, packages/shallot's
//      eleven) don't ride the entry doc's Commands block, so they take direction 1 alone;
//   3. every root `scripts/*` entry file is reachable — a script target itself, or cited by
//      path (an import specifier, a `resolve()` string, a doc reference) anywhere outside
//      itself. A file nothing reaches is an orphan and deletes outright.
//
// `--root <dir>` points the check at an alternate tree (fixture-driven proof; check-boundary.ts
// carries the same flag for the same reason).

export type Violation = { file: string; script: string; detail: string };

// Entry points invoked by path (`bun run evals/setup.ts red-box`), not declared as a package.json
// script — direction 1/3's `scripts/*` scope doesn't reach `evals/`, and without coverage they
// ride no gate at all. Direction 2's citation coverage is the property that matters (a reader has
// to find the command somewhere); extend it to these by name, same corpus.
//
// The entry-point set is DERIVED from the tree, not hand-listed: every `.ts` file directly under
// `evals/` (not in a subdir like `harness/` or `tasks/`) that isn't a `.test.ts` file is a
// path-invoked entry point. Asking git makes the scope identical in every checkout (same law as
// check-docs' doc scan). Mutation proof: adding a tracked `evals/<new>.ts` reds this arm (not
// cited in AGENTS.md or .claude/rules/*.md), witnessed 2026-08-25, exit 1; removing it greens.
//
//
// Empty-population guard: a tree with no `evals/` dir (a fixture root) yields no entry points,
// which is valid — the gate's other arms still run. A tree with `evals/` where `git ls-files`
// fails is an error (the derivation needs a git checkout to scope its entry-point set). A tree
// with `evals/` where the filter yields no direct entry point (all matches in subdirs or
// .test.ts) is also an error — the citation arm would be vacuously green, which is the exact
// fail-open the derivation was introduced to remove. Same law as check-docs' sibling arms:// each derivation carries its empty-population guard in the same diff.
async function derivePathEntryPoints(rootDir: string): Promise<string[]> {
    const evalsDir = resolve(rootDir, "evals");
    if (!existsSync(evalsDir)) return [];
    const tracked = Bun.spawnSync(["git", "ls-files", "-z", "evals/*.ts"], { cwd: rootDir });
    if (!tracked.success) {
        console.error(
            "✗ `git ls-files` failed — check-scripts needs a git checkout to derive evals entry points.",
        );
        process.exit(1);
    }
    const entryPoints = tracked.stdout
        .toString()
        .split("\0")
        .filter(Boolean)
        .filter((f) => {
            const rel = f.slice("evals/".length);
            return !rel.includes("/") && !rel.endsWith(".test.ts");
        });
    if (entryPoints.length === 0) {
        console.error(
            "✗ `evals/` exists but `git ls-files` yielded no direct entry point — the evals citation arm would be vacuously green.",
        );
        process.exit(1);
    }
    return entryPoints;
}

export function escapeRegExp(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export async function readScripts(pkgPath: string): Promise<Record<string, string>> {
    const pkg = (await Bun.file(pkgPath).json()) as { scripts?: Record<string, string> };
    return pkg.scripts ?? {};
}

// Expand the root `workspaces` globs (bun's own resolution: a literal dir, or `<base>/*`) into
// every member's package.json path.
export async function workspacePkgPaths(rootDir: string, patterns: string[]): Promise<string[]> {
    const paths: string[] = [];
    for (const pattern of patterns) {
        const starIdx = pattern.indexOf("*");
        if (starIdx === -1) {
            paths.push(resolve(rootDir, pattern, "package.json"));
            continue;
        }
        const base = pattern.slice(0, starIdx).replace(/\/$/, "");
        const rest = pattern.slice(base.length + 1);
        const glob = new Glob(rest);
        for await (const match of glob.scan({ cwd: resolve(rootDir, base), onlyFiles: false })) {
            paths.push(resolve(rootDir, base, match, "package.json"));
        }
    }
    return paths.filter((p) => existsSync(p));
}

// Direction 1 — every declared script resolves to an existing file/dir, or a real delegate.
export async function checkExists(pkgPaths: string[]): Promise<Violation[]> {
    const violations: Violation[] = [];
    for (const pkgPath of pkgPaths) {
        const dir = dirname(pkgPath);
        const scripts = await readScripts(pkgPath);
        for (const [name, cmd] of Object.entries(scripts)) {
            for (const segment of cmd.split("&&")) {
                // `bunx <pkg>` resolves an npm-published binary (like `npx`) — never a repo-local
                // file, never a delegate to a declared script. It's an external command, exactly
                // like a bare `tsc` / `playwright test` segment below: unchecked here on purpose.
                const isBunx = /\bbunx\s+/.test(segment);
                if (isBunx) continue;
                const m = segment.match(/\bbun\s+(?:(run|test)\s+)?(?:--cwd\s+(\S+)\s+)?(\S+)/);
                if (!m) continue;
                const [, verb, cwdArg, token] = m;
                if (token.startsWith("-")) continue;
                const base = cwdArg ? resolve(dir, cwdArg) : dir;
                if (token.includes("/") || token.includes(".")) {
                    const target = resolve(base, token);
                    if (!existsSync(target)) {
                        violations.push({
                            file: pkgPath,
                            script: name,
                            detail: `target "${token}" does not exist (resolved ${target})`,
                        });
                    }
                } else if (verb !== "test") {
                    // a bare token after `run` (or no verb) is a delegate: `bun run --cwd X
                    // name` or `bun run name` must name a script the target package.json
                    // declares. `bun test <name>` bare tokens are filters, not delegates.
                    const targetPkgPath = resolve(base, "package.json");
                    if (!existsSync(targetPkgPath)) continue;
                    const targetScripts = await readScripts(targetPkgPath);
                    if (!(token in targetScripts)) {
                        violations.push({
                            file: pkgPath,
                            script: name,
                            detail: `delegates to script "${token}" not declared in ${targetPkgPath}`,
                        });
                    }
                }
            }
        }
    }
    return violations;
}

// Direction 2 — every root script name is cited in AGENTS.md or .claude/rules/*.md, plus any
// path-invoked entry point named in `pathEntryPoints` (defaults to none — callers that want the
// `evals/*.ts` coverage pass the derived set explicitly, `run()` below does).
export async function checkDocs(
    rootDir: string,
    rootScripts: Record<string, string>,
    pathEntryPoints: string[] = [],
): Promise<Violation[]> {
    const docPaths = [resolve(rootDir, "AGENTS.md")];
    const rulesDir = resolve(rootDir, ".claude/rules");
    if (existsSync(rulesDir)) {
        const glob = new Glob("*.md");
        for await (const match of glob.scan({ cwd: rulesDir })) {
            docPaths.push(resolve(rulesDir, match));
        }
    }
    const texts = await Promise.all(
        docPaths.filter((p) => existsSync(p)).map((p) => Bun.file(p).text()),
    );
    const combined = texts.join("\n");

    const violations: Violation[] = [];
    for (const name of Object.keys(rootScripts)) {
        // `:` namespaces a script name (`test:install`) rather than separating one — `\b` treats
        // it as a word boundary, so a bare `\b`-terminated match on "foo" is satisfied by a
        // citation of "foo:bar" alone. Require the char right after the name to not continue a
        // longer script-name token (word char, `:`, or `-`), so a namespaced sibling's citation
        // can't stand in for its own.
        const cited = new RegExp(`\\bbun\\s+(?:run\\s+)?${escapeRegExp(name)}(?![\\w:-])`).test(
            combined,
        );
        if (!cited) {
            violations.push({
                file: "package.json",
                script: name,
                detail: `not cited as \`bun ${name}\` / \`bun run ${name}\` in AGENTS.md or .claude/rules/*.md`,
            });
        }
    }
    for (const entry of pathEntryPoints) {
        // Same citation shape as a script name, but the token is a path (`evals/setup.ts`), so no
        // word-boundary trailing guard is needed — a `/` and `.` already bound it on both sides.
        const cited = new RegExp(`\\bbun\\s+(?:run\\s+)?${escapeRegExp(entry)}\\b`).test(combined);
        if (!cited) {
            violations.push({
                file: entry,
                script: entry,
                detail: `not cited as \`bun ${entry}\` / \`bun run ${entry}\` in AGENTS.md or .claude/rules/*.md`,
            });
        }
    }
    return violations;
}

// A source-and-doc corpus for direction 3's citation search: the doc surface check-docs.ts
// already asserts is clean, plus every `.ts`/`.mjs` file under the trees a `scripts/*` file's
// citation could realistically live in. Keyed by repo-relative path so a candidate can exclude
// its own text (a file's self-reference to its own name doesn't count as a citation).
// `scripts/test-changed.ts` is the changed-path selector wired into stage-close and release gates.
export async function loadCorpus(rootDir: string): Promise<Map<string, string>> {
    const files = new Map<string, string>();
    const docTargets = [
        "AGENTS.md",
        "README.md",
        "examples/AGENTS.md",
        "packages/shallot/AGENTS.md",
        "packages/shallot/README.md",
        "evals/README.md",
    ];
    for (const rel of docTargets) {
        const full = resolve(rootDir, rel);
        if (existsSync(full)) files.set(rel, await Bun.file(full).text());
    }
    const rulesDir = resolve(rootDir, ".claude/rules");
    if (existsSync(rulesDir)) {
        const glob = new Glob("*.md");
        for await (const match of glob.scan({ cwd: rulesDir })) {
            const rel = `.claude/rules/${match}`;
            files.set(rel, await Bun.file(resolve(rulesDir, match)).text());
        }
    }
    for (const dir of ["scripts", "packages/shallot", "examples", "evals"]) {
        const base = resolve(rootDir, dir);
        if (!existsSync(base)) continue;
        const glob = new Glob("**/*.{ts,mjs}");
        for await (const match of glob.scan({ cwd: base })) {
            const rel = `${dir}/${match}`;
            if (!files.has(rel)) files.set(rel, await Bun.file(resolve(base, match)).text());
        }
    }
    return files;
}

// Direction 3 — every root `scripts/*` entry file is reachable.
export async function checkReachable(
    rootDir: string,
    rootScripts: Record<string, string>,
): Promise<Violation[]> {
    const scriptsDir = resolve(rootDir, "scripts");
    if (!existsSync(scriptsDir)) return [];
    const glob = new Glob("*.{ts,mjs}");
    const files: string[] = [];
    for await (const match of glob.scan({ cwd: scriptsDir })) {
        // A `.test.ts` file's reachability mechanism is `bun test`'s own glob discovery, not a
        // script target / import / doc citation — it's not an "entry file" in this direction's
        // sense (check-pack.ts excludes the same suffix from the tarball for the same reason).
        if (match.endsWith(".test.ts")) continue;
        files.push(match);
    }

    const targeted = new Set<string>();
    for (const cmd of Object.values(rootScripts)) {
        for (const m of cmd.matchAll(/scripts\/([\w.-]+\.(?:ts|mjs))/g)) targeted.add(m[1]);
    }

    const corpus = await loadCorpus(rootDir);
    const violations: Violation[] = [];
    for (const file of files) {
        if (targeted.has(file)) continue;
        const extMatch = file.match(/\.(ts|mjs)$/);
        const ext = extMatch ? extMatch[1] : "ts";
        const stem = file.slice(0, file.length - ext.length - 1);
        // Reachability's property is a *path* citation — a quoted mention of the bare stem
        // (`"orphan-test-file"` in prose) is not one, so require a literal `/` immediately
        // before the stem: `scripts/<stem>` (doc/prose form) or `./<stem>` / `../…/<stem>`
        // (an import specifier). The trailing guard keeps the match from landing mid-filename
        // (`<stem>-extra.ts`, `<stem>.tsx`) without demanding a specific closing character, so
        // both quoted citations and unquoted prose (`scripts/gltf-conformance.ts --write`) count.
        const pattern = new RegExp(`/${escapeRegExp(stem)}(?:\\.${ext})?(?![\\w.-])`);
        const selfKey = `scripts/${file}`;
        let cited = false;
        for (const [key, text] of corpus) {
            if (key === selfKey) continue;
            if (pattern.test(text)) {
                cited = true;
                break;
            }
        }
        if (!cited) {
            violations.push({
                file: selfKey,
                script: file,
                detail: "not a script target, not cited by path anywhere outside itself",
            });
        }
    }
    return violations;
}

export async function run(rootDir: string): Promise<{
    pkgPaths: string[];
    rootScripts: Record<string, string>;
    existsViolations: Violation[];
    docViolations: Violation[];
    reachViolations: Violation[];
}> {
    const rootPkgPath = resolve(rootDir, "package.json");
    const rootPkg = (await Bun.file(rootPkgPath).json()) as {
        scripts?: Record<string, string>;
        workspaces?: string[];
    };
    const rootScripts = rootPkg.scripts ?? {};
    const pkgPaths = [rootPkgPath, ...(await workspacePkgPaths(rootDir, rootPkg.workspaces ?? []))];

    // Only assert citation coverage for entry points this tree actually has — a fixture root
    // with no `evals/` carries nothing to cite, and that's not a violation of its own.
    const pathEntryPoints = await derivePathEntryPoints(rootDir);

    const existsViolations = await checkExists(pkgPaths);
    const docViolations = await checkDocs(rootDir, rootScripts, pathEntryPoints);
    const reachViolations = await checkReachable(rootDir, rootScripts);

    return { pkgPaths, rootScripts, existsViolations, docViolations, reachViolations };
}

if (import.meta.main) {
    const rootArgIdx = process.argv.indexOf("--root");
    const rootDir = resolve(
        rootArgIdx >= 0 ? process.argv[rootArgIdx + 1] : resolve(import.meta.dir, ".."),
    );

    const { pkgPaths, rootScripts, existsViolations, docViolations, reachViolations } =
        await run(rootDir);

    const total = existsViolations.length + docViolations.length + reachViolations.length;

    if (total > 0) {
        if (existsViolations.length > 0) {
            console.error(
                `✗ ${existsViolations.length} script(s) resolving to a missing target:\n`,
            );
            for (const v of existsViolations) {
                console.error(`  ${v.file} → "${v.script}": ${v.detail}`);
            }
        }
        if (docViolations.length > 0) {
            console.error(
                `\n✗ ${docViolations.length} root script(s) not cited in AGENTS.md or .claude/rules/*.md:\n`,
            );
            for (const v of docViolations) console.error(`  "${v.script}": ${v.detail}`);
        }
        if (reachViolations.length > 0) {
            console.error(`\n✗ ${reachViolations.length} scripts/* file(s) unreachable:\n`);
            for (const v of reachViolations) console.error(`  ${v.file}: ${v.detail}`);
        }
        console.error(
            "\nA script name is sanctioned routine (suite-speed — a sanctioned script is one the suite " +
                "can run fast) — fix the target, cite it, or delete the alias.",
        );
        process.exit(1);
    }

    console.log(
        `✓ script surface clean (${pkgPaths.length} package.json, ${Object.keys(rootScripts).length} root scripts)`,
    );
}
