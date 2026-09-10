import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Glob } from "bun";

// The script surface is asserted two ways rather than left to accrete silently:
//
//   1. every declared script (root + every workspace package.json) resolves to an existing
//      file/directory, or delegates (`bun run --cwd <dir> <name>` / `bun run <name>`) to a
//      script actually declared at that target;
//   2. every `scripts/*` file is reached by a package.json script, another file's import, or a
//      workflow, or is a `.test.ts`/`.probes.ts` sibling of a reached file. Prose citations do
//      not count. A file nothing reaches is an orphan and deletes outright.
//
// `--root <dir>` points the check at an alternate tree (fixture-driven proof; check-boundary.ts
// carries the same flag for the same reason).

export type Violation = { file: string; script: string; detail: string };

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

const SIBLING = /\.(?:test|probes)\.ts$/;
const IMPORT_ROOTS = ["scripts", "src", "bin", "tests", "examples", "evals", "packages"];

// Every import/require specifier in the source trees, with the importing file's repo-relative path.
async function importSpecifiers(rootDir: string): Promise<{ from: string; spec: string }[]> {
    const specs: { from: string; spec: string }[] = [];
    const glob = new Glob("**/*.{ts,mjs,js,svelte}");
    for (const dir of IMPORT_ROOTS) {
        const base = resolve(rootDir, dir);
        if (!existsSync(base)) continue;
        for await (const match of glob.scan({ cwd: base })) {
            if (match.includes("node_modules/") || match.startsWith("dist/")) continue;
            const text = await Bun.file(resolve(base, match)).text();
            for (const m of text.matchAll(
                /(?:\bfrom\s+|\bimport\s*\(\s*|\brequire\s*\(\s*|\bimport\s+)["'`]([^"'`]+)["'`]/g,
            )) {
                specs.push({ from: `${dir}/${match}`, spec: m[1] });
            }
        }
    }
    return specs;
}

// A `bun test <paths...>` command reaches every test file under a listed directory.
function testDirs(commands: string[]): string[] {
    const dirs: string[] = [];
    for (const cmd of commands) {
        for (const segment of cmd.split("&&")) {
            const m = /\bbun\s+test\s+(.+)$/.exec(segment.trim());
            if (!m) continue;
            for (const token of m[1].split(/\s+/)) {
                if (token && !token.startsWith("-")) dirs.push(token.replace(/^\.\//, ""));
            }
        }
    }
    return dirs;
}

// Direction 2 — every `scripts/*` file is reached by a script, an import, a workflow, or a reached sibling.
export async function checkReachable(rootDir: string, commands: string[]): Promise<Violation[]> {
    const scriptsDir = resolve(rootDir, "scripts");
    if (!existsSync(scriptsDir)) return [];
    const files: string[] = [];
    for await (const match of new Glob("*.{ts,mjs}").scan({ cwd: scriptsDir })) files.push(match);

    const reached = new Set<string>();
    for (const cmd of commands) {
        for (const m of cmd.matchAll(/scripts\/([\w.-]+\.(?:ts|mjs))/g)) reached.add(m[1]);
    }
    const workflowsDir = resolve(rootDir, ".github/workflows");
    if (existsSync(workflowsDir)) {
        for await (const match of new Glob("*.{yml,yaml}").scan({ cwd: workflowsDir })) {
            const text = await Bun.file(resolve(workflowsDir, match)).text();
            for (const m of text.matchAll(/scripts\/([\w.-]+\.(?:ts|mjs))/g)) reached.add(m[1]);
        }
    }
    const specs = await importSpecifiers(rootDir);
    for (const file of files) {
        const stem = file.replace(/\.(ts|mjs)$/, "");
        const pattern = new RegExp(`/${escapeRegExp(stem)}(?:\\.(?:ts|mjs|js))?$`);
        const self = `scripts/${file}`;
        if (specs.some(({ from, spec }) => from !== self && pattern.test(spec))) reached.add(file);
    }
    const tested = testDirs(commands);

    const violations: Violation[] = [];
    for (const file of files) {
        if (reached.has(file)) continue;
        if (SIBLING.test(file)) {
            const base = file.replace(SIBLING, "");
            if (reached.has(`${base}.ts`) || reached.has(`${base}.mjs`)) continue;
            if (
                file.endsWith(".test.ts") &&
                tested.some((dir) => `scripts/${file}`.startsWith(`${dir.replace(/\/$/, "")}/`))
            )
                continue;
        }
        violations.push({
            file: `scripts/${file}`,
            script: file,
            detail: "not reached by a package.json script, an import, a workflow, or a reached sibling",
        });
    }
    return violations;
}

export async function run(rootDir: string): Promise<{
    pkgPaths: string[];
    rootScripts: Record<string, string>;
    existsViolations: Violation[];
    reachViolations: Violation[];
}> {
    const rootPkgPath = resolve(rootDir, "package.json");
    const rootPkg = (await Bun.file(rootPkgPath).json()) as {
        scripts?: Record<string, string>;
        workspaces?: string[];
    };
    const rootScripts = rootPkg.scripts ?? {};
    const pkgPaths = [rootPkgPath, ...(await workspacePkgPaths(rootDir, rootPkg.workspaces ?? []))];
    const commands: string[] = [];
    for (const pkgPath of pkgPaths) commands.push(...Object.values(await readScripts(pkgPath)));

    const existsViolations = await checkExists(pkgPaths);
    const reachViolations = await checkReachable(rootDir, commands);
    return { pkgPaths, rootScripts, existsViolations, reachViolations };
}

if (import.meta.main) {
    const rootArgIdx = process.argv.indexOf("--root");
    const rootDir = resolve(
        rootArgIdx >= 0 ? process.argv[rootArgIdx + 1] : resolve(import.meta.dir, ".."),
    );

    const { pkgPaths, rootScripts, existsViolations, reachViolations } = await run(rootDir);

    if (existsViolations.length + reachViolations.length > 0) {
        if (existsViolations.length > 0) {
            console.error(
                `✗ ${existsViolations.length} script(s) resolving to a missing target:\n`,
            );
            for (const v of existsViolations) {
                console.error(`  ${v.file} → "${v.script}": ${v.detail}`);
            }
        }
        if (reachViolations.length > 0) {
            console.error(`\n✗ ${reachViolations.length} scripts/* file(s) unreachable:\n`);
            for (const v of reachViolations) console.error(`  ${v.file}: ${v.detail}`);
        }
        console.error("\nFix the target, wire the script, or delete it.");
        process.exit(1);
    }

    console.log(
        `✓ script surface clean (${pkgPaths.length} package.json, ${Object.keys(rootScripts).length} root scripts)`,
    );
}
