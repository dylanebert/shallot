// Prints the *.node.ts files a change reaches, against a base revision (default origin/main).
// `bun test --changed` does not follow imports through the package's own name
// (oven-sh/bun#44162), and an allocation gate reaches engine code only through the entries it
// bundles and spawns, so this script selects both instead.
import { existsSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import {
    ALLOCATION_BUNDLE_ASSETS,
    allocationBuildConfig,
} from "../diagnostics/first-person-allocation/allocation";
import { normalizePath } from "../src/project/paths";

const ROOT = resolve(import.meta.dir, "..");
const GATE = /sampleAllocation|prepareAllocationSampler/;
const ENTRY = /["'`](?:\$\{import\.meta\.dir\}\/)?([^"'`\s$]*\.entry\.ts)["'`]/g;
const SCRIPT = /\.[cm]?[jt]sx?$/;
const transpiler = new Bun.Transpiler({ loader: "tsx" });

/** A git listing's paths, NUL-delimited so none is quoted. */
function git(args: readonly string[], root = ROOT): string[] {
    const [command, ...rest] = args;
    const run = Bun.spawnSync(["git", command, "-z", ...rest], { cwd: root, stderr: "pipe" });
    if (run.exitCode !== 0) throw new Error(`git ${args.join(" ")}: ${run.stderr.toString()}`);
    return run.stdout.toString().split("\0").filter(Boolean);
}

function files(dirs: readonly string[], root = ROOT): string[] {
    return git(["ls-files", "--cached", "--others", "--exclude-standard", ...dirs], root).filter(
        (path) => path.endsWith(".node.ts") && existsSync(resolve(root, path)),
    );
}

/** Every `*.node.ts` under src, examples, scripts and diagnostics, relative to the root. */
export function nodeTests(root = ROOT): string[] {
    return files(["src", "examples", "scripts", "diagnostics"], root).sort();
}

/** Paths that differ from `base` or are untracked, relative to the root. */
export function changedPaths(base: string, root = ROOT): string[] {
    return [
        ...git(["diff", "--name-only", base], root),
        ...git(["ls-files", "--others", "--exclude-standard"], root),
    ];
}

/** Every `*.node.ts` under src, examples and diagnostics that samples allocation, relative to the root. */
export function allocationGates(): string[] {
    return files(["src", "examples", "diagnostics"])
        .filter((path) => GATE.test(readFileSync(resolve(ROOT, path), "utf8")))
        .sort();
}

/**
 * Each test's absolute static import closure inside the checkout, itself included: static and
 * string-literal dynamic imports, resolved as the runtime resolves them, package self-references
 * included. Installed packages end the walk. An import that does not resolve throws, naming its file.
 */
export function importClosures(tests: readonly string[], root = ROOT): Map<string, Set<string>> {
    const imports = new Map<string, string[]>();
    const edges = (file: string): string[] => {
        let found = imports.get(file);
        if (found) return found;
        found = [];
        imports.set(file, found);
        if (!SCRIPT.test(file)) return found;
        for (const { path: specifier } of transpiler.scanImports(readFileSync(file, "utf8"))) {
            let target: string;
            try {
                target = Bun.resolveSync(specifier, dirname(file));
            } catch (error) {
                throw new Error(
                    `${normalizePath(relative(root, file))}: cannot resolve "${specifier}"`,
                    {
                        cause: error,
                    },
                );
            }
            const path = normalizePath(relative(root, target));
            if (
                isAbsolute(target) &&
                !path.startsWith("../") &&
                !isAbsolute(path) &&
                !path.split("/").includes("node_modules")
            ) {
                found.push(target);
            }
        }
        return found;
    };
    return new Map(
        tests.map((test) => {
            const closure = new Set<string>();
            const pending = [resolve(root, test)];
            for (let file = pending.pop(); file !== undefined; file = pending.pop()) {
                if (closure.has(file)) continue;
                closure.add(file);
                pending.push(...edges(file));
            }
            return [test, closure] as const;
        }),
    );
}

/** The `.entry.ts` files a gate names, resolved beside it; a gate naming none that exists throws. */
export async function gateEntries(gate: string): Promise<string[]> {
    const dir = dirname(resolve(ROOT, gate));
    const named = [...readFileSync(resolve(ROOT, gate), "utf8").matchAll(ENTRY)].map((match) =>
        resolve(dir, match[1]),
    );
    const entries = [];
    for (const entry of new Set(named)) if (await Bun.file(entry).exists()) entries.push(entry);
    if (entries.length === 0) throw new Error(`${gate}: names no .entry.ts file that exists`);
    return entries;
}

async function inputs(
    gate: string,
    config: Bun.BuildConfig | Promise<Bun.BuildConfig>,
): Promise<string[]> {
    let built: Bun.BuildOutput;
    try {
        built = await Bun.build({ ...(await config), metafile: true });
    } catch (error) {
        const logs = error instanceof AggregateError ? error.errors : [error];
        throw new Error(`${gate}: bundle failed: ${logs.map(String).join("\n")}`, { cause: error });
    }
    return Object.keys(built.metafile!.inputs).map((input) => resolve(ROOT, input));
}

/**
 * Each gate's absolute input paths: its own static imports, those of the entries it samples, and the
 * assets the harness copies beside every bundle.
 */
export async function gateInputs(gates: readonly string[]): Promise<Map<string, Set<string>>> {
    const resolved = await Promise.all(
        gates.map(async (gate) => {
            const own = inputs(gate, {
                entrypoints: [resolve(ROOT, gate)],
                target: "bun",
                external: ["bun:*"],
            });
            const entries = (await gateEntries(gate)).map((entry) =>
                inputs(gate, allocationBuildConfig(entry)),
            );
            const files = (await Promise.all([own, ...entries])).flat();
            return [gate, new Set([...files, ...ALLOCATION_BUNDLE_ASSETS])] as const;
        }),
    );
    return new Map(resolved);
}

/** The tests whose inputs include a changed path, given relative to the root. */
export function selectTests(
    inputs: Map<string, Set<string>>,
    changed: readonly string[],
    root = ROOT,
): string[] {
    const paths = changed.map((path) => resolve(root, path));
    return [...inputs]
        .filter(([, files]) => paths.some((path) => files.has(path)))
        .map(([test]) => test);
}

/**
 * The node tests a change reaches through their import closures, and the allocation gates whose
 * bundles it reaches, each once and sorted. Gates are bundled only when a path under src, examples
 * or diagnostics changed.
 */
export async function changedTests(changed: readonly string[]): Promise<string[]> {
    const selected = new Set(selectTests(importClosures(nodeTests()), changed));
    if (changed.some((path) => /^(src|examples|diagnostics)\//.test(path))) {
        for (const gate of selectTests(await gateInputs(allocationGates()), changed)) {
            selected.add(gate);
        }
    }
    return [...selected].sort();
}

if (import.meta.main) {
    for (const test of await changedTests(changedPaths(process.argv[2] ?? "origin/main")))
        console.log(`./${test}`);
}
