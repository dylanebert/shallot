// Prints the allocation gates whose bundled inputs differ from a base revision (default origin/main).
// `bun test --changed` cannot select them: a gate reaches engine code only through the entries it
// bundles and spawns, and those import the engine by the package's own name.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
    ALLOCATION_BUNDLE_ASSETS,
    allocationBuildConfig,
} from "../diagnostics/first-person-allocation/allocation";

const ROOT = resolve(import.meta.dir, "..");
const GATE = /sampleAllocation|prepareAllocationSampler/;
const ENTRY = /["'`](?:\$\{import\.meta\.dir\}\/)?([^"'`\s$]*\.entry\.ts)["'`]/g;

function git(args: string[]): string[] {
    const run = Bun.spawnSync(["git", ...args], { cwd: ROOT, stderr: "pipe" });
    if (run.exitCode !== 0) throw new Error(`git ${args.join(" ")}: ${run.stderr.toString()}`);
    return run.stdout.toString().split("\n").filter(Boolean);
}

/** Every `*.node.ts` under src, examples and diagnostics that samples allocation, relative to the root. */
export function allocationGates(): string[] {
    return git([
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "src",
        "examples",
        "diagnostics",
    ])
        .filter(
            (path) =>
                path.endsWith(".node.ts") && GATE.test(readFileSync(resolve(ROOT, path), "utf8")),
        )
        .sort();
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

/** The gates whose inputs include a changed path, given relative to the root. */
export function selectGates(
    inputs: Map<string, Set<string>>,
    changed: readonly string[],
): string[] {
    const paths = changed.map((path) => resolve(ROOT, path));
    return [...inputs]
        .filter(([, files]) => paths.some((path) => files.has(path)))
        .map(([gate]) => gate);
}

if (import.meta.main) {
    const changed = git(["diff", "--name-only", process.argv[2] ?? "origin/main"]);
    const selected = selectGates(await gateInputs(allocationGates()), changed);
    for (const gate of selected) console.log(`./${gate}`);
}
