import { Glob } from "bun";
import { EXAMPLE_GATES, type ExampleGate } from "./example-gates";
import { skipReason } from "./verify";

const WHOLE_ROSTER = new Set(["bun.lock"]);

export function selectExampleGates(paths: string[]): ExampleGate[] {
    if (paths.some((path) => WHOLE_ROSTER.has(path) || path.endsWith("package.json"))) {
        return [...EXAMPLE_GATES];
    }
    return EXAMPLE_GATES.filter((row) =>
        paths.some((path) => row.covers.some((cover) => new Glob(cover).match(path))),
    );
}

interface Args {
    base?: string;
    diff?: string;
    dryRun: boolean;
    all: boolean;
}

function parseArgs(argv: string[]): Args {
    const args: Args = { dryRun: false, all: false };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === "--dry-run") args.dryRun = true;
        else if (arg === "--all") args.all = true;
        else if (arg === "--base" || arg === "--diff") {
            const value = argv[++i];
            if (!value) throw new Error(`${arg} needs a ref`);
            if (arg === "--base") args.base = value;
            else args.diff = value;
        } else if (arg === "--help" || arg === "-h") {
            console.log(
                `Usage: bun run test:changed -- --base <ref> --diff <ref> [--dry-run]\n       bun run test:changed -- --all [--dry-run]`,
            );
            process.exit(0);
        } else throw new Error(`unknown argument: ${arg}`);
    }
    if (!args.all && (!args.base || !args.diff)) {
        throw new Error("--base <ref> and --diff <ref> are required unless --all is used");
    }
    return args;
}

async function changedPaths(base: string, diff: string): Promise<string[]> {
    const proc = Bun.spawn(["git", "diff", "--name-only", "--diff-filter=ACMR", base, diff], {
        stdout: "pipe",
        stderr: "pipe",
    });
    const [stdout, stderr, code] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    if (code !== 0) throw new Error(stderr.trim() || `git diff exited ${code}`);
    return stdout.split("\n").filter(Boolean);
}

function printPlan(paths: string[], rows: ExampleGate[]): void {
    console.log("Changed-path example gate plan:");
    if (paths.length > 0) for (const path of paths) console.log(`  changed: ${path}`);
    if (rows.length === 0) console.log("  selected: nothing");
    else for (const row of rows) console.log(`  selected: ${row.dir} -> ${row.gate}`);
}

async function runGate(row: ExampleGate): Promise<{ ok: boolean; warnings: number }> {
    const proc = Bun.spawn(["sh", "-c", row.gate], { stdout: "pipe", stderr: "pipe" });
    const [stdout, stderr, code] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    if (stdout) process.stdout.write(stdout);
    if (stderr) process.stderr.write(stderr);
    let warnings = 0;
    for (const match of `${stdout}\n${stderr}`.matchAll(/⚠ (\d+) console warning\(s\):/g)) {
        warnings += Number(match[1]);
    }
    return { ok: code === 0, warnings };
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
    const args = parseArgs(argv);
    const paths = args.all ? [] : await changedPaths(args.base!, args.diff!);
    const rows = args.all ? [...EXAMPLE_GATES] : selectExampleGates(paths);
    printPlan(paths, rows);
    if (args.dryRun || rows.length === 0) return 0;

    const skip = skipReason();
    if (skip) {
        console.log(
            `SKIP: selected example gates need native hardware (${skip}); no gate was run.`,
        );
        return 0;
    }

    let allPass = true;
    for (const row of rows) {
        const result = await runGate(row);
        console.log(`${result.ok ? "PASS" : "FAIL"}: ${row.dir} (${result.warnings} warnings)`);
        allPass = result.ok && allPass;
    }
    return allPass ? 0 : 1;
}

if (import.meta.main) {
    main()
        .then((code) => process.exit(code))
        .catch((error) => {
            console.error(error instanceof Error ? error.message : error);
            process.exit(2);
        });
}
