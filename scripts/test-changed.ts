import { resolve } from "node:path";
import { Glob } from "bun";
import { EXAMPLE_GATES, type ExampleGate } from "./example-gates";
import { type CpuGate, OCEAN_CPU_GATES } from "./ocean-oracle-gates";
import { REPO_ROOT, skipReason, type VerifyCommandDeps, verifyCommand } from "./verify";

const WHOLE_ROSTER = new Set(["bun.lock"]);
const DISPLAY_REQUIRED_ENV = "SHALLOT_DISPLAY_REQUIRED";

function wholeRoster(paths: string[]): boolean {
    return paths.some((path) => WHOLE_ROSTER.has(path) || path.endsWith("package.json"));
}

export function selectExampleGates(paths: string[]): ExampleGate[] {
    if (wholeRoster(paths)) return [...EXAMPLE_GATES];
    return EXAMPLE_GATES.filter((row) =>
        paths.some((path) => row.covers.some((cover) => new Glob(cover).match(path))),
    );
}

export function selectCpuGates(paths: string[]): CpuGate[] {
    if (wholeRoster(paths)) return [...OCEAN_CPU_GATES];
    return OCEAN_CPU_GATES.filter((row) =>
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
    if (!args.all && (!args.base || !args.diff))
        throw new Error("--base <ref> and --diff <ref> are required unless --all is used");
    return args;
}

export async function changedPaths(base: string, diff: string): Promise<string[]> {
    // Deletions still select the gate whose cover path disappeared; omitting D makes an oracle
    // deletion look like an empty, green selection.
    const proc = Bun.spawn(["git", "diff", "--name-only", "--diff-filter=ACMRD", base, diff], {
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

function printPlan(paths: string[], cpu: CpuGate[], display: ExampleGate[]): void {
    console.log("Changed-path gate plan:");
    for (const path of paths) console.log(`  changed: ${path}`);
    for (const row of cpu) console.log(`  CPU: ${row.name} -> bun run ${row.script}`);
    for (const row of display) console.log(`  display: ${row.dir} -> ${row.gate}`);
    if (cpu.length + display.length === 0) console.log("  selected: nothing");
}

export async function runCommand(
    command: string,
    deps: VerifyCommandDeps = {},
): Promise<{ ok: boolean; warnings: number }> {
    let cwd = REPO_ROOT;
    let argv = ["sh", "-c", command];
    const projectGate = /^bun run --cwd ([\w./-]+) gate$/.exec(command);
    if (projectGate) {
        const project = resolve(REPO_ROOT, projectGate[1]);
        const manifest = await Bun.file(resolve(project, "package.json")).json();
        const gate: unknown = manifest.scripts?.gate;
        if (typeof gate !== "string") throw new Error(`${project}: missing manifest gate`);
        if (/\bshallot\s+verify\b/.test(gate)) {
            // Only the simple, literal command shape is rewritable. Shell operators,
            // expansion and quoting must not be silently discarded.
            if (!/^bunx shallot verify(?: [\w./:=+-]+)*$/.test(gate)) {
                throw new Error(`${project}: unsupported verify gate composition: ${gate}`);
            }
            cwd = project;
            argv = await verifyCommand(gate.split(" ").slice(3), deps);
            console.log(`manifest gate: ${project}/package.json -> ${gate}`);
            console.log(`verify transport: cwd=${cwd} argv=${JSON.stringify(argv)}`);
        }
    }
    const proc = Bun.spawn(argv, { cwd, stdout: "pipe", stderr: "pipe", env: { ...process.env } });
    const [stdout, stderr, code] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
    ]);
    if (stdout) process.stdout.write(stdout);
    if (stderr) process.stderr.write(stderr);
    let warnings = 0;
    for (const match of `${stdout}\n${stderr}`.matchAll(
        /^[\t ]*(?:⚠ (\d+) console warning\(s\):|warnings \((\d+)\):)[\t ]*\r?$/gm,
    ))
        warnings += Number(match[1] ?? match[2]);
    return { ok: code === 0, warnings };
}

export interface MainDeps {
    paths?: (base: string, diff: string) => Promise<string[]>;
    run?: (command: string) => Promise<{ ok: boolean; warnings: number }>;
    displaySkip?: () => string | null;
    displayRequired?: boolean;
}

export async function main(argv = process.argv.slice(2), deps: MainDeps = {}): Promise<number> {
    const args = parseArgs(argv);
    const paths = args.all ? [] : await (deps.paths ?? changedPaths)(args.base!, args.diff!);
    const cpu = args.all ? [...OCEAN_CPU_GATES] : selectCpuGates(paths);
    const display = args.all ? [...EXAMPLE_GATES] : selectExampleGates(paths);
    printPlan(paths, cpu, display);
    if (args.dryRun) return 0;
    if (cpu.length + display.length === 0) {
        console.log("PASS: no changed-path rows selected.");
        return 0;
    }

    const run = deps.run ?? runCommand;
    let allPass = true;
    for (const row of cpu) {
        const result = await run(`bun run ${row.script}`);
        console.log(`${result.ok ? "PASS" : "FAIL"}: CPU ${row.name}`);
        allPass = result.ok && allPass;
    }

    if (display.length === 0) return allPass ? 0 : 1;
    const skip = (deps.displaySkip ?? skipReason)();
    if (skip) {
        const required = deps.displayRequired ?? process.env[DISPLAY_REQUIRED_ENV] === "1";
        console.log(
            `${required ? "FAIL" : "UNAVAILABLE"}: selected display gates need native hardware (${skip}); no display gate was run.`,
        );
        if (!required && cpu.length > 0 && allPass)
            console.log("PASS: selected CPU rows passed; display rows were unavailable.");
        return required || !allPass ? 1 : 0;
    }

    try {
        for (const row of display) {
            const result = await run(row.gate);
            console.log(
                `${result.ok ? "PASS" : "FAIL"}: display ${row.dir} (${result.warnings} warnings)`,
            );
            allPass = result.ok && allPass;
        }
    } finally {
    }
    if (allPass) console.log("PASS: all selected rows passed.");
    return allPass ? 0 : 1;
}

if (import.meta.main)
    main()
        .then((code) => process.exit(code))
        .catch((error) => {
            console.error(error instanceof Error ? error.message : error);
            process.exit(2);
        });
