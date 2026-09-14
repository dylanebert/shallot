import { resolve } from "node:path";
import { cargoTestExecutable, cargoTestTargetExecutables } from "../src/harness/verdict";

const root = resolve(process.env.SHALLOT_PROJECT_ROOT ?? resolve(import.meta.dir, ".."));
const subjectByPackage: Record<string, string> = {
    "shallot-audio": "crates/audio",
    "shallot-physics": "crates/physics",
};

function subjectFor(packageName: string): string {
    const subject = subjectByPackage[packageName];
    if (subject === undefined) throw new Error(`unsupported Cargo package: ${packageName}`);
    return subject;
}

export interface CargoTestPartition {
    filter: string;
    tests: readonly string[];
}

const partitionCache = new Map<string, readonly CargoTestPartition[]>();

function listTests(executable: string): string[] {
    const proc = Bun.spawnSync([executable, "--list"], {
        cwd: root,
        stdout: "pipe",
        stderr: "pipe",
    });
    if (!proc.success) {
        const detail = proc.stderr.toString().trim() || proc.stdout.toString().trim();
        throw new Error(`libtest --list failed${detail ? `: ${detail}` : ""}`);
    }
    const tests = proc.stdout
        .toString()
        .split("\n")
        .flatMap((line) => {
            const match = line.match(/^(.+): test$/);
            return match === null ? [] : [match[1]];
        });
    if (tests.length === 0) throw new Error("libtest --list produced an empty test population");
    if (new Set(tests).size !== tests.length) {
        throw new Error("libtest --list produced duplicate test names");
    }
    return tests;
}

/**
 * Discover the complete current libtest population and partition it by its stable Rust module
 * prefix. A partition is admitted only when the discovered sets are non-empty, exclusive, and
 * their union is exactly the list emitted by this executable.
 */
export function discoverCargoTestPartitions(packageName: string): readonly CargoTestPartition[] {
    if (packageName !== "shallot-audio") {
        throw new Error(`unsupported audio Cargo package: ${packageName}`);
    }
    const subject = subjectFor(packageName);
    const key = `${root}\u0000${packageName}`;
    const cached = partitionCache.get(key);
    if (cached !== undefined) return cached;

    const executable = cargoTestExecutable(root, subject);
    const tests = listTests(executable);
    const byModule = new Map<string, string[]>();
    for (const test of tests) {
        const separator = test.indexOf("::");
        if (separator <= 0) {
            throw new Error(`libtest test has no stable module prefix: ${test}`);
        }
        const module = test.slice(0, separator);
        const members = byModule.get(module) ?? [];
        members.push(test);
        byModule.set(module, members);
    }

    const partitions = [...byModule.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([filter, members]) => ({ filter: `${filter}::`, tests: members }));
    const union = partitions.flatMap((partition) => partition.tests);
    if (
        partitions.some((partition) => partition.tests.length === 0) ||
        union.length !== tests.length ||
        new Set(union).size !== tests.length ||
        [...union].sort().join("\n") !== [...tests].sort().join("\n")
    ) {
        throw new Error("libtest partitions are not a non-empty exclusive exact union");
    }
    partitionCache.set(key, partitions);
    return partitions;
}

function killProcessGroup(pid: number): void {
    // detached makes the direct libtest child the leader of its own POSIX process group. Kill the
    // group first so a test that planted a descendant cannot survive the row timeout.
    try {
        process.kill(-pid, "SIGKILL");
    } catch {
        // The child may have exited between the timeout and the group signal.
    }
}

async function runDirectExecutable(
    executable: string,
    args: readonly string[],
    label: string,
): Promise<void> {
    const child = Bun.spawn([executable, ...args], {
        cwd: root,
        stdout: "inherit",
        stderr: "inherit",
        detached: true,
    });
    let timedOut = false;
    const timeout = setTimeout(() => {
        timedOut = true;
        killProcessGroup(child.pid);
        try {
            child.kill("SIGKILL");
        } catch {
            // The process-group signal is authoritative; this is only a race-safe fallback.
        }
    }, 19_500);
    const exitCode = await child.exited;
    clearTimeout(timeout);
    if (timedOut) throw new Error(`libtest partition ${label} exceeded the 20s row budget`);
    if (exitCode !== 0) {
        throw new Error(`libtest partition ${label} exited with ${exitCode}`);
    }
}

/** Run already-compiled direct libtest binaries; Cargo is never the timed-row parent. */
export async function runCargoTest(packageName: string, ...args: string[]): Promise<void> {
    const subject = subjectFor(packageName);
    if (packageName === "shallot-audio") {
        if (args.length !== 1)
            throw new Error("audio Cargo test needs exactly one module partition");
        await runDirectExecutable(cargoTestExecutable(root, subject), args, args[0]);
        return;
    }
    const targets = args.flatMap((arg, index) => (arg === "--test" ? [args[index + 1] ?? ""] : []));
    const executables = cargoTestTargetExecutables(root, subject, targets);
    for (const [index, executable] of executables.entries()) {
        await runDirectExecutable(executable, [], targets[index]);
    }
}
