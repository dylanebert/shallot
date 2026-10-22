import { existsSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

// Attribution outside JavaScript, from the processes that spend it and with no privileges: `cputrace`
// polls each WebKit, Metal compiler and browser process's CPU time every millisecond on the wall clock,
// and `sample` takes the GPU and web content processes' stacks from the handoff.
const PROCESSES = [
    "com.apple.WebKit.GPU",
    "com.apple.WebKit.WebContent",
    "com.apple.WebKit.Networking",
    "MTLCompilerService",
    "Safari.app/Contents/MacOS/Safari",
    "Playwright.app/Contents/MacOS/Playwright",
];
// From the first plugin hook, through the precompile drain and the handoff to the settled frames.
const SAMPLE_SECONDS = 6;
// From the handoff through the stall, so its stacks are the stall's alone.
const HANDOFF_SECONDS = 1;
const SAMPLE_INTERVAL_MS = 2;

function tool(outDir: string): string {
    const source = resolve(import.meta.dir, "cputrace.c");
    const binary = join(outDir, "cputrace");
    if (!existsSync(binary) || statSync(binary).mtimeMs < statSync(source).mtimeMs) {
        const built = Bun.spawnSync(["clang", "-O2", source, "-o", binary], { stderr: "pipe" });
        if (built.exitCode !== 0) throw new Error(`cputrace: ${built.stderr.toString()}`);
    }
    return binary;
}

const pids = (pattern: string): number[] =>
    Bun.spawnSync(["pgrep", "-f", pattern], { stdout: "pipe" })
        .stdout.toString()
        .split("\n")
        .filter(Boolean)
        .map(Number);

// Processes that were running before this run belong to other clients: another app's WebKit, or Metal
// compilers serving other processes. A cold run's browser processes all start after this.
const preexisting = new Set(PROCESSES.flatMap(pids));

export type Trace = { cpu: string; samples: string[] };

/** Starts the CPU timeline for one load; `beacon` takes stacks from the page's first plugin hook and its handoff. */
export function startTrace(outDir: string, name: string) {
    const cpu = join(outDir, `${name}.cpu.ndjson`);
    const poller = Bun.spawn([tool(outDir), cpu, "1000", ...PROCESSES], {
        stderr: "inherit",
        env: { ...process.env, CPUTRACE_EXCLUDE: [...preexisting].join(",") },
    });
    const samplers: Promise<unknown>[] = [];
    const samples: string[] = [];
    const sampled = new Set<string>();
    let stopped: Promise<Trace> | undefined;
    return {
        /** `hook` from the first plugin hook, then `handoff` from the handoff, each sampled once. */
        beacon(window: "hook" | "handoff") {
            if (sampled.has(window)) return;
            sampled.add(window);
            const seconds = window === "hook" ? SAMPLE_SECONDS : HANDOFF_SECONDS;
            for (const [role, pattern] of [
                ["gpu", "com.apple.WebKit.GPU"],
                ["web", "com.apple.WebKit.WebContent"],
            ] as const)
                for (const pid of pids(pattern).filter((pid) => !preexisting.has(pid))) {
                    const file = join(outDir, `${name}.${window}.${role}-${pid}.sample.txt`);
                    samples.push(file);
                    const child = Bun.spawn(
                        ["sample", String(pid), String(seconds), String(SAMPLE_INTERVAL_MS), "-mayDie", "-f", file],
                        { stdout: "ignore", stderr: "ignore" },
                    );
                    samplers.push(child.exited);
                }
        },
        stop(): Promise<Trace> {
            stopped ??= (async () => {
                await Promise.all(samplers);
                poller.kill("SIGTERM");
                await poller.exited;
                return { cpu, samples };
            })();
            return stopped;
        },
    };
}

type Line = { t: number; pid: number; name: string; cpu: number; threads: [string, number][] };
const role = (name: string) =>
    name.includes("GPU")
        ? "gpu"
        : name.includes("WebContent")
          ? "web"
          : name.includes("MTLCompiler")
            ? "compiler"
            : name.includes("Networking")
              ? "network"
              : "ui";
export const ROLES = ["gpu", "compiler", "web", "ui", "network"] as const;

/**
 * CPU milliseconds per process role in each window, from the poller's deltas. A delta is CPU spent since
 * the previous poll (1 ms earlier), so it is charged to the window containing its poll time. Times are
 * epoch milliseconds. The web content process's main thread is reported apart, as `web-main`.
 */
export function cpuIn(file: string, windows: [string, number, number][]) {
    const lines = readFileSync(file, "utf8")
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Line);
    return windows.map(([label, from, to]) => {
        const totals: Record<string, number> = {};
        const threads: Record<string, number> = {};
        for (const line of lines) {
            const t = line.t / 1000;
            if (t <= from || t > to) continue;
            const r = role(line.name);
            totals[r] = (totals[r] ?? 0) + line.cpu / 1e6;
            for (const [thread, ns] of line.threads) {
                const key = `${r}:${thread.replace(/#\d+$/, "") || "<unnamed>"}`;
                threads[key] = (threads[key] ?? 0) + ns / 1e6;
            }
        }
        const round = (value: number) => Math.round(value * 10) / 10;
        return {
            label,
            ms: round(to - from),
            cpu: Object.fromEntries(ROLES.map((r) => [r, round(totals[r] ?? 0)])),
            top: Object.entries(threads)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([thread, value]) => `${thread} ${round(value)}`),
        };
    });
}
