import { existsSync } from "node:fs";
import { createServer } from "node:net";
import { CLI, REPO_ROOT, skipReason } from "./verify";
import { type Bridge, start as startBridge } from "./wsl-bridge";

// Re-runnable startup-cost measurement over `shallot verify examples/gym --timings`: phase-checkpoint
// wall time (server boot → first page load → harness ready → run → capture → teardown, plus the
// harness-install probe and the resource-timing readout `bin/verify.ts` already emits under
// `--timings`) across N sequential runs, and — with `--transform` — one additional run's per-plugin
// dev-server transform attribution, read from Vite's own `DEBUG=vite:plugin-transform` event stream (a
// signal that already exists; this adds no instrument). Instrument-only: no lever, no `dev.ts` edit.
//
// Spawns `shallot verify` directly rather than going through this repo's `verify()`/`spawnVerify`
// (`scripts/verify.ts`), for the same reason stage 3a's driver did: `--transform` needs an explicit
// `env: { DEBUG: … }` on the child (a `process.env.DEBUG = …` set after this process's own startup does
// not reach a Bun.spawn child by default — verified empirically), and `--timings` prints as plain lines
// outside verify()'s parsed JSON result, so a caller reading the phases back needs the raw stdout
// `spawnVerify` doesn't expose. Both modes use one spawn path so they stay one source of truth.
//
// Display-gated like every `shallot verify` run (testing.md): skips honestly when no real GPU is
// reachable (`skipReason()`), and drives the WSL bridge automatically where one is needed.
//
// A DIFFERENTIAL against a prior tag needs that tag's own `bin/verify.ts` to emit the same `--timings`
// phase checkpoints — a HEAD-only reading here is a single-side number, not evidence of a regression.
// Pre-port tags don't have the checkpoints; this script can't invent them for a repo it isn't run from.
// Back-porting is a one-time, ~15-line diff over `verify()`'s existing `drive()` calls (the checkpoint
// pushes + `formatTimings`), applied to a throwaway worktree of the older tag and never committed to
// it. Copy this script into that worktree, `bun install && bun run build` there, run it there too, and
// diff the two printed tables by hand — this script only ever measures the repo it is run from.
//
// Sequential by design (testing.md's bench-cost note: a cold per-process module-transform graph is
// exactly what this reads, and a parallel run would corrupt it by contending on the same transform).

interface Args {
    dir: string;
    query: string[];
    runs: number;
    transform: boolean;
}

function help(): void {
    console.log(`Usage: bun run scripts/boot-cost.ts [options]

Runs \`shallot verify\` with --timings N times and prints per-phase wall time (median across runs) plus
the resource-timing readout. With --transform, also runs once under DEBUG=vite:plugin-transform and
prints per-plugin dev-transform CPU time, reconciled against the child process's own total CPU time.

Options:
  --dir <path>    project to verify (default: examples/gym)
  --query <k=v>   --query passthrough to verify (repeatable; default: scenario=render, seed=1,
                  warmup=60, frames=240 — the params this repo's headline startup number used)
  --runs <n>      number of --timings runs to median across (default: 3)
  --transform     also run the plugin-transform attribution (one extra run)`);
}

function parseArgs(argv: string[]): Args {
    if (argv.includes("--help") || argv.includes("-h")) {
        help();
        process.exit(0);
    }
    const out: Args = {
        dir: "examples/gym",
        query: ["scenario=render", "seed=1", "warmup=60", "frames=240"],
        runs: 3,
        transform: false,
    };
    let queryOverridden = false;
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        const take = (): string => {
            if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
            return argv[++i];
        };
        switch (arg) {
            case "--dir":
                out.dir = take();
                break;
            case "--query":
                if (!queryOverridden) {
                    out.query = [];
                    queryOverridden = true;
                }
                out.query.push(take());
                break;
            case "--runs":
                out.runs = Number.parseInt(take(), 10);
                break;
            case "--transform":
                out.transform = true;
                break;
            default:
                throw new Error(`unknown option: ${arg}`);
        }
    }
    return out;
}

const isWSL = process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

const freePort = (): Promise<number> =>
    new Promise((res, rej) => {
        const s = createServer();
        s.on("error", rej);
        s.listen(0, "127.0.0.1", () => {
            const p = (s.address() as { port: number }).port;
            s.close(() => res(p));
        });
    });

interface Spawned {
    stdout: string;
    stderr: string;
    exitCode: number | null;
    cpuUserMs: number | null;
    cpuSysMs: number | null;
}

/** one `shallot verify <dir> --json <extra>` run, WSL-bridge-routed or native, with `extraEnv` reaching
 *  the child explicitly (a plain `process.env` mutation would not — see the header note). `bridge`,
 *  when passed, is reused rather than started fresh (a sweep's-worth of runs share one browser boot). */
async function spawnVerify(
    dir: string,
    extra: string[],
    extraEnv: Record<string, string>,
    bridge: Bridge | null,
): Promise<Spawned> {
    const cmd = bridge
        ? [
              "node",
              bridge.bundle,
              "verify",
              dir,
              "--json",
              "--connect",
              bridge.connectUrl,
              "--port",
              String(await freePort()),
              ...extra,
          ]
        : ["bun", CLI, "verify", dir, "--json", ...extra];
    const proc = Bun.spawn(cmd, {
        cwd: REPO_ROOT,
        stdout: "pipe",
        stderr: "pipe",
        env: { ...process.env, ...extraEnv },
    });
    const [stdout, stderr] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
    ]);
    await proc.exited;
    const usage = proc.resourceUsage();
    return {
        stdout,
        stderr,
        exitCode: proc.exitCode,
        cpuUserMs: usage ? Number(usage.cpuTime.user) / 1000 : null,
        cpuSysMs: usage ? Number(usage.cpuTime.system) / 1000 : null,
    };
}

interface Phase {
    name: string;
    ms: number;
}

/** parse `formatTimings`/`formatExtraTimings`'s printed lines ("  name  123ms") out of a verify
 *  `--timings` stdout capture. Pure text parsing — the phase math itself stays owned by `bin/verify.ts`. */
export function parsePhases(stdout: string): Phase[] {
    const phases: Phase[] = [];
    for (const line of stdout.split("\n")) {
        const m = / {2}([a-z][a-z ()]*?) {2,}(\d+)ms\s*$/.exec(line);
        if (m) phases.push({ name: m[1], ms: Number(m[2]) });
    }
    return phases;
}

export interface Resources {
    count: number;
    totalMs: number;
    saturated: boolean;
}

export function parseResources(stdout: string): Resources | null {
    const m = / {2}resources {2}(\d+) requests, ([\d.]+)ms total(.*)$/m.exec(stdout);
    if (!m) return null;
    return { count: Number(m[1]), totalMs: Number(m[2]), saturated: /buffer full/.test(m[3]) };
}

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

async function runTimings(args: Args, bridge: Bridge | null): Promise<void> {
    const extra = [...args.query.flatMap((q) => ["--query", q]), "--timings"];
    const byPhase = new Map<string, number[]>();
    const resources: Resources[] = [];
    for (let i = 0; i < args.runs; i++) {
        const { stdout, exitCode } = await spawnVerify(args.dir, extra, {}, bridge);
        const phases = parsePhases(stdout);
        const res = parseResources(stdout);
        console.log(
            `run ${i + 1}/${args.runs}: exitCode=${exitCode} ${phases.length} phases parsed`,
        );
        for (const p of phases) byPhase.set(p.name, [...(byPhase.get(p.name) ?? []), p.ms]);
        if (res) resources.push(res);
        else
            console.log(
                "  (no resource-timing line found — check the run actually reached --timings)",
            );
    }
    console.log("\nphase medians:");
    for (const [name, samples] of byPhase) {
        console.log(
            `  ${name.padEnd(24)} ${median(samples).toFixed(0).padStart(6)}ms  [${samples.join(", ")}]`,
        );
    }
    if (resources.length > 0) {
        const counts = resources.map((r) => r.count);
        const totals = resources.map((r) => r.totalMs);
        console.log(
            `\nresources: ${median(counts)} requests (median), ${median(totals).toFixed(0)}ms total (median)` +
                (resources.some((r) => r.saturated)
                    ? " — SATURATED on at least one run, re-read"
                    : ""),
        );
    }
}

async function runTransform(args: Args, bridge: Bridge | null): Promise<void> {
    const extra = [...args.query.flatMap((q) => ["--query", q]), "--timings"];
    const { stdout, stderr, exitCode, cpuUserMs, cpuSysMs } = await spawnVerify(
        args.dir,
        extra,
        { DEBUG: "vite:plugin-transform", NO_COLOR: "1" },
        bridge,
    );
    console.log(`\ntransform run: exitCode=${exitCode}`);
    const lineRe = /vite:plugin-transform\s+([\d.]+)ms\s+(\S+)\s+(.+?)\s*$/;
    const byPlugin = new Map<string, { ms: number; n: number }>();
    for (const line of stderr.split("\n")) {
        const m = lineRe.exec(line);
        if (!m) continue;
        const e = byPlugin.get(m[2]) ?? { ms: 0, n: 0 };
        e.ms += Number(m[1]);
        e.n += 1;
        byPlugin.set(m[2], e);
    }
    console.log("per-plugin transform totals (naive sum — see the reconciliation below):");
    for (const [plugin, e] of [...byPlugin.entries()].sort((a, b) => b[1].ms - a[1].ms)) {
        console.log(`  ${plugin.padEnd(28)} ${e.ms.toFixed(1).padStart(10)}ms  (${e.n} events)`);
    }
    if (cpuUserMs !== null && cpuSysMs !== null) {
        const total = cpuUserMs + cpuSysMs;
        console.log(
            `\nchild process CPU: user=${cpuUserMs.toFixed(1)}ms sys=${cpuSysMs.toFixed(1)}ms total=${total.toFixed(1)}ms`,
        );
        const grandSum = [...byPlugin.values()].reduce((s, e) => s + e.ms, 0);
        if (grandSum > total) {
            console.log(
                `  the naive per-event sum (${grandSum.toFixed(1)}ms) EXCEEDS this total — proof of concurrency\n` +
                    `  double-counting (overlapping async hooks), not a real reading. Trust only a plugin whose own\n` +
                    `  hook is synchronous (no async/await) — its own summed duration is real serial CPU time.`,
            );
        }
    } else {
        console.log("\nchild process CPU: resourceUsage() unavailable");
    }
    const phases = parsePhases(stdout);
    if (phases.length > 0) {
        console.log("\n--timings phase block from the same run (for reconciliation):");
        for (const p of phases) console.log(`  ${p.name}  ${p.ms}ms`);
    }
}

async function main(): Promise<void> {
    const skip = skipReason();
    if (skip) {
        console.log(`skipped: ${skip}`);
        return;
    }
    const args = parseArgs(process.argv.slice(2));
    const bridge = isWSL ? await startBridge() : null;
    try {
        await runTimings(args, bridge);
        if (args.transform) await runTransform(args, bridge);
    } finally {
        if (bridge) await bridge.teardown();
    }
}

await main();
