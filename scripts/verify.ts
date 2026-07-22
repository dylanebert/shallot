import { existsSync } from "node:fs";
import { createServer } from "node:net";
import { resolve } from "node:path";
import { type Bridge, bridgePrereq, start as startBridge } from "./wsl-bridge";

// Shared thin wrapper the repo bench/flows scripts drive the shipped gate through. `shallot verify` boots
// the target (an ejected vite app — the gym or a flow project), picks its own port, runs the published
// `window.__harness` in a real browser, and prints a JSON Result under `--json`. These scripts spawn it,
// pull that JSON off stdout, and interpret it — no server boot, no port logic, no Playwright here.

const repoRoot = resolve(import.meta.dir, "..");
export const REPO_ROOT = repoRoot;
export const CLI = resolve(repoRoot, "packages/shallot/bin/cli.ts");

/** one named check inside a verify Verdict (the published protocol's shape on the wire). */
export interface Check {
    name: string;
    ok: boolean;
    detail?: string;
    data?: Record<string, number>;
}

/** the `--memory` leak sample verify reports (informational, never gates). */
export interface Memory {
    start: number;
    end: number;
    growthPerSecond: number;
    leak: boolean;
    gcCount: number;
    gcPauseMs: number;
}

/** the Verdict a project's harness returns, as verify serializes it. `metrics` is a pass-through extra a
 *  gym scenario fills with the profiler measurement; a driver casts it to `BenchmarkMeasurement`. */
export interface Verdict {
    ok?: boolean;
    checks?: Check[];
    metrics?: unknown;
    [extra: string]: unknown;
}

/** the `shallot verify --json` Result (bin/verify.ts). A setup failure emits `{ pass:false, error }`. */
export interface VerifyResult {
    pass: boolean;
    error?: string;
    hardware?: string;
    verdict?: Verdict;
    memory?: Memory | null;
    errors?: string[];
    booted?: boolean;
    rendered?: boolean;
}

const isWSL = process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

// verify needs a conformant WebGPU adapter, which WSL's software adapter isn't (testing.md). On WSL the
// `wsl-bridge` drives the Windows host's real-GPU Chrome — proceed when the host has the interop + node/bun
// the bridge needs, else skip honestly naming what's missing. Elsewhere: native hardware, skip only a
// headless Linux box with no display. Returns a human reason to skip, or null to proceed.
export function skipReason(): string | null {
    if (isWSL) return bridgePrereq();
    if (process.platform === "linux" && !(process.env.DISPLAY || process.env.WAYLAND_DISPLAY)) {
        return "no display";
    }
    return null;
}

/** `--query k=v` for each entry. */
export function queryFlags(query: string[]): string[] {
    return query.flatMap((q) => ["--query", q]);
}

/** find the JSON verify emits under --json: the last stdout line parsing to an object with a boolean
 *  `pass`. Vite/console chatter shares stdout, so scan for it rather than assume the last line. */
export function extractResult(stdout: string): VerifyResult | null {
    let found: VerifyResult | null = null;
    for (const line of stdout.split("\n")) {
        const t = line.trim();
        if (!t.startsWith("{")) continue;
        try {
            const o = JSON.parse(t);
            if (o && typeof o.pass === "boolean") found = o as VerifyResult;
        } catch {
            // not the JSON line — keep scanning
        }
    }
    return found;
}

// On WSL the same drive logic runs, but against the host's real-GPU browser over the `wsl-bridge`: the verify
// CLI is bundled for node (Bun's Playwright client can't drive a connected browser here) and pointed at the
// bridge's ws endpoint with `--connect`. The bridge is started once and reused across a sweep's cells.
let bridge: Promise<Bridge> | null = null;
function wslBridge(): Promise<Bridge> {
    if (!bridge) bridge = startBridge();
    return bridge;
}

/** Tear down the shared WSL bridge if a sweep started one, so the driving process can exit. The bridge's
 *  rendezvous + client TCP servers and the host browser-server subprocess stay live between `verify` calls
 *  (reused across a sweep) and keep the event loop alive past the last verdict — without this a driver that
 *  ends by draining its loop (rather than `process.exit`) hangs. A sweep calls it once when done; native
 *  runs never start a bridge, so it's a no-op there. Idempotent — safe to call more than once. */
export async function teardownBridge(): Promise<void> {
    const started = bridge;
    if (!started) return;
    bridge = null;
    try {
        await (await started).teardown();
    } catch {
        // best-effort — the bridge's own rendezvous watchdog is the backstop if this can't complete
    }
}

/** spawn `shallot verify <dir> --json <extra>` from the repo root and return the parsed Result (null if
 *  none was emitted — a crash before verify could report). Echoes verify's stdout so a single run shows
 *  its full envelope; `quiet` suppresses the echo for a many-cell sweep where the blobs drown the table. */
export async function verify(
    dir: string,
    extra: string[] = [],
    quiet = false,
): Promise<VerifyResult | null> {
    const cmd = isWSL ? await wslCmd(dir, extra) : ["bun", CLI, "verify", dir, "--json", ...extra];
    const proc = Bun.spawn(cmd, { cwd: repoRoot, stdout: "pipe", stderr: "inherit" });
    const stdout = await new Response(proc.stdout).text();
    if (!quiet) process.stdout.write(stdout);
    await proc.exited;
    return extractResult(stdout);
}

// the WSL spawn: the node-bundled verify, driving the bridge's remote browser. A fixed `--port` skips the
// CLI's own `Bun.serve` port probe (undefined under node) — vite binds it on WSL and the host browser reaches
// it back over localhost forwarding, so the port must be one both sides agree on.
async function wslCmd(dir: string, extra: string[]): Promise<string[]> {
    const b = await wslBridge();
    return [
        "node",
        b.bundle,
        "verify",
        dir,
        "--json",
        "--connect",
        b.connectUrl,
        ...(extra.includes("--port") ? [] : ["--port", String(await bridgePort())]),
        ...extra,
    ];
}

const bridgePort = (): Promise<number> =>
    new Promise((res, rej) => {
        const s = createServer();
        s.on("error", rej);
        s.listen(0, "127.0.0.1", () => {
            const p = (s.address() as { port: number }).port;
            s.close(() => res(p));
        });
    });
