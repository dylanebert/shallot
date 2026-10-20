import { spawn, spawnSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
    accessSync,
    appendFileSync,
    constants,
    cpSync,
    createWriteStream,
    existsSync,
    lstatSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    realpathSync,
    renameSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import { basename, delimiter, dirname, isAbsolute, relative, resolve } from "node:path";
import { classifyAdapter } from "../src/engine/runtime/adapter";
import { normalizePath } from "../src/project/paths";
import { CEILING } from "../src/project/testing.js";

const ROOT = resolve(import.meta.dir, "..");
const SETUP_COMMAND = "bash scripts/native-linux-setup.sh";
const BUN_INSTALL = "curl -fsSL https://bun.sh/install | bash";
const RUSTUP_INSTALL = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh";
const LINUX_TARGET = "x86_64-unknown-linux-gnu";
const ISOLATION_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
};
const ANSI_ESCAPE = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, "g");

export interface LinuxPrerequisiteProbe {
    commandExists(name: string): boolean;
    hasX11Headers(): boolean;
    softwareVulkanIcd(): string | null;
}

function executableOnPath(name: string): boolean {
    for (const directory of (process.env.PATH ?? "").split(delimiter)) {
        if (!directory) continue;
        try {
            accessSync(resolve(directory, name), constants.X_OK);
            return true;
        } catch {}
    }
    return false;
}

function configuredSoftwareIcd(): string | null {
    const configured = process.env.VK_ICD_FILENAMES;
    if (configured) {
        const path = configured
            .split(delimiter)
            .find((entry) => /(?:lvp|swrast)/i.test(basename(entry)));
        return path && existsSync(path) ? path : null;
    }
    for (const directory of ["/usr/share/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"]) {
        if (!existsSync(directory)) continue;
        const candidate = readdirSync(directory)
            .filter((entry) => /(?:lvp|swrast).*\.json$/i.test(entry))
            .map((entry) => resolve(directory, entry))
            .find(existsSync);
        if (candidate) return candidate;
    }
    return null;
}

const defaultProbe: LinuxPrerequisiteProbe = {
    commandExists: executableOnPath,
    hasX11Headers: () =>
        spawnSync("pkg-config", ["--exists", "x11"], { stdio: "ignore" }).status === 0,
    softwareVulkanIcd: configuredSoftwareIcd,
};

export function requireLinuxPrerequisites(
    probe: LinuxPrerequisiteProbe = defaultProbe,
    hardware = false,
): string {
    const softwareIcd = probe.softwareVulkanIcd();
    const missingBun = ["bun", "bunx"].filter((command) => !probe.commandExists(command));
    const missingRust = ["cargo", "rustc"].filter((command) => !probe.commandExists(command));
    const missingSystem = [
        "cc",
        "pkg-config",
        "cmake",
        "ninja",
        "Xvfb",
        "xcompmgr",
        "tar",
        "python3",
    ].filter((command) => !probe.commandExists(command));
    if (!probe.hasX11Headers()) missingSystem.push("X11 development headers (libx11-dev)");
    if (!hardware && !softwareIcd)
        missingSystem.push("Mesa software Vulkan ICD (mesa-vulkan-drivers)");

    const missing = [...missingBun, ...missingRust, ...missingSystem];
    if (missing.length) {
        const routes = [
            ...(missingBun.length ? [`Bun: \`${BUN_INSTALL}\``] : []),
            ...(missingRust.length ? [`Rust: \`${RUSTUP_INSTALL}\``] : []),
            ...(missingSystem.length ? [`system packages: \`${SETUP_COMMAND}\``] : []),
        ];
        throw new Error(
            `missing Linux native smoke prerequisites: ${missing.join(", ")}; install via ${routes.join("; ")}${!hardware && !softwareIcd ? " (or set VK_ICD_FILENAMES to a Mesa lvp ICD)" : ""}`,
        );
    }
    return softwareIcd ?? "";
}

function parseArgs(args: string[]) {
    const options: {
        consumer?: string;
        logs?: string;
        tarball?: string;
        staleRuntime?: string;
        plantStale: boolean;
        hardware: boolean;
        portableOnly: boolean;
        help: boolean;
    } = { plantStale: false, hardware: false, portableOnly: false, help: false };
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === "--help" || arg === "-h") options.help = true;
        else if (arg === "--hardware") options.hardware = true;
        else if (arg === "--plant-stale") options.plantStale = true;
        else if (arg === "--portable-only") options.portableOnly = true;
        else if (["--consumer", "--logs", "--tarball", "--stale-runtime"].includes(arg)) {
            const value = args[++index];
            if (!value || value.startsWith("--")) throw new Error(`${arg} requires a path`);
            if (arg === "--consumer") options.consumer = value;
            else if (arg === "--logs") options.logs = value;
            else if (arg === "--tarball") options.tarball = value;
            else options.staleRuntime = value;
        } else throw new Error(`unknown option: ${arg}`);
    }
    if (!options.help && !options.consumer) throw new Error("--consumer <directory> is required");
    if (options.staleRuntime && options.plantStale)
        throw new Error("--stale-runtime and --plant-stale are separate evidence modes");
    return options;
}

function cleanEnvironment(
    source: NodeJS.ProcessEnv,
    additions: NodeJS.ProcessEnv = {},
): NodeJS.ProcessEnv {
    const env = { ...source };
    delete env.DISPLAY;
    delete env.WAYLAND_DISPLAY;
    delete env.WAYLAND_SOCKET;
    delete env.SHALLOT_DEV_URL;
    delete env.CEF_PATH;
    Object.assign(env, additions);
    return env;
}

function slug(value: string): string {
    return value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "");
}

class SmokeFailure extends Error {
    constructor(step: string, logPath: string, reason: string) {
        super(`${step} failed: ${reason}. Log: ${logPath}`);
    }
}

interface CommandResult {
    output: string;
}

async function runLogged(
    step: string,
    command: string,
    args: string[],
    cwd: string,
    env: NodeJS.ProcessEnv,
    logs: string,
): Promise<CommandResult> {
    const logPath = resolve(logs, `${slug(step)}.log`);
    const log = createWriteStream(logPath);
    const chunks: string[] = [];
    const started = performance.now();
    let spawnError: Error | undefined;
    const child = spawn(command, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"] });
    const record = (stream: "stdout" | "stderr", data: Buffer) => {
        const text = data.toString();
        chunks.push(text);
        log.write(`[${stream}] ${text}`);
    };
    child.stdout.on("data", (data: Buffer) => record("stdout", data));
    child.stderr.on("data", (data: Buffer) => record("stderr", data));
    child.once("error", (error) => {
        spawnError = error;
    });
    const result = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>(
        (resolveExit) => child.once("close", (code, signal) => resolveExit({ code, signal })),
    );
    await new Promise<void>((resolveEnd) => log.end(resolveEnd));
    if (spawnError || result.code !== 0) {
        throw new SmokeFailure(
            step,
            logPath,
            spawnError?.message ?? `exit ${result.code ?? result.signal}`,
        );
    }
    const durationMs = performance.now() - started;
    console.info(`${step}: ${(durationMs / 1000).toFixed(1)}s (log: ${logPath})`);
    return { output: chunks.join("") };
}

function groupExists(pid: number): boolean {
    try {
        process.kill(-pid, 0);
        return true;
    } catch (error) {
        return (error as NodeJS.ErrnoException).code !== "ESRCH";
    }
}

/** Terminate a detached POSIX process group, escalating after a bounded grace period. */
export async function terminateProcessGroup(pid: number, graceMs = 500): Promise<void> {
    if (process.platform === "win32") {
        spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], { stdio: "ignore" });
        return;
    }
    try {
        process.kill(-pid, "SIGTERM");
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }
    const deadline = Date.now() + graceMs;
    while (groupExists(pid) && Date.now() < deadline) await Bun.sleep(25);
    if (groupExists(pid)) {
        try {
            process.kill(-pid, "SIGKILL");
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
        }
    }
}

function processGroup(pid: number): number | null {
    if (process.platform === "darwin") {
        const result = spawnSync("ps", ["-o", "pgid=", "-p", String(pid)], { encoding: "utf8" });
        return result.status === 0 ? Number(result.stdout.trim()) : null;
    }
    try {
        const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
        const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
        return Number(fields[2]);
    } catch {
        return null;
    }
}

// LaunchServices receives this explicitly via open --env, including under App Translocation.
const MAC_OWNER = `SHALLOT_SMOKE_OWNER=${randomUUID()}`;

export function assertNoSmokeProcesses(pids: number[]): void {
    if (pids.length) throw new Error(`smoke processes still running: ${pids.join(", ")}`);
}

export function macOwnedPids(owner = MAC_OWNER): number[] {
    const result = spawnSync("ps", ["eww", "-axo", "pid=,command="], { encoding: "utf8" });
    if (result.status !== 0) throw new Error(`cannot inventory smoke processes: ${result.stderr}`);
    return result.stdout.split("\n").flatMap((row) => {
        const match = row.trim().match(/^(\d+)\s+(.+)$/);
        return match?.[2].split(/\s+/).includes(owner) ? [Number(match[1])] : [];
    });
}

export function macSmokePids(consumer: string): number[] {
    const root = `${realpathSync.native(consumer)}/`;
    const result = spawnSync("ps", ["-wwaxo", "pid=,comm="], { encoding: "utf8" });
    if (result.status !== 0)
        throw new Error(`cannot inventory consumer processes: ${result.stderr}`);
    const pids = new Set(macOwnedPids());
    for (const row of result.stdout.split("\n")) {
        const match = row.trim().match(/^(\d+)\s+(.+)$/);
        if (!match) continue;
        try {
            if (realpathSync.native(match[2]).startsWith(root)) pids.add(Number(match[1]));
        } catch {}
    }
    return [...pids];
}

async function stopMacOwnedProcesses(consumer: string): Promise<void> {
    for (const signal of ["SIGTERM", "SIGKILL"] as const) {
        for (const pid of macSmokePids(consumer)) {
            try {
                process.kill(pid, signal);
            } catch (error) {
                if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
            }
        }
        const deadline = Date.now() + 1000;
        while (macSmokePids(consumer).length && Date.now() < deadline) await Bun.sleep(25);
    }
    assertNoSmokeProcesses(macSmokePids(consumer));
}

export function mainShellPid(group: number, executable: string): number | null {
    if (process.platform === "win32") {
        return (
            windowsProcesses().find(
                (row) =>
                    row.executablePath?.toLowerCase() === executable.toLowerCase() &&
                    !row.commandLine?.includes("--type="),
            )?.processId ?? null
        );
    }
    if (process.platform === "darwin") {
        if (!existsSync(executable)) return null;
        const expected = realpathSync.native(executable);
        const rows = spawnSync("ps", ["-axo", "pid=,pgid=,comm="], { encoding: "utf8" }).stdout;
        for (const row of rows.split("\n")) {
            const match = row.trim().match(/^(\d+)\s+(\d+)\s+(.+)$/);
            if (!match) continue;
            try {
                if (realpathSync.native(match[3]) === expected) return Number(match[1]);
            } catch {}
        }
        return null;
    }
    const expected = realpathSync.native(executable);
    for (const entry of readdirSync("/proc")) {
        if (!/^\d+$/.test(entry)) continue;
        const pid = Number(entry);
        if (processGroup(pid) !== group) continue;
        try {
            const actual = realpathSync.native(`/proc/${pid}/exe`);
            const args = readFileSync(`/proc/${pid}/cmdline`, "utf8").split("\0");
            if (actual === expected && !args.some((arg) => arg.startsWith("--type="))) return pid;
        } catch {}
    }
    return null;
}

export async function stopMainShell(group: number, executable: string): Promise<void> {
    const pid = mainShellPid(group, executable);
    if (!pid) return;
    try {
        process.kill(pid, "SIGTERM");
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ESRCH") return;
        throw error;
    }
    if (process.platform === "win32") return;
    const deadline = Date.now() + 1000;
    while (processGroup(pid) !== null && Date.now() < deadline) await Bun.sleep(25);
    if (processGroup(pid) !== null) {
        try {
            process.kill(pid, "SIGKILL");
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
        }
    }
}

export async function withoutProject<T>(project: string, action: () => Promise<T>): Promise<T> {
    const hidden = `${project}-unavailable-${randomUUID()}`;
    renameSync(project, hidden);
    try {
        return await action();
    } finally {
        renameSync(hidden, project);
    }
}

interface MarkerReport {
    nonce: string;
    mode: string;
    href: string;
    pixel?: number[];
    requireHardware?: boolean;
    cpuReady?: boolean;
    hooks?: string[];
    refusal?: string;
    adapter?: Record<string, string | boolean>;
    error?: string;
    userAgent?: string;
    run?: string;
}

interface Observer {
    origin: string;
    wait(nonce: string): Promise<MarkerReport>;
    close(): Promise<void>;
}

export async function startObserver(logs: string): Promise<Observer> {
    const logPath = resolve(logs, "observer.log");
    writeFileSync(logPath, "");
    const pending = new Map<string, { resolve(report: MarkerReport): void }>();
    const runs = new Set<string>();
    const server = createServer((request, response) => {
        response.setHeader("Access-Control-Allow-Origin", "*");
        response.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
        response.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
        response.setHeader("Access-Control-Allow-Headers", "content-type");
        if (request.method === "OPTIONS") {
            response.writeHead(204).end();
            return;
        }
        if (request.method !== "POST" || request.url !== "/marker") {
            response.writeHead(404).end();
            return;
        }
        const chunks: Buffer[] = [];
        let size = 0;
        request.on("data", (chunk: Buffer) => {
            size += chunk.length;
            if (size <= 16_384) chunks.push(chunk);
        });
        request.on("end", () => {
            if (size > 16_384) {
                response.writeHead(413).end();
                return;
            }
            try {
                const report = JSON.parse(Buffer.concat(chunks).toString("utf8")) as MarkerReport;
                appendFileSync(logPath, `${JSON.stringify(report)}\n`);
                if (!report.run || runs.has(report.run)) {
                    response.writeHead(409).end("missing or repeated page-run marker");
                    return;
                }
                runs.add(report.run);
                const waiter = pending.get(report.nonce);
                if (waiter) {
                    waiter.resolve(report);
                    pending.delete(report.nonce);
                }
                response.writeHead(200, { "Content-Type": "text/plain" }).end("observed");
            } catch {
                response.writeHead(400).end();
            }
        });
    });
    await new Promise<void>((resolveListen, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolveListen);
    });
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("observer did not bind TCP");
    return {
        origin: `http://127.0.0.1:${address.port}`,
        wait(nonce) {
            let resolveReport!: (report: MarkerReport) => void;
            const promise = new Promise<MarkerReport>((resolve) => {
                resolveReport = resolve;
            });
            pending.set(nonce, { resolve: resolveReport });
            return promise;
        },
        close: () =>
            new Promise<void>((resolveClose, reject) =>
                server.close((error) => (error ? reject(error) : resolveClose())),
            ),
    };
}

interface VirtualDisplay {
    display: string;
    stop(): Promise<void>;
}

function displayPathExists(path: string): boolean {
    try {
        lstatSync(path);
        return true;
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
        throw error;
    }
}

/** Own one lock-arbitrated Xvfb process group; never remove display paths ourselves. */
export async function startVirtualDisplay(
    logs: string,
    env: NodeJS.ProcessEnv,
    probe = { pathExists: displayPathExists, timeoutMs: 15_000 },
): Promise<VirtualDisplay> {
    const step = "virtual display";
    const logPath = resolve(logs, "virtual-display.log");
    let number = 99;
    while (
        number < 1099 &&
        (probe.pathExists(`/tmp/.X${number}-lock`) || probe.pathExists(`/tmp/.X11-unix/X${number}`))
    )
        number++;
    if (number === 1099)
        throw new SmokeFailure(step, logPath, "no unused display number in 99..1098");
    const log = createWriteStream(logPath);
    // Like Debian xvfb-run: inherited SIGUSR1 ignore asks Xserver to notify its parent.
    // Normal Xserver locking arbitrates competing launches after our occupancy check.
    const script = `trap 'printf "ready\\n"' USR1
(trap '' USR1; exec Xvfb "$@" >&2) &
server=$!
wait "$server"
status=$?
while kill -0 "$server" 2>/dev/null; do
    wait "$server"
    status=$?
done
exit "$status"`;
    const child = spawn(
        "sh",
        [
            "-c",
            script,
            "shallot-xvfb",
            `:${number}`,
            "-screen",
            "0",
            "1280x720x24",
            "-nolisten",
            "tcp",
            "-ac",
        ],
        { env: cleanEnvironment(env), detached: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    let ready = false;
    let spawnError: Error | undefined;
    let output = "";
    child.stdout.on("data", (data: Buffer) => {
        const text = data.toString();
        output += text;
        log.write(`[stdout] ${text}`);
        ready ||= /(?:^|\n)ready\n/.test(output);
    });
    child.stderr.on("data", (data: Buffer) => log.write(`[stderr] ${data.toString()}`));
    child.once("error", (error) => {
        spawnError = error;
    });
    let closeResult: { code: number | null; signal: NodeJS.Signals | null } | undefined;
    const exited = new Promise<void>((resolveExit) => {
        child.once("close", (code, signal) => {
            closeResult = { code, signal };
            resolveExit();
        });
    });
    const deadline = Date.now() + probe.timeoutMs;
    while (!ready && !closeResult && !spawnError && Date.now() < deadline) await Bun.sleep(25);
    if (!ready || closeResult || spawnError) {
        const reason =
            spawnError?.message ??
            (closeResult
                ? `Xvfb exited (${closeResult.code ?? closeResult.signal})`
                : `Xvfb did not signal readiness within ${probe.timeoutMs}ms`);
        if (child.pid) await terminateProcessGroup(child.pid, 100);
        await Promise.race([exited, Bun.sleep(1500)]);
        await new Promise<void>((resolveEnd) => log.end(resolveEnd));
        throw new SmokeFailure(step, logPath, reason);
    }
    const display = `:${number}`;
    console.info(`virtual display: ${display} (log: ${logPath})`);
    return {
        display,
        async stop() {
            if (child.pid) await terminateProcessGroup(child.pid);
            await Promise.race([exited, Bun.sleep(1500)]);
            await new Promise<void>((resolveEnd) => log.end(resolveEnd));
        },
    };
}

async function startCompositor(logs: string, env: NodeJS.ProcessEnv): Promise<() => Promise<void>> {
    const log = createWriteStream(resolve(logs, "compositor.log"));
    const child = spawn("xcompmgr", [], { env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    child.stdout.pipe(log, { end: false });
    child.stderr.pipe(log, { end: false });
    let spawnError: Error | undefined;
    child.once("error", (error) => {
        spawnError = error;
    });
    const stop = async () => {
        if (child.pid) await terminateProcessGroup(child.pid);
        await new Promise<void>((resolveEnd) => log.end(resolveEnd));
    };
    try {
        // A compositor owner makes Chromium select an ARGB visual instead of the root visual.
        await runLogged(
            "compositor ready",
            "python3",
            [
                "-c",
                `
import ctypes as c, time
x = c.CDLL("libX11.so.6")
x.XOpenDisplay.argtypes = [c.c_char_p]
x.XOpenDisplay.restype = c.c_void_p
x.XInternAtom.argtypes = [c.c_void_p, c.c_char_p, c.c_int]
x.XInternAtom.restype = c.c_ulong
x.XGetSelectionOwner.argtypes = [c.c_void_p, c.c_ulong]
x.XGetSelectionOwner.restype = c.c_ulong
x.XCloseDisplay.argtypes = [c.c_void_p]
d = x.XOpenDisplay(None)
assert d, "cannot open smoke display"
try:
    atom = x.XInternAtom(d, b"_NET_WM_CM_S0", 0)
    deadline = time.monotonic() + 5
    while not x.XGetSelectionOwner(d, atom):
        assert time.monotonic() < deadline, "X11 compositor did not acquire its selection"
        time.sleep(.025)
    print("X11 compositor selection acquired")
finally:
    x.XCloseDisplay(d)
`,
            ],
            ROOT,
            env,
            logs,
        );
        if (spawnError) throw spawnError;
        return stop;
    } catch (error) {
        await stop();
        throw error;
    }
}

function writePage(
    project: string,
    observer: string,
    mode: string,
    nonce: string,
    hardware = false,
): void {
    const report = JSON.stringify({ nonce, mode, href: "" });
    const script = `import { createApp } from '@dylanebert/shallot/app';
const report = ${report};
report.href = location.href;
report.userAgent = navigator.userAgent;
report.run = crypto.randomUUID();
report.requireHardware = globalThis.__shallotRequireHardwareGpu;
try {
    const status = document.createElement('p');
    document.body.append(status);
    const cpu = await createApp({ defaults: false, plugins: [], loading: { show() { status.textContent = 'CPU-only startup ready'; }, update() {} } });
    cpu.dispose();
    if (status.textContent !== 'CPU-only startup ready') throw new Error('CPU-only startup failed');
    report.cpuReady = true;
    const policy = Object.getOwnPropertyDescriptor(globalThis, '__shallotRequireHardwareGpu');
    if (!policy || policy.writable || policy.configurable) throw new Error('native GPU policy must be readonly host data');
    const frame = document.createElement('iframe');
    frame.srcdoc = '<!doctype html><title>policy context</title>';
    await new Promise((resolve) => { frame.onload = resolve; document.body.append(frame); });
    if (frame.contentWindow.__shallotRequireHardwareGpu !== report.requireHardware) throw new Error('native subframe policy differs');
    frame.remove();
    const hooks = report.hooks = [];
    let splashError;
    let gpuApp;
    try {
        gpuApp = await createApp({ defaults: false, plugins: [{ name: 'StartupProbe', gpu: {},
            prepareGpu() { hooks.push('prepare'); }, initialize() { hooks.push('initialize'); }, warm() { hooks.push('warm'); }
        }], loading: { show() {}, update() {}, error(error) { splashError = String(error); status.textContent = splashError; } } });
    } catch (error) {
        if (!report.requireHardware || !String(error).includes('Hardware WebGPU is required:')) throw error;
    }
    if (report.requireHardware && !${hardware}) {
        if (gpuApp || hooks.length || !splashError?.includes('fallback') || status.textContent !== splashError) throw new Error('native software refusal did not reach splash before plugin hooks');
        report.refusal = splashError;
    } else {
        if (!gpuApp || hooks.join(',') !== 'prepare,initialize,warm' || splashError) throw new Error('GPU startup failed');
        gpuApp.dispose();
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("no WebGPU adapter");
        const { vendor, architecture, device: deviceName, description, isFallbackAdapter } = adapter.info;
        report.adapter = { vendor, architecture, device: deviceName, description, isFallbackAdapter };
        const device = await adapter.requestDevice();
        const canvas = document.querySelector("canvas");
        const context = canvas.getContext("webgpu");
        context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: "opaque" });
        await new Promise((resolve, reject) => {
            device.lost.then((info) => reject(new Error("WebGPU device lost: " + info.message)));
            device.addEventListener("uncapturederror", (event) => reject(event.error));
            requestAnimationFrame(() => {
                try {
                    const encoder = device.createCommandEncoder();
                    const pass = encoder.beginRenderPass({ colorAttachments: [{
                        view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store",
                        clearValue: { r: 0, g: 1, b: 0, a: 1 },
                    }] });
                    pass.end();
                    device.queue.submit([encoder.finish()]);
                    const probe = document.createElement("canvas");
                    probe.width = probe.height = 1;
                    const pixels = probe.getContext("2d");
                    pixels.drawImage(canvas, 0, 0, 1, 1);
                    report.pixel = Array.from(pixels.getImageData(0, 0, 1, 1).data);
                    resolve();
                } catch (error) { reject(error); }
            });
        });
    }
} catch (error) { report.error = String(error); }
fetch(${JSON.stringify(`${observer}/marker`)}, { method: "POST", headers: { "Content-Type": "text/plain" }, body: JSON.stringify(report) }).catch((error) => console.error("smoke observer", error));`;
    writeFileSync(
        resolve(project, "index.html"),
        `<!doctype html><html><head><meta charset="utf-8"><title>Shallot native smoke</title></head><body><canvas width="64" height="64"></canvas><script type="module">${script}</script></body></html>\n`,
    );
}

export function assertCanvasImage(report: Pick<MarkerReport, "pixel" | "error" | "adapter">): void {
    if (report.error || JSON.stringify(report.pixel) !== "[0,255,0,255]") {
        throw new Error(
            `WebGPU canvas image failed: ${report.error ?? JSON.stringify(report.pixel)}; adapter: ${JSON.stringify(report.adapter)}`,
        );
    }
}

export function assertStartup(
    report: Pick<
        MarkerReport,
        "requireHardware" | "cpuReady" | "hooks" | "refusal" | "error" | "pixel" | "adapter"
    >,
    hardware = false,
): void {
    if (report.error || report.cpuReady !== true)
        throw new Error(
            `Native startup failed: ${report.error ?? "CPU-only startup was not observed"}`,
        );
    if (hardware) {
        const verdict = classifyAdapter({ present: !!report.adapter, info: report.adapter });
        if (verdict.class !== "real")
            throw new Error(`Hardware adapter required: ${verdict.reason}`);
        if (
            report.requireHardware !== true ||
            report.refusal ||
            JSON.stringify(report.hooks) !== '["prepare","initialize","warm"]'
        )
            throw new Error("Hardware startup was not observed with the default native GPU policy");
        assertCanvasImage(report);
    } else if (report.requireHardware === true) {
        if (
            JSON.stringify(report.hooks) !== "[]" ||
            !report.refusal?.includes("Hardware WebGPU is required:") ||
            !report.refusal.includes("fallback")
        )
            throw new Error("Native software refusal was not observed before plugin hooks");
    } else if (report.requireHardware === false) {
        if (JSON.stringify(report.hooks) !== '["prepare","initialize","warm"]' || report.refusal)
            throw new Error("Deliberate software startup was not observed");
        assertCanvasImage(report);
    } else {
        throw new Error("Native GPU policy was not published");
    }
}

function cliUrl(output: string): string | null {
    for (const line of output.split(/\r?\n/)) {
        const clean = line.replace(ANSI_ESCAPE, "");
        const match = clean.match(/\bLocal:\s*(https?:\/\/\S+)/);
        if (match) return match[1];
    }
    return null;
}

/** Prefix whole lines per stream: a writer may split one line across chunks, and a chunk prefix would break it. */
export function streamLines(write: (stream: "stdout" | "stderr", line: string) => void) {
    const partial = { stdout: "", stderr: "" };
    return {
        push(stream: "stdout" | "stderr", text: string) {
            const parts = (partial[stream] + text).split(/\r?\n/);
            partial[stream] = parts.pop() ?? "";
            for (const line of parts) write(stream, line);
        },
        flush() {
            for (const stream of ["stdout", "stderr"] as const)
                if (partial[stream]) write(stream, partial[stream]);
            partial.stdout = partial.stderr = "";
        },
    };
}

function assetOrigin(output: string): string | null {
    for (const line of output.split(/\r?\n/)) {
        const match = line.match(/\[shallot\] asset server on (https?:\/\/\S+)/);
        if (match) return match[1];
    }
    return null;
}

function urlOrigin(value: string, label: string): string {
    try {
        return new URL(value).origin;
    } catch {
        throw new Error(`${label} is not an absolute URL: ${value}`);
    }
}

export function assertStandaloneOrigin(
    pageHref: string,
    assetServerOrigin: string | null,
    viteOrigin: string | null,
): void {
    if (!assetServerOrigin) throw new Error("shell did not report its asset-server origin");
    const pageOrigin = urlOrigin(pageHref, "page location.href");
    const shellOrigin = urlOrigin(assetServerOrigin, "shell asset server origin");
    if (pageOrigin !== shellOrigin)
        throw new Error(`page ran at ${pageOrigin}, not the shell asset server ${shellOrigin}`);
    if (viteOrigin && pageOrigin === urlOrigin(viteOrigin, "Vite URL"))
        throw new Error("standalone preview loaded the dev Vite origin");
}

function lastDiagnostic(output: string): string | null {
    const lines = output
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
    const cause = [...lines]
        .reverse()
        .find((line) =>
            /unsupported CEF API version|CEF API mismatch|API hash|failed to initialize|404 Not Found/i.test(
                line,
            ),
        );
    return cause?.slice(-500) ?? null;
}

interface LaunchResult {
    report: MarkerReport;
    output: string;
    viteOrigin: string | null;
}

async function launchAndObserve(
    step: string,
    project: string,
    logs: string,
    observer: Observer,
    environment: NodeJS.ProcessEnv,
    mode: string,
    nonce: string,
    args: string[],
    viteOrigin: string | null,
    expected: "vite" | "asset-server",
    hardware = false,
    mac?: {
        executable: string;
        app?: string;
        portable: boolean;
        cwd?: string;
        windows?: boolean;
        direct?: boolean;
    },
): Promise<LaunchResult> {
    const logPath = resolve(logs, `${slug(step)}.log`);
    const log = createWriteStream(logPath);
    let transcript = "";
    let spawnError: Error | undefined;
    let started = false;
    let shellStartedAt: number | undefined;
    let resolveStarted!: () => void;
    const launchStart = new Promise<void>((resolveStart) => {
        resolveStarted = resolveStart;
    });
    const commandStartedAt = performance.now();
    // Register before spawn: the page can report before the CLI emits its launch line.
    const marker = observer.wait(nonce);
    const nativeLog = resolve(logs, `${slug(step)}-app.log`);
    if (mac && !mac.windows) {
        writeFileSync(nativeLog, "");
        const bin = resolve(logs, "launch-bin");
        mkdirSync(bin, { recursive: true });
        writeFileSync(
            resolve(bin, "open"),
            '#!/bin/sh\nexec /usr/bin/open --env "SHALLOT_SMOKE_OWNER=$SHALLOT_SMOKE_OWNER" --stdout "$SHALLOT_SMOKE_NATIVE_LOG" --stderr "$SHALLOT_SMOKE_NATIVE_LOG" "$@"\n',
            { mode: 0o755 },
        );
        environment = {
            ...environment,
            PATH: `${bin}${delimiter}${environment.PATH}`,
            SHALLOT_SMOKE_NATIVE_LOG: nativeLog,
            SHALLOT_SMOKE_OWNER: MAC_OWNER.split("=")[1],
        };
    }
    const child = spawn(
        mac?.direct ? mac.executable : mac?.app ? "open" : "bunx",
        mac?.direct
            ? []
            : mac?.app
              ? ["-n", "-W", "--stdout", nativeLog, "--stderr", nativeLog, mac.app]
              : ["shallot", ...args],
        {
            cwd: mac?.cwd ?? project,
            env: environment,
            detached: process.platform !== "win32",
            stdio: ["ignore", "pipe", "pipe"],
        },
    );
    const lines = streamLines((stream, line) => log.write(`[${stream}] ${line}\n`));
    const record = (stream: "stdout" | "stderr", data: Buffer) => {
        const text = data.toString();
        transcript += text;
        lines.push(stream, text);
        if (!started && /running\s+game\.\.\./.test(transcript)) {
            started = true;
            shellStartedAt = performance.now();
            resolveStarted();
        }
    };
    child.stdout.on("data", (data: Buffer) => record("stdout", data));
    child.stderr.on("data", (data: Buffer) => record("stderr", data));
    child.once("error", (error) => {
        spawnError = error;
    });
    if (mac?.app || mac?.direct) {
        started = true;
        shellStartedAt = performance.now();
        resolveStarted();
    }
    let exit: { code: number | null; signal: NodeJS.Signals | null } | undefined;
    const exited = new Promise<void>((resolveExit) => {
        child.once("close", (code, signal) => {
            exit = { code, signal };
            resolveExit();
        });
    });

    let failure: string | undefined;
    let report: MarkerReport | undefined;
    try {
        if (!child.pid) throw new Error("CLI did not start");
        const beforeLaunch = await Promise.race([
            launchStart.then(() => "launching" as const),
            exited.then(() => "exited" as const),
        ]);
        if (beforeLaunch !== "launching") {
            failure =
                spawnError?.message ??
                `CLI exited before launching the shell (${exit?.code ?? exit?.signal})`;
        } else {
            let timer: ReturnType<typeof setTimeout> | undefined;
            try {
                const result = await Promise.race([
                    marker.then((value) => ({ kind: "report" as const, value })),
                    exited.then(() => ({ kind: "exit" as const })),
                    new Promise<{ kind: "timeout" }>((resolveTimeout) => {
                        timer = setTimeout(
                            () => resolveTimeout({ kind: "timeout" }),
                            CEILING.startup,
                        );
                    }),
                ]);
                if (result.kind === "report") {
                    report = result.value;
                    if (mac?.app) {
                        const actual = realpathSync.native(mac.executable);
                        if (!actual.startsWith(`${realpathSync.native(mac.app)}/Contents/MacOS/`))
                            throw new Error(`main executable is outside copied bundle: ${actual}`);
                        const pid = mainShellPid(child.pid, mac.executable);
                        if (!pid)
                            throw new Error(
                                `running main executable was not found inside copied bundle: ${mac.executable}`,
                            );
                        log.write(
                            `[identity] running main pid=${pid} executable=${realpathSync.native(mac.executable)}\n`,
                        );
                        console.info(
                            `${step}: running main pid=${pid} executable=${realpathSync.native(mac.executable)}`,
                        );
                    }
                } else if (result.kind === "timeout")
                    failure = `page script did not report its nonce within CEILING.startup (${CEILING.startup}ms)`;
                else
                    failure =
                        spawnError?.message ??
                        `CLI exited before the page marker (${exit?.code ?? exit?.signal})`;
            } finally {
                if (timer) clearTimeout(timer);
            }
        }
    } catch (error) {
        failure = error instanceof Error ? error.message : String(error);
    } finally {
        try {
            if (child.pid) {
                await stopMainShell(child.pid, mac?.executable ?? shellExecutable(project, mode));
                await Promise.race([exited, Bun.sleep(2000)]);
            }
        } finally {
            if (child.pid) await terminateProcessGroup(child.pid);
            if (mac?.windows) await stopWindowsProcesses(dirname(project));
            else if (mac) await stopMacOwnedProcesses(dirname(project));
            await Promise.race([exited, Bun.sleep(1500)]);
            lines.flush();
            if (mac && !mac.windows) log.write(readFileSync(nativeLog, "utf8"));
            await new Promise<void>((resolveEnd) => log.end(resolveEnd));
        }
    }

    const output = readFileSync(logPath, "utf8");
    const origin = assetOrigin(output);
    if (failure) {
        const index = standaloneIndex(project, mode, mac);
        const missingAssets = index && !existsSync(index);
        const cause = lastDiagnostic(output);
        throw new SmokeFailure(
            step,
            logPath,
            `${missingAssets ? "standalone bundle is missing dist/index.html; " : ""}${failure}${cause ? `; last native diagnostic: ${cause}` : ""}`,
        );
    }
    if (!report) throw new SmokeFailure(step, logPath, "observer returned no report");
    if (report.nonce !== nonce || report.mode !== mode)
        throw new SmokeFailure(
            step,
            logPath,
            `observer marker did not match this launch (mode ${report.mode})`,
        );
    try {
        assertStartup(report, hardware);
    } catch (error) {
        throw new SmokeFailure(
            step,
            logPath,
            error instanceof Error ? error.message : String(error),
        );
    }
    const observedOrigin = urlOrigin(report.href, "page location.href");
    const resolvedViteOrigin = viteOrigin ?? (expected === "vite" ? cliUrl(output) : null);
    if (expected === "vite") {
        if (!resolvedViteOrigin || observedOrigin !== urlOrigin(resolvedViteOrigin, "Vite URL"))
            throw new SmokeFailure(
                step,
                logPath,
                `page ran at ${observedOrigin}, not the dev Vite origin ${resolvedViteOrigin ?? "(not reported)"}`,
            );
    } else {
        try {
            if (mac?.windows && !mac.portable) {
                if (new URL(report.href).origin !== "http://shallot.localhost")
                    throw new Error(`unexpected WebView2 origin: ${report.href}`);
            } else if (mac && !mac.portable) assertSystemOrigin(report.href);
            else assertStandaloneOrigin(report.href, origin, viteOrigin);
        } catch (error) {
            throw new SmokeFailure(
                step,
                logPath,
                error instanceof Error ? error.message : String(error),
            );
        }
    }
    if (report.requireHardware !== (environment.SHALLOT_ALLOW_SOFTWARE_GPU !== "1"))
        throw new SmokeFailure(step, logPath, "native GPU policy did not match launch environment");
    const prelaunchMs = (shellStartedAt ?? performance.now()) - commandStartedAt;
    const durationMs = performance.now() - (shellStartedAt ?? commandStartedAt);
    console.info(`${step}: CLI work before shell launch ${(prelaunchMs / 1000).toFixed(2)}s`);
    console.info(
        `${step}: CPU startup, ${report.refusal ? "software refusal before plugin hooks" : "GPU startup and WebGPU canvas readback"} verified in ${(durationMs / 1000).toFixed(2)}s at ${report.href}; marker ${report.nonce}/${report.run}; GPU: ${report.refusal ?? JSON.stringify(report.adapter)}; runtime: ${report.userAgent} (log: ${logPath})`,
    );
    return { report, output, viteOrigin: resolvedViteOrigin };
}

function nativeOutput(project: string, release: boolean, dev = false): string {
    return resolve(
        project,
        "build",
        "linux",
        `${release ? "release" : "debug"}-portable${dev ? "-dev" : ""}`,
    );
}

function shellExecutable(project: string, mode: string): string {
    const outputs =
        mode === "dev"
            ? [nativeOutput(project, false, true), nativeOutput(project, false)]
            : [nativeOutput(project, mode === "release-preview")];
    const shells = outputs.map((directory) => resolve(directory, basename(project)));
    return shells.find(existsSync) ?? shells[0];
}

async function bundleIdentity(
    label: string,
    project: string,
    logs: string,
    env: NodeJS.ProcessEnv,
    release: boolean,
    dev = false,
): Promise<void> {
    const candidates = dev
        ? [nativeOutput(project, release, true), nativeOutput(project, release)]
        : [nativeOutput(project, release)];
    const outputDir =
        candidates.find(
            (candidate) =>
                existsSync(resolve(candidate, basename(project))) &&
                existsSync(resolve(candidate, "cef/libcef.so")),
        ) ?? candidates[0];
    const shell = resolve(outputDir, basename(project));
    const cef = resolve(outputDir, "cef/libcef.so");
    if (!existsSync(shell) || !existsSync(cef))
        throw new SmokeFailure(
            label,
            resolve(logs, `${slug(label)}.log`),
            "bundle is missing its shell or bundled cef/libcef.so",
        );
    const hash = createHash("sha256").update(readFileSync(shell)).digest("hex");
    const code =
        "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); f=lib.cef_version_info; f.argtypes=[ctypes.c_int]; f.restype=ctypes.c_int; print('.'.join(str(f(i)) for i in range(3)))";
    const version = await runLogged(
        `${label} CEF identity`,
        "python3",
        ["-c", code, cef],
        project,
        {
            ...env,
            LD_LIBRARY_PATH: [dirname(cef), env.LD_LIBRARY_PATH].filter(Boolean).join(delimiter),
        },
        logs,
    );
    console.info(`${label}: shell sha256=${hash} CEF=${version.output.trim()} (${outputDir})`);
}

function plantStaleArtifacts(consumer: string, project: string, targetDir: string): void {
    const buildRoot = resolve(targetDir, LINUX_TARGET);
    const releaseBuild = resolve(buildRoot, "release", "build");
    for (const name of ["cef-dll-sys-000-smoke-stale", "cef-dll-sys-001-sibling-stale"]) {
        const runtime = resolve(releaseBuild, name, "out", "cef_linux_x86_64");
        mkdirSync(runtime, { recursive: true });
        writeFileSync(resolve(runtime, "libcef.so"), `stale ${name}\n`);
    }
    for (const bundle of ["debug-portable", "debug-portable-dev", "release-portable"]) {
        const staleBundle = resolve(project, "build/linux", bundle);
        mkdirSync(resolve(staleBundle, "cef"), { recursive: true });
        writeFileSync(resolve(staleBundle, "cef/libcef.so"), "stale consumer bundle\n");
        mkdirSync(resolve(staleBundle, "dist"), { recursive: true });
        writeFileSync(resolve(staleBundle, "dist/index.html"), "stale page\n");
    }
    console.info(`planted stale release runtimes and consumer bundles under ${consumer}`);
}

function plantOlderRuntime(targetDir: string, source: string): void {
    if (!existsSync(resolve(source, "libcef.so")))
        throw new Error(`older runtime source has no libcef.so: ${source}`);
    const destination = resolve(
        targetDir,
        LINUX_TARGET,
        "release/build/cef-dll-sys-000-older-runtime/out/cef_linux_x86_64",
    );
    mkdirSync(dirname(destination), { recursive: true });
    cpSync(source, destination, { recursive: true, dereference: false });
    console.info(`copied older CEF runtime into scratch target: ${destination}`);
}

async function prepareTarball(
    consumer: string,
    logs: string,
    suppliedTarball: string | undefined,
    env: NodeJS.ProcessEnv,
): Promise<string> {
    const sourceTarball = resolve(consumer, "shallot-source.tgz");
    if (suppliedTarball) {
        if (!existsSync(suppliedTarball))
            throw new Error(`tarball does not exist: ${suppliedTarball}`);
        cpSync(suppliedTarball, sourceTarball);
    } else {
        const packDir = resolve(consumer, ".pack-source");
        rmSync(packDir, { recursive: true, force: true });
        mkdirSync(packDir, { recursive: true });
        await runLogged(
            "pack checkout",
            process.execPath,
            ["pm", "pack", "--ignore-scripts", "--destination", packDir, "--quiet"],
            ROOT,
            env,
            logs,
        );
        const archive = readdirSync(packDir).find((entry) => entry.endsWith(".tgz"));
        if (!archive) throw new Error(`bun pm pack produced no tarball in ${packDir}`);
        cpSync(resolve(packDir, archive), sourceTarball);
    }

    const unpacked = resolve(consumer, ".pack-stamp");
    rmSync(unpacked, { recursive: true, force: true });
    mkdirSync(unpacked, { recursive: true });
    await runLogged(
        "unpack package",
        "tar",
        ["-xzf", sourceTarball, "-C", unpacked],
        consumer,
        env,
        logs,
    );
    const packageDir = resolve(unpacked, "package");
    const manifestPath = resolve(packageDir, "package.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as { version?: string };
    if (!manifest.version || typeof manifest.version !== "string")
        throw new Error("packed package has no string version to stamp");
    const originalVersion = manifest.version.split("+")[0];
    const unpublishedVersion = `${originalVersion}-native-smoke.${Date.now().toString(36)}.${process.pid}`;
    manifest.version = unpublishedVersion;
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    const stamped = resolve(consumer, "shallot-smoke.tgz");
    await runLogged(
        "stamp unpublished package",
        "tar",
        ["-czf", stamped, "-C", unpacked, "package"],
        consumer,
        env,
        logs,
    );
    console.info(
        `packed source version ${originalVersion} stamped unpublished as ${unpublishedVersion}`,
    );
    return stamped;
}

async function createConsumer(
    consumer: string,
    logs: string,
    tarball: string,
    env: NodeJS.ProcessEnv,
): Promise<{ project: string; targetDir: string }> {
    const project = resolve(consumer, "game");
    const targetDir = resolve(consumer, "cargo-target");
    rmSync(project, { recursive: true, force: true });
    mkdirSync(project, { recursive: true });
    const manifest = {
        name: "shallot-native-smoke",
        version: "1.0.0",
        private: true,
        type: "module",
        dependencies: {
            "@dylanebert/shallot": `file:../${basename(tarball)}`,
            vite: "^8.3.1",
        },
    };
    writeFileSync(resolve(project, "package.json"), `${JSON.stringify(manifest, null, 2)}\n`);
    writeFileSync(
        resolve(project, "vite.config.ts"),
        `import { shallot } from '@dylanebert/shallot/vite';\nexport default { plugins: [shallot()], server: { headers: ${JSON.stringify(ISOLATION_HEADERS)} }, preview: { headers: ${JSON.stringify(ISOLATION_HEADERS)} } };\n`,
    );
    await runLogged(
        "install packed consumer",
        process.execPath,
        ["install", "--prefer-offline", "--no-progress"],
        project,
        env,
        logs,
    );
    const installed = resolve(project, "node_modules/@dylanebert/shallot");
    const nativeCrate = resolve(installed, "crates/native");
    if (!existsSync(nativeCrate))
        throw new Error(`packed consumer has no native crate: ${nativeCrate}`);
    const targetLink = resolve(nativeCrate, "target");
    rmSync(targetLink, { recursive: true, force: true });
    mkdirSync(targetDir, { recursive: true });
    symlinkSync(targetDir, targetLink, process.platform === "win32" ? "junction" : "dir");
    console.info(`Cargo target cache: ${targetDir}`);
    return { project, targetDir };
}

async function runSmoke(options: ReturnType<typeof parseArgs>, icd: string): Promise<void> {
    if (process.platform !== "linux") throw new Error("native Linux smoke must run on Linux");
    const consumer = resolve(options.consumer!);
    const fromRoot = normalizePath(relative(ROOT, consumer));
    const insideCheckout =
        fromRoot === "" ||
        (!isAbsolute(fromRoot) && fromRoot !== ".." && !fromRoot.startsWith("../"));
    if (insideCheckout) throw new Error(`--consumer must be outside the checkout: ${consumer}`);
    const logs = resolve(options.logs ?? resolve(consumer, "logs"));
    mkdirSync(consumer, { recursive: true });
    mkdirSync(logs, { recursive: true });
    const runtimeData = resolve(consumer, "runtime-data");
    rmSync(runtimeData, { recursive: true, force: true });
    const env = cleanEnvironment(process.env, {
        XDG_DATA_HOME: runtimeData,
        XDG_SESSION_TYPE: "x11",
        ...(options.hardware ? {} : { VK_ICD_FILENAMES: icd, LIBGL_ALWAYS_SOFTWARE: "1" }),
    });
    if (options.hardware) {
        delete env.LIBGL_ALWAYS_SOFTWARE;
    }

    const tarball = await prepareTarball(consumer, logs, options.tarball, env);
    const { project, targetDir } = await createConsumer(consumer, logs, tarball, env);
    if (options.staleRuntime) plantOlderRuntime(targetDir, resolve(options.staleRuntime));
    else if (options.plantStale) plantStaleArtifacts(consumer, project, targetDir);

    const display = await startVirtualDisplay(logs, env);
    let observer: Observer | undefined;
    let stopCompositor: (() => Promise<void>) | undefined;
    try {
        const defaultEnv = cleanEnvironment(env, { DISPLAY: display.display });
        delete defaultEnv.SHALLOT_ALLOW_SOFTWARE_GPU;
        const launchEnv = options.hardware
            ? defaultEnv
            : { ...defaultEnv, SHALLOT_ALLOW_SOFTWARE_GPU: "1" };
        stopCompositor = await startCompositor(logs, launchEnv);
        observer = await startObserver(logs);
        let devOrigin: string | null = null;
        const debugNonce = randomUUID();
        writePage(project, observer.origin, "debug-preview", debugNonce, options.hardware);
        const debugBuild = await runLogged(
            "build debug standalone",
            "bunx",
            ["shallot", "build", "--target", "linux", "--portable"],
            project,
            env,
            logs,
        );
        if (!debugBuild.output.includes("done."))
            throw new SmokeFailure(
                "build debug standalone",
                resolve(logs, "build-debug-standalone.log"),
                "installed CLI did not report a completed build",
            );
        await bundleIdentity("debug standalone bundle", project, logs, env, false);

        if (!options.hardware)
            await launchAndObserve(
                "launch default hardware policy",
                project,
                logs,
                observer,
                defaultEnv,
                "debug-preview",
                debugNonce,
                ["preview", "--target", "linux", "--portable"],
                null,
                "asset-server",
            );

        const devNonce = randomUUID();
        writePage(project, observer.origin, "dev", devNonce, options.hardware);
        const dev = await launchAndObserve(
            "launch dev",
            project,
            logs,
            observer,
            launchEnv,
            "dev",
            devNonce,
            ["dev", "--target", "linux", "--portable"],
            null,
            "vite",
            options.hardware,
        );
        devOrigin = dev.viteOrigin;
        if (!devOrigin)
            throw new SmokeFailure(
                "launch dev",
                resolve(logs, "launch-dev.log"),
                "Vite did not report a local URL",
            );
        const devOutputAssetOrigin = assetOrigin(dev.output);
        if (
            devOutputAssetOrigin &&
            urlOrigin(devOutputAssetOrigin, "dev shell asset origin") ===
                urlOrigin(devOrigin, "Vite URL")
        )
            throw new SmokeFailure(
                "launch dev",
                resolve(logs, "launch-dev.log"),
                "dev page unexpectedly used the shell asset server",
            );
        await bundleIdentity("dev shell bundle", project, logs, env, false, true);

        await launchAndObserve(
            "launch debug preview",
            project,
            logs,
            observer,
            launchEnv,
            "debug-preview",
            debugNonce,
            ["preview", "--target", "linux", "--portable"],
            devOrigin,
            "asset-server",
            options.hardware,
        );

        const releaseNonce = randomUUID();
        writePage(project, observer.origin, "release-preview", releaseNonce, options.hardware);
        const releaseBuild = await runLogged(
            "build source release standalone",
            "bunx",
            ["shallot", "build", "--target", "linux", "--portable", "--release"],
            project,
            env,
            logs,
        );
        const releaseLog = resolve(logs, "build-source-release-standalone.log");
        if (
            !/prebuilt:\s+(?:no matching release|offline|source checkout|checksum mismatch|extract failed|cache error)/i.test(
                releaseBuild.output,
            )
        )
            throw new SmokeFailure(
                "build source release standalone",
                releaseLog,
                "build output did not prove a prebuilt miss and source fallback",
            );
        if (/prebuilt:\s+(?:cache hit|downloaded \+ SHA256 verified)/i.test(releaseBuild.output))
            throw new SmokeFailure(
                "build source release standalone",
                releaseLog,
                "a prebuilt shell was used instead of the candidate source release",
            );
        if (!releaseBuild.output.includes("done."))
            throw new SmokeFailure(
                "build source release standalone",
                releaseLog,
                "installed CLI did not report a completed build",
            );
        await bundleIdentity("release source bundle", project, logs, env, true);

        await launchAndObserve(
            "launch release preview",
            project,
            logs,
            observer,
            launchEnv,
            "release-preview",
            releaseNonce,
            ["preview", "--target", "linux", "--portable", "--release"],
            devOrigin,
            "asset-server",
            options.hardware,
        );
    } finally {
        try {
            if (observer) await observer.close();
        } finally {
            try {
                if (stopCompositor) await stopCompositor();
            } finally {
                await display.stop();
            }
        }
    }
}

export function assertSystemOrigin(href: string): void {
    const url = new URL(href);
    if (url.protocol !== "shallot:" || url.host !== "localhost")
        throw new Error(`page ran at ${href}, not shallot://localhost/`);
}

/** The debug preview's bundled page; release builds embed theirs. */
export function standaloneIndex(
    project: string,
    mode: string,
    mac?: { executable: string; windows?: boolean },
): string | null {
    if (mode !== "debug-preview") return null;
    return mac
        ? resolve(
              dirname(mac.executable),
              mac.windows ? "dist/index.html" : "../Resources/dist/index.html",
          )
        : resolve(project, "build/linux/debug-portable/dist/index.html");
}

export function macApp(project: string, portable: boolean, release: boolean, dev = false): string {
    return resolve(
        project,
        "build/mac",
        `${release ? "release" : "debug"}-${portable ? "portable" : "system"}${dev ? "-dev" : ""}`,
        `${basename(project)}.app`,
    );
}

async function runMacSmoke(options: ReturnType<typeof parseArgs>): Promise<void> {
    if (process.arch !== "arm64") throw new Error("Mac smoke requires aarch64-apple-darwin");
    if (options.plantStale || options.staleRuntime)
        throw new Error("stale runtime fixtures are Linux-only");
    const consumer = resolve(options.consumer!);
    const rel = normalizePath(relative(ROOT, consumer));
    if (!rel || (!isAbsolute(rel) && rel !== ".." && !rel.startsWith("../")))
        throw new Error("--consumer must be outside the checkout");
    const logs = resolve(options.logs ?? resolve(consumer, "logs"));
    mkdirSync(logs, { recursive: true });
    const env = cleanEnvironment(process.env);
    delete env.SHALLOT_ALLOW_SOFTWARE_GPU;
    const display = await runLogged(
        "Mac hardware identity",
        "system_profiler",
        ["SPDisplaysDataType"],
        consumer,
        env,
        logs,
    );
    console.info(display.output);
    const tarball = await prepareTarball(consumer, logs, options.tarball, env);
    const { project } = await createConsumer(consumer, logs, tarball, env);
    const observer = await startObserver(logs);
    const findings: string[] = [];
    try {
        if (options.portableOnly) console.info("WKWebView: not run (--portable-only)");
        for (const portable of options.portableOnly ? [true] : [false, true]) {
            const backend = portable ? "CEF" : "WKWebView";
            const flags = ["--target", "mac", ...(portable ? ["--portable"] : [])];
            let viteOrigin: string | null = null;
            try {
                for (const mode of ["dev", "debug-preview", "release-preview"] as const) {
                    const nonce = randomUUID();
                    writePage(project, observer.origin, mode, nonce, true);
                    if (process.env.SHALLOT_SMOKE_WITHHOLD_MARKER === "1")
                        writeFileSync(
                            resolve(project, "index.html"),
                            "<!doctype html><title>Withheld marker</title>",
                        );
                    const release = mode === "release-preview";
                    if (mode !== "dev") {
                        const build = await runLogged(
                            `${backend} build ${mode}`,
                            "bunx",
                            ["shallot", "build", ...flags, ...(release ? ["--release"] : [])],
                            project,
                            env,
                            logs,
                        );
                        if (
                            release &&
                            !/prebuilt:\s+(?:no matching release|offline|source checkout|checksum mismatch|extract failed|cache error)/i.test(
                                build.output,
                            )
                        )
                            throw new Error("release did not prove a source fallback");
                    }
                    const app = macApp(project, portable, release, mode === "dev");
                    const executable = resolve(app, "Contents/MacOS", basename(project));
                    const result = await launchAndObserve(
                        `${backend} launch ${mode}`,
                        project,
                        logs,
                        observer,
                        env,
                        mode,
                        nonce,
                        [
                            mode === "dev" ? "dev" : "preview",
                            ...flags,
                            ...(release ? ["--release"] : []),
                        ],
                        viteOrigin,
                        mode === "dev" ? "vite" : "asset-server",
                        true,
                        { executable, portable },
                    );
                    const plist = portable
                        ? resolve(
                              app,
                              "Contents/Frameworks/Chromium Embedded Framework.framework/Resources/Info.plist",
                          )
                        : "/System/Library/Frameworks/WebKit.framework/Resources/Info.plist";
                    const identity = await runLogged(
                        `${backend} ${mode} runtime identity`,
                        "/usr/libexec/PlistBuddy",
                        ["-c", "Print :CFBundleShortVersionString", plist],
                        project,
                        env,
                        logs,
                    );
                    console.info(`${backend} ${mode} runtime version: ${identity.output.trim()}`);
                    if (mode === "dev") viteOrigin = result.viteOrigin;
                    if (release) {
                        const copy = resolve(
                            consumer,
                            `installed-${backend}`,
                            `${basename(project)}.app`,
                        );
                        mkdirSync(dirname(copy), { recursive: true });
                        rmSync(copy, { recursive: true, force: true });
                        await runLogged(
                            `${backend} copy release`,
                            "ditto",
                            [app, copy],
                            project,
                            env,
                            logs,
                        );
                        // Relocation does not require downloaded-app execution.
                        spawnSync("xattr", ["-dr", "com.apple.quarantine", copy]);
                        const quarantine = spawnSync("xattr", ["-lr", copy], { encoding: "utf8" });
                        if (
                            quarantine.status !== 0 ||
                            quarantine.stdout.includes("com.apple.quarantine")
                        )
                            throw new Error(
                                `copied launch must not carry quarantine: ${quarantine.stderr}`,
                            );
                        const copiedExecutable = resolve(copy, "Contents/MacOS", basename(project));
                        const cwd = resolve(consumer, "unrelated-launch-directory");
                        mkdirSync(cwd, { recursive: true });
                        const launchCopy = (step: string) =>
                            launchAndObserve(
                                step,
                                project,
                                logs,
                                observer,
                                env,
                                mode,
                                nonce,
                                [],
                                viteOrigin,
                                "asset-server",
                                true,
                                { executable: copiedExecutable, app: copy, portable, cwd },
                            );
                        await withoutProject(project, async () => {
                            console.info(
                                `${backend} copied release: original project/build/assets unavailable; cwd=${cwd}`,
                            );
                            await launchCopy(`${backend} open copied release`);
                        });
                        const savedExecutable = `${copiedExecutable}.independent`;
                        renameSync(copiedExecutable, savedExecutable);
                        symlinkSync(executable, copiedExecutable);
                        try {
                            if (
                                realpathSync.native(copiedExecutable) !==
                                realpathSync.native(executable)
                            )
                                throw new Error(
                                    "dependent-copy control does not point to source executable",
                                );
                            await withoutProject(project, async () => {
                                if (existsSync(copiedExecutable))
                                    throw new Error("dependent executable remained available");
                                try {
                                    await launchCopy(`${backend} dependent copy negative control`);
                                } catch (error) {
                                    if (
                                        !(error instanceof SmokeFailure) ||
                                        !/CLI exited before the page marker/.test(error.message)
                                    )
                                        throw error;
                                    console.info(
                                        `${backend} dependent copy negative control: expected RED: ${error.message}`,
                                    );
                                    return;
                                }
                                throw new Error("build-tree-dependent copy unexpectedly passed");
                            });
                        } finally {
                            rmSync(copiedExecutable, { force: true });
                            renameSync(savedExecutable, copiedExecutable);
                        }
                        const assessmentCopy = resolve(
                            consumer,
                            `gatekeeper-${backend}`,
                            `${basename(project)}.app`,
                        );
                        rmSync(dirname(assessmentCopy), { recursive: true, force: true });
                        mkdirSync(dirname(assessmentCopy), { recursive: true });
                        cpSync(copy, assessmentCopy, { recursive: true, verbatimSymlinks: true });
                        await runLogged(
                            `${backend} quarantine copy`,
                            "xattr",
                            [
                                "-w",
                                "com.apple.quarantine",
                                `0083;${Math.floor(Date.now() / 1000).toString(16)};ShallotSmoke;${randomUUID()}`,
                                assessmentCopy,
                            ],
                            project,
                            env,
                            logs,
                        );
                        const verdict = spawnSync(
                            "spctl",
                            ["--assess", "--type", "execute", "-v", assessmentCopy],
                            { encoding: "utf8" },
                        );
                        if (verdict.error) throw verdict.error;
                        const evidence = `Gatekeeper ${backend}: spctl --assess --type execute -v ${assessmentCopy}; exit ${verdict.status}; ${verdict.stdout}${verdict.stderr}`;
                        writeFileSync(resolve(logs, `${backend}-gatekeeper.log`), evidence);
                        console.info(evidence);
                    }
                }
            } catch (error) {
                const finding = `${backend}: ${error instanceof Error ? error.message : String(error)}`;
                findings.push(finding);
                console.error(finding);
            }
        }
    } finally {
        await observer.close();
    }
    assertNoSmokeProcesses(macSmokePids(consumer));
    if (findings.length) throw new Error(findings.join("\n"));
    console.info(
        `Mac packed native smoke passed: hardware startup and canvas readback in ${options.portableOnly ? "CEF" : "both backends"}, dev, debug/release preview and copied release via open; Gatekeeper verdicts recorded.`,
    );
}

interface WindowsProcess {
    processId: number;
    executablePath?: string;
    commandLine?: string;
}

function windowsProcesses(): WindowsProcess[] {
    const result = spawnSync(
        "powershell.exe",
        [
            "-NoProfile",
            "-Command",
            "ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process | Select-Object processId,executablePath,commandLine)",
        ],
        { encoding: "utf8" },
    );
    if (result.status !== 0)
        throw new Error(`cannot inventory Windows processes: ${result.stderr}`);
    return JSON.parse(result.stdout);
}

export function windowsSmokePids(consumer: string): number[] {
    const root = `${normalizePath(resolve(consumer)).toLowerCase()}/`;
    return windowsProcesses()
        .filter((row) => {
            const executable = normalizePath(row.executablePath ?? "").toLowerCase();
            const command = normalizePath(row.commandLine ?? "").toLowerCase();
            return (
                row.processId !== process.pid &&
                (executable.startsWith(root) || command.includes(root))
            );
        })
        .map((row) => row.processId);
}

async function stopWindowsProcesses(consumer: string): Promise<void> {
    for (const pid of windowsSmokePids(consumer)) await terminateProcessGroup(pid);
    const deadline = Date.now() + 5000;
    while (windowsSmokePids(consumer).length && Date.now() < deadline) await Bun.sleep(100);
    assertNoSmokeProcesses(windowsSmokePids(consumer));
}

export function windowsOutput(
    project: string,
    portable: boolean,
    release: boolean,
    dev = false,
): string {
    return resolve(
        project,
        "build/windows",
        `${release ? "release" : "debug"}-${portable ? "portable" : "system"}${dev ? "-dev" : ""}`,
    );
}

async function runWindowsSmoke(options: ReturnType<typeof parseArgs>): Promise<void> {
    if (process.arch !== "x64") throw new Error("Windows smoke requires x86_64-pc-windows-msvc");
    if (options.plantStale || options.staleRuntime)
        throw new Error("stale fixtures are Linux-only");
    const consumer = resolve(options.consumer!);
    const rel = normalizePath(relative(ROOT, consumer));
    if (!rel || (!isAbsolute(rel) && rel !== ".." && !rel.startsWith("../")))
        throw new Error("--consumer must be outside the checkout");
    const logs = resolve(options.logs ?? resolve(consumer, "logs"));
    mkdirSync(logs, { recursive: true });
    const env = cleanEnvironment(process.env, { SHALLOT_ALLOW_SOFTWARE_GPU: "1" });
    console.info(
        "Windows native: software GPU explicitly allowed (SHALLOT_ALLOW_SOFTWARE_GPU=1); adapter identity follows each marker.",
    );
    const tarball = await prepareTarball(consumer, logs, options.tarball, env);
    const { project } = await createConsumer(consumer, logs, tarball, env);
    const observer = await startObserver(logs);
    try {
        if (options.portableOnly) console.info("WebView2: not run (--portable-only)");
        for (const portable of options.portableOnly ? [true] : [false, true]) {
            const backend = portable ? "CEF" : "WebView2";
            const flags = ["--target", "windows", ...(portable ? ["--portable"] : [])];
            let viteOrigin: string | null = null;
            for (const mode of ["dev", "debug-preview", "release-preview"] as const) {
                const nonce = randomUUID();
                writePage(project, observer.origin, mode, nonce);
                if (process.env.SHALLOT_SMOKE_WITHHOLD_MARKER === "1")
                    writeFileSync(
                        resolve(project, "index.html"),
                        "<!doctype html><title>Withheld marker</title>",
                    );
                const release = mode === "release-preview";
                if (mode !== "dev") {
                    const build = await runLogged(
                        `${backend} build ${mode}`,
                        "bunx",
                        ["shallot", "build", ...flags, ...(release ? ["--release"] : [])],
                        project,
                        env,
                        logs,
                    );
                    if (
                        release &&
                        !/prebuilt:\s+(?:no matching release|offline|source checkout|checksum mismatch|extract failed|cache error)/i.test(
                            build.output,
                        )
                    )
                        throw new Error("release did not prove a source fallback");
                }
                const output = windowsOutput(project, portable, release, mode === "dev");
                const executable = resolve(output, `${basename(project)}.exe`);
                const result = await launchAndObserve(
                    `${backend} launch ${mode}`,
                    project,
                    logs,
                    observer,
                    env,
                    mode,
                    nonce,
                    [
                        mode === "dev" ? "dev" : "preview",
                        ...flags,
                        ...(release ? ["--release"] : []),
                    ],
                    viteOrigin,
                    mode === "dev" ? "vite" : "asset-server",
                    false,
                    { executable, portable, windows: true },
                );
                if (mode === "dev") viteOrigin = result.viteOrigin;
                if (release) {
                    const copy = resolve(consumer, `installed-${backend}`);
                    rmSync(copy, { recursive: true, force: true });
                    cpSync(output, copy, { recursive: true });
                    const cwd = resolve(consumer, "unrelated-launch-directory");
                    mkdirSync(cwd, { recursive: true });
                    await withoutProject(project, async () => {
                        console.info(
                            `${backend} copied release: source project unavailable; cwd=${cwd}`,
                        );
                        await launchAndObserve(
                            `${backend} copied release`,
                            project,
                            logs,
                            observer,
                            env,
                            mode,
                            nonce,
                            [],
                            viteOrigin,
                            "asset-server",
                            false,
                            {
                                executable: resolve(copy, `${basename(project)}.exe`),
                                portable,
                                windows: true,
                                direct: true,
                                cwd,
                            },
                        );
                    });
                }
            }
        }
    } finally {
        await stopWindowsProcesses(consumer);
        await observer.close();
        assertNoSmokeProcesses(windowsSmokePids(consumer));
    }
    console.info(
        "Windows packed native smoke passed: dev, debug/release preview and copied release with source unavailable.",
    );
}

function usage(): string {
    return `Usage: bun scripts/native-smoke.ts --consumer <directory> [options]\n\nOptions:\n  --consumer <directory>  Scratch consumer/cache root, outside the checkout (required)\n  --logs <directory>      Retained step logs (default: <consumer>/logs)\n  --tarball <path>        Test an existing Shallot npm tarball instead of packing this checkout\n  --hardware              Require hardware on Linux's owned Xvfb display (Mac always requires hardware)\n  --portable-only         Mac/Windows: run CEF alone; the log names the omitted system backend\n  --plant-stale           Plant stale runtimes and bundles to exercise warm-cache selection\n  --stale-runtime <path>  Copy an older libcef.so tree into the release-profile Cargo output\n  --help                  Show this help\n\nLinux setup: ${SETUP_COMMAND}\nMac: Apple Silicon; system WKWebView and portable CEF, copied release via open, quarantined Gatekeeper assessment.\nWindows: x86_64-pc-windows-msvc; system WebView2 and portable CEF, copied release with source unavailable. Software GPU explicitly allowed.\nSet SHALLOT_SMOKE_WITHHOLD_MARKER=1 on Mac/Windows to plant a missing-marker failure.\n`;
}

async function main(): Promise<void> {
    let options: ReturnType<typeof parseArgs>;
    try {
        options = parseArgs(Bun.argv.slice(2));
    } catch (error) {
        console.error(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
        process.exitCode = 2;
        return;
    }
    if (options.help) {
        console.info(usage());
        return;
    }
    try {
        if (process.platform === "win32") {
            await runWindowsSmoke(options);
            return;
        }
        if (process.platform === "darwin") {
            await runMacSmoke(options);
            return;
        }
        const icd = requireLinuxPrerequisites(defaultProbe, options.hardware);
        console.info(
            `setup: ${options.hardware ? `composited Xvfb; hardware required (Vulkan ICD: ${process.env.VK_ICD_FILENAMES ?? "system loader default"})` : `composited Xvfb + Mesa software Vulkan (${icd})`}; launch ceiling ${CEILING.startup}ms; CEF download and native compile use the CI job timeout`,
        );
        await runSmoke(options, icd);
        console.info(
            `Linux packed native smoke passed: ${options.hardware ? "hardware startup" : "default software refusal; deliberate software startup"} and canvas readback in dev, debug preview, and source-release preview.`,
        );
    } catch (error) {
        console.error(`FAIL: ${error instanceof Error ? error.message : String(error)}`);
        process.exitCode = 1;
    }
}

if (import.meta.main) await main();
