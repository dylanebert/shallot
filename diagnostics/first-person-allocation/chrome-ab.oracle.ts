// Stage 4 spike: does desktop Chrome 153 fault on Omarchy's RTX 4090 with any WebGPU page, or only with
// first-person's? Alternates desktop Chrome 153 and Chromium 152, each opening, in its own fresh browser,
// (a) first-person's built page and (b) a control page with no Shallot, on the display seat's monitor.
// Each launch records the adapter the page itself acquired, the WebGPU backend chrome://gpu reports, the
// frames the page submitted, any device loss with its message, and the kernel's NVRM lines over its window.
//
// Run by path on the seat host, only when the person starts it; it takes the declared monitor while it runs:
//   SHALLOT_DISPLAY_SEAT=<monitor> bun diagnostics/first-person-allocation/chrome-ab.oracle.ts [--dry]
//     [--arms=<project>,<project>…]
// `--dry` does everything but open a browser. `--arms` narrows instead: each named project, a first-person
// variant with one candidate disabled, is built beside first-person (the baseline) and the control, and
// every page gets ARM_LAUNCHES launches, interleaved by round, alternating browser by round.
//
// The browsers are the desktop's own wrappers, so they read the same Omarchy flag files the failing spike
// runs read, each with a fresh profile and attached over CDP as the spike was. The pages request no input,
// so the run never asks for pointer lock.

import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { resolveViteCli } from "../../src/project/vite-command";
import {
    confirmOnDisplay,
    declaredMonitor,
    openOnDisplay,
    parseMonitors,
    SEAT_WINDOW_CLASS,
} from "./display-seat";

const BROWSERS = [
    { name: "chrome", path: "/usr/bin/google-chrome-stable", major: "153" },
    { name: "chromium", path: "/usr/bin/chromium", major: "152" },
] as const;
type BrowserName = (typeof BROWSERS)[number]["name"];
/** "first-person", "control", or an arm's project name */
type PageName = string;

const ARMS = (process.argv.find((arg) => arg.startsWith("--arms="))?.slice("--arms=".length) ?? "")
    .split(",")
    .filter(Boolean);
/** runs per browser in the A/B; each run opens both pages, each in its own launch. */
const RUNS = 3;
/** launches per page when narrowing: at a baseline rate near 4 in 19, 14 clean launches have about a 4%
 * chance of all missing the fault. */
const ARM_LAUNCHES = 14;
/** how long each page is observed once loaded: the spike's losses all came within ten seconds of launch. */
const OBSERVE_MS = ARMS.length > 0 ? 12_000 : 15_000;
/** one launch's bound: start, load, observe, read chrome://gpu, close. */
const LAUNCH_MS = ARMS.length > 0 ? 40_000 : 45_000;
/** the whole run's bound: the A/B's 12 launches at their bound plus the build; 25 minutes when narrowing. */
const TOTAL_MS = ARMS.length > 0 ? 25 * 60_000 : 12 * LAUNCH_MS + 60_000;
const PORT = 4177;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const PROJECT = resolve(import.meta.dir, "../../examples/first-person");
/** an arm's page name: its project directory, less the scratch prefix. */
const armName = (project: string) =>
    project
        .split("/")
        .pop()!
        .replace(/^zz-ab-/, "");

// The control is the WebGPU samples' helloTriangle (github.com/webgpu/webgpu-samples, sample/helloTriangle/
// main.ts, triangle.vert.wgsl, red.frag.wgsl at e040ec1a), inlined as one page. One change: it requests a
// core adapter, as the engine does, rather than the sample's `featureLevel: 'compatibility'`, which may
// select a different backend.
const CONTROL_PAGE = `<!doctype html>
<meta charset="utf-8">
<title>webgpu control</title>
<style>html,body{margin:0;height:100%;background:#000}canvas{width:100%;height:100%;display:block}</style>
<canvas></canvas>
<script type="module">
const canvas = document.querySelector("canvas");
const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();
if (!device) throw new Error("control: no WebGPU device");
const context = canvas.getContext("webgpu");
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: presentationFormat });
const pipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: device.createShaderModule({ code: \`@vertex
fn main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2(0.0, 0.5), vec2(-0.5, -0.5), vec2(0.5, -0.5));
  return vec4f(pos[VertexIndex], 0.0, 1.0);
}\` }) },
  fragment: { module: device.createShaderModule({ code: \`@fragment
fn main() -> @location(0) vec4f { return vec4(1.0, 0.0, 0.0, 1.0); }\` }), targets: [{ format: presentationFormat }] },
  primitive: { topology: "triangle-list" },
});
function frame() {
  const commandEncoder = device.createCommandEncoder();
  const textureView = context.getCurrentTexture().createView();
  const passEncoder = commandEncoder.beginRenderPass({
    colorAttachments: [{ view: textureView, clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }],
  });
  passEncoder.setPipeline(pipeline);
  passEncoder.draw(3);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
`;

/** What every document records about its own WebGPU use, without acquiring anything itself. */
const OBSERVER = `(() => {
  const o = (window.__ab = { adapters: [], lost: [], errors: [], submitFrames: 0, frames: 0 });
  let submitted = false;
  const tick = () => { o.frames++; if (submitted) o.submitFrames++; submitted = false; requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  if (!navigator.gpu) return;
  const requestAdapter = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function (options) {
    const adapter = await requestAdapter.call(this, options);
    const i = adapter?.info;
    o.adapters.push(adapter ? { vendor: i?.vendor, architecture: i?.architecture, device: i?.device,
      description: i?.description, fallback: i?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null } : null);
    return adapter;
  };
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (descriptor) {
    const device = await requestDevice.call(this, descriptor);
    const at = performance.now();
    device.lost.then((info) => o.lost.push({ reason: info.reason, message: info.message, ms: Math.round(performance.now()) }));
    device.addEventListener("uncapturederror", (e) => o.errors.push(String(e.error?.message ?? e.error).slice(0, 300)));
    o.deviceAt = Math.round(at);
    return device;
  };
  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function (buffers) { submitted = true; return submit.call(this, buffers); };
})();`;

interface Launch {
    run: number;
    browser: BrowserName;
    version: string;
    page: PageName;
    startedAt: string;
    adapters: unknown[];
    backend: string[];
    frames: number;
    submitFrames: number;
    lost: { reason: string; message: string; ms: number }[];
    errors: string[];
    /** warnings, errors and page errors, each prefixed with its milliseconds since the launch began */
    console: string[];
    /** responses that failed, as `<status> <url>` */
    failedResponses: string[];
    /** the loss the console reported, when the observer's own read missed it */
    consoleLoss: string | null;
    failureScreen: string | null;
    nvrm: string[];
    refusal?: string;
}

const started = performance.now();
const remaining = () => TOTAL_MS - (performance.now() - started);

function run(cmd: string, args: string[]): string {
    const out = Bun.spawnSync([cmd, ...args], { stdout: "pipe", stderr: "pipe" });
    if (out.exitCode !== 0)
        throw new Error(
            `${cmd} ${args.join(" ")} exited ${out.exitCode}: ${out.stderr.toString().trim()}`,
        );
    return out.stdout.toString();
}

/** A Proton game holding the GPU is the confound this run exists to exclude. */
function refuseOpenGame(): void {
    const apps = run("nvidia-smi", [
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ]);
    const games = apps.split("\n").filter((line) => /\.exe\s*$/i.test(line));
    if (games.length > 0)
        throw new Error(
            `a game holds the GPU, so this is not the game-closed comparison: ${games.join("; ")}`,
        );
}

function kernelLines(since: number, until: number): string[] {
    const log = run("journalctl", [
        "-k",
        "--no-pager",
        "-o",
        "short-iso",
        `--since=@${Math.floor(since / 1000)}`,
        `--until=@${Math.ceil(until / 1000) + 2}`,
    ]);
    return log.split("\n").filter((line) => /NVRM/.test(line));
}

async function startPreview(outDir: string): Promise<{ close(): Promise<void> }> {
    const child = Bun.spawn(
        [
            process.execPath,
            resolveViteCli(PROJECT),
            "preview",
            "--host",
            "127.0.0.1",
            "--port",
            String(PORT),
            "--strictPort",
            "--outDir",
            outDir,
        ],
        { cwd: PROJECT, stdout: "ignore", stderr: "inherit" },
    );
    const close = async () => {
        if (child.exitCode !== null) return;
        child.kill();
        await child.exited;
    };
    const deadline = performance.now() + 20_000;
    while (performance.now() < deadline) {
        if (child.exitCode !== null)
            throw new Error(`vite preview exited with status ${child.exitCode}`);
        const ok = await fetch(`${ORIGIN}/webgpu-control.html`).then(
            (r) => {
                void r.body?.cancel();
                return r.ok;
            },
            () => false,
        );
        if (ok) return { close };
        await Bun.sleep(100);
    }
    await close();
    throw new Error(`vite preview did not become ready at ${ORIGIN}`);
}

/**
 * chrome://gpu's WebGPU status and the adapters its Dawn Info lists, as "<Discrete GPU>  Vulkan backend -
 * NVIDIA …". The page fills Dawn Info asynchronously, so it is polled for up to eight seconds.
 */
async function webgpuBackend(context: import("playwright").BrowserContext): Promise<string[]> {
    const page = await context.newPage();
    try {
        await page.goto("chrome://gpu", { timeout: 10_000 });
        for (let i = 0; i < 16; i++) {
            await page.waitForTimeout(500);
            const lines = await page.evaluate(() => {
                const parts: string[] = [];
                const walk = (node: Node) => {
                    if (node.nodeType === Node.TEXT_NODE) parts.push(node.textContent ?? "");
                    const root = (node as Element).shadowRoot;
                    if (root) walk(root);
                    for (const child of node.childNodes) walk(child);
                };
                walk(document.body);
                return parts.map((line) => line.trim()).filter(Boolean);
            });
            const adapters = lines.filter((line) => /^<[^>]+>\s+\S+ backend - /.test(line));
            if (adapters.length > 0 || i === 15) {
                const status = lines[lines.indexOf("WebGPU:") + 1] ?? "unread";
                return [`WebGPU: ${status}`, ...adapters];
            }
        }
        return [];
    } finally {
        await page.close();
    }
}

async function waitForNoSeatWindow(): Promise<void> {
    for (let i = 0; i < 50; i++) {
        const open = JSON.parse(run("hyprctl", ["clients", "-j"])) as { class?: string }[];
        if (!open.some((row) => row.class === SEAT_WINDOW_CLASS)) return;
        await Bun.sleep(100);
    }
    throw new Error(`a ${SEAT_WINDOW_CLASS} window is still open after its browser closed`);
}

async function launchOnce(
    declared: string,
    browser: (typeof BROWSERS)[number],
    version: string,
    page: PageName,
    runIndex: number,
    shots: string,
): Promise<Launch> {
    const record: Launch = {
        run: runIndex,
        browser: browser.name,
        version,
        page,
        startedAt: new Date().toISOString(),
        adapters: [],
        backend: [],
        frames: 0,
        submitFrames: 0,
        lost: [],
        errors: [],
        console: [],
        failedResponses: [],
        consoleLoss: null,
        failureScreen: null,
        nvrm: [],
    };
    const since = Date.now();
    const placement = await openOnDisplay(declared);
    const profile = mkdtempSync(join(tmpdir(), "shallot-chrome-ab-"));
    // Its own process group, so closing it reaches exactly the processes this launch started.
    const child = spawn(
        browser.path,
        [
            `--user-data-dir=${profile}`,
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-mode",
            ...placement.args,
            "about:blank",
        ],
        { detached: true, stdio: "ignore" },
    );
    const pgid = child.pid;
    if (pgid === undefined) throw new Error(`${browser.path} did not start`);
    let cdp: import("playwright").Browser | undefined;
    const bound = setTimeout(() => {
        record.refusal = `the launch exceeded its ${LAUNCH_MS} ms bound`;
        try {
            process.kill(-pgid, "SIGKILL");
        } catch {}
    }, LAUNCH_MS);
    try {
        let port: string | undefined;
        for (let i = 0; i < 100 && port === undefined; i++) {
            try {
                port = readFileSync(join(profile, "DevToolsActivePort"), "utf8").split("\n")[0];
            } catch {
                await Bun.sleep(100);
            }
        }
        if (!port) throw new Error(`${browser.name} wrote no DevToolsActivePort`);
        const { chromium } = await import("playwright");
        cdp = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { timeout: 10_000 });
        const context = cdp.contexts()[0];
        const tab = await context.newPage();
        await confirmOnDisplay(placement);
        const at = () => Date.now() - since;
        tab.on("console", (m) => {
            if (m.type() === "error" || m.type() === "warning")
                record.console.push(`${at()} ${m.type()}: ${m.text().slice(0, 300)}`);
        });
        tab.on("pageerror", (e) =>
            record.console.push(`${at()} pageerror: ${e.message.slice(0, 300)}`),
        );
        tab.on("response", (r) => {
            if (r.status() >= 400) record.failedResponses.push(`${r.status()} ${r.url()}`);
        });
        await tab.addInitScript({ content: OBSERVER });
        await tab.goto(`${ORIGIN}/${page === "control" ? "webgpu-control.html" : `${page}/`}`, {
            timeout: 10_000,
        });
        await tab.waitForTimeout(OBSERVE_MS);
        const observe = async () => {
            const seen = await tab.evaluate(
                () => (window as unknown as { __ab: Record<string, unknown> }).__ab,
            );
            Object.assign(record, {
                adapters: seen.adapters,
                frames: seen.frames,
                submitFrames: seen.submitFrames,
                lost: seen.lost,
                errors: seen.errors,
            });
            const screen = await tab.evaluate(() => document.body.innerText);
            if (/Something went wrong/.test(screen)) record.failureScreen = screen.slice(0, 400);
        };
        await observe();
        await tab.screenshot({ path: join(shots, `${runIndex}-${browser.name}-${page}.png`) });
        record.backend = await webgpuBackend(context);
        // A stalled page can report its loss after the observation window (run 1's Chromium first-person
        // logged it while chrome://gpu was read), so the page is read again just before the browser closes.
        await observe();
    } catch (error) {
        record.refusal ??= error instanceof Error ? error.message.split("\n")[0] : String(error);
    } finally {
        clearTimeout(bound);
        await cdp?.close().catch(() => {});
        try {
            process.kill(-pgid, "SIGTERM");
        } catch {}
        for (let i = 0; i < 50 && child.exitCode === null && child.signalCode === null; i++)
            await Bun.sleep(100);
        try {
            process.kill(-pgid, "SIGKILL");
        } catch {}
        rmSync(profile, { recursive: true, force: true });
        await waitForNoSeatWindow();
        record.nvrm = kernelLines(since, Date.now());
        record.consoleLoss =
            record.console.find((line) =>
                /GPU device lost|Instance reference no longer exists/.test(line),
            ) ?? null;
    }
    return record;
}

/** every sign of a fault one launch can show, so the tally and the reading are computed from the same rows. */
function faultsOf(launch: Launch): string[] {
    const faults: string[] = [];
    if (launch.refusal !== undefined) faults.push(`refused: ${launch.refusal}`);
    if (launch.lost.length > 0) faults.push(`device lost: ${launch.lost[0].message}`);
    else if (launch.consoleLoss !== null)
        faults.push(`device lost (console): ${launch.consoleLoss}`);
    if (launch.nvrm.length > 0) faults.push(`kernel: ${launch.nvrm.join(" | ")}`);
    if (launch.failureScreen !== null) faults.push("failure screen");
    if (launch.adapters.length === 0 || launch.adapters.includes(null)) faults.push("no adapter");
    if (launch.submitFrames === 0) faults.push("no frame submitted GPU work");
    return faults;
}

/** the reading, decided before the run and printed at its start. */
const READING_RULE =
    ARMS.length > 0
        ? `reading: a launch faults on a refusal, a device loss (observer or console), an NVRM kernel line, the failure screen, no adapter, or no frame that submitted GPU work. A faulted control means the browser or the driver, and the arms decide nothing about the engine. An arm reads clean, a suspect, at 0 faults in its ${ARM_LAUNCHES} launches while the first-person baseline faults. An arm that faults is not separated. If the baseline is clean too, the run decides nothing.`
        : "reading: a faulted control means the browser or the driver; faults only on first-person mean the engine's workload on this driver; clean on both means the earlier faults did not recur.";

function verdict(launches: readonly Launch[]): string {
    const pages = [...new Set(launches.map((l) => l.page))];
    const tally = (page: PageName, browser?: BrowserName) => {
        const rows = launches.filter((l) => l.page === page && (!browser || l.browser === browser));
        return { faulted: rows.filter((l) => faultsOf(l).length > 0).length, of: rows.length };
    };
    const lines = pages.map((page) => {
        const all = tally(page);
        const per = BROWSERS.map((b) => {
            const t = tally(page, b.name);
            return `${b.name} ${t.faulted}/${t.of}`;
        }).join(", ");
        return `  ${page}: ${all.faulted}/${all.of} (${per})`;
    });
    const faulted = launches
        .filter((l) => faultsOf(l).length > 0)
        .map((l) => `  run ${l.run} ${l.browser} ${l.page}: ${faultsOf(l).join("; ")}`);
    const control = tally("control");
    const baseline = tally("first-person");
    const arms = pages.filter((page) => page !== "control" && page !== "first-person");
    let reading: string;
    if (control.faulted > 0) {
        reading =
            "A control page, which has no Shallot, faulted: the cause is the browser or the driver, not the engine.";
    } else if (baseline.faulted === 0) {
        reading =
            arms.length > 0
                ? "The first-person baseline was clean: this run decides nothing."
                : "Every launch was clean: the earlier faults did not recur.";
    } else if (arms.length === 0) {
        const browsers = BROWSERS.filter((b) => tally("first-person", b.name).faulted > 0).map(
            (b) => b.name,
        );
        reading = `Only first-person faulted, in ${browsers.join(" and ")}: the engine's workload is exposed on this driver; bisect it.`;
    } else {
        const rate = baseline.faulted / baseline.of;
        reading = arms
            .map((arm) => {
                const t = tally(arm);
                if (t.faulted > 0) return `${arm}: not separated (${t.faulted}/${t.of}).`;
                if (t.of < ARM_LAUNCHES)
                    return `${arm}: 0/${t.of}, short of ${ARM_LAUNCHES}, so not read.`;
                return `${arm}: clean 0/${t.of} against baseline ${baseline.faulted}/${baseline.of}, a suspect (at the baseline's rate, ${(100 * (1 - rate) ** t.of).toFixed(0)}% of ${t.of}-launch arms would be clean by chance).`;
            })
            .join("\n");
    }
    return ["faulted launches by page:", ...lines, ...faulted, reading].join("\n");
}

const dry = process.argv.includes("--dry");
const declared = process.env.SHALLOT_DISPLAY_SEAT?.trim();
if (!declared)
    throw new Error(
        "display seat unavailable: SHALLOT_DISPLAY_SEAT names no monitor, and this run takes the one it names",
    );

const versions = new Map<BrowserName, string>();
for (const browser of BROWSERS) {
    const version = run(browser.path, ["--version"]).trim();
    if (!new RegExp(`\\b${browser.major}\\.`).test(version))
        throw new Error(
            `${browser.path} reports ${version}, not ${browser.major}: this A/B is Chrome 153 against Chromium 152`,
        );
    versions.set(browser.name, version);
}
// A dry run reports an open game rather than refusing, so it can be prepared while the person plays.
if (dry) {
    try {
        refuseOpenGame();
    } catch (error) {
        console.info(`[chrome-ab] dry run, not refused: ${(error as Error).message}`);
    }
} else refuseOpenGame();
// Read-only before anything is built: Hyprland answers and the declared monitor is live. Each launch then
// declares the seat's placement rule and verifies the window landed, as the seat does.
if (!process.env.HYPRLAND_INSTANCE_SIGNATURE)
    throw new Error(
        `the display seat declares ${declared}, but HYPRLAND_INSTANCE_SIGNATURE is unset, so no placement can be verified`,
    );
const monitor = declaredMonitor(parseMonitors(run("hyprctl", ["monitors", "-j"])), declared);
if ("refused" in monitor) throw new Error(monitor.refused);

const outDir = mkdtempSync(join(tmpdir(), "shallot-chrome-ab-build-"));
const shots = mkdtempSync(join(tmpdir(), "shallot-chrome-ab-shots-"));
let preview: { close(): Promise<void> } | undefined;
try {
    const { buildWeb } = await import("../../src/project/build");
    // Each project is built under its own base, so the pages share one preview without sharing assets.
    const projects = [PROJECT, ...ARMS.map((arm) => resolve(import.meta.dir, "../..", arm))];
    const names = projects.map((project, i) => (i === 0 ? "first-person" : armName(project)));
    for (const [i, project] of projects.entries())
        await buildWeb(project, { outDir: join(outDir, names[i]) }, ["--base", `/${names[i]}/`]);
    writeFileSync(join(outDir, "webgpu-control.html"), CONTROL_PAGE);
    preview = await startPreview(outDir);
    for (const path of [...names.map((name) => `/${name}/`), "/webgpu-control.html"]) {
        const response = await fetch(ORIGIN + path);
        if (!response.ok) throw new Error(`${ORIGIN}${path} answered ${response.status}`);
        await response.body?.cancel();
    }
    console.info(`[chrome-ab] ${READING_RULE}`);
    console.info(
        `[chrome-ab] seat ${monitor.name} (${monitor.width}x${monitor.height} @ ${monitor.refreshRate.toFixed(2)} Hz); ${[...versions.values()].join(" / ")}; bound ${Math.round(TOTAL_MS / 1000)} s`,
    );
    if (dry) {
        console.info(
            `[chrome-ab] dry run: built and served ${names.map((name) => `/${name}/`).join(", ")} and /webgpu-control.html; no browser opened`,
        );
    } else {
        const launches: Launch[] = [];
        const schedule: { run: number; browser: (typeof BROWSERS)[number]; page: PageName }[] = [];
        if (ARMS.length === 0) {
            for (let run = 1; run <= RUNS; run++)
                // The page order alternates by run, so neither page always runs first in a fresh GPU state.
                for (const browser of BROWSERS)
                    for (const page of run % 2 === 1
                        ? ["first-person", "control"]
                        : ["control", "first-person"])
                        schedule.push({ run, browser, page });
        } else {
            // Every round opens every page once in one browser; the browser alternates by round and the page
            // order rotates, so time-dependent effects spread across pages and browsers.
            const pages = [...names, "control"];
            for (let run = 1; run <= ARM_LAUNCHES; run++) {
                const browser = BROWSERS[(run - 1) % BROWSERS.length];
                for (let i = 0; i < pages.length; i++)
                    schedule.push({ run, browser, page: pages[(i + run - 1) % pages.length] });
            }
        }
        console.info(`[chrome-ab] ${schedule.length} launches planned`);
        for (const { run, browser, page } of schedule) {
            if (remaining() < LAUNCH_MS) {
                console.info(
                    `[chrome-ab] the ${Math.round(TOTAL_MS / 1000)} s bound stopped the run after ${launches.length} of ${schedule.length} launches`,
                );
                break;
            }
            refuseOpenGame();
            const launch = await launchOnce(
                declared,
                browser,
                versions.get(browser.name)!,
                page,
                run,
                shots,
            );
            launches.push(launch);
            console.info(`[chrome-ab] ${JSON.stringify(launch)}`);
        }
        writeFileSync(join(shots, "launches.json"), JSON.stringify(launches, null, 2));
        console.info(`[chrome-ab] records and screenshots: ${shots}`);
        console.info(`[chrome-ab] ${verdict(launches)}`);
    }
} finally {
    await preview?.close();
    rmSync(outDir, { recursive: true, force: true });
    if (dry) rmSync(shots, { recursive: true, force: true });
}
