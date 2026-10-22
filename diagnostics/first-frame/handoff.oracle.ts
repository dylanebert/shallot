import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    renameSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { BROWSER_CONFIG, TIMEOUT } from "@dylanebert/shallot/testing";
import {
    type HandoffReport,
    HANDOFF_TIMING,
    installPrimes,
    observeHandoff,
    PRIMES,
    summarizeHandoff,
} from "./probe.mjs";
import { cpuIn, ROLES, startTrace, type Trace } from "./trace";
import { type BrowserContext, chromium, type Page, webkit } from "playwright";
import { bunChild } from "../../scripts/bun-child";

// Per example, the gap from the loading screen's handoff to the first steady frame, with what ran in between:
// each system's setup and first update, GPU calls and the UI mount. Warm runs load each example once unmeasured
// first; cold runs measure the first load from an empty GPU cache.
const USAGE = `Usage: bun diagnostics/first-frame/handoff.oracle.ts <chrome|webkit|safari> [--headed] [--cold] [--trace] [--prime=<kind,...>] [example ...]
  chrome          installed Google Chrome, headless unless --headed; --cold gives each example a fresh profile
  webkit          Playwright WebKit, headless; --cold moves
                  $(getconf DARWIN_USER_CACHE_DIR)com.apple.WebKit.GPU+org.webkit.Playwright aside and launches
                  a fresh browser per example, then restores the cache
  safari          the running Safari.app, in one window of its own whose URL this run sets; always headed.
                  --cold asks before quitting Safari, moves
                  $(getconf DARWIN_USER_CACHE_DIR)com.apple.WebKit.GPU+com.apple.Safari aside, quits and relaunches
                  Safari before each example, then restores the cache and relaunches Safari
  --trace         webkit or safari: record each WebKit, Metal compiler and browser process's CPU time
                  every millisecond and their stacks from the handoff, and print CPU per process in each
                  segment of the handoff; tracing perturbs the timing it attributes, and leaves out
                  processes running before the run, so it traces a cold run's fresh browser
  --prime=<kinds> a what-if that models an engine move, from ${PRIMES.join(", ")}: probe.mjs's PRIMES
A headed run takes the screen: keep its window frontmost and uncovered, since an occluded window's frames throttle.`;

const args = process.argv.slice(2);
const browserName = args[0];
const headed = args.includes("--headed") || browserName === "safari";
const cold = args.includes("--cold");
const tracing = args.includes("--trace");
const primes = (args.find((arg) => arg.startsWith("--prime="))?.slice("--prime=".length) ?? "")
    .split(",")
    .filter(Boolean);
const only = args.slice(1).filter((arg) => !arg.startsWith("--"));
const unknown = args
    .slice(1)
    .filter(
        (arg) =>
            arg.startsWith("--") &&
            arg !== "--headed" &&
            arg !== "--cold" &&
            arg !== "--trace" &&
            !arg.startsWith("--prime="),
    );
if (
    (browserName !== "chrome" && browserName !== "webkit" && browserName !== "safari") ||
    unknown.length ||
    primes.some((prime) => !(PRIMES as readonly string[]).includes(prime)) ||
    (browserName === "webkit" && headed) ||
    (browserName === "chrome" && tracing)
)
    throw new Error(USAGE);

const root = resolve(import.meta.dir, "../..");
const vite = resolve(root, "node_modules/vite/bin/vite.js");
const examples = only.length
    ? only
    : readdirSync(resolve(root, "examples"), { withFileTypes: true })
          .filter((entry) => entry.isDirectory())
          .map((entry) => entry.name)
          .sort();
const FRAMES = 120;
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const outDir = join(tmpdir(), "shallot-first-frame");
mkdirSync(outDir, { recursive: true });
const outFile = join(
    outDir,
    `${browserName}${headed ? "-headed" : ""}-${cold ? "cold" : "warm"}${primes.length ? `-prime-${primes.join("+")}` : ""}-${stamp}.jsonl`,
);

// The GPU process's shader caches for this browser: a cold run measures each example without them.
const safariCache = `${Bun.spawnSync(["getconf", "DARWIN_USER_CACHE_DIR"], { stdout: "pipe" }).stdout.toString().trim()}com.apple.WebKit.GPU+${browserName === "webkit" ? "org.webkit.Playwright" : "com.apple.Safari"}`;
const safariAside = `${safariCache}.shallot-aside`;
if (browserName !== "chrome" && cold && existsSync(safariAside))
    throw new Error(
        `${safariAside} is left from an interrupted cold run. Quit the browser and restore it before running again:\n  rm -rf '${safariCache}' && mv '${safariAside}' '${safariCache}'`,
    );

// A cold Safari run quits Safari before each example; it asks first, before the builds.
const safariWasRunning =
    browserName === "safari" &&
    Bun.spawnSync(["pgrep", "-f", "/Safari.app/Contents/MacOS/Safari"], { stdout: "pipe" })
        .exitCode === 0;
if (browserName === "safari" && cold && safariWasRunning) {
    const answer = prompt(
        "A cold run quits Safari now and relaunches it for each example, closing your Safari windows; it relaunches Safari when done. Quit Safari? [y/N]",
    );
    if (answer?.trim().toLowerCase() !== "y")
        throw new Error("Cold Safari run declined; Safari left running.");
}

async function freePort(): Promise<number> {
    const reservation = createServer();
    await new Promise<void>((done, fail) => {
        reservation.once("error", fail);
        reservation.listen(0, "127.0.0.1", done);
    });
    const address = reservation.address();
    if (!address || typeof address === "string") throw new Error("No free port");
    await new Promise<void>((done) => reservation.close(() => done()));
    return address.port;
}

async function until(
    ready: () => Promise<boolean>,
    what: string,
    timeout: number = TIMEOUT.startup,
): Promise<void> {
    const started = performance.now();
    while (!(await ready().catch(() => false))) {
        if (performance.now() - started > timeout) throw new Error(`${what} timed out`);
        await Bun.sleep(100);
    }
}

type HandoffRecord = Parameters<typeof summarizeHandoff>[0] & {
    handoff: number;
    timeOrigin: number;
    events: { kind: string; label: string; time: number; bytes?: number; end?: number }[];
    drains: { label: string; at: number; done: number }[];
    measures: { name: string; start: number; duration: number }[];
};

function detailOf(record: HandoffRecord, report: HandoffReport, adapter: string) {
    const ms = (value: number) => Math.round(value * 10) / 10;
    const after = record.measures.filter((m) => m.start >= record.handoff);
    const runs = (suffix: string) =>
        after
            .filter((m) => m.name.startsWith("shallot:system:") && m.name.endsWith(suffix))
            .map((m) => ({
                name: m.name.slice("shallot:system:".length, -suffix.length),
                ms: m.duration,
            }))
            .sort((a, b) => b.ms - a.ms);
    const setups = runs(":setup");
    const updates = runs(":first-update");
    const frameOf = (time: number) => record.frames.filter((start) => start <= time).length;
    const bytesIn = (kind: string, frame: number) =>
        record.events
            .filter((e) => e.kind === kind && frameOf(e.time) === frame)
            .reduce((sum, e) => sum + (e.bytes ?? 0), 0);
    const created = record.events.filter((e) => e.kind === "pipeline");
    const pipelines = created.filter((e) => e.time >= record.handoff);
    const under = created.filter((e) => e.time < record.handoff);
    const sync = under.filter((e) => e.end !== undefined);
    const relative = (time: number) => ms(time - record.handoff);
    const firstFrames = (kind: string) =>
        record.events
            .filter((e) => e.kind === kind && frameOf(e.time) >= 1 && frameOf(e.time) <= 4)
            .map((e) => `${frameOf(e.time)}:${ms((e.end ?? e.time) - e.time)}`);
    const top = (list: { name: string; ms: number }[]) =>
        list
            .filter((r) => r.ms >= 1)
            .slice(0, 4)
            .map((r) => `${r.name} ${ms(r.ms)}`);
    const sorted = [...report.intervals].sort((a, b) => a - b);
    return {
        adapter,
        gap: report.gap,
        loading: ms(record.handoff),
        intervals: report.intervals.slice(0, 4),
        steadyInterval: sorted[Math.floor(sorted.length / 2)],
        // Frame 0 is the handoff to the first frame's start; then each frame's callback time.
        frameCpu: [
            ms((record.frames[0] ?? record.handoff) - record.handoff),
            ...record.frames.slice(0, 4).map((start, i) => ms(record.ends[i]! - start)),
        ],
        ui: ms(after.find((m) => m.name === "shallot:ui-mount")?.duration ?? 0),
        setup: {
            count: setups.length,
            ms: ms(setups.reduce((a, r) => a + r.ms, 0)),
            top: top(setups),
        },
        firstUpdate: { ms: ms(updates.reduce((a, r) => a + r.ms, 0)), top: top(updates) },
        pipelines: [...new Set(pipelines.map((p) => `${frameOf(p.time)}:${p.label}`))],
        // Under the screen: creation calls, and the last one's return relative to the handoff.
        created: {
            count: under.length,
            lastEnd: sync.length ? relative(Math.max(...sync.map((e) => e.end!))) : undefined,
        },
        // Each drain's start and completion relative to the handoff; a handoff drain longer than a
        // frame is GPU-process work still queued when the screen came down.
        drains: record.drains
            .map((d) => ({ label: d.label, at: relative(d.at), done: relative(d.done) }))
            .sort((a, b) => a.at - b.at),
        acquire: firstFrames("acquire"),
        submit: firstFrames("submit"),
        stamps: record.stamps.slice(0, 6).map(relative),
        frameStarts: record.frames.slice(0, 6).map(relative),
        frameEnds: record.ends.slice(0, 6).map(relative),
        // Epoch milliseconds of the handoff, to align a process trace's wall clock.
        handoffEpoch: record.timeOrigin + record.handoff,
        navigation: relative(0),
        // Under the screen at full progress: the precompile drain, `Loading.complete` and one frame.
        tail: ms(
            record.handoff -
                Math.max(
                    ...record.measures
                        .filter((m) => m.name.startsWith("shallot:plugin:"))
                        .map((m) => m.start + m.duration),
                ),
        ),
        decodes: record.events.filter((e) => e.kind === "decode").length,
        frame1AllocBytes: bytesIn("allocation", 1),
        frame1UploadBytes: bytesIn("upload", 1),
        steadyUploadBytes: bytesIn("upload", FRAMES - 1),
    };
}

type Detail = ReturnType<typeof detailOf>;

/** CPU per process in each segment of one load, from its trace. */
function segmentsOf(detail: Detail, trace: Trace) {
    const at = (relative: number) => detail.handoffEpoch + relative;
    const settled = detail.gap;
    const [start, end] = [detail.frameStarts[0] ?? 0, detail.frameEnds[0] ?? 0];
    return {
        ...trace,
        segments: cpuIn(trace.cpu, [
            ["under the screen", at(detail.navigation), at(0)],
            ["handoff to frame 1", at(0), at(start)],
            ["frame 1 callback", at(start), at(end)],
            ["frame 1 end to settled", at(end), at(settled)],
            ["settled, next 500 ms", at(settled), at(settled + 500)],
        ]),
    };
}

/** Loads `url` under the measurement probe and returns its record. */
async function measure(page: Page, url: string, traceName?: string) {
    const handoff = await observeHandoff(page, { frames: FRAMES, primes });
    const trace = traceName ? startTrace(outDir, traceName) : undefined;
    if (trace)
        await page.exposeFunction("__shallotHandoffBeacon", (window: "hook" | "handoff") =>
            trace.beacon(window),
        );
    await page.goto(url);
    const report = await handoff.report();
    const record = await page.evaluate(() =>
        (
            window as unknown as { __shallotHandoff: { read(): HandoffRecord } }
        ).__shallotHandoff.read(),
    );
    const adapter = await page.evaluate(async () => {
        const info = (await navigator.gpu?.requestAdapter())?.info;
        return info ? `${info.vendor} ${info.architecture}`.trim() : "none";
    });
    const detail = detailOf(record, report, adapter);
    return trace ? { ...detail, trace: segmentsOf(detail, await trace.stop()) } : detail;
}

// Safari runs the same measurement probe and primes as Playwright.
let probe = primes.length
    ? `(${installPrimes.toString()})(${JSON.stringify({ ...HANDOFF_TIMING, primes })});`
    : "";
await observeHandoff(
    {
        async addInitScript(fn: (arg: unknown) => void, arg: unknown) {
            probe += `(${fn.toString()})(${JSON.stringify(arg)});`;
        },
    } as unknown as Page,
    { frames: FRAMES },
);
const REPORT_PATH = "/__shallot-handoff-report";
const BEACON_PATH = "/__shallot-handoff-beacon";
// Posts the record once the observed frames have run, with the adapter and any page errors.
const reporter = `(() => {
    const errors = [];
    ${tracing ? `window.__shallotHandoffBeacon = (window) => fetch("${BEACON_PATH}?window=" + window, { method: "POST" });` : ""}
    addEventListener("error", (event) => errors.push(String(event.message)));
    addEventListener("unhandledrejection", (event) => errors.push(String(event.reason?.message ?? event.reason)));
    const post = (body) => fetch("${REPORT_PATH}", { method: "POST", body: JSON.stringify({ ...body, errors }) });
    const started = performance.now();
    const send = async () => {
        const handoff = window.__shallotHandoff;
        if (!handoff) return void post({ failure: "the handoff probe did not install: " + window.__shallotProbeError });
        if (handoff.framesAfterHandoff() < ${FRAMES}) {
            if (performance.now() - started < ${TIMEOUT.browser - 5000}) return void setTimeout(send, 100);
            return void post({ failure: handoff.read().handoff === undefined ? "no handoff" : "too few frames after the handoff" });
        }
        const info = (await navigator.gpu?.requestAdapter())?.info;
        const adapter = info ? (info.vendor + " " + info.architecture).trim() : "none";
        post({ record: handoff.read(), adapter });
    };
    send();
})();`;

type SafariReport = { record?: HandoffRecord; adapter?: string; failure?: string; errors: string[] };

/** Serves a built example through its preview server, with the probe as the page's first script. */
function injectingProxy(
    upstream: string,
    received: (body: SafariReport) => void,
    beacon?: (window: "hook" | "handoff") => void,
) {
    return Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        async fetch(request) {
            const url = new URL(request.url);
            if (url.pathname === REPORT_PATH) {
                received(await request.json());
                return new Response("ok");
            }
            if (url.pathname === BEACON_PATH) {
                beacon?.(url.searchParams.get("window") === "handoff" ? "handoff" : "hook");
                return new Response("ok");
            }
            const response = await fetch(`${upstream}${url.pathname}${url.search}`);
            // fetch has already decoded the body, so its encoding and length no longer describe it.
            const headers = new Headers(response.headers);
            headers.delete("content-length");
            headers.delete("content-encoding");
            if (!headers.get("content-type")?.includes("text/html"))
                return new Response(response.body, { status: response.status, headers });
            const html = (await response.text()).replace(
                /<head[^>]*>/i,
                (head) =>
                    `${head}<script>try { ${probe} } catch (error) { window.__shallotProbeError = String(error); }\n${reporter}</script>`,
            );
            return new Response(html, { status: response.status, headers });
        },
    });
}

function osascript(script: string): string {
    const result = Bun.spawnSync(["osascript", "-e", script], { stdout: "pipe", stderr: "pipe" });
    if (result.exitCode !== 0) throw new Error(`osascript: ${result.stderr.toString().trim()}`);
    return result.stdout.toString().trim();
}
const safariRunning = () =>
    Bun.spawnSync(["pgrep", "-f", "/Safari.app/Contents/MacOS/Safari"], { stdout: "pipe" })
        .exitCode === 0;
async function quitSafari(): Promise<void> {
    osascript('tell application "Safari" to quit');
    await until(async () => !safariRunning(), "Safari quit");
}

console.log(`Building ${examples.length} examples; nothing takes the screen yet.`);
for (const example of examples) {
    const build = Bun.spawn({
        ...bunChild([vite, "build", "--logLevel", "error"]),
        cwd: resolve(root, "examples", example),
        stdout: "ignore",
        stderr: "inherit",
    });
    if ((await build.exited) !== 0) throw new Error(`${example}: vite build failed`);
}
if (headed)
    console.log(`Measuring from now; ${browserName} takes the screen until the summary prints.`);

const results: Record<string, unknown>[] = [];
// Chrome profiles go once the run ends; Chrome can still be writing one as its context closes.
const removeAfterExit: string[] = [];
const cleanups: (() => Promise<void> | void)[] = [];
async function cleanup() {
    while (cleanups.length) await cleanups.pop()!();
}
process.on("SIGINT", () => {
    void cleanup().finally(() => process.exit(130));
});

try {
    // One browser for every warm load; a cold run opens a fresh one per example.
    let open: () => Promise<{
        load(url: string, traceName?: string): Promise<Record<string, unknown>>;
        close(): Promise<void>;
    }>;
    if (browserName === "safari") {
        // A Safari this run launched is quit when it ends, as the cold path also does.
        if (!safariWasRunning && !cold)
            cleanups.push(async () => {
                if (safariRunning()) await quitSafari();
            });
        if (cold) {
            if (safariRunning()) await quitSafari();
            if (existsSync(safariCache)) renameSync(safariCache, safariAside);
            cleanups.push(async () => {
                if (safariRunning()) await quitSafari();
                rmSync(safariCache, { recursive: true, force: true });
                if (existsSync(safariAside)) renameSync(safariAside, safariCache);
                if (safariWasRunning) osascript('tell application "Safari" to activate');
            });
        }
        open = async () => {
            // Safari's own window for this run: only its URL is ever set, and it closes when the run is done.
            const window = osascript(`tell application "Safari"
                make new document with properties {URL:"about:blank"}
                activate
                set bounds of front window to {0, 0, 1280, 800}
                return id of front window
            end tell`);
            return {
                async load(url, traceName) {
                    let received: SafariReport | undefined;
                    const trace = traceName ? startTrace(outDir, traceName) : undefined;
                    const proxy = injectingProxy(
                        url,
                        (body) => {
                            received = body;
                        },
                        (window) => trace?.beacon(window),
                    );
                    try {
                        osascript(
                            `tell application "Safari" to set URL of current tab of window id ${window} to "http://127.0.0.1:${proxy.port}/"`,
                        );
                        await until(
                            async () => received !== undefined,
                            "the page's handoff report",
                            TIMEOUT.browser,
                        );
                        const { record, adapter, failure, errors } = received!;
                        if (!record)
                            throw new Error(`${failure}; page errors: ${errors.join(" | ") || "none"}`);
                        const detail = detailOf(record, summarizeHandoff(record), adapter!);
                        return {
                            ...detail,
                            ...(trace ? { trace: segmentsOf(detail, await trace.stop()) } : {}),
                            errors,
                        };
                    } finally {
                        proxy.stop(true);
                        await trace?.stop();
                    }
                },
                async close() {
                    osascript(`tell application "Safari" to close window id ${window}`);
                    if (!cold) return;
                    // A cold load needs a GPU process that has compiled nothing.
                    await quitSafari();
                    rmSync(safariCache, { recursive: true, force: true });
                },
            };
        };
    } else {
        if (browserName === "webkit" && cold) {
            if (existsSync(safariCache)) renameSync(safariCache, safariAside);
            cleanups.push(() => {
                rmSync(safariCache, { recursive: true, force: true });
                if (existsSync(safariAside)) renameSync(safariAside, safariCache);
            });
        }
        const launch = async (): Promise<BrowserContext> => {
            if (browserName === "webkit") {
                const browser = await webkit.launch();
                const context = await browser.newContext({
                    viewport: { width: 1280, height: 720 },
                });
                context.on("close", () => void browser.close());
                return context;
            }
            const profile = mkdtempSync(join(outDir, "chrome-profile-"));
            const context = await chromium.launchPersistentContext(profile, {
                channel: "chrome",
                headless: !headed,
                viewport: { width: 1280, height: 720 },
                ...BROWSER_CONFIG.use.launchOptions,
            });
            context.on("close", () => removeAfterExit.push(profile));
            return context;
        };
        open = async () => {
            const context = await launch();
            return {
                async load(url, traceName) {
                    const page = await context.newPage();
                    page.setDefaultTimeout(TIMEOUT.browser);
                    await page.bringToFront();
                    try {
                        return await measure(page, url, traceName);
                    } finally {
                        await page.close();
                    }
                },
                async close() {
                    await context.close();
                    await context.browser()?.close();
                    // A cold load needs a GPU process that has compiled nothing.
                    if (browserName === "webkit" && cold)
                        rmSync(safariCache, { recursive: true, force: true });
                },
            };
        };
    }

    let warm: Awaited<ReturnType<typeof open>> | undefined;
    if (!cold) {
        warm = await open();
        cleanups.push(() => warm!.close());
    }
    for (const example of examples) {
        const url = `http://127.0.0.1:${await freePort()}`;
        const preview = Bun.spawn({
            ...bunChild([
                vite,
                "preview",
                "--host",
                "127.0.0.1",
                "--port",
                url.split(":")[2]!,
                "--strictPort",
            ]),
            cwd: resolve(root, "examples", example),
            stdout: "ignore",
            stderr: "inherit",
        });
        const session = warm ?? (await open());
        try {
            await until(async () => (await fetch(url)).ok, `${example} preview`);
            if (warm) await session.load(url);
            const detail = await session.load(
                url,
                tracing ? `${browserName}-${cold ? "cold" : "warm"}-${stamp}-${example}` : undefined,
            );
            results.push({ example, browser: browserName, headed, cold, ...detail });
            console.log(`${example}: gap ${detail.gap} ms`);
        } catch (error) {
            results.push({ example, browser: browserName, headed, cold, error: String(error) });
            console.log(`${example}: ${error}`);
        } finally {
            if (!warm) await session.close();
            preview.kill();
            await preview.exited;
        }
    }
} finally {
    await cleanup();
    await Bun.sleep(removeAfterExit.length ? 1000 : 0);
    for (const profile of removeAfterExit) rmSync(profile, { recursive: true, force: true });
}

writeFileSync(outFile, `${results.map((result) => JSON.stringify(result)).join("\n")}\n`);
const cell = (value: unknown) => (value === undefined ? "" : String(value));
console.log(
    [
        `\n${browserName}${headed ? " headed" : ""} ${cold ? "cold" : "warm"}${primes.length ? ` primed ${primes.join(", ")}` : ""}, ${results[0]?.adapter ?? ""}`,
        "| Example | Gap ms | Navigation to handoff ms | 1st interval ms | Frame-1 CPU ms | First updates ms | UI ms | Late pipelines | Setups |",
        "|---|---|---|---|---|---|---|---|---|",
        ...results.map((r) =>
            r.error
                ? `| ${r.example} | error: ${r.error} |`
                : `| ${r.example} | ${cell(r.gap)} | ${cell(r.loading)} | ${cell((r.intervals as number[])[0])} | ${cell((r.frameCpu as number[])[1])} | ${cell((r.firstUpdate as { ms: number }).ms)} | ${cell(r.ui)} | ${(r.pipelines as string[]).length} | ${(r.setup as { count: number }).count} |`,
        ),
        ...results.flatMap((r) => {
            const trace = r.trace as ReturnType<typeof segmentsOf> | undefined;
            if (!trace) return [];
            return [
                `\n${r.example}: CPU ms per process in each segment (traced; samples in ${outDir})`,
                `| Segment | ms | ${ROLES.join(" | ")} | Top threads |`,
                `|---|---|${ROLES.map(() => "---|").join("")}---|`,
                ...trace.segments.map(
                    (segment) =>
                        `| ${segment.label} | ${segment.ms} | ${ROLES.map((role) => segment.cpu[role]).join(" | ")} | ${segment.top.join(", ")} |`,
                ),
            ];
        }),
        `\nFull records: ${outFile}`,
    ].join("\n"),
);
