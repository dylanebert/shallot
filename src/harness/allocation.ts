import {
    existsSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    realpathSync,
    rmSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { SourceMap } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import { classifyAdapter } from "../engine/runtime/adapter";
import { CROSS_ORIGIN_ISOLATION } from "../project/vite";
import { attribute, originalPosition, subjectSite } from "./allocation-sampler.mjs";
import { CAPTURE_CONTRACT } from "./capture";
import { adapterFacts } from "./driver";
import { LAUNCH_MODES, launchOptions, launchPlan } from "./launch";
import { resolveSeat } from "./seat";

export interface AllocationSite {
    /** function name and original source line */
    site: string;
    /** sampled bytes over the measured window, collected objects included */
    bytes: number;
}

export interface AllocationWindow {
    label: string;
    /** allocating sites in the bundled subject, most bytes first; builtin bytes credit their caller */
    sites: readonly AllocationSite[];
}

export interface AllocationSample {
    runtime: string;
    warm: number;
    frames: number;
    /** windows read after `warm` frames, after twice that, and an A/A repeat, each after a collection */
    windows: readonly AllocationWindow[];
    /** sites of the bundle's `control` literal, attributed as the windows are; empty means the sampler saw nothing */
    control: readonly AllocationSite[];
}

export interface TransitionSample extends AllocationSample {
    /** every byte of the one frame that spawns, collected objects included */
    spawn: readonly AllocationSite[];
    /** every byte of the one frame that despawns, collected objects included */
    despawn: readonly AllocationSite[];
    /** the spawn frame of a second cycle at the same high-water mark */
    spawnAgain: readonly AllocationSite[];
    /** the despawn frame of that second cycle */
    despawnAgain: readonly AllocationSite[];
    /** every byte of the chunk of steady frames right after each of those four event frames, in order */
    afterEvents: readonly AllocationWindow[];
    /** objects allocated from spawn through despawn still live after a full collection */
    survivors: readonly AllocationSite[];
}

const SAMPLER = resolve(import.meta.dir, "allocation-sampler.mjs");

/**
 * Lowered V8 tier thresholds (defaults 400 and 3,000 in Node 26), so every function in a stepped loop
 * reaches TurboFan inside the warm: warm-up boxing and Maglev-only literals are JIT transitions, not
 * steady-state cost. An unoptimized path allocates more, never less, so tiering can only redden a reading.
 */
export const TIER_FLAGS = [
    "--invocation-count-for-maglev=10",
    "--invocation-count-for-turbofan=50",
];

export const windowBytes = (window: AllocationWindow) =>
    window.sites.reduce((total, row) => total + row.bytes, 0);

/** True only when every window reads zero bytes at zero sites. */
export const allocatesNothing = (sample: AllocationSample) =>
    sample.windows.every((window) => window.sites.length === 0 && windowBytes(window) === 0);

/**
 * Bundle `entry` for Node, build its default export with `input`, step it `warm` frames, collect,
 * then sample `frames` more under V8's sampling heap profiler. Needs the `node` requirement resolved.
 */
export function sampleAllocation(
    entry: string,
    options: { warm?: number; frames?: number; input?: string } = {},
): Promise<AllocationSample> {
    return runSampler(entry, options, []);
}

/**
 * As {@link sampleAllocation}, for an entry whose subject also has `spawn()` and `despawn()`: warms paired
 * cycles, samples each event frame and the chunk after it, the steady windows after them, and one cycle's live survivors.
 */
export function sampleTransition(
    entry: string,
    options: { warm?: number; frames?: number; input?: string } = {},
): Promise<TransitionSample> {
    return runSampler(entry, options, ["transition"]) as Promise<TransitionSample>;
}

async function runSampler(
    entry: string,
    { warm = 600, frames = 600, input = "" }: { warm?: number; frames?: number; input?: string },
    mode: string[],
): Promise<AllocationSample> {
    const dir = mkdtempSync(join(tmpdir(), "shallot-allocation-"));
    try {
        writeFileSync(join(dir, "input.txt"), input);
        const built = await Bun.build({
            entrypoints: [entry],
            outdir: dir,
            target: "node",
            format: "esm",
            sourcemap: "linked",
            naming: "subject.mjs",
        });
        if (!built.success)
            throw new Error(`allocation bundle failed: ${built.logs.map(String).join("\n")}`);
        const proc = Bun.spawn(
            [
                "node",
                "--expose-gc",
                "--enable-source-maps",
                ...TIER_FLAGS,
                SAMPLER,
                join(dir, "subject.mjs"),
                String(warm),
                String(frames),
                join(dir, "input.txt"),
                ...mode,
            ],
            { stdout: "pipe", stderr: "pipe" },
        );
        const [stdout, stderr, code] = await Promise.all([
            new Response(proc.stdout).text(),
            new Response(proc.stderr).text(),
            proc.exited,
        ]);
        if (code !== 0) throw new Error(`allocation sampler exited ${code}: ${stderr.trim()}`);
        return JSON.parse(stdout) as AllocationSample;
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

/** a page sample: the steady windows and control of {@link AllocationSample}, and the adapter it ran on. */
export interface PageSample extends AllocationSample {
    /** the positively identified real adapter the display seat resolved on */
    adapter: string;
    /** the run frame's own site, where the control's literal is credited */
    loopSite: string;
}

type CallFrame = Parameters<typeof subjectSite>[0] & { url: string };
type FrameCount = { __shallotFrames: number };

// The page's run frame: the engine's frame loop, `run`'s `frame` method in the app module. A method keeps its
// property name through minification, so a production build's profile still names it.
const FRAME_LOOP = { source: resolve(import.meta.dir, "../engine/app/index.ts"), name: "frame" };

/** frames the control span steps under the breakpoint. */
const CONTROL_FRAMES = 60;

/** the unsampled span the page's presented rate is read over, before any profiler is attached. */
const RATE_SPAN_MS = 500;

/**
 * Build `projectDir` for the web as `shallot run` does, with source maps, serve it in-process, and open it in
 * the display seat's headed Chromium, resolving the seat on the adapter the page reaches. The page's own
 * frame loop is the run frame, attributed by the same rule as {@link sampleAllocation}. Once the loop runs,
 * wait `warm` frames, then sample `frames` after warm, after twice that, and an A/A repeat, each after a
 * collection. The control runs last, over {@link CONTROL_FRAMES} frames: a conditional breakpoint where the
 * frame loop begins allocates one literal per frame. The page's presented rate is read first and reported in
 * `runtime`; a page that cannot supply the whole frame budget before `deadline` refuses as inconclusive
 * rather than timing out mid-warm. Every wait ends by `deadline`, a `performance.now()` time, and the build,
 * server and browser are gone when it returns. Needs the `display` requirement resolved.
 */
export async function samplePage(
    projectDir: string,
    { warm, frames, deadline }: { warm: number; frames: number; deadline: number },
): Promise<PageSample> {
    if (!Number.isInteger(frames) || frames <= 0 || !Number.isInteger(warm) || warm < frames)
        throw new Error("page sampler: needs integer warm >= frames > 0");
    const plan = launchPlan(process.platform, "display");
    if ("refused" in plan) throw new Error(`display seat unavailable: ${plan.refused}`);
    const declared = process.env.SHALLOT_DISPLAY_SEAT?.trim();
    if (!declared) throw new Error("display seat unavailable: no headed display is declared");
    const app = realpathSync(FRAME_LOOP.source);
    const remaining = () => Math.max(1, deadline - performance.now());
    // Every slow call races the deadline, so a hang still reaches `finally` before the row's budget ends.
    // Either way the refusal names the call: a call that ran out of time proves nothing about the claim, so
    // it is inconclusive, and Playwright's own `TimeoutError` is the same refusal wearing a product-red face.
    const bounded = <T>(what: string, work: Promise<T>): Promise<T> => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        const late = new Promise<never>((_, reject) => {
            timer = setTimeout(
                () => reject(new Error(`inconclusive: ${what} did not finish by the deadline`)),
                remaining(),
            );
        });
        const named = work.catch((error: unknown) => {
            if (error instanceof Error && error.name === "TimeoutError")
                throw new Error(`inconclusive: ${what} timed out: ${error.message.split("\n")[0]}`);
            throw error;
        });
        return Promise.race([named, late]).finally(() => clearTimeout(timer));
    };

    const outDir = mkdtempSync(join(tmpdir(), "shallot-page-"));
    let server: ReturnType<typeof Bun.serve> | undefined;
    let browser: import("playwright").Browser | undefined;
    try {
        // A second small static server beside the fixtures': none of those serves a directory with the
        // cross-origin isolation headers the build needs.
        server = Bun.serve({
            port: 0,
            fetch(request) {
                const path = new URL(request.url).pathname;
                const file = join(outDir, path === "/" ? "index.html" : decodeURIComponent(path));
                if (
                    !file.startsWith(`${outDir}${sep}`) ||
                    !existsSync(file) ||
                    !statSync(file).isFile()
                )
                    return new Response("not found", { status: 404 });
                const body = Bun.file(file);
                return new Response(body, {
                    headers: { ...CROSS_ORIGIN_ISOLATION, "Content-Type": body.type },
                });
            },
        });
        // Loaded here, as playwright is, so Node allocation rows never load vite on import.
        const { buildWeb } = await import("../cli/build");
        await bounded("the web build", buildWeb(projectDir, { outDir, sourcemap: true }));
        const origin = `http://localhost:${server.port}`;
        const maps = new Map<string, { map: SourceMap; base: string }>();
        for (const file of readdirSync(outDir, { recursive: true }) as string[]) {
            const mapFile = join(outDir, `${file}.map`);
            if (file.endsWith(".js") && existsSync(mapFile))
                maps.set(`${origin}/${file.split(sep).join("/")}`, {
                    map: new SourceMap(JSON.parse(readFileSync(mapFile, "utf8"))),
                    base: dirname(mapFile),
                });
        }
        const siteOf = (frame: CallFrame) => {
            const built = maps.get(frame.url);
            return built && subjectSite(frame, built.map, built.base, frame.url);
        };
        const isFrameLoop = (frame: CallFrame) => {
            const built = maps.get(frame.url);
            return (
                frame.functionName === FRAME_LOOP.name &&
                built !== undefined &&
                originalPosition(frame, built.map, built.base)?.source === app
            );
        };
        const runSite = (frame: CallFrame) => (isFrameLoop(frame) ? siteOf(frame) : undefined);

        const { chromium } = await import("playwright");
        const options = launchOptions(plan);
        const tiers = `--js-flags=${TIER_FLAGS.join(" ")}`;
        browser = await chromium.launch({
            ...options,
            args: [...options.args, tiers],
            timeout: remaining(),
        });
        const page = await bounded(
            "newPage",
            browser.newPage({
                viewport: { width: CAPTURE_CONTRACT.width, height: CAPTURE_CONTRACT.height },
                deviceScaleFactor: CAPTURE_CONTRACT.deviceScale,
            }),
        );
        page.setDefaultTimeout(remaining());
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        // The harness's frame count: its own rAF callback, outside the run frame, never wrapping the engine's.
        await page.addInitScript(() => {
            const counter = window as unknown as FrameCount;
            counter.__shallotFrames = 0;
            const tick = () => {
                counter.__shallotFrames++;
                requestAnimationFrame(tick);
            };
            requestAnimationFrame(tick);
        });
        const cdp = await bounded("newCDPSession", page.context().newCDPSession(page));
        await bounded("goto", page.goto(`${origin}/`, { waitUntil: "load", timeout: remaining() }));
        const facts = await bounded("adapterFacts", adapterFacts(page));
        const seat = resolveSeat("display", {
            display: { source: declared, browser: { launch: plan, adapter: facts } },
        });
        if (!seat.ok) throw new Error(seat.reason);

        const frameCount = () =>
            bounded(
                "the frame count",
                page.evaluate(() => (window as unknown as FrameCount).__shallotFrames),
            );
        const advance = async (count: number) => {
            const now = await frameCount();
            await bounded(
                `${count} frames`,
                page.waitForFunction(
                    (target) => (window as unknown as FrameCount).__shallotFrames >= target,
                    now + count,
                    { polling: 50, timeout: remaining() },
                ),
            );
        };

        // The page is ready once a CPU profile of its frames holds the frame loop, whose position is also
        // where the control's breakpoint goes.
        await bounded("Profiler.enable", cdp.send("Profiler.enable"));
        let loop: CallFrame | undefined;
        while (loop === undefined) {
            await bounded("Profiler.start", cdp.send("Profiler.start"));
            await advance(30);
            const { profile } = await bounded("Profiler.stop", cdp.send("Profiler.stop"));
            loop = profile.nodes.find((node) => isFrameLoop(node.callFrame))?.callFrame;
        }
        await bounded("Profiler.disable", cdp.send("Profiler.disable"));
        const loopSite = siteOf(loop);
        if (loopSite === undefined) throw new Error("inconclusive: the frame loop has no site");

        // The page's own presented rate is this row's premise, read over a short unsampled span before any
        // profiler is attached. Every remaining step is counted from this function's own arguments, so a
        // page too slow to supply them refuses here by name instead of timing out inside a wait. The warm is
        // never shrunk to fit: its length is what makes three agreeing windows a steadiness claim.
        const before = await frameCount();
        const spanStart = performance.now();
        await Bun.sleep(RATE_SPAN_MS);
        const rate = (((await frameCount()) - before) * 1000) / (performance.now() - spanStart);
        const budget = 2 * warm + 2 * frames + CONTROL_FRAMES;
        const required = (budget * 1000) / remaining();
        if (rate < required)
            throw new Error(
                `inconclusive: the page presents at ${rate.toFixed(1)} Hz, below the ${required.toFixed(1)} Hz needed to step this row's ${budget} frames (warm ${warm}, three ${frames}-frame windows and a ${CONTROL_FRAMES}-frame control) before the deadline`,
            );

        await bounded("HeapProfiler.enable", cdp.send("HeapProfiler.enable"));
        const sample = async (count: number) => {
            await bounded("collectGarbage", cdp.send("HeapProfiler.collectGarbage"));
            await bounded(
                "startSampling",
                cdp.send("HeapProfiler.startSampling", {
                    samplingInterval: 1,
                    includeObjectsCollectedByMajorGC: true,
                    includeObjectsCollectedByMinorGC: true,
                }),
            );
            await advance(count);
            const { profile } = await bounded(
                "stopSampling",
                cdp.send("HeapProfiler.stopSampling"),
            );
            return attribute(profile, runSite, siteOf);
        };
        await advance(warm);
        const atWarm = await sample(frames);
        await advance(warm - frames);
        const atDoubleWarm = await sample(frames);
        const repeat = await sample(frames);

        await bounded("Debugger.enable", cdp.send("Debugger.enable"));
        const { breakpointId } = await bounded(
            "setBreakpointByUrl",
            cdp.send("Debugger.setBreakpointByUrl", {
                url: loop.url,
                lineNumber: loop.lineNumber,
                columnNumber: loop.columnNumber,
                condition: "(globalThis.__shallotControl = { frame: 0 }), false",
            }),
        );
        // The breakpoint's condition runs in the debugger on every frame, which slows the page; 60 frames prove
        // the rule sees bytes under the frame loop as well as a full window would.
        const control = await sample(CONTROL_FRAMES);
        await bounded("removeBreakpoint", cdp.send("Debugger.removeBreakpoint", { breakpointId }));
        await bounded("Debugger.disable", cdp.send("Debugger.disable"));

        if (errors.length > 0)
            throw new Error(`inconclusive: the page threw:\n${errors.slice(0, 20).join("\n")}`);
        return {
            runtime: `chromium ${browser.version()} ${LAUNCH_MODES[plan.seat]} ${tiers} at ${rate.toFixed(1)} Hz`,
            adapter: classifyAdapter(facts).identity,
            loopSite,
            warm,
            frames,
            windows: [
                { label: `after warm ${warm}`, sites: atWarm },
                { label: `after warm ${2 * warm}`, sites: atDoubleWarm },
                { label: "A/A repeat", sites: repeat },
            ],
            control,
        };
    } finally {
        await browser?.close();
        server?.stop(true);
        rmSync(outDir, { recursive: true, force: true });
    }
}

/** Empty when `sites` names exactly `named`; otherwise the unnamed sites by bytes and the named ones absent. */
export function siteSetMismatch(
    label: string,
    sites: readonly AllocationSite[],
    named: readonly string[],
): string {
    const extra = sites.filter((row) => !named.includes(row.site));
    const missing = named.filter((site) => !sites.some((row) => row.site === site));
    if (extra.length === 0 && missing.length === 0) return "";
    return [
        `${label}: ${extra.length} unnamed sites, ${missing.length} named sites absent`,
        ...extra.map((row) => `  + ${String(row.bytes).padStart(8)}  ${row.site}`),
        ...missing.map((site) => `  - ${site}`),
    ].join("\n");
}

/** Per-window totals, then the heaviest window's sites as a table, for a failure message. */
export function siteTable(sample: AllocationSample, limit = 30): string {
    const perFrame = (bytes: number) => (bytes / sample.frames).toFixed(1);
    const totals = sample.windows.map(
        (window) =>
            `${window.label}: ${windowBytes(window)} bytes (${perFrame(windowBytes(window))}/f) at ${window.sites.length} sites`,
    );
    const heaviest = sample.windows.reduce((a, b) => (windowBytes(b) > windowBytes(a) ? b : a));
    const rows = heaviest.sites
        .slice(0, limit)
        .map(
            (row) =>
                `  ${String(row.bytes).padStart(10)}  ${perFrame(row.bytes).padStart(8)}/f  ${row.site}`,
        );
    const more =
        heaviest.sites.length > limit ? [`  … ${heaviest.sites.length - limit} more sites`] : [];
    return [
        ...totals,
        `heaviest, ${heaviest.label}, over ${sample.frames} frames, ${sample.runtime}`,
        ...rows,
        ...more,
    ].join("\n");
}
