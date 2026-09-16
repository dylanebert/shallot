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
import { MissingPremise } from "./verdict";

export interface AllocationSite {
    /** function name and original source line */
    site: string;
    /** sampled bytes over the measured window, collected objects included */
    bytes: number;
    /** allocations at this site over the window: one sample per allocation at the one-byte interval */
    count: number;
}

export interface AllocationWindow {
    label: string;
    /** allocating sites in the bundled subject, most bytes first; builtin bytes credit their caller */
    sites: readonly AllocationSite[];
    /**
     * the fewest frames this window can have stepped. The Node sampler steps its frames itself, so this is
     * exact and equals {@link framesAtMost}; a page window is bracketed instead, because nothing in the
     * page can be sampled once per frame reliably enough to count them (see {@link derivedFrames}).
     */
    frames: number;
    /** the most frames this window can have stepped; equal to {@link frames} where that is exact */
    framesAtMost: number;
}

export interface AllocationSample {
    runtime: string;
    warm: number;
    /** frames each window was asked to step; what it stepped is the window's own `frames` */
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

export const windowBytes = (window: { sites: readonly AllocationSite[] }) =>
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

/** one collection in the steady windows, read from V8's GC trace over CDP */
export interface Collection {
    /** "minor" for a scavenge, "major" for a mark-compact */
    kind: "minor" | "major";
    /** trace clock microseconds at the start of the pause */
    at: number;
    /** pause milliseconds */
    pause: number;
    /** bytes promoted out of the young generation by this collection */
    promoted: number;
    /** V8's own reason for the collection */
    reason: string;
    /** true when this is the collection the harness forced before a window, not the page's own */
    forced: boolean;
}

/** one presented-frame interval above one and a half refresh periods, and the collection inside it */
export interface LongFrame {
    /** trace clock microseconds at the start of the frame that ran long */
    at: number;
    /** the interval in milliseconds */
    gap: number;
    /** the collection inside the interval, or null when none */
    collection: Collection | null;
}

/** the GC and frame-pacing evidence read from the trace over the steady windows */
export interface PlayTrace {
    /** the refresh period the long-frame threshold is derived from: the median frame interval, ms */
    period: number;
    /** presented-frame intervals seen over the traced span */
    frames: number;
    collections: readonly Collection[];
    longFrames: readonly LongFrame[];
}

/** a page sample: the steady windows and control of {@link AllocationSample}, and the adapter it ran on. */
export interface PageSample extends AllocationSample {
    /** the positively identified real adapter the display seat resolved on */
    adapter: string;
    /** the run frame's own site, where the control's literal is credited */
    loopSite: string;
    /** sites of a fourth window's allocations still live after a full collection */
    survivors: readonly AllocationSite[];
    /** the control span, bracketed as the windows are */
    controlSpan: AllocationWindow;
    /** the GC trace and long-frame correlation over the steady windows */
    trace: PlayTrace;
}

type CallFrame = Parameters<typeof subjectSite>[0] & { url: string; scriptId: string };
type FrameCount = { __shallotFrames: number };

// The page's run frame: the engine's frame loop, `run`'s `frame` method in the app module. A method keeps its
// property name through minification, so a production build's profile still names it.
const FRAME_LOOP = { source: resolve(import.meta.dir, "../engine/app/index.ts"), name: "frame" };

/**
 * The harness's frame counter, injected into every document. It is read over CDP to step and bracket
 * windows; it is deliberately not an allocation, because an allocation cannot be counted reliably here —
 * see {@link derivedFrames}.
 */
const FRAME_TICK_SCRIPT = `(() => {
    const counter = window;
    counter.__shallotFrames = 0;
    const tick = () => {
        counter.__shallotFrames++;
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
})();
`;

/** frames the control span steps under the breakpoint. */
const CONTROL_FRAMES = 60;

/** one span the page's presented rate is read over. Each rate is the lower of two of them. */
const RATE_SPAN_MS = 250;

/** how often a frame wait re-reads the page's frame count; each wait can overshoot by this much. */
const ADVANCE_POLL_MS = 50;

/** the trace categories the GC and frame-pacing evidence is read from. */
const TRACE_CATEGORIES = ["devtools.timeline", "disabled-by-default-v8.gc"];

/** a long frame is a presentation interval above this multiple of the measured refresh period. */
const LONG_FRAME = 1.5;

/**
 * the trace's name for one presented animation frame. The page runs more than one rAF callback (the
 * harness's frame counter beside the engine's loop), and Chromium emits `FireAnimationFrame` per
 * callback, so the callback event does not count frames; the presentation event is emitted once per
 * frame that reached the display, which is also what a dropped frame is missing.
 */
const FRAME_EVENT = "AnimationFrame::Presentation";

/**
 * V8's reasons for the collections the harness's own CDP calls cause: `HeapProfiler.collectGarbage`
 * before each window, and the sampling profiler's own start and stop. Neither is steady play's, so both
 * are reported beside the gate rather than counted by it.
 */
const FORCED_REASONS = ["low memory notification", "sampling profiler"];

/** one raw trace event, as `Tracing.dataCollected` delivers it. */
type TraceEvent = {
    name: string;
    ts: number;
    dur?: number;
    pid?: number;
    tid?: number;
    args?: Record<string, unknown>;
};

/**
 * the collections, presented frames and long frames in `events`. `MinorGC` and `MajorGC` carry the pause
 * on the devtools timeline; `V8.GCTraceGCNVP` carries the same collection's reason and promoted bytes,
 * matched by order. {@link FRAME_EVENT} is the page's own presented frame, in the same trace clock, so a
 * long frame and a collection are compared without crossing clocks. Every read is restricted to the one
 * thread the collections ran on, so another process's frames never enter the correlation.
 */
function readTrace(events: readonly TraceEvent[]): PlayTrace {
    const gc = events.find((event) => event.name === "MinorGC" || event.name === "MajorGC");
    const onThread = (event: TraceEvent) =>
        gc === undefined || (event.pid === gc.pid && event.tid === gc.tid);
    const nvp: { reason: string; promoted: number }[] = [];
    for (const event of events) {
        if (event.name !== "V8.GCTraceGCNVP" || !onThread(event)) continue;
        const value = (event.args as { value?: string } | undefined)?.value;
        if (typeof value !== "string") continue;
        const parsed = JSON.parse(value) as { promoted?: number; reason?: string };
        nvp.push({ reason: parsed.reason ?? "", promoted: parsed.promoted ?? 0 });
    }
    const collections: Collection[] = [];
    let index = 0;
    for (const event of events
        .filter(
            (event) => (event.name === "MinorGC" || event.name === "MajorGC") && onThread(event),
        )
        .sort((a, b) => a.ts - b.ts)) {
        const detail = nvp[index++];
        collections.push({
            kind: event.name === "MajorGC" ? "major" : "minor",
            at: event.ts,
            pause: (event.dur ?? 0) / 1000,
            promoted: detail?.promoted ?? 0,
            reason: detail?.reason ?? "",
            forced: FORCED_REASONS.includes(detail?.reason ?? ""),
        });
    }
    const presented = events
        .filter((event) => event.name === FRAME_EVENT && onThread(event))
        .sort((a, b) => a.ts - b.ts);
    const gaps: number[] = [];
    for (let i = 1; i < presented.length; i++)
        gaps.push((presented[i].ts - presented[i - 1].ts) / 1000);
    const sorted = [...gaps].sort((a, b) => a - b);
    const period = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : 0;
    const longFrames: LongFrame[] = [];
    for (let i = 1; i < presented.length; i++) {
        const gap = (presented[i].ts - presented[i - 1].ts) / 1000;
        if (period === 0 || gap <= LONG_FRAME * period) continue;
        const from = presented[i - 1].ts;
        const to = presented[i].ts;
        longFrames.push({
            at: from,
            gap,
            collection: collections.find((row) => row.at >= from && row.at <= to) ?? null,
        });
    }
    return { period, frames: gaps.length, collections, longFrames };
}

/** the control's one literal per frame, allocated where the frame loop begins and never breaking. */
const CONTROL_CONDITION = "(globalThis.__shallotControl = { frame: 0 }), false";

/**
 * Build `projectDir` for the web as `shallot run` does, with source maps, serve it in-process, and open it in
 * the display seat's headed Chromium, resolving the seat on the adapter the page reaches. The page's own
 * frame loop is the run frame, attributed by the same rule as {@link sampleAllocation}. Once the loop runs,
 * wait `warm` frames, then sample `frames` after warm, after twice that, and an A/A repeat, each after a
 * collection. The control runs last, over {@link CONTROL_FRAMES} frames: a conditional breakpoint where the
 * frame loop begins allocates one literal per frame. Once the loop is found, the page's presented rate is
 * read under both of those conditions and reported in `runtime`, and the calls that step no frame are timed
 * once; a page that cannot supply the whole frame budget and that reserve before `deadline` refuses its
 * premise rather than timing out mid-warm. Every wait ends by `deadline`, a `performance.now()` time, and
 * the build, server and browser are gone when it returns. Needs the `display` requirement resolved.
 */
export async function samplePage(
    projectDir: string,
    { warm, frames, deadline }: { warm: number; frames: number; deadline: number },
): Promise<PageSample> {
    if (!Number.isInteger(frames) || frames <= 0 || !Number.isInteger(warm) || warm < frames)
        throw new Error("page sampler: needs integer warm >= frames > 0");
    const plan = launchPlan(process.platform, "display");
    if ("refused" in plan) throw new MissingPremise(`display seat unavailable: ${plan.refused}`);
    const declared = process.env.SHALLOT_DISPLAY_SEAT?.trim();
    if (!declared)
        throw new MissingPremise("display seat unavailable: no headed display is declared");
    const app = realpathSync(FRAME_LOOP.source);
    const remaining = () => Math.max(1, deadline - performance.now());
    // The premise is the page's measured rate, and it is proved partway through. Before it, a call that runs
    // out of time may be a host too slow to hold the premise at all, so it refuses. After it, the row has
    // just proved this host steps frames fast enough, so a call that then runs out of time is the page
    // hanging: a failure of the claim, which is exactly what this row exists to notice.
    let premiseHeld = false;
    const outOfTime = (message: string) =>
        premiseHeld ? new Error(message) : new MissingPremise(message);
    // Every slow call races the deadline, so a hang still reaches `finally` before the row's budget ends.
    // Either way the error names the call.
    const bounded = <T>(what: string, work: Promise<T>): Promise<T> => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        const late = new Promise<never>((_, reject) => {
            timer = setTimeout(
                () => reject(outOfTime(`${what} did not finish by the deadline`)),
                remaining(),
            );
        });
        const named = work.catch((error: unknown) => {
            if (error instanceof Error && error.name === "TimeoutError")
                throw outOfTime(`${what} timed out: ${error.message.split("\n")[0]}`);
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
        // The harness's frame count and its per-frame sentinel: its own rAF callback, outside the run frame,
        // never wrapping the engine's.
        await page.addInitScript({ content: FRAME_TICK_SCRIPT });
        const cdp = await bounded("newCDPSession", page.context().newCDPSession(page));
        await bounded("goto", page.goto(`${origin}/`, { waitUntil: "load", timeout: remaining() }));
        const facts = await bounded("adapterFacts", adapterFacts(page));
        const seat = resolveSeat("display", {
            display: { source: declared, browser: { launch: plan, adapter: facts } },
        });
        if (!seat.ok) throw new MissingPremise(seat.reason);

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
                    { polling: ADVANCE_POLL_MS, timeout: remaining() },
                ),
            );
        };

        // The page is ready once a CPU profile of its frames holds the frame loop, whose position is also
        // where the control's breakpoint goes.
        const findFrameLoop = async (): Promise<CallFrame> => {
            for (;;) {
                await bounded("Profiler.start", cdp.send("Profiler.start"));
                await advance(30);
                const { profile } = await bounded("Profiler.stop", cdp.send("Profiler.stop"));
                const found = profile.nodes.find((node) => isFrameLoop(node.callFrame))?.callFrame;
                if (found !== undefined) return found;
            }
        };
        await bounded("Profiler.enable", cdp.send("Profiler.enable"));
        const loop = await findFrameLoop();
        await bounded("Profiler.disable", cdp.send("Profiler.disable"));
        const loopSite = siteOf(loop);
        if (loopSite === undefined) throw new Error("the frame loop has no site");

        // One call site per operation, so the premise probe below times exactly the calls the row repeats.
        const collect = () => bounded("collectGarbage", cdp.send("HeapProfiler.collectGarbage"));
        const startSampling = () =>
            bounded(
                "startSampling",
                cdp.send("HeapProfiler.startSampling", {
                    samplingInterval: 1,
                    includeObjectsCollectedByMajorGC: true,
                    includeObjectsCollectedByMinorGC: true,
                }),
            );
        const stopSampling = () => bounded("stopSampling", cdp.send("HeapProfiler.stopSampling"));
        const debuggerEnable = () => bounded("Debugger.enable", cdp.send("Debugger.enable"));
        const debuggerDisable = () => bounded("Debugger.disable", cdp.send("Debugger.disable"));
        const setLoopBreakpoint = () =>
            bounded(
                "setBreakpointByUrl",
                cdp.send("Debugger.setBreakpointByUrl", {
                    url: loop.url,
                    lineNumber: loop.lineNumber,
                    columnNumber: loop.columnNumber,
                    condition: CONTROL_CONDITION,
                }),
            );
        const removeLoopBreakpoint = (breakpointId: string) =>
            bounded("removeBreakpoint", cdp.send("Debugger.removeBreakpoint", { breakpointId }));
        // A window is bracketed, not measured. The counter is read before the profiler starts and after
        // it stops, and `advance` only returns once the page has stepped the frames asked for while the
        // profiler was running, so the frames inside the profiled span are at least what was asked for and
        // at most the counter's delta. What the span stepped exactly is derived from the declared rows by
        // `derivedFrames`, because nothing in the page can be sampled once per frame reliably enough to
        // count them: V8 attributes occasional tiering and deoptimisation allocations to whichever frame is
        // running, so a per-frame allocation site over-reports, nondeterministically and by a margin larger
        // than the bracket itself.
        const sample = async (count: number) => {
            await collect();
            const before = await frameCount();
            await startSampling();
            await advance(count);
            const { profile } = await stopSampling();
            const after = await frameCount();
            return {
                sites: attribute(profile, runSite, siteOf),
                frames: count,
                framesAtMost: after - before,
            };
        };
        // the survivor window: sampling without the collected-object classes, then a full collection
        // before the profile is read, so what remains is what the window allocated and still holds.
        const sampleSurvivors = async (count: number) => {
            await collect();
            await bounded(
                "startSampling for survivors",
                cdp.send("HeapProfiler.startSampling", { samplingInterval: 1 }),
            );
            await advance(count);
            await collect();
            const { profile } = await stopSampling();
            return attribute(profile, runSite, siteOf);
        };
        const timed = async <T>(work: () => Promise<T>): Promise<[T, number]> => {
            const at = performance.now();
            const value = await work();
            return [value, performance.now() - at];
        };

        // This row's premise, and the last point at which a slow host is still a missing premise rather than
        // a hung page. A rate read on an idle page licenses a frame budget the sampled, broken-at page cannot
        // supply, so each rate is read under the condition its frames are spent under, and each is the lower
        // of two spans: the same discipline the row applies to its own windows. The warm is never shrunk to
        // fit, because its length is what makes three agreeing windows a steadiness claim.
        const spanRate = async () => {
            const before = await frameCount();
            const spanStart = performance.now();
            await Bun.sleep(RATE_SPAN_MS);
            return (((await frameCount()) - before) * 1000) / (performance.now() - spanStart);
        };
        const slowerOfTwoSpans = async () => Math.min(await spanRate(), await spanRate());

        await bounded("HeapProfiler.enable", cdp.send("HeapProfiler.enable"));
        const [, collectMs] = await timed(collect);
        const [, startMs] = await timed(startSampling);
        const sampledRate = await slowerOfTwoSpans();
        const [, stopMs] = await timed(stopSampling);
        const [, enableMs] = await timed(debuggerEnable);
        const [probed, setMs] = await timed(setLoopBreakpoint);
        const controlRate = await slowerOfTwoSpans();
        const [, removeMs] = await timed(() => removeLoopBreakpoint(probed.breakpointId));
        const [, disableMs] = await timed(debuggerDisable);

        // The time the row still spends on calls that step no frame, measured rather than guessed: the probe
        // above ran one whole sampling cycle and one whole debugger cycle and timed each call in them. What
        // remains is four sampling cycles — the three windows and the control — one more debugger cycle, and
        // six frame waits, each of which may overshoot the target by one poll. `HeapProfiler.enable` is not
        // in it: the probe paid it once, for the row.
        const reserve =
            5 * (collectMs + startMs + stopMs) +
            collectMs +
            enableMs +
            setMs +
            removeMs +
            disableMs +
            7 * ADVANCE_POLL_MS;
        const budget = 2 * warm + 3 * frames + CONTROL_FRAMES;
        const frameMs = (count: number, rate: number) => (count * 1000) / rate;
        const needed =
            frameMs(2 * warm + 3 * frames, sampledRate) +
            frameMs(CONTROL_FRAMES, controlRate) +
            reserve;
        const left = remaining();
        const required = (budget * 1000) / Math.max(1, left - reserve);
        if (needed > left)
            throw new MissingPremise(
                `the page presents at ${sampledRate.toFixed(1)} Hz under the heap sampler and ${controlRate.toFixed(1)} Hz under the control breakpoint, so this row's ${budget} frames (warm ${warm}, three ${frames}-frame windows, a ${frames}-frame survivor window and a ${CONTROL_FRAMES}-frame control) need ${needed.toFixed(0)} ms, including ${reserve.toFixed(0)} ms of calls that step no frame, against the ${left.toFixed(0)} ms left before the deadline; an even ${required.toFixed(1)} Hz would be required`,
            );
        // Past here the host has proved it steps frames fast enough, so running out of time is the page's.
        premiseHeld = true;

        await advance(warm);
        // the GC and frame-pacing trace spans the three steady windows: the collections, their pauses and
        // promoted bytes, and the rAF intervals, all on one trace clock
        const events: TraceEvent[] = [];
        const onTrace = (batch: { value: unknown[] }) => {
            events.push(...(batch.value as TraceEvent[]));
        };
        cdp.on("Tracing.dataCollected", onTrace);
        const traced = new Promise<void>((resolve) =>
            cdp.once("Tracing.tracingComplete", () => resolve()),
        );
        await bounded(
            "Tracing.start",
            cdp.send("Tracing.start", {
                traceConfig: {
                    recordMode: "recordContinuously",
                    includedCategories: TRACE_CATEGORIES,
                },
            }),
        );
        const atWarm = await sample(frames);
        await advance(warm - frames);
        const atDoubleWarm = await sample(frames);
        const repeat = await sample(frames);
        await bounded("Tracing.end", cdp.send("Tracing.end"));
        await bounded("the trace", traced);
        cdp.off("Tracing.dataCollected", onTrace);
        const trace = readTrace(events);
        const survivors = await sampleSurvivors(frames);

        await debuggerEnable();
        const { breakpointId } = await setLoopBreakpoint();
        // The breakpoint's condition runs in the debugger on every frame, which slows the page; 60 frames prove
        // the rule sees bytes under the frame loop as well as a full window would.
        const control = await sample(CONTROL_FRAMES);
        await removeLoopBreakpoint(breakpointId);
        await debuggerDisable();

        if (errors.length > 0)
            throw new Error(`the page threw:\n${errors.slice(0, 20).join("\n")}`);
        return {
            runtime: `chromium ${browser.version()} ${LAUNCH_MODES[plan.seat]} ${tiers} at ${sampledRate.toFixed(1)} Hz sampled, ${controlRate.toFixed(1)} Hz under the control breakpoint`,
            adapter: classifyAdapter(facts).identity,
            loopSite,
            survivors,
            trace,
            warm,
            frames,
            windows: [
                { label: `after warm ${warm}`, ...atWarm },
                { label: `after warm ${2 * warm}`, ...atDoubleWarm },
                { label: "A/A repeat", ...repeat },
            ],
            control: control.sites,
            controlSpan: { label: "control", ...control },
        };
    } finally {
        await browser?.close();
        server?.stop(true);
        rmSync(outDir, { recursive: true, force: true });
    }
}

/**
 * A measured site is `<function> <file>:<line>`; a declaration names the `<file>:<line>` half, the one a
 * minified production build keeps.
 */
export const where = (site: string): string => site.slice(site.indexOf(" ") + 1);

/**
 * The frames one window stepped, derived from the sanctioned sites rather than measured.
 *
 * Each sanctioned site allocates a fixed number of objects per frame, so over a window of F frames a row
 * declaring `count` must read exactly `count * F` allocations. Every row therefore divides to the same F,
 * and F must fall inside the window's bracket. That agreement is what makes the reading exact: one extra
 * allocation, in one frame, at one site either leaves a remainder or moves that row's quotient off the
 * others, and either way it is named.
 *
 * This is derived rather than measured because nothing in the page can be sampled once per frame reliably
 * enough to count them. A per-frame allocation site over-reports: V8 attributes occasional tiering and
 * deoptimisation allocations to whichever JS frame is running, nondeterministically and by more than the
 * bracket's own width, so such a site is an upper bound on frames and never a count. An inflated
 * denominator would shrink every per-frame figure and red the count assertion at the sanctioned sites
 * instead of naming its own cause, so it is not used.
 *
 * Returns the frames, or the reason no single count is consistent with the window.
 */
export function derivedFrames(
    window: AllocationWindow,
    sanctions: readonly { site: string; count: number }[],
): { frames: number } | { reason: string } {
    const quotients = new Map<number, string[]>();
    const ragged: string[] = [];
    for (const row of sanctions) {
        const seen = window.sites.find((site) => where(site.site) === row.site);
        if (seen === undefined) continue;
        if (seen.count % row.count !== 0) {
            // The rate range the bracket allows, so the reading says what the site actually allocates
            // rather than only that the declared count does not divide it.
            const low = (seen.count / window.framesAtMost).toFixed(2);
            const high = (seen.count / window.frames).toFixed(2);
            ragged.push(
                `  ${row.site} read ${seen.count} allocations, which is not a whole number of frames at the declared ${row.count}×/f; over this window's ${window.frames} to ${window.framesAtMost} frames that is ${low} to ${high} per frame`,
            );
            continue;
        }
        const frames = seen.count / row.count;
        quotients.set(frames, [
            ...(quotients.get(frames) ?? []),
            `  ${row.site}: ${seen.count} allocations at ${row.count}×/f is ${frames} frames`,
        ]);
    }
    if (ragged.length > 0)
        return {
            reason: `${window.label}: a sanctioned site did not allocate a whole number of frames' worth:\n${ragged.join("\n")}`,
        };
    if (quotients.size === 0)
        return { reason: `${window.label}: no sanctioned site allocated, so it measured nothing` };
    if (quotients.size > 1)
        return {
            reason: `${window.label}: the sanctioned sites disagree on how many frames this window stepped, so at least one of them is off its declared count:\n${[
                ...quotients.values(),
            ]
                .flat()
                .join("\n")}`,
        };
    const frames = [...quotients.keys()][0];
    if (frames < window.frames || frames > window.framesAtMost)
        return {
            reason: `${window.label}: the sanctioned sites agree on ${frames} frames, which is outside the ${window.frames} to ${window.framesAtMost} frames the page's own counter bracketed this window at, so they agree on something other than the frame count`,
        };
    return { frames };
}

/**
 * The declared-site conditions over a page sample's windows: nothing allocates outside the two
 * declarations, no declared site is stale, and every sanctioned site reads its derived count exactly.
 *
 * Membership and staleness are the same for both classes. Only the count differs: a sanction is what the
 * platform forces, so its samples must equal `count * frames` exactly — no rounding, no tolerance, because
 * one extra allocation in one frame of one window is the defect the count exists to catch. That is
 * asserted by {@link derivedFrames}, which fails unless every row agrees on one frame count inside the
 * window's bracket. A red-circle is real, unwanted allocation the person deferred to a named later gate,
 * so it carries no count and holds at any count until that gate closes it.
 *
 * Returns one message per broken condition, empty when all hold.
 */
export function declaredSiteFailures(
    windows: readonly AllocationWindow[],
    sanctions: readonly { site: string; count: number }[],
    redCircles: readonly { site: string }[],
): string[] {
    const declared = new Set([...sanctions, ...redCircles].map((row) => row.site));
    const failures: string[] = [];
    const undeclared = windows.flatMap((window) =>
        window.sites
            .filter((row) => !declared.has(where(row.site)))
            .map((row) => `  ${window.label}: ${row.bytes} B at ${row.site}`),
    );
    if (undeclared.length > 0)
        failures.push(
            `warm page frames allocate outside the sanctions and red circles:\n${undeclared.join("\n")}`,
        );
    const stale: string[] = [];
    for (const row of [
        ...sanctions.map((row) => ({ site: row.site, noun: "sanction" })),
        ...redCircles.map((row) => ({ site: row.site, noun: "red-circle" })),
    ])
        for (const window of windows)
            if (!window.sites.some((site) => where(site.site) === row.site))
                stale.push(`  ${window.label}: ${row.noun} ${row.site} allocates nothing`);
    if (stale.length > 0) failures.push(`stale declared rows:\n${stale.join("\n")}`);
    for (const window of windows) {
        const derived = derivedFrames(window, sanctions);
        if ("reason" in derived) failures.push(derived.reason);
    }
    return failures;
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
    const perFrame = (bytes: number, frames: number) => (bytes / frames).toFixed(1);
    const totals = sample.windows.map(
        (window) =>
            `${window.label}: ${windowBytes(window)} bytes (${perFrame(windowBytes(window), window.frames)}/f) over ${window.frames} frames at ${window.sites.length} sites`,
    );
    const heaviest = sample.windows.reduce((a, b) => (windowBytes(b) > windowBytes(a) ? b : a));
    const rows = heaviest.sites
        .slice(0, limit)
        .map(
            (row) =>
                `  ${String(row.bytes).padStart(10)}  ${perFrame(row.bytes, heaviest.frames).padStart(8)}/f  ${row.site}`,
        );
    const more =
        heaviest.sites.length > limit ? [`  … ${heaviest.sites.length - limit} more sites`] : [];
    return [
        ...totals,
        `heaviest, ${heaviest.label}, over ${heaviest.frames} frames, ${sample.runtime}`,
        ...rows,
        ...more,
    ].join("\n");
}
