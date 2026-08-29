// Node-runnable driver for the RUM slow-frame intake proof, and (spec `shallot-compile-vitals`
// S2) the `pipeline_compile` two-sided wire proof — both in `scripts/rum-intake-check.ts`.
// Bundled to a node target and spawned with `node` — `scripts/wsl-bridge.ts`'s documented fact 2:
// Bun's Playwright client hangs on this platform (`chromium.launch`/`chromium.connect` never
// resolve), Node's client doesn't. No real GPU is needed here: the RUM sampler
// (`site/rum-runtime.ts`) is rAF-based and main-thread, independent of WebGPU init, so a plain
// local headless launch (software adapter, WebGPU unavailable) is enough to drive it — no
// wsl-bridge host tunnel, no real-hardware contention.
//
// Two instruments per scenario, not one — the machine this runs on shares a box with unrelated
// concurrent work (other specs' dev servers, capture harnesses), so a bare "did any intake
// request carry slow_frame within N seconds" is exposed to ambient main-thread jank unrelated to
// the driven busy-loop. `called`/`callDuration` monkey-patch `DD_RUM.addDurationVital` in-page and
// read its actual arguments back immediately (~1s, no flush wait) — a structural proof of what the
// sampler decided, immune to system contention. `reportedOnWire` is the live network capture: the
// wire proof the spec asks for, required for the above-threshold direction (the direction that
// actually catches a transport-level defect — this proof caught S1's raw-rAF-timestamp-as-epoch
// bug, `site/rum-runtime.ts`'s fix). The below-threshold direction's "no such request leaves" is
// entailed by `called === false` (the SDK cannot batch a vital it was never told about) rather than
// a bounded wait proving a negative over a shared, noisy machine.
//
// `runCompileVitalScenario` (below) runs the analogous two-sided proof for `pipeline_compile`:
// a synthetic `performance.measure` emitted in-page, prefixed or not, is real enough to drive
// `site/rum-runtime.ts`'s `PerformanceObserver` without a real GPU device or a forced compile.
//
// argv: [baseUrl, slug]. Every request to the Datadog RUM intake host
// (`browser-intake-datadoghq.com`, confirmed 2026-08-25 against the served CDN bundle) is
// intercepted and fulfilled locally — never forwarded, so nothing this check does reaches the
// real Datadog org. Prints one JSON line to stdout:
//   { below: ScenarioResult, above: ScenarioResult,
//     compileVitalPrefixed: CompileVitalScenarioResult, compileVitalUnprefixed: CompileVitalScenarioResult }

import { chromium } from "playwright";
// This driver is bundled `--target node` (never browser) and runs in a Node subprocess — the
// same bundle-graph concern `site/rum-compile-vitals.ts`'s docblock records for the *browser*
// bundle doesn't apply here, so importing the real constant is fine and is the wire's own source
// of truth rather than a second hand-copied string.
import { PIPELINE_COMPILE_MEASURE_PREFIX } from "../packages/shallot/src/engine/runtime/gpu";

const INTAKE_HOST_RE = /browser-intake-datadoghq\.com/;
const BELOW_THRESHOLD_MS = 20;
// The measured rAF delta undershoots the busy-loop's own wall-clock duration in this headless,
// display-less environment — a probed sweep (0/20/30/40/50/65ms: never reports; 90ms: 66.7ms
// delta; 120ms: 100ms delta) shows rAF timestamps aren't linear with block duration below
// ~90ms here (no real compositor/vsync to derive them from). 150ms sits well past that knee,
// comfortably clear of the 50ms threshold either way it's measured.
const ABOVE_THRESHOLD_MS = 150;

// The SDK batches and flushes on a periodic timer, not on `page.goto("about:blank")` (measured
// 2026-08-25: a navigate-away 1.5s after the busy-loop sent nothing; live runs against the real
// Dogfood intake flushed ~29-30s after session start, consistently). No documented constant to
// derive this from (the CDN bundle is minified, no source map), so this is an empirical floor with
// real margin, not a tuned fit: poll well past the observed interval and stop as soon as a match
// lands, rather than sleeping the ceiling every time.
const FLUSH_POLL_MS = 2_000;
const FLUSH_CEILING_MS = 45_000;
// The negative direction's network sanity window: short on purpose (well under one flush cycle) —
// it isn't waited out to a real flush, since "no request" can't be proven by a bounded wait ending
// early. `called === false` is the actual proof; this window only surfaces an immediate anomaly.
const NEGATIVE_SANITY_WAIT_MS = 8_000;

interface ScenarioResult {
    busyMs: number;
    called: boolean;
    callDuration: number | null;
    reportedOnWire: boolean;
    // S2p (`shallot-demo-startup-stall`): the LoAF attribution context attached to the vital, so
    // the above-threshold direction can assert the correlator actually saw the busy-loop's own
    // long-animation-frame entry rather than reading a false "idle main thread" (site/rum-loaf.ts,
    // site/rum-runtime.ts's deferred read).
    loafEntryCount: number | null;
}

type RumWindow = Window & {
    // biome-ignore lint/style/useNamingConvention: DD_RUM is the SDK's own global name
    DD_RUM?: {
        addDurationVital: (
            name: string,
            opts: { duration: number; context?: Record<string, unknown> },
        ) => void;
    };
    __vitalCalls?: { name: string; duration: number; context?: Record<string, unknown> }[];
};

async function runScenario(
    url: string,
    busyMs: number,
    expectReport: boolean,
): Promise<ScenarioResult> {
    const browser = await chromium.launch({ headless: true });
    try {
        const context = await browser.newContext();
        const page = await context.newPage();
        const bodies: string[] = [];

        // every intake request is intercepted and fulfilled locally — never forwarded.
        await page.route("**/*", async (route) => {
            const req = route.request();
            if (INTAKE_HOST_RE.test(req.url())) {
                bodies.push(req.postData() ?? "");
                await route.fulfill({ status: 202, body: "" });
            } else {
                await route.continue();
            }
        });

        await page.goto(url, { waitUntil: "load" });
        await page.waitForFunction(
            () => typeof (window as unknown as RumWindow).DD_RUM?.addDurationVital === "function",
            null,
            { timeout: 30_000 },
        );

        // monkey-patch the real call so we can read back what the sampler actually told the SDK,
        // immediately — no dependency on the SDK's own batch/flush timing.
        await page.evaluate(() => {
            const w = window as unknown as RumWindow;
            w.__vitalCalls = [];
            const orig = w.DD_RUM?.addDurationVital.bind(w.DD_RUM);
            if (w.DD_RUM && orig) {
                w.DD_RUM.addDurationVital = (name, opts) => {
                    w.__vitalCalls?.push({ name, duration: opts.duration, context: opts.context });
                    return orig(name, opts);
                };
            }
        });

        // let a couple of natural frames pass first, so the sampler has a real previous
        // timestamp to diff against (`rum-sampler.ts`'s `sampleFrame` first-frame rule).
        await page.waitForTimeout(300);

        await page.evaluate((ms) => {
            const start = performance.now();
            // deliberate main-thread busy-loop — the forced (or not-forced) slow frame.
            while (performance.now() - start < ms) {
                // spin
            }
        }, busyMs);

        // a couple more rAF ticks for the sampler to observe the delta, plus `rum-runtime.ts`'s
        // own deferred LoAF read (`LOAF_ATTRIBUTION_DELAY_MS`) before it calls addDurationVital —
        // margin above that deferral, not just above the sampler's own rAF cadence.
        await page.waitForTimeout(1_500);

        const calls = await page.evaluate(
            () => (window as unknown as RumWindow).__vitalCalls ?? [],
        );
        const slowFrameCalls = calls.filter((c) => c.name === "slow_frame");
        const called = slowFrameCalls.length > 0;
        const callDuration = slowFrameCalls[0]?.duration ?? null;
        const loafEntryCount =
            (slowFrameCalls[0]?.context?.loafEntryCount as number | undefined) ?? null;

        if (expectReport) {
            // wire proof required: poll for the periodic batch flush to reach the intake host,
            // rather than a fixed sleep — stop the moment a match lands.
            const deadline = Date.now() + FLUSH_CEILING_MS;
            while (Date.now() < deadline && !bodies.some((b) => b.includes("slow_frame"))) {
                await page.waitForTimeout(FLUSH_POLL_MS);
            }
        } else {
            // sanity window only — not waited out to a real flush (see module docblock).
            await page.waitForTimeout(NEGATIVE_SANITY_WAIT_MS);
        }

        return {
            busyMs,
            called,
            callDuration,
            reportedOnWire: bodies.some((b) => b.includes("slow_frame")),
            loafEntryCount,
        };
    } finally {
        await browser.close();
    }
}

interface CompileVitalScenarioResult {
    /** Whether `DD_RUM.addDurationVital` was called with `"pipeline_compile"` — the structural
     * proof, read back immediately (no flush wait), same shape as `runScenario`'s `called`. */
    called: boolean;
    /** Whether the intercepted intake body carried a `pipeline_compile` vital — the wire proof,
     * required for the positive direction (this is what caught S1's own
     * raw-rAF-timestamp-as-epoch bug for `slow_frame`, the analogous bug this arm exists to catch
     * for `pipeline_compile`). */
    reportedOnWire: boolean;
}

// The `pipeline_compile` two-sided wire proof (spec `shallot-compile-vitals` S2): a synthetic
// `performance.measure` emitted in-page, prefixed or not, drives `site/rum-runtime.ts`'s own
// `PerformanceObserver({ type: "measure", buffered: true })` — this is the same mechanism a real
// forced compile drives (`gpu.ts`'s `reportCompile`, called from both `compileValidated`'s serial
// path and `precompileAll`'s batch-then-bisect fast path), just without needing a real GPU device.
// Two instruments, same reason `runScenario` above carries two: `called` is read back
// immediately after the synthetic measure with no flush wait, so it's the primary verdict for
// the negative direction — a wire-only negative check bounded by a *sanity* window (not a full
// flush wait, since "no request" can't be proven by a bounded wait ending early) passed with the
// prefix filter itself deleted, discovered by mutation-testing this arm: the SDK's flush cadence
// in this environment outlasted the sanity window either way, so the wire read never distinguished
// a working filter from a disabled one. `reportedOnWire` stays the required check for the
// positive direction, which does cross the network boundary.
async function runCompileVitalScenario(
    url: string,
    prefixed: boolean,
): Promise<CompileVitalScenarioResult> {
    const browser = await chromium.launch({ headless: true });
    try {
        const context = await browser.newContext();
        const page = await context.newPage();
        const bodies: string[] = [];

        await page.route("**/*", async (route) => {
            const req = route.request();
            if (INTAKE_HOST_RE.test(req.url())) {
                bodies.push(req.postData() ?? "");
                await route.fulfill({ status: 202, body: "" });
            } else {
                await route.continue();
            }
        });

        await page.goto(url, { waitUntil: "load" });
        await page.waitForFunction(
            () => typeof (window as unknown as RumWindow).DD_RUM?.addDurationVital === "function",
            null,
            { timeout: 30_000 },
        );

        // monkey-patch the real call so we can read back what the observer actually told the
        // SDK, immediately — no dependency on the SDK's own batch/flush timing (same shape as
        // `runScenario` above).
        await page.evaluate(() => {
            const w = window as unknown as RumWindow;
            w.__vitalCalls = [];
            const orig = w.DD_RUM?.addDurationVital.bind(w.DD_RUM);
            if (w.DD_RUM && orig) {
                w.DD_RUM.addDurationVital = (name, opts) => {
                    w.__vitalCalls?.push({ name, duration: opts.duration, context: opts.context });
                    return orig(name, opts);
                };
            }
        });

        // a couple of natural frames first — `rum-runtime.ts`'s own observer is constructed at
        // module top level, before this, so by the time the page has loaded it's already
        // observing; this margin is just to let the page settle, not to wait for the observer.
        await page.waitForTimeout(300);

        const measureName = prefixed
            ? `${PIPELINE_COMPILE_MEASURE_PREFIX}intake-check-synthetic`
            : "intake-check-synthetic-unprefixed";
        await page.evaluate((name) => {
            const start = performance.now();
            performance.measure(name, { start, end: start + 5 });
        }, measureName);

        // the observer's callback is synchronous on `measure` (no `LOAF_ATTRIBUTION_DELAY_MS`
        // deferral the slow_frame path has), so a short margin is enough for `called`.
        await page.waitForTimeout(500);

        const calls = await page.evaluate(
            () => (window as unknown as RumWindow).__vitalCalls ?? [],
        );
        const called = calls.some((c) => c.name === "pipeline_compile");

        if (prefixed) {
            // wire proof required for the positive direction — poll for the periodic batch flush
            // to reach the intake host, stopping the moment a match lands.
            const deadline = Date.now() + FLUSH_CEILING_MS;
            while (Date.now() < deadline && !bodies.some((b) => b.includes("pipeline_compile"))) {
                await page.waitForTimeout(FLUSH_POLL_MS);
            }
        } else {
            // negative direction: `called === false` is the whole proof (the SDK cannot batch a
            // vital it was never told about), so no flush wait is owed here at all.
        }

        return { called, reportedOnWire: bodies.some((b) => b.includes("pipeline_compile")) };
    } finally {
        await browser.close();
    }
}

async function main(): Promise<void> {
    const [baseUrl, slug] = process.argv.slice(2);
    if (!baseUrl || !slug) {
        console.error("usage: node rum-intake-driver.js <baseUrl> <slug>");
        process.exit(2);
    }
    const url = `${baseUrl}/${slug}/`;
    const below = await runScenario(url, BELOW_THRESHOLD_MS, false);
    const above = await runScenario(url, ABOVE_THRESHOLD_MS, true);
    const compileVitalPrefixed = await runCompileVitalScenario(url, true);
    const compileVitalUnprefixed = await runCompileVitalScenario(url, false);
    console.log(
        JSON.stringify({
            below,
            above,
            compileVitalPrefixed,
            compileVitalUnprefixed,
        }),
    );
}

if (import.meta.main) {
    main().catch((err) => {
        console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
        process.exit(1);
    });
}
