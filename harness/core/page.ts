import { test, expect, type Page } from "@playwright/test";

// The one Playwright entry point. The launcher (gym / lab) serializes a PageSpec as
// HARNESS_SPEC, this drives the page through the single `window.__harness` contract, and
// the result comes back on stdout as a __HARNESS_RESULT__ envelope. There is no mode
// branch: every example — a gym scenario — returns its Verdict from
// `__harness.run(opts)`. The console is a diagnostics channel only (error capture +
// `[phys…]` progress), never the result transport.
//
// Self-contained by design: no relative imports, so the WSL→Windows staging copies this
// file alone. The Verdict is opaque JSON here — only the launcher decodes its shape.

interface PageSpec {
    url: string;
    warmup: number;
    frames: number;
    timeoutMs: number;
    // how long to wait for the scene to build + settle (window.__harness.ready). A heavy scenario (a
    // 65536-body physics pile) outlasts the 30s default; raise it via the launcher's --timeout.
    readyTimeoutMs: number;
    sampleMemory: boolean;
    sampleAlloc: boolean;
    // when set, write a post-run canvas screenshot to this (run-dir-relative) path
    screenshot?: string;
}

// one allocating call site, attributed by the heap sampling profiler — the hunt tool's "where".
interface Allocator {
    name: string;
    location: string;
    bytes: number;
}

// what `window.__probeAlloc(windowMs)` returns to a scenario's assert: heap growth + GC activity over
// a no-forced-GC window, plus the top allocators by sampled self-size. The assert reads `Compute.frame`
// itself for the per-frame denominator; this stays frame-agnostic (it only knows wall-clock).
interface AllocProbe {
    heapDelta: number;
    gcCount: number;
    gcPauseMs: number;
    top: Allocator[];
}

interface MemoryStats {
    start: number;
    end: number;
    growthPerFrame: number;
    leak: boolean;
    gcCount: number;
    gcPauseMs: number;
}

interface ResultEnvelope {
    ok: boolean;
    hardware: string;
    verdict?: unknown;
    memory?: MemoryStats | null;
    errors?: string[];
    fatal?: string;
}

// Match real failure signatures only — "adapter limits" / "requestAdapter" are normal
// startup chatter, not failures.
const ERR_HINT =
    /\b(?:wgsl|shader compilation|pipeline.*invalid|destroyed|validation error|device.*lost|uncaptured|GPUValidationError|GPUInternalError|exceeds the max|crashed)\b/i;

function loadSpec(): PageSpec {
    const raw = process.env.HARNESS_SPEC;
    if (!raw) throw new Error("HARNESS_SPEC env var not set");
    return JSON.parse(raw) as PageSpec;
}

function emit(envelope: ResultEnvelope): void {
    console.log(`__HARNESS_RESULT__${JSON.stringify(envelope)}__HARNESS_RESULT__`);
}

async function readHardware(page: Page): Promise<string> {
    return page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "unknown";
        const info = adapter.info;
        return (
            [info.vendor, info.architecture, info.device, info.description]
                .filter(Boolean)
                .join(" / ") || "unknown"
        );
    });
}

type Metrics = { metrics: { name: string; value: number }[] };
const metricOf = (m: Metrics, name: string): number =>
    m.metrics.find((x) => x.name === name)?.value ?? 0;

// CDP sampling-profile node: a call frame + its self-allocated bytes + children. Reduce the tree to
// the top allocators by self-size — the attributed "where" the GC-pause hunt reads.
interface HeapNode {
    callFrame: { functionName: string; url: string; lineNumber: number };
    selfSize: number;
    children?: HeapNode[];
}

function topAllocators(node: HeapNode | undefined, n: number): Allocator[] {
    const by = new Map<string, Allocator>();
    const walk = (h: HeapNode | undefined): void => {
        if (!h) return;
        if (h.selfSize > 0) {
            const cf = h.callFrame;
            const name = cf.functionName || "(anonymous)";
            const location = `${cf.url.split("/").pop() ?? cf.url}:${cf.lineNumber + 1}`;
            const key = `${name}@${location}`;
            const a = by.get(key);
            if (a) a.bytes += h.selfSize;
            else by.set(key, { name, location, bytes: h.selfSize });
        }
        if (h.children) for (const c of h.children) walk(c);
    };
    walk(node);
    return [...by.values()].sort((a, b) => b.bytes - a.bytes).slice(0, n);
}

// Samples JS heap via CDP while the run is in flight. Example-agnostic — it watches the
// heap, not the scenario. Returns null when sampling isn't possible (non-Chromium, etc.).
async function sampleMemory(
    page: Page,
    warmup: number,
    frames: number,
): Promise<MemoryStats | null> {
    try {
        const cdp = await page.context().newCDPSession(page);
        await cdp.send("Performance.enable");

        const initial = await cdp.send("Performance.getMetrics");
        const gcCountStart =
            metricOf(initial, "MinorGCCount") + metricOf(initial, "MajorGCCount");
        const gcDurStart =
            metricOf(initial, "MinorGCDuration") + metricOf(initial, "MajorGCDuration");

        const samples: { frame: number; heap: number }[] = [];
        const interval = 50;
        let frame = 0;
        const total = warmup + frames;

        while (frame < total) {
            // Force a GC before each sample so the heap reflects *retained* memory, not the
            // transient per-frame allocations (profiler sample arrays, overlay text churn)
            // that pile up uncollected when no natural GC fires in the short window — those
            // read as linear growth and trip the leak heuristic on a scenario that holds
            // nothing. With GC, the slope measures a real leak. collectGarbage needs no
            // domain enable; swallow if the backend lacks it (sampling still proceeds).
            await cdp.send("HeapProfiler.collectGarbage").catch(() => {});
            const metrics = await cdp.send("Performance.getMetrics");
            const heap = metrics.metrics.find((m) => m.name === "JSHeapUsedSize");
            if (heap) samples.push({ frame, heap: heap.value });
            frame += interval;
            await page.waitForTimeout(interval * 16);
        }

        const final_ = await cdp.send("Performance.getMetrics");
        const gcCount =
            metricOf(final_, "MinorGCCount") + metricOf(final_, "MajorGCCount") - gcCountStart;
        const gcPauseMs =
            (metricOf(final_, "MinorGCDuration") +
                metricOf(final_, "MajorGCDuration") -
                gcDurStart) *
            1000;

        await cdp.send("Performance.disable");
        await cdp.detach();

        if (samples.length < 2) return null;

        // Drop the cold-start sample before fitting: the first reading is taken before the
        // workload allocates its steady-state working set (pipeline caches, profiler maps),
        // so it biases the slope upward on a scenario that retains nothing. A leak is
        // steady-state growth — same reason the timing measure discards warmup frames.
        const fit = samples.length > 2 ? samples.slice(1) : samples;
        const start = fit[0].heap;
        const end = fit[fit.length - 1].heap;
        const n = fit.length;
        const sumX = fit.reduce((s, p) => s + p.frame, 0);
        const sumY = fit.reduce((s, p) => s + p.heap, 0);
        const sumXY = fit.reduce((s, p) => s + p.frame * p.heap, 0);
        const sumX2 = fit.reduce((s, p) => s + p.frame * p.frame, 0);
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const leak = slope > 1024;

        return { start, end, growthPerFrame: slope, leak, gcCount, gcPauseMs };
    } catch {
        return null;
    }
}

// Expose `window.__probeAlloc(windowMs)` — the GC-pause hunt's measurement primitive. Each call forces
// one GC for a clean baseline, then samples allocations for `windowMs` WITHOUT forcing GC (so transient
// per-frame allocation is visible, not collected away), and returns heap growth + GC count + the top
// allocators by sampled self-size. The scenario's assert ramps a work knob between calls and reads the
// slope. Best-effort: a CDP/HeapProfiler failure leaves the binding uninstalled and the assert skips.
async function installAllocProbe(page: Page): Promise<void> {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Performance.enable");
    await page.exposeFunction("__probeAlloc", async (windowMs: number): Promise<AllocProbe> => {
        await cdp.send("HeapProfiler.collectGarbage").catch(() => {});
        const m0 = await cdp.send("Performance.getMetrics");
        const heap0 = metricOf(m0, "JSHeapUsedSize");
        const gc0 = metricOf(m0, "MinorGCCount") + metricOf(m0, "MajorGCCount");
        const dur0 = metricOf(m0, "MinorGCDuration") + metricOf(m0, "MajorGCDuration");

        // Enable + sample only for the window. The HeapProfiler's allocation-tracking instrumentation taxes
        // every system's timing while enabled, so leaving it on for the whole run inflates the other axes'
        // per-system CPU p99 (it tripped submission's even-pacing spike guard). Confine it to the window.
        // 4 KB average sampling interval (vs the 32 KB default) — fine enough to attribute a per-frame
        // allocation to its stack without taxing the window.
        await cdp.send("HeapProfiler.enable").catch(() => {});
        await cdp.send("HeapProfiler.startSampling", { samplingInterval: 4096 }).catch(() => {});
        await new Promise((r) => setTimeout(r, windowMs)); // no forced GC — the engine RAF loop runs
        const m1 = await cdp.send("Performance.getMetrics");
        const profile = await cdp
            .send("HeapProfiler.stopSampling")
            .then((p) => (p as { profile?: { head: HeapNode } }).profile?.head)
            .catch(() => undefined);
        await cdp.send("HeapProfiler.disable").catch(() => {});

        return {
            heapDelta: metricOf(m1, "JSHeapUsedSize") - heap0,
            gcCount: metricOf(m1, "MinorGCCount") + metricOf(m1, "MajorGCCount") - gc0,
            gcPauseMs:
                (metricOf(m1, "MinorGCDuration") + metricOf(m1, "MajorGCDuration") - dur0) * 1000,
            top: topAllocators(profile, 8),
        };
    });
}

test("harness", async ({ page }) => {
    const spec = loadSpec();
    test.setTimeout(spec.timeoutMs + 30_000);

    const errors: string[] = [];
    const transcript: { kind: string; text: string; when: number }[] = [];
    const t0 = Date.now();
    const record = (kind: string, text: string) => {
        const when = Date.now() - t0;
        transcript.push({ kind, text, when });
        if (kind === "err" || kind === "page-error" || ERR_HINT.test(text)) errors.push(text);
    };

    page.on("console", (msg) => {
        const text = msg.text();
        // stream progress markers straight through so a long run shows life
        if (text.startsWith("[phys")) {
            process.stdout.write(`  ${text}\n`);
            return;
        }
        const type = msg.type();
        const kind = type === "error" ? "err" : type === "warning" ? "warn" : "log";
        record(kind, text);
    });
    page.on("pageerror", (err) => {
        record("page-error", `${err.name}: ${err.message}\n${err.stack ?? ""}`);
    });
    page.on("crash", () => record("page-error", "page crashed"));

    const dumpTranscript = (header: string) => {
        process.stdout.write(`\n${header}\n`);
        for (const l of transcript) {
            process.stdout.write(
                `  [${String(l.when).padStart(6)}ms ${l.kind.padEnd(10)}] ${l.text}\n`,
            );
        }
    };

    await page.goto(spec.url);

    try {
        await page.waitForFunction(() => window.__harness?.ready === true, null, {
            timeout: spec.readyTimeoutMs,
        });
    } catch {
        dumpTranscript("=== full transcript (window.__harness never became ready) ===");
        emit({
            ok: false,
            hardware: "unknown",
            errors,
            fatal: `window.__harness.ready never became true within ${spec.readyTimeoutMs}ms`,
        });
        throw new Error("window.__harness.ready never became true");
    }

    const hasWebGPU = await page.evaluate(async () => {
        if (!navigator.gpu) return false;
        return (await navigator.gpu.requestAdapter()) !== null;
    });
    expect(hasWebGPU).toBe(true);

    const hardware = await readHardware(page);
    if (spec.sampleAlloc) {
        try {
            await installAllocProbe(page);
        } catch (err) {
            record("warn", `alloc probe unavailable: ${err instanceof Error ? err.message : err}`);
        }
    }
    const memPromise = spec.sampleMemory
        ? sampleMemory(page, spec.warmup, spec.frames)
        : Promise.resolve(null);

    let verdict: unknown;
    try {
        verdict = await page.evaluate(
            (opts) => window.__harness!.run(opts),
            { warmup: spec.warmup, frames: spec.frames },
        );
    } catch (err) {
        const memory = await memPromise;
        const fatal = err instanceof Error ? err.message : String(err);
        dumpTranscript("=== full transcript (__harness.run threw) ===");
        emit({ ok: false, hardware, memory, errors, fatal });
        throw new Error(`__harness.run failed: ${fatal}`);
    }

    const memory = await memPromise;
    if (errors.length > 0) dumpTranscript("=== full transcript (had errors) ===");

    // optional visual smoke test: the scene's RAF loop keeps rendering after run() returns, so awaiting a
    // couple of frames lands a fresh one, then a viewport screenshot captures it. Written to the run dir;
    // the core copies it back. Best-effort — a screenshot failure never fails the run.
    if (spec.screenshot) {
        try {
            await page.evaluate(
                () => new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r()))),
            );
            await page.screenshot({ path: spec.screenshot });
        } catch (err) {
            record("warn", `screenshot failed: ${err instanceof Error ? err.message : String(err)}`);
        }
    }

    emit({ ok: errors.length === 0, hardware, verdict, memory, errors: errors.length ? errors : undefined });
    expect(errors).toEqual([]);
});
