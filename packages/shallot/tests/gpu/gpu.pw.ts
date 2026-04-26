import { test, expect, type Page } from "@playwright/test";

interface GpuStats {
    avg: number;
    median: number;
    p5: number;
    p95: number;
    min: number;
    max: number;
    samples: number;
    passes: Record<string, number>;
}

interface CpuStats {
    systems: Record<string, number>;
    total: number;
}

interface FrameStats {
    avg: number;
    median: number;
    p5: number;
    p95: number;
    min: number;
    max: number;
    samples: number;
    clampedFrames: number;
    avgFixedSteps: number;
    maxPending: number;
}

interface MemoryStats {
    start: number;
    end: number;
    growthPerFrame: number;
    leak: boolean;
    gcCount: number;
    gcPauseMs: number;
}

interface CompileStats {
    totalMs: number;
    pipelines: Record<string, number>;
}

interface Measurement {
    gpu: GpuStats | null;
    cpu: CpuStats | null;
    frame: FrameStats | null;
    compile: CompileStats | null;
    frames: number;
}

interface RunResult {
    pipeline: string;
    scenario: string;
    effects: string[];
    camera: string;
    layout: string;
    count: number | undefined;
    warmup: number;
    frames: number;
    gpu: Omit<GpuStats, "passes"> | null;
    passes: Record<string, number>;
    cpu: CpuStats | null;
    frame: FrameStats | null;
    memory: MemoryStats | null;
}

interface FullResults {
    timestamp: string;
    hardware: string;
    runs: RunResult[];
    compile: CompileStats | null;
}

function getConfig() {
    const scenario = process.env.GPU_SCENARIO || "";
    const testVariant = process.env.GPU_TEST || (scenario === "physics" ? "box" : "");
    const pipeline = process.env.GPU_PIPELINE || "raster";
    const warmup = parseInt(process.env.GPU_WARMUP || "60", 10);
    const frames = parseInt(process.env.GPU_FRAMES || "500", 10);
    const count = process.env.GPU_COUNT ? parseInt(process.env.GPU_COUNT, 10) : undefined;
    const effectsStr = process.env.GPU_EFFECTS || "none";
    const effects =
        effectsStr === "none"
            ? []
            : effectsStr === "all"
              ? [
                    "tonemap",
                    "fxaa",
                    "vignette",
                    "bloom",
                    "lensflare",
                    "godrays",
                    "posterize",
                    "dither",
                    "skylab",
                    "sky",
                    "sun",
                    "stars",
                    "moon",
                    "haze",
                    "clouds",
                    "shadows",
                    "reflections",
                ]
              : effectsStr.split(",").map((s) => s.trim().toLowerCase());
    const camera = (process.env.GPU_CAMERA || "static") as "static" | "pan";
    const layout = (process.env.GPU_LAYOUT || "lorenz") as "lorenz" | "grid";
    const room = process.env.GPU_ROOM || "";
    const shapes = process.env.GPU_SHAPES || "";
    return {
        scenario,
        testVariant,
        pipeline,
        warmup,
        frames,
        count,
        effects,
        camera,
        layout,
        room,
        shapes,
    };
}

function buildUrl(cfg: ReturnType<typeof getConfig>): string {
    const params: string[] = [];
    if (cfg.scenario) params.push(`scenario=${cfg.scenario}`);
    if (cfg.testVariant && !cfg.testVariant.includes(",")) params.push(`test=${cfg.testVariant}`);
    if (cfg.room) params.push(`room=${cfg.room}`);
    if (cfg.count !== undefined) params.push(`count=${cfg.count}`);
    if (cfg.effects.length > 0) params.push(`effects=${cfg.effects.join(",")}`);
    if (cfg.camera !== "static") params.push(`camera=${cfg.camera}`);
    if (cfg.layout !== "lorenz") params.push(`layout=${cfg.layout}`);
    if (cfg.shapes) params.push(`shapes=${cfg.shapes}`);
    params.push(`pipeline=${cfg.pipeline}`);
    const qs = params.length > 0 ? `?${params.join("&")}` : "";
    return `http://localhost:3002/${qs}`;
}

async function settle(page: Page, timeout: number): Promise<void> {
    await page.waitForFunction(() => window.__benchmark?.ready === true, null, { timeout });
}

async function measure(page: Page, warmup: number, frames: number): Promise<Measurement> {
    return page.evaluate(({ warmup, frames }) => window.__benchmark.measure(warmup, frames), {
        warmup,
        frames,
    });
}

function printResults(
    label: string,
    effects: string[],
    camera: string,
    r: Measurement,
    mem: MemoryStats | null,
): void {
    const bar = "=".repeat(40);
    console.log(`\n${bar}`);
    const effectsLabel = effects.length > 0 ? ` [${effects.join(",")}]` : "";
    const cameraLabel = camera !== "static" ? ` (${camera})` : "";
    console.log(`  ${label}${effectsLabel}${cameraLabel} Results`);
    console.log(bar);
    console.log(`  Frames measured: ${r.frames}`);
    if (r.gpu) {
        console.log(`  GPU samples:     ${r.gpu.samples}`);
        console.log(`  GPU Avg:    ${r.gpu.avg.toFixed(2)} ms`);
        console.log(`  GPU Median: ${r.gpu.median.toFixed(2)} ms`);
        console.log(`  GPU P5:     ${r.gpu.p5.toFixed(2)} ms   (best 5%)`);
        console.log(`  GPU P95:    ${r.gpu.p95.toFixed(2)} ms   (worst 5%)`);
        console.log(`  GPU Range:  [${r.gpu.min.toFixed(2)} - ${r.gpu.max.toFixed(2)}] ms`);
        const passes = Object.entries(r.gpu.passes).sort((a, b) => b[1] - a[1]);
        if (passes.length > 0) {
            console.log(`  Passes:`);
            for (const [name, ms] of passes) {
                console.log(`    ${name.padEnd(20)} ${ms.toFixed(2)} ms`);
            }
        }
    } else {
        console.log(`  GPU timing unavailable (no timestamp-query support)`);
    }
    if (r.frame) {
        console.log(`  Frame Avg:  ${r.frame.avg.toFixed(2)} ms`);
        console.log(`  Frame P95:  ${r.frame.p95.toFixed(2)} ms`);
        console.log(`  Clamped:    ${r.frame.clampedFrames} frames`);
        console.log(`  Avg steps:  ${r.frame.avgFixedSteps.toFixed(1)}`);
        console.log(`  Pending:  ${r.frame.maxPending} max`);
    }
    if (r.cpu) {
        console.log(`  CPU total:  ${r.cpu.total.toFixed(2)} ms`);
        const systems = Object.entries(r.cpu.systems).sort((a, b) => b[1] - a[1]);
        for (const [name, ms] of systems) {
            console.log(`    ${name.padEnd(20)} ${ms.toFixed(2)} ms`);
        }
    }
    if (mem) {
        const mb = (b: number) => (b / 1024 / 1024).toFixed(1);
        console.log(`  Memory:  ${mb(mem.start)} → ${mb(mem.end)} MB`);
        console.log(
            `  Growth:  ${(mem.growthPerFrame / 1024).toFixed(2)} KB/frame${mem.leak ? " ⚠ LEAK" : ""}`,
        );
        console.log(`  GC:      ${mem.gcCount} collections, ${mem.gcPauseMs.toFixed(1)} ms total`);
    }
    console.log(`${bar}\n`);
}

async function sampleMemory(
    page: Page,
    warmup: number,
    frames: number,
): Promise<MemoryStats | null> {
    try {
        const cdp = await page.context().newCDPSession(page);
        await cdp.send("Performance.enable");

        const getMetric = (
            metrics: { metrics: { name: string; value: number }[] },
            name: string,
        ): number => metrics.metrics.find((m) => m.name === name)?.value ?? 0;

        const initial = await cdp.send("Performance.getMetrics");
        const gcCountStart =
            getMetric(initial, "MinorGCCount") + getMetric(initial, "MajorGCCount");
        const gcDurStart =
            getMetric(initial, "MinorGCDuration") + getMetric(initial, "MajorGCDuration");

        const samples: { frame: number; heap: number }[] = [];
        const interval = 50;
        let frame = 0;
        const total = warmup + frames;

        while (frame < total) {
            const metrics = await cdp.send("Performance.getMetrics");
            const heap = metrics.metrics.find((m) => m.name === "JSHeapUsedSize");
            if (heap) samples.push({ frame, heap: heap.value });
            frame += interval;
            await page.waitForTimeout(interval * 16);
        }

        const final_ = await cdp.send("Performance.getMetrics");
        const gcCount =
            getMetric(final_, "MinorGCCount") + getMetric(final_, "MajorGCCount") - gcCountStart;
        const gcPauseMs =
            (getMetric(final_, "MinorGCDuration") +
                getMetric(final_, "MajorGCDuration") -
                gcDurStart) *
            1000;

        await cdp.send("Performance.disable");
        await cdp.detach();

        if (samples.length < 2) return null;

        const start = samples[0].heap;
        const end = samples[samples.length - 1].heap;
        const n = samples.length;
        const sumX = samples.reduce((s, p) => s + p.frame, 0);
        const sumY = samples.reduce((s, p) => s + p.heap, 0);
        const sumXY = samples.reduce((s, p) => s + p.frame * p.heap, 0);
        const sumX2 = samples.reduce((s, p) => s + p.frame * p.frame, 0);
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const leak = slope > 1024;

        return { start, end, growthPerFrame: slope, leak, gcCount, gcPauseMs };
    } catch {
        return null;
    }
}

test("gpu", async ({ page }) => {
    const cfg = getConfig();
    const perPipeline = 30000 + (cfg.warmup + cfg.frames) * 50;
    test.setTimeout(perPipeline + 30000);

    const gpuErrors: string[] = [];
    page.on("console", (msg) => {
        const text = msg.text();
        if (text.startsWith("[phys")) {
            console.log(text);
        } else if (
            msg.type() === "error" &&
            /webgpu|wgsl|shader|pipeline|destroyed|validation/i.test(text)
        ) {
            gpuErrors.push(text);
        }
    });

    await page.goto(buildUrl(cfg));
    await settle(page, 30000);

    const hasWebGPU = await page.evaluate(async () => {
        if (!navigator.gpu) return false;
        return (await navigator.gpu.requestAdapter()) !== null;
    });
    expect(hasWebGPU).toBe(true);

    const hardware = await page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "unknown";
        const info = adapter.info;
        return (
            [info.vendor, info.architecture, info.device, info.description]
                .filter(Boolean)
                .join(" / ") || "unknown"
        );
    });

    const scenarioSlug = cfg.scenario || "default";
    const memPromise = sampleMemory(page, cfg.warmup, cfg.frames);
    const results = await measure(page, cfg.warmup, cfg.frames);
    const mem = await memPromise;

    printResults(`${scenarioSlug}/${cfg.pipeline}`, cfg.effects, cfg.camera, results, mem);

    if (results.compile) {
        const sorted = Object.entries(results.compile.pipelines).sort((a, b) => b[1] - a[1]);
        console.log(`\n  Compile: ${results.compile.totalMs.toFixed(0)} ms total`);
        for (const [name, ms] of sorted.slice(0, 10)) {
            console.log(`    ${ms.toFixed(1)} ms  ${name}`);
        }
        if (sorted.length > 10) console.log(`    ... +${sorted.length - 10} more`);
    }

    const run: RunResult = {
        pipeline: cfg.pipeline,
        scenario: scenarioSlug,
        effects: cfg.effects,
        camera: cfg.camera,
        layout: cfg.layout,
        count: cfg.count,
        warmup: cfg.warmup,
        frames: cfg.frames,
        gpu: results.gpu
            ? {
                  avg: results.gpu.avg,
                  median: results.gpu.median,
                  p5: results.gpu.p5,
                  p95: results.gpu.p95,
                  min: results.gpu.min,
                  max: results.gpu.max,
                  samples: results.gpu.samples,
              }
            : null,
        passes: results.gpu?.passes ?? {},
        cpu: results.cpu,
        frame: results.frame,
        memory: mem,
    };

    const fullResults: FullResults = {
        timestamp: new Date().toISOString(),
        hardware,
        runs: [run],
        compile: results.compile,
    };
    console.log(`__BENCH_JSON__${JSON.stringify(fullResults)}__BENCH_JSON__`);

    expect(gpuErrors).toEqual([]);
});
