import { isDegradedBootMessage } from "@dylanebert/shallot/harness";
import { expect, type Page, test } from "@playwright/test";
import { adapterName, SOFTWARE } from "./gpu-adapter";

// The showcase measurement instrument: two windows, both read through
// `window.__benchmark.measure(warmup, frames)` (ProfilePlugin's published surface) — idle orbit (no input)
// and a scripted carve drag driven by real pointer events (`page.mouse`, never `locator.fill()`, which
// issues no pointer event and so can't see an interaction cost). Committed here rather than run ad hoc,
// since a figure whose tooling isn't recorded isn't reproducible. Reports frame-interval percentiles, the
// CPU/fence/gap decomposition, and the `voxel:emit` / `sear:*` GPU pass spans.
//
// Gates nothing (no arbitrary wall-clock band — on this seat's 240 Hz monitor the display
// clock dominates both windows identically: p50/p95/p99 ≈ 4.2/4.3/4.3 ms, idle vs carve indistinguishable).
// The only assertions here are instrument-health ones (the page raised no error, the profiler actually
// reported passes) — never a threshold on the numbers this file prints.

const WARMUP = 30;
const FRAMES = 180;

interface PassReading {
    occMs: number;
    occP95: number;
    occP99: number;
    perFrameMs: number;
    firesPerFrame: number;
}

interface WindowReading {
    frame: {
        median: number;
        p95: number;
        p99: number;
        stddev: number;
        cpuMs: number;
        fenceMs: number;
        gapMs: number;
    } | null;
    gpu: Record<string, PassReading>;
    cpuSystems: Record<string, number>;
}

async function measureWindow(page: Page, drive: () => Promise<void>): Promise<WindowReading> {
    const measured = page.evaluate(
        ([warmup, frames]) => window.__benchmark!.measure(warmup, frames),
        [WARMUP, FRAMES],
    );
    await drive();
    const stats = await measured;
    return {
        frame: stats.frame
            ? {
                  median: stats.frame.median,
                  p95: stats.frame.p95,
                  p99: stats.frame.p99,
                  stddev: stats.frame.stddev,
                  cpuMs: stats.frame.cpuMs,
                  fenceMs: stats.frame.fenceMs,
                  gapMs: stats.frame.gapMs,
              }
            : null,
        gpu: stats.gpu?.passes ?? {},
        cpuSystems: stats.cpu?.systems ?? {},
    };
}

function printReading(label: string, r: WindowReading): void {
    console.log(`--- ${label} ---`);
    if (r.frame) {
        console.log(
            `frame(ms): median=${r.frame.median} p95=${r.frame.p95} p99=${r.frame.p99} ` +
                `stddev=${r.frame.stddev} cpu=${r.frame.cpuMs} fence=${r.frame.fenceMs} gap=${r.frame.gapMs}`,
        );
    } else {
        console.log("frame: no samples (window too short or profiler not draining)");
    }
    for (const [name, pass] of Object.entries(r.gpu)) {
        console.log(
            `gpu ${name}: occMs=${pass.occMs} occP95=${pass.occP95} occP99=${pass.occP99} ` +
                `perFrameMs=${pass.perFrameMs} firesPerFrame=${pass.firesPerFrame}`,
        );
    }
    for (const [name, ms] of Object.entries(r.cpuSystems)) {
        console.log(`cpu ${name}: ${ms} ms/frame`);
    }
}

// wait `ms`, holding the driver's async shape (used to span the idle window while `measure` counts frames
// on the page side — no assertion rides on this delay's exact length).
function wait(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

test("measure — idle orbit vs carve drag (recorded, never gated)", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (message) => {
        if (message.type() === "error" || isDegradedBootMessage(message.text())) {
            errors.push(`[console.${message.type()}] ${message.text()}`);
        }
    });

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`voxel measure adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    await page.waitForFunction(
        () => typeof window.__voxelGate === "function" && window.__benchmark?.ready === true,
        null,
        { timeout: 120_000 },
    );

    // idle window: no input, the pointer tool's default orbit view sitting still. `measure`'s own promise
    // is the real condition awaited below; this wait only gives the browser wall-clock time to advance it.
    const DriveMs = 3000;
    const idle = await measureWindow(page, () => wait(DriveMs));

    // carve-drag window: switch to the terrain tool (key B) and drag a real pointer stroke across the
    // canvas centre — left-drag adds, so the surface keeps growing rather than carving toward empty air.
    await page.keyboard.press("b");
    const canvas = page.locator("canvas");
    const box = await canvas.boundingBox();
    if (!box) throw new Error("voxel canvas not found — cannot drive a carve drag");
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;

    const carve = await measureWindow(page, async () => {
        await page.mouse.move(cx - 60, cy);
        await page.mouse.down();
        const steps = 24;
        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            await page.mouse.move(cx - 60 + 120 * t, cy + 30 * Math.sin(t * Math.PI * 2), {
                steps: 3,
            });
            await wait(DriveMs / steps);
        }
        await page.mouse.up();
    });
    await page.keyboard.press("v"); // restore the pointer tool for any test run after this one

    printReading("idle orbit", idle);
    printReading("carve drag", carve);

    expect(errors, errors.join("\n")).toEqual([]);
    // instrument health, not a perf threshold: the profiler must have drained at least one real frame in
    // each window, or every number above is a silent zero rather than a reading.
    expect(idle.frame, "idle window: profiler reported no frame samples").not.toBeNull();
    expect(carve.frame, "carve window: profiler reported no frame samples").not.toBeNull();
});
