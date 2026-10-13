import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Browser, type BrowserContext, chromium, type Page } from "playwright";

const durationMs = 12_000;
const categories = "cc,viz,gpu,benchmark,blink,blink.user_timing,toplevel";
const preferred = "navigator.gpu.getPreferredCanvasFormat()";
const arms = [
    { name: "baseline", browser: "chrome", mode: "webgpu" },
    { name: "pixels-2x-css", browser: "chrome", mode: "webgpu", pixels: 2 },
    { name: "pixels-1x-css", browser: "chrome", mode: "webgpu", pixels: 1 },
    { name: "ozone-x11", browser: "chrome", mode: "webgpu", ozone: "x11" },
    { name: "opaque-alpha", browser: "chrome", mode: "webgpu", alpha: "opaque" },
    { name: "alternative-format", browser: "chrome", mode: "webgpu", alternate: true },
    { name: "chromium", browser: "chromium", mode: "webgpu" },
    { name: "css-only", browser: "chrome", mode: "css" },
] as const;

type Arm = (typeof arms)[number];
const selfTest = process.argv.includes("--self-test");
const selected = process.argv.slice(2).filter((argument) => argument !== "--self-test");
const chosen = selected.length ? arms.filter((arm) => selected.includes(arm.name)) : arms;
if (!chosen.length || selected.some((name) => !arms.some((arm) => arm.name === name))) {
    throw new Error(`Arm names: ${arms.map(({ name }) => name).join(", ")}`);
}

function html(arm: Arm): string {
    const alteration =
        "alternate" in arm && arm.alternate
            ? `format = format === 'rgba8unorm' ? 'bgra8unorm' : 'rgba8unorm';`
            : "";
    const pixelSizing =
        "pixels" in arm
            ? `canvas.width = Math.round(innerWidth * ${arm.pixels}); canvas.height = Math.round(innerHeight * ${arm.pixels});`
            : "canvas.width = Math.round(innerWidth * devicePixelRatio); canvas.height = Math.round(innerHeight * devicePixelRatio);";
    const configure = `
    (async () => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      const info = adapter.info;
      const device = await adapter.requestDevice();
      const gpuErrors = [];
      device.addEventListener('uncapturederror', event => gpuErrors.push(String(event.error?.message || event.error)));
      device.lost.then(info => gpuErrors.push('device lost: ' + info.reason + ' ' + info.message));
      window.__gpuErrors = gpuErrors;
      const canvas = document.querySelector('canvas');
      const context = canvas.getContext('webgpu');
      let format = ${preferred}; ${alteration}
      ${pixelSizing}
      context.configure({ device, format, alphaMode: '${"alpha" in arm ? arm.alpha : "premultiplied"}' });
      let stopped = false;
      let animationFrame = 0;
      window.stopAnimation = () => { stopped = true; cancelAnimationFrame(animationFrame); };
      function draw(t) {
        if (stopped) return;
        const texture = context.getCurrentTexture();
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), clearValue: {r:0.08,g:0.1,b:0.18,a:1}, loadOp:'clear', storeOp:'store' }] });
        const x = Math.floor((t * 0.12) % canvas.width);
        pass.setScissorRect(x, Math.floor(canvas.height/2)-20, Math.min(90,canvas.width-x), 40);
        pass.setPipeline(pipeline);
          pass.draw(3);
        pass.end(); device.queue.submit([encoder.finish()]); animationFrame = requestAnimationFrame(draw);
      }
      const module = device.createShaderModule({code: 'struct O{@builtin(position) p:vec4f}; @vertex fn vs(@builtin(vertex_index) i:u32)->O{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3)); var o:O; o.p=vec4f(p[i],0,1); return o;} @fragment fn fs()->@location(0) vec4f{return vec4f(1,1,1,1);}' });
      const pipeline = device.createRenderPipeline({layout:'auto',vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format}]},primitive:{topology:'triangle-list'}});
      window.runInfo = { mode:'webgpu', adapter:{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:info.isFallbackAdapter ?? adapter.isFallbackAdapter}, format, alphaMode:'${"alpha" in arm ? arm.alpha : "premultiplied"}' };
      animationFrame = requestAnimationFrame(draw);
    })().catch(error => { window.runError = String(error.stack || error); });
  `;
    if (arm.mode === "css")
        return `<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#141a2e}.bar{position:absolute;left:0;top:calc(50% - 20px);width:90px;height:40px;background:white;animation:move 3s linear infinite}@keyframes move{to{transform:translateX(3840px)}}</style><div class="bar"></div><script>window.runInfo={mode:'css'};window.stopAnimation=()=>document.getAnimations().forEach(animation=>animation.cancel())</script>`;
    return `<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#141a2e}canvas{display:block;width:100vw;height:100vh}</style><canvas></canvas><script>${configure}</script>`;
}

type TraceSession = Awaited<ReturnType<Browser["newBrowserCDPSession"]>>;
type TraceEvent = {
    name?: string;
    ph?: string;
    ts?: number;
    dur?: number;
    args?: { data?: { name?: string } };
};
type TraceComplete = { stream?: string };

function stopTrace(session: TraceSession): Promise<TraceComplete> {
    const complete = new Promise<TraceComplete>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("Tracing did not complete")), 30_000);
        session.once("Tracing.tracingComplete", (event) => {
            clearTimeout(timeout);
            resolve(event);
        });
    });
    return session.send("Tracing.end").then(() => complete);
}

async function traceText(
    session: TraceSession,
    completePromise: Promise<TraceComplete>,
): Promise<string> {
    const complete = await completePromise;
    if (!complete.stream) throw new Error("Tracing completed without a stream");
    let text = "";
    while (true) {
        const part = await session.send("IO.read", { handle: complete.stream, size: 10_000_000 });
        text += part.data;
        if (part.eof) break;
    }
    await session.send("IO.close", { handle: complete.stream });
    return text;
}

function traceMark(events: TraceEvent[], name: string): number | undefined {
    return events.find(
        (event) =>
            (event.name === "UserTiming" && event.args?.data?.name === name) || event.name === name,
    )?.ts;
}

function longSwapsInWindow(events: TraceEvent[], start: number, end: number): TraceEvent[] {
    return events.filter(
        (event) =>
            event.name === "SkiaOutputSurfaceImplOnGpu::SwapBuffers" &&
            event.ph === "X" &&
            (event.dur ?? 0) > 10_000 &&
            (event.ts ?? -Infinity) >= start &&
            (event.ts ?? Infinity) < end,
    );
}

function verifyTraceWindowSelection(): void {
    const events: TraceEvent[] = [
        { name: "UserTiming", ts: 100, args: { data: { name: "measurement-start" } } },
        { name: "SkiaOutputSurfaceImplOnGpu::SwapBuffers", ph: "X", ts: 99, dur: 20_000 },
        { name: "SkiaOutputSurfaceImplOnGpu::SwapBuffers", ph: "X", ts: 100, dur: 11_000 },
        { name: "UserTiming", ts: 200, args: { data: { name: "measurement-end" } } },
        { name: "SkiaOutputSurfaceImplOnGpu::SwapBuffers", ph: "X", ts: 200, dur: 30_000 },
    ];
    const start = traceMark(events, "measurement-start");
    const end = traceMark(events, "measurement-end");
    if (start !== 100 || end !== 200 || longSwapsInWindow(events, start, end).length !== 1)
        throw new Error("Synthetic trace-window boundary check failed");
}

verifyTraceWindowSelection();

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

async function run(arm: Arm) {
    const profile = await mkdtemp(join(tmpdir(), "shallot-webgpu-stall-"));
    let browser: Browser | undefined;
    let context: BrowserContext | undefined;
    const server = createServer((_request, response) => {
        response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        response.end(html(arm));
    });
    try {
        await new Promise<void>((resolve, reject) => {
            server.once("error", reject);
            server.listen(0, "127.0.0.1", resolve);
        });
        const address = server.address();
        if (!address || typeof address === "string")
            throw new Error("Could not bind diagnostic server");
        const executablePath = arm.browser === "chromium" ? "/usr/bin/chromium" : undefined;
        const ozone = "ozone" in arm ? arm.ozone : "wayland";
        const launchArgs = [
            "--kiosk",
            "--start-fullscreen",
            "--enable-unsafe-webgpu",
            `--ozone-platform=${ozone}`,
        ];
        context = await chromium.launchPersistentContext(profile, {
            ...(executablePath ? { executablePath } : { channel: "chrome" }),
            headless: false,
            viewport: null,
            args: launchArgs,
            env: { ...process.env, WAYLAND_DISPLAY: "wayland-1" },
            timeout: 30_000,
        });
        browser = context.browser() ?? undefined;
        await context.addInitScript(() => {
            const frames: number[] = [];
            (window as any).__frames = frames;
            (window as any).__recordFrames = false;
            function sample(t: number) {
                if ((window as any).__recordFrames) frames.push(t);
                requestAnimationFrame(sample);
            }
            requestAnimationFrame(sample);
        });
        const page: Page = await context.newPage();
        const windowSession = await context.newCDPSession(page);
        const { windowId } = await windowSession.send("Browser.getWindowForTarget");
        await windowSession.send("Browser.setWindowBounds", {
            windowId,
            bounds: { windowState: "fullscreen" },
        });
        await page.waitForFunction(
            () =>
                innerWidth === screen.width &&
                innerHeight === screen.height &&
                devicePixelRatio > 1.2,
            undefined,
            { timeout: 10_000 },
        );
        const browserSession = await browser!.newBrowserCDPSession();
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        const gpuInfo = await browserSession.send("SystemInfo.getInfo");
        const commandLine = String(gpuInfo.commandLine ?? "");
        const launchBackendFlags = commandLine
            .split(" ")
            .filter(
                (arg) =>
                    arg.startsWith("--ozone-platform") ||
                    arg.startsWith("--use-gl") ||
                    arg.startsWith("--use-angle") ||
                    arg.startsWith("--render-node-override"),
            );
        await page.goto(`http://localhost:${address.port}/`);
        await page.evaluate(
            () =>
                new Promise<void>((resolve) => {
                    const poll = () =>
                        (window as any).runInfo || (window as any).runError
                            ? resolve()
                            : setTimeout(poll, 20);
                    poll();
                }),
        );
        const initial = await page.evaluate(() => ({
            info: (window as any).runInfo,
            error: (window as any).runError,
            dpr: devicePixelRatio,
            viewport: { width: innerWidth, height: innerHeight },
            screen: { width: screen.width, height: screen.height },
            canvas: document.querySelector("canvas")
                ? {
                      width: (document.querySelector("canvas") as HTMLCanvasElement).width,
                      height: (document.querySelector("canvas") as HTMLCanvasElement).height,
                  }
                : null,
            userAgent: navigator.userAgent,
        }));
        if (initial.error) throw new Error(`${arm.name}: ${initial.error}`);
        if (
            initial.viewport.width !== initial.screen.width ||
            initial.viewport.height !== initial.screen.height ||
            initial.dpr < 1.2
        )
            throw new Error(`${arm.name}: fullscreen Wayland dimensions/DPR not established`);
        if (
            initial.canvas &&
            (initial.canvas.width !==
                Math.round(
                    initial.viewport.width *
                        ("pixels" in arm ? arm.pixels : 1) *
                        ("pixels" in arm ? 1 : initial.dpr),
                ) ||
                initial.canvas.height !==
                    Math.round(
                        initial.viewport.height *
                            ("pixels" in arm ? arm.pixels : 1) *
                            ("pixels" in arm ? 1 : initial.dpr),
                    ))
        )
            throw new Error(
                `${arm.name}: canvas dimensions do not match the measured layout and DPR`,
            );
        const adapterInfo = initial.info?.adapter;
        if (
            arm.mode === "webgpu" &&
            (adapterInfo?.vendor?.toLowerCase() !== "nvidia" ||
                adapterInfo?.isFallbackAdapter !== false)
        )
            throw new Error(
                `${arm.name}: rejected adapter=${JSON.stringify(adapterInfo)} renderer=${gpuInfo.gpu?.auxAttributes?.glRenderer} devices=${JSON.stringify(gpuInfo.gpu?.devices)} ozoneFlags=${JSON.stringify(launchBackendFlags)} dpr=${initial.dpr} viewport=${JSON.stringify(initial.viewport)} ua=${initial.userAgent}`,
            );
        if (
            !gpuInfo.gpu?.devices?.some((device: { deviceString?: string }) =>
                /NVIDIA GeForce RTX 4090/.test(device.deviceString ?? ""),
            )
        )
            throw new Error(`${arm.name}: real RTX 4090 GPU backend not identified`);
        await page.waitForTimeout(500);
        await browserSession.send("Tracing.start", {
            categories,
            transferMode: "ReturnAsStream",
            options: "record-as-much-as-possible",
        });
        await page.evaluate(() => {
            (window as any).__frames.length = 0;
            performance.mark("measurement-start");
            (window as any).__recordFrames = true;
        });
        await page.waitForTimeout(durationMs);
        const measurement = await page.evaluate(() => {
            performance.mark("measurement-end");
            (window as any).__recordFrames = false;
            (window as any).stopAnimation?.();
            return {
                frames: (window as any).__frames as number[],
                gpuErrors: (window as any).__gpuErrors as string[] | undefined,
            };
        });
        const traceComplete = stopTrace(browserSession);
        await page.close();
        const { frames: frameTimes, gpuErrors } = measurement;
        const trace = JSON.parse(await traceText(browserSession, traceComplete)) as {
            traceEvents?: TraceEvent[];
        };
        const start = traceMark(trace.traceEvents ?? [], "measurement-start");
        const end = traceMark(trace.traceEvents ?? [], "measurement-end");
        if (start === undefined || end === undefined || start >= end)
            throw new Error(`${arm.name}: measurement trace boundaries missing or invalid`);
        const intervals = frameTimes.slice(1).map((time, i) => time - frameTimes[i]!);
        if (!intervals.length) throw new Error(`${arm.name}: no rAF samples in measurement window`);
        if (gpuErrors?.length)
            throw new Error(`${arm.name}: WebGPU runtime errors: ${gpuErrors.join("; ")}`);
        const intervalMedian = median(intervals);
        const gaps = intervals.filter((gap) => gap > intervalMedian * 2.5);
        const gapTimes = frameTimes
            .slice(1)
            .flatMap((time, i) => (time - frameTimes[i]! > intervalMedian * 2.5 ? [time] : []));
        const gapPeriods = gapTimes.slice(1).map((time, i) => time - gapTimes[i]!);
        const gapPeriodMedian = median(gapPeriods) || null;
        const gapDurations = gaps;
        const longSwaps = longSwapsInWindow(trace.traceEvents ?? [], start, end).map((event) => ({
            ms: (event.dur ?? 0) / 1000,
            ts: event.ts,
        }));
        const swapPeriods = longSwaps
            .slice(1)
            .map((swap, i) => ((swap.ts ?? 0) - (longSwaps[i]!.ts ?? 0)) / 1000);
        const swapPeriodMedian = median(swapPeriods) || null;
        const periodComparison = {
            gapMedianMs: gapPeriodMedian,
            swapMedianMs: swapPeriodMedian,
            medianDifferenceMs:
                gapPeriodMedian !== null && swapPeriodMedian !== null
                    ? Math.abs(gapPeriodMedian - swapPeriodMedian)
                    : null,
        };
        if (initial.error) throw new Error(`${arm.name}: ${initial.error}`);
        if (errors.length) throw new Error(`${arm.name}: ${errors.join("; ")}`);
        const result = {
            arm: arm.name,
            browser: arm.browser,
            durationMs,
            traceCategories: categories,
            initial,
            browserGpu: {
                devices: gpuInfo.gpu?.devices,
                backend: gpuInfo.gpu?.auxAttributes?.displayType,
                renderer: gpuInfo.gpu?.auxAttributes?.glRenderer,
                webgpu: gpuInfo.gpu?.featureStatus?.webgpu,
                gpuCompositing: gpuInfo.gpu?.featureStatus?.gpu_compositing,
                hardwareSupportsVulkan: gpuInfo.gpu?.auxAttributes?.hardwareSupportsVulkan,
                launchBackendFlags,
                swiftshaderFlagPresent: commandLine.includes("--enable-unsafe-swiftshader"),
            },
            frames: intervals.length,
            intervalMs: {
                min: Math.min(...intervals),
                max: Math.max(...intervals),
                median: intervalMedian,
                gapsOver2_5x: gaps.length,
                gapMin: gaps.length ? Math.min(...gaps) : null,
                gapMax: gaps.length ? Math.max(...gaps) : null,
                gapDurationsMs: gapDurations,
                gapPeriodMs: {
                    min: gapPeriods.length ? Math.min(...gapPeriods) : null,
                    max: gapPeriods.length ? Math.max(...gapPeriods) : null,
                    median: gapPeriodMedian,
                },
            },
            swaps: {
                longOver10ms: longSwaps.length,
                durationsMs: longSwaps.map((item) => item.ms),
                periodMs: {
                    min: swapPeriods.length ? Math.min(...swapPeriods) : null,
                    max: swapPeriods.length ? Math.max(...swapPeriods) : null,
                    median: swapPeriodMedian,
                },
                medianMs: median(longSwaps.map((item) => item.ms)) || null,
                periodComparison,
            },
            errors,
            traceEvents: trace.traceEvents?.length ?? 0,
        };
        console.log(JSON.stringify(result));
        await browserSession.detach();
    } finally {
        if (context) await context.close().catch(() => {});
        if (browser) await browser.close().catch(() => {});
        await new Promise<void>((resolve) => server.close(() => resolve()));
        await rm(profile, { recursive: true, force: true });
    }
}

if (!selfTest) {
    const failures: string[] = [];
    for (const arm of chosen) {
        try {
            await run(arm);
        } catch (error) {
            const reason = error instanceof Error ? error.message : String(error);
            failures.push(`${arm.name}: ${reason}`);
            console.log(JSON.stringify({ arm: arm.name, status: "failed", reason }));
        }
    }
    if (failures.length)
        throw new Error(`${failures.length} failed arm(s): ${failures.join(" | ")}`);
}
