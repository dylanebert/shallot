import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Browser, type BrowserContext, chromium, type Page } from "playwright";

const pageDurationMs = 12_000;
const flagDurationMs = 6_000;
const categories = "cc,viz,gpu,benchmark,blink,blink.user_timing,toplevel";
const preferred = "navigator.gpu.getPreferredCanvasFormat()";
// All arms own disposable profiles; the workaround suite uses Chromium at the person's request.
const arms = [
    { name: "baseline", browser: "chrome", mode: "webgpu" },
    { name: "pixels-2x-css", browser: "chrome", mode: "webgpu", pixels: 2 },
    { name: "pixels-1x-css", browser: "chrome", mode: "webgpu", pixels: 1 },
    { name: "ozone-x11", browser: "chrome", mode: "webgpu", ozone: "x11" },
    { name: "opaque-alpha", browser: "chrome", mode: "webgpu", alpha: "opaque" },
    { name: "alternative-format", browser: "chrome", mode: "webgpu", alternate: true },
    { name: "chromium", browser: "chromium", mode: "webgpu" },
    { name: "css-only", browser: "chrome", mode: "css" },
    { name: "page-baseline", browser: "chromium", mode: "webgpu", plain:true },
    { name: "canvas-opacity", browser: "chromium", mode: "webgpu", plain:true, style: "opacity:0.999" },
    { name: "canvas-transform", browser: "chromium", mode: "webgpu", plain:true, style: "transform:rotateY(0.01deg)" },
    { name: "dom-overlap", browser: "chromium", mode: "webgpu", plain:true, overlap: true },
    { name: "offscreen-worker", browser: "chromium", mode: "webgpu", plain:true, worker: true },
    { name: "bitmaprenderer", browser: "chromium", mode: "webgpu", plain:true, bitmap: true },
    // These consumers agree at 153.0.8010.52 (brief) and 152.0.7977.82 (run binary).
    // https://chromium.googlesource.com/chromium/src/+/refs/tags/153.0.8010.52/ui/gl/gl_implementation.cc#L123
    { name: "angle-vulkan", browser: "chromium", mode: "webgpu", plain:true, flags:["--use-angle=vulkan"] },
    // Preserve Playwright's baseline feature while adding only Graphite; Linux's support check refuses it.
    // https://chromium.googlesource.com/chromium/src/+/refs/tags/153.0.8010.52/gpu/config/gpu_finch_features.cc#L574
    { name: "skia-graphite", browser: "chromium", mode: "webgpu", plain:true, flags:["--enable-features=CDPScreenshotNewSurface,SkiaGraphite"] },
    // None disables promotion strategies, not the primary plane.
    // https://chromium.googlesource.com/chromium/src/+/refs/tags/153.0.8010.52/chrome/browser/about_flags.cc#L548
    // https://chromium.googlesource.com/chromium/src/+/refs/tags/153.0.8010.52/components/viz/host/renderer_settings_creation.cc#L86
    // https://chromium.googlesource.com/chromium/src/+/refs/tags/153.0.8010.52/components/viz/common/display/overlay_strategy.cc#L13
    { name: "overlay-strategies-none", browser: "chromium", mode: "webgpu", plain:true, flags:["--enable-hardware-overlays"] },
    // Compared with angle-vulkan, only Ozone changes; neither switch guarantees a hardware adapter.
    // https://chromium.googlesource.com/chromium/src/+/refs/tags/153.0.8010.52/ui/ozone/public/ozone_switches.cc#L9
    { name: "x11-angle-vulkan", browser: "chromium", mode: "webgpu", plain:true, ozone:"x11", flags:["--use-angle=vulkan"] },
    // The one added switch enables WebGPU and bypasses its adapter blocklist; it does not guarantee hardware.
    // https://chromium.googlesource.com/chromium/src/+/refs/tags/153.0.8010.52/gpu/command_buffer/service/service_utils.cc#L273
    { name: "x11-angle-vulkan-unsafe", browser: "chromium", mode: "webgpu", plain:true, ozone:"x11", flags:["--use-angle=vulkan","--enable-unsafe-webgpu"] },
] as const;

type Arm = (typeof arms)[number];
const selfTest = process.argv.includes("--self-test");
const selected = process.argv.slice(2).filter((argument) => argument !== "--self-test");
const chosen = selected.length ? arms.filter((arm) => selected.includes(arm.name)) : [];
if ((!selfTest && !chosen.length) || selected.some((name) => !arms.some((arm) => arm.name === name))) {
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
      const timestamp = adapter.features.has('timestamp-query');
      const device = await adapter.requestDevice({ requiredFeatures: timestamp ? ['timestamp-query'] : [] });
      const gpuErrors = [];
      device.addEventListener('uncapturederror', event => gpuErrors.push(String(event.error?.message || event.error)));
      device.lost.then(info => gpuErrors.push('device lost: ' + info.reason + ' ' + info.message));
      globalThis.__gpuErrors = gpuErrors;
      const visible = ${"worker" in arm ? "globalThis.__canvas" : "document.querySelector('canvas')"};
      const canvas = ${"bitmap" in arm ? "new OffscreenCanvas(visible.width, visible.height)" : "visible"};
      const bitmapContext = ${"bitmap" in arm ? "visible.getContext('bitmaprenderer')" : "null"};
      ${"bitmap" in arm ? "if (!bitmapContext) throw new Error('No bitmaprenderer context');" : ""}
      const context = canvas.getContext('webgpu');
      if (!context) throw new Error('No WebGPU canvas context');
      let format = ${preferred}; ${alteration}
      ${"worker" in arm || "bitmap" in arm ? "" : pixelSizing}
      context.configure({ device, format, alphaMode: '${"alpha" in arm ? arm.alpha : "premultiplied"}' });
      let stopped = false;
      let animationFrame = 0;
      const cpuMs = [], gpuMs = [], renderedTimes = [];
      globalThis.__cost = { cpuMs, gpuMs, renderedTimes, timestamp };
      const query = timestamp ? device.createQuerySet({type:'timestamp',count:2}) : null;
      const resolve = timestamp ? device.createBuffer({size:256,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}) : null;
      const read = timestamp ? device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}) : null;
      let timingPending = false, timingDone = Promise.resolve(), frame = 0;
      globalThis.stopAnimation = () => { stopped = true; cancelAnimationFrame(animationFrame); };
      function draw(t) {
        if (stopped) return;
        animationFrame = requestAnimationFrame(draw);
        const record = globalThis.__recordFrames && performance.now() >= globalThis.__measurementStart && performance.now() < globalThis.__measurementEnd;
        const cpuStart = performance.now();
        const timed = record && timestamp && !timingPending && frame++ % 60 === 0;
        const texture = context.getCurrentTexture();
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), clearValue: {r:0.08,g:0.1,b:0.18,a:1}, loadOp:'clear', storeOp:'store' }], ...(timed ? {timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}} : {}) });
        const x = Math.floor((t * 0.12) % canvas.width);
        pass.setScissorRect(x, Math.floor(canvas.height/2)-20, Math.min(90,canvas.width-x), 40);
        pass.setPipeline(pipeline);
          pass.draw(3);
        pass.end();
        if (timed) { encoder.resolveQuerySet(query,0,2,resolve,0); encoder.copyBufferToBuffer(resolve,0,read,0,16); }
        device.queue.submit([encoder.finish()]);
        if (bitmapContext) bitmapContext.transferFromImageBitmap(canvas.transferToImageBitmap());
        if (record) { cpuMs.push(performance.now()-cpuStart); renderedTimes.push(t); }
        if (timed) {
          timingPending = true;
          timingDone = read.mapAsync(GPUMapMode.READ).then(() => {
            const times = new BigUint64Array(read.getMappedRange());
            gpuMs.push(Number(times[1]-times[0])/1e6);
            read.unmap(); timingPending = false;
          }).catch(error => gpuErrors.push(String(error)));
        }
      }
      const module = device.createShaderModule({code: 'struct O{@builtin(position) p:vec4f}; @vertex fn vs(@builtin(vertex_index) i:u32)->O{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3)); var o:O; o.p=vec4f(p[i],0,1); return o;} @fragment fn fs()->@location(0) vec4f{return vec4f(1,1,1,1);}' });
      const pipeline = device.createRenderPipeline({layout:'auto',vertex:{module,entryPoint:'vs'},fragment:{module,entryPoint:'fs',targets:[{format}]},primitive:{topology:'triangle-list'}});
      globalThis.runInfo = { mode:'webgpu', adapter:{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:info.isFallbackAdapter ?? adapter.isFallbackAdapter}, format, alphaMode:'${"alpha" in arm ? arm.alpha : "premultiplied"}' };
      globalThis.finishMeasurement = async () => { stopAnimation(); await timingDone; return {cost:__cost,errors:__gpuErrors}; };
      animationFrame = requestAnimationFrame(draw);
    })().catch(error => { globalThis.runError = String(error.stack || error); });
  `;
    if (arm.mode === "css")
        return `<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#141a2e}.bar{position:absolute;left:0;top:calc(50% - 20px);width:90px;height:40px;background:white;animation:move 3s linear infinite}@keyframes move{to{transform:translateX(3840px)}}</style><div class="bar"></div><script>window.runInfo={mode:'css'};window.stopAnimation=()=>document.getAnimations().forEach(animation=>animation.cancel())</script>`;
    const setup = `const canvas = document.querySelector('canvas'); ${pixelSizing}`;
    const script = "worker" in arm ? `${setup}
      const worker = new Worker(URL.createObjectURL(new Blob([${JSON.stringify(`
        onmessage = event => {
          if (event.data.canvas) { globalThis.__canvas = event.data.canvas; ${configure} }
          if (event.data.start !== undefined) {
            globalThis.__measurementStart = event.data.start-performance.timeOrigin;
            globalThis.__measurementEnd = event.data.end-performance.timeOrigin;
            globalThis.__recordFrames = true;
            postMessage({ready:true});
          }
          if (event.data.stop) finishMeasurement().then(result => postMessage(result));
        };
        const poll = setInterval(() => {
          if (globalThis.runInfo || globalThis.runError) { postMessage({info:globalThis.runInfo,error:globalThis.runError}); clearInterval(poll); }
        },20);
      `)}], {type:'text/javascript'})));
      worker.onerror = event => { window.runError = 'Worker error: ' + event.message; };
      worker.onmessage = event => {
        if (event.data.info) window.runInfo = event.data.info;
        if (event.data.error) window.runError = event.data.error;
        if (event.data.ready) window.__workerReady = true;
        if (event.data.cost) { window.__cost = event.data.cost; window.__gpuErrors = event.data.errors; window.__finished = true; }
      };
      const offscreen = canvas.transferControlToOffscreen();
      worker.postMessage({canvas:offscreen},[offscreen]);
      window.prepareMeasurement = (start,end) => worker.postMessage({start,end});
      window.finishMeasurement = () => { worker.postMessage({stop:true}); };
` : `${setup}${configure}`;
    return `<!doctype html><meta charset="utf-8"><style>*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#141a2e}canvas{display:block;width:100vw;height:100vh;${"style" in arm ? arm.style : ""}}</style><canvas></canvas>${"overlap" in arm ? '<div style="position:absolute;inset:0;background:rgba(255,255,255,0.001);pointer-events:none"></div>' : ""}<script>${script}</script>`;
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

function mean(values: number[]): number | null {
    return values.length ? values.reduce((sum,value) => sum+value,0)/values.length : null;
}

function percentile(values: number[], fraction: number): number | null {
    const sorted = [...values].sort((a,b) => a-b);
    return sorted.length ? sorted[Math.ceil(sorted.length*fraction)-1]! : null;
}

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

async function run(arm: Arm) {
    const durationMs = "flags" in arm ? flagDurationMs : pageDurationMs;
    if (process.env.SHALLOT_DISPLAY_SEAT !== "DP-1")
        throw new Error("Declare SHALLOT_DISPLAY_SEAT=DP-1 for this physical-display oracle");
    const profile = await mkdtemp(join(tmpdir(), "shallot-webgpu-stall-"));
    let browser: Browser | undefined;
    let context: BrowserContext | undefined;
    let page: Page | undefined;
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
            ...("plain" in arm ? [] : ["--enable-unsafe-webgpu"]),
            `--ozone-platform=${ozone}`,
            ...("flags" in arm ? arm.flags : []),
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
        page = context.pages()[0] ?? await context.newPage();
        await page.addInitScript(() => {
            const frames: number[] = [];
            (window as any).__frames = frames;
            (window as any).__recordFrames = false;
            (window as any).__displayInterruptions = [];
            for (const event of ["blur","focus","visibilitychange"]) {
                window.addEventListener(event, () => {
                    if ((window as any).__recordFrames)
                        (window as any).__displayInterruptions.push({event,t:performance.now(),focused:document.hasFocus(),visibility:document.visibilityState});
                });
            }
            function sample(t: number) {
                if ((window as any).__recordFrames) {
                    frames.push(t);
                    if (!document.hasFocus() || document.visibilityState !== "visible")
                        (window as any).__displayInterruptions.push({event:"rAF",t,focused:document.hasFocus(),visibility:document.visibilityState});
                }
                requestAnimationFrame(sample);
            }
            requestAnimationFrame(sample);
        });
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
        const browserVersion = await browserSession.send("Browser.getVersion");
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
        await page.waitForFunction(() => (window as any).runInfo || (window as any).runError, undefined, {timeout:15_000});
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
        if (initial.error) throw new Error(`${arm.name}: ${initial.error}; renderer=${gpuInfo.gpu?.auxAttributes?.glRenderer}; webgpu=${gpuInfo.gpu?.featureStatus?.webgpu}; backendFlags=${JSON.stringify(launchBackendFlags)}`);
        if (
            initial.viewport.width !== initial.screen.width ||
            initial.viewport.height !== initial.screen.height ||
            initial.dpr < 1.2
        )
            throw new Error(`${arm.name}: fullscreen dimensions/DPR not established`);
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
            throw new Error(`${arm.name}: SystemInfo did not identify the RTX 4090`);
        await page.waitForTimeout(500);
        let presentationWitness: {whiteFraction:number} | undefined;
        if ("worker" in arm || "bitmap" in arm) {
            const screenshot = await page.screenshot();
            const pixels = Bun.spawn(["magick","png:-","-colorspace","sRGB","-threshold","95%","-format","%[fx:mean]","info:"], {stdin:screenshot,stdout:"pipe",stderr:"pipe"});
            const whiteFraction = Number(await new Response(pixels.stdout).text());
            if (await pixels.exited !== 0 || !Number.isFinite(whiteFraction) || whiteFraction <= 0 || whiteFraction >= 0.01)
                throw new Error(`${arm.name}: fullscreen screenshot did not contain the rendered white bar`);
            presentationWitness = {whiteFraction};
        }
        const performanceSession = await context.newCDPSession(page);
        await performanceSession.send("Performance.enable");
        const beforeMetrics = await performanceSession.send("Performance.getMetrics");
        await page.evaluate((durationMs) => {
            const start = performance.timeOrigin+performance.now()+1_000;
            (window as any).__measurementStart = start-performance.timeOrigin;
            (window as any).__measurementEnd = start+durationMs-performance.timeOrigin;
            (window as any).prepareMeasurement?.(start,start+durationMs);
        },durationMs);
        if ("worker" in arm) await page.waitForFunction(() => (window as any).__workerReady,undefined,{timeout:5_000});
        await browserSession.send("Tracing.start",  {
            categories,
            transferMode: "ReturnAsStream",
            options: "record-as-much-as-possible",
        });
        await page.evaluate(() => new Promise<void>((resolve,reject) => {
            setTimeout(() => {
                if (!document.hasFocus() || document.visibilityState !== "visible") { reject(new Error("Diagnostic window is not focused/visible")); return; }
                (window as any).__frames.length = 0;
                (window as any).__recordFrames = true;
                performance.mark("measurement-start");
                resolve();
            },Math.max(0,(window as any).__measurementStart-performance.now()));
        }));
        await page.evaluate(() => new Promise<void>((resolve) => {
            setTimeout(() => {
                performance.mark("measurement-end");
                (window as any).__recordFrames = false;
                resolve();
            },Math.max(0,(window as any).__measurementEnd-performance.now()));
        }));
        const measurement = await page.evaluate(async () => {
            const startOffsetMs = performance.getEntriesByName("measurement-start")[0]!.startTime-(window as any).__measurementStart;
            const endOffsetMs = performance.getEntriesByName("measurement-end")[0]!.startTime-(window as any).__measurementEnd;
            const result = await (window as any).finishMeasurement?.();
            if (result) (window as any).__finished = true;
            (window as any).stopAnimation?.();
            return { frames: (window as any).__frames as number[], displayInterruptions:(window as any).__displayInterruptions, focused:document.hasFocus(), visibility:document.visibilityState, startOffsetMs, endOffsetMs };
        });
        const afterMetrics = await performanceSession.send("Performance.getMetrics");
        await page.waitForFunction(() => (window as any).__finished || (window as any).runInfo.mode === "css", undefined, {timeout:5_000});
        const costs = await page.evaluate(() => ({
            cost: (window as any).__cost as {cpuMs:number[];gpuMs:number[];renderedTimes:number[];timestamp:boolean} | undefined,
            gpuErrors: (window as any).__gpuErrors as string[] | undefined,
            runtimeError:(window as any).runError as string | undefined,
        }));
        await performanceSession.detach();
        const trace = JSON.parse(await traceText(browserSession, stopTrace(browserSession))) as {
            traceEvents?: TraceEvent[];
        };
        const { frames: frameTimes } = measurement;
        if (measurement.displayInterruptions.length || !measurement.focused || measurement.visibility !== "visible")
            throw new Error(`${arm.name}: display interrupted: ${JSON.stringify(measurement.displayInterruptions)}`);
        if (measurement.startOffsetMs < -1 || measurement.startOffsetMs > 4.165 || Math.abs(measurement.endOffsetMs) > 4.165)
            throw new Error(`${arm.name}: measurement boundary missed a display interval: start=${measurement.startOffsetMs} ms end=${measurement.endOffsetMs} ms`);
        const { cost, gpuErrors, runtimeError } = costs;
        if (runtimeError) throw new Error(`${arm.name}: ${runtimeError}`);
        // Keep the last window alive through trace delivery and session detach; finally closes it.
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
        const elapsedSeconds = (end-start)/1e6;
        const metric = (metrics: {name:string;value:number}[], name:string) => {
            const value = metrics.find(item => item.name === name)?.value;
            if (value === undefined) throw new Error(`${arm.name}: missing ${name} performance metric`);
            return value;
        };
        const renderedFrames = cost?.renderedTimes.length ?? intervals.length;
        const result = {
            arm: arm.name,
            browser: arm.browser,
            browserVersion,
            durationMs,
            readingRule: { gap:"rAF interval > 2.5 x this arm's median", longSwap:"SwapBuffers > 10 ms", removal:"zero gaps and zero long swaps", uncapped:"no deliberate cap; cadence must match the 240.084 Hz display" },
            traceCategories: categories,
            requestedLaunchArgs:launchArgs,
            commandLine,
            displayInterruptions:measurement.displayInterruptions,
            presentationWitness,
            scheduledBoundaryOffsetsMs:{start:measurement.startOffsetMs,end:measurement.endOffsetMs},
            cost: {
                rafP99Ms: percentile(intervals,0.99),
                rafHz: intervals.length / elapsedSeconds,
                renderHz: renderedFrames / elapsedSeconds,
                rateScope:"rAF callbacks and render-loop submissions, not independently counted displayed frames",
                mainThreadTaskMsPerFrame: (metric(afterMetrics.metrics,"TaskDuration")-metric(beforeMetrics.metrics,"TaskDuration"))*1000/renderedFrames,
                mainThreadMetricBracketMs: (metric(afterMetrics.metrics,"Timestamp")-metric(beforeMetrics.metrics,"Timestamp"))*1000,
                mainThreadScope:"Performance TaskDuration bracket includes preparation outside the trace marks",
                drawCpuMeanMs: mean(cost?.cpuMs ?? []),
                drawCpuP99Ms: percentile(cost?.cpuMs ?? [],0.99),
                drawCpuThread: "worker" in arm ? "worker" : "main",
                gpuRenderPassMeanMs: mean(cost?.gpuMs ?? []),
                gpuTimestampSamples: cost?.gpuMs.length ?? 0,
                gpuTimestampSupported: cost?.timestamp ?? false,
                gpuScope:"sampled clear/draw render pass, excluding browser compositing and bitmap conversion",
            },
            initial,
            browserGpu: {
                devices: gpuInfo.gpu?.devices,
                compositorDisplayType: gpuInfo.gpu?.auxAttributes?.displayType,
                glRenderer: gpuInfo.gpu?.auxAttributes?.glRenderer,
                webgpu: gpuInfo.gpu?.featureStatus?.webgpu,
                featureStatus: gpuInfo.gpu?.featureStatus,
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
            compositorWallTime: ["FinishPaintCurrentFrame","FinishPaintRenderPass","SwapBuffers"].map(name => {
                const durations = (trace.traceEvents ?? []).filter(event => event.name === `SkiaOutputSurfaceImplOnGpu::${name}` && event.ph === "X")
                    .map(event => Math.max(0, Math.min(end,(event.ts ?? 0)+(event.dur ?? 0))-Math.max(start,event.ts ?? 0))/1000).filter(ms => ms > 0);
                const total = durations.reduce((sum,ms) => sum+ms,0);
                return {name, count:durations.length,totalMs:total,meanMs:mean(durations),p99Ms:percentile(durations,0.99),msPerRenderedFrame:total/renderedFrames};
            }),
            compositorScope:"GPU-process wall time including waits, not device execution; scopes are not summed",
            traceEvents: trace.traceEvents?.length ?? 0,
        };
        await browserSession.detach();
        console.log(JSON.stringify(result));
    } finally {
        if (page && !page.isClosed()) await page.close().catch(() => {});
        if (context) await context.close().catch(() => {});
        if (browser) await browser.close().catch(() => {});
        await new Promise<void>((resolve) => server.close(() => resolve()));
        await rm(profile, { recursive: true, force: true });
    }
}

if (selfTest) {
    for (const arm of arms) {
        const script = html(arm).match(/<script>([\s\S]*)<\/script>/)?.[1];
        if (!script) throw new Error(`${arm.name}: generated page script missing`);
        new Function(script);
    }
    if (percentile([1,2,3,4],0.99) !== 4 || mean([1,3]) !== 2)
        throw new Error("Synthetic cost-statistics check failed");
    console.log("Trace-window boundaries, cost statistics and generated page syntax checked; no browser run");
} else {
    for (const arm of chosen) {
        try {
            await run(arm);
        } catch (error) {
            const reason = error instanceof Error ? error.message : String(error);
            console.log(JSON.stringify({ arm: arm.name, status: "failed", reason }));
            throw error;
        }
    }
}
