export const HANDOFF_TIMING = {
    handoff: "shallot:handoff",
    plugin: "shallot:plugin:",
};

/** @typedef {{ handoff: number | undefined, timeOrigin: number, frames: number[], ends: number[], stamps: number[], events: { kind: string, label: string, time: number, bytes?: number, end?: number }[], drains: { label: string, at: number, done: number }[], measures: { name: string, start: number, duration: number }[] }} HandoffRecord */
/** @typedef {{ gap: number, intervals: number[] }} HandoffReport */

/** What-ifs that model an engine move under the loading screen with no game system: each primes one
 * kind of first-use work WebKit's GPU process would otherwise do on the first frame after the handoff.
 * `canvas` sizes each canvas's backing store to its displayed size before it is configured, so the
 * first frame's sizing does not resize a configured canvas (this one runs whenever a canvas is
 * configured, which is on the first frame); `bundles` binds each render pipeline in a throwaway render
 * bundle as it is created; `device` executes one throwaway pass of bundled and direct indirect draws. */
export const PRIMES = /** @type {const} */ (["canvas", "bundles", "device"]);

/**
 * Install the manual oracle's measurement probe before navigating.
 * @param {import("playwright/test").Page} page
 * @param {{ frames: number, primes?: readonly string[] }} options
 */
export async function observeHandoff(page, { frames, primes = [] }) {
    if (primes.length) await page.addInitScript(installPrimes, { ...HANDOFF_TIMING, primes });
    await page.addInitScript(installHandoffProbe, HANDOFF_TIMING);
    return {
        async report() {
            await page.waitForFunction(
                (count) =>
                    /** @type {any} */ (window).__shallotHandoff.framesAfterHandoff() >= count,
                frames,
                { polling: 100 },
            );
            const record = await page.evaluate(() => /** @type {any} */ (window).__shallotHandoff.read());
            return summarizeHandoff(record);
        },
    };
}

/** Runs in the page before its scripts. @param {typeof HANDOFF_TIMING} timing */
function installHandoffProbe(timing) {
    /** @type {HandoffRecord["events"]} */
    const events = [];
    /** @type {number[]} */
    const frames = [];
    /** @type {number[]} */
    const ends = [];
    /** @type {number[]} */
    const stamps = [];
    let lastStamp = -1;
    const handoff = () => performance.getEntriesByName(timing.handoff, "mark")[0]?.startTime;
    // Pipeline and shader creation is noted from the page's start, so what was created under the screen
    // is visible; other kinds only after the handoff.
    /** @param {string} kind @param {string} label @param {number} [bytes] */
    const note = (kind, label, bytes) => {
        if (kind === "pipeline" || handoff() !== undefined) {
            const event = { kind, label, time: performance.now(), bytes };
            events.push(event);
            return event;
        }
    };
    // Each drain is the time from `at` until the queue reports all prior work done. In WebKit a pipeline
    // created synchronously returns at once while its GPU process compiles it in order with later
    // messages, so a drain at the handoff is the GPU process's backlog that JavaScript cannot see.
    /** @type {HandoffRecord["drains"]} */
    const drains = [];
    /** @type {GPUDevice | undefined} */
    let gpuDevice;
    /** @param {string} label */
    const drain = (label) => {
        const at = performance.now();
        gpuDevice?.queue.onSubmittedWorkDone().then(
            () => drains.push({ label, at, done: performance.now() }),
            () => {},
        );
    };
    const requestDevice = globalThis.GPUAdapter?.prototype.requestDevice;
    if (requestDevice)
        GPUAdapter.prototype.requestDevice = async function (/** @type {any[]} */ ...args) {
            const created = await requestDevice.apply(this, args);
            gpuDevice ??= created;
            return created;
        };
    const mark = performance.mark.bind(performance);
    let hooked = false;
    performance.mark = (/** @type {any[]} */ ...args) => {
        const entry = mark(...args);
        // A tracer takes stacks from the first plugin hook, so they cover the drain, handoff and first frames.
        if (!hooked && String(args[0]).startsWith(timing.plugin)) {
            hooked = true;
            /** @type {any} */ (window).__shallotHandoffBeacon?.("hook");
        }
        if (args[0] === timing.handoff) {
            drain("handoff");
            /** @type {any} */ (window).__shallotHandoffBeacon?.("handoff");
        }
        return entry;
    };
    /** @param {any} target @param {string} key @param {string} kind @param {(args: any[]) => [string, number?]} describe */
    const wrap = (target, key, kind, describe) => {
        const original = target?.[key];
        if (typeof original !== "function") return;
        target[key] = function (/** @type {any[]} */ ...args) {
            const [label, bytes] = describe(args);
            const event = note(kind, label, bytes);
            const result = original.apply(this, args);
            // An async creation's end is when its promise settles; a synchronous call's is its return.
            if (event && result instanceof Promise)
                result.then(
                    () => (event.end = performance.now()),
                    () => {},
                );
            else if (event) event.end = performance.now();
            return result;
        };
    };
    /** @param {any[]} args @returns {[string]} */
    const labelled = (args) => [args[0]?.label || "<unlabelled>"];
    /** @param {any} size */
    const extent = (size) =>
        Array.isArray(size)
            ? size.reduce((a, b) => a * b, 1)
            : (size?.width ?? 1) * (size?.height ?? 1) * (size?.depthOrArrayLayers ?? 1);
    const device = globalThis.GPUDevice?.prototype;
    for (const key of [
        "createRenderPipeline", "createComputePipeline", "createRenderPipelineAsync",
        "createComputePipelineAsync", "createShaderModule",
    ]) wrap(device, key, "pipeline", labelled);
    wrap(device, "createBuffer", "allocation", (args) => [args[0]?.label || "<unlabelled buffer>", args[0]?.size ?? 0]);
    wrap(device, "createTexture", "allocation", (args) => [args[0]?.label || "<unlabelled texture>", extent(args[0]?.size)]);
    const queue = globalThis.GPUQueue?.prototype;
    wrap(queue, "writeBuffer", "upload", (args) => [args[0]?.label || "<unlabelled buffer>", args[4] ?? (args[2]?.byteLength ?? 0) - (args[3] ?? 0)]);
    wrap(queue, "writeTexture", "upload", (args) => [args[0]?.texture?.label || "<unlabelled texture>", args[1]?.byteLength ?? 0]);
    wrap(queue, "copyExternalImageToTexture", "upload", (args) => [args[1]?.texture?.label || "<unlabelled texture>", extent(args[2])]);
    wrap(globalThis.GPUCanvasContext?.prototype, "getCurrentTexture", "acquire", () => ["getCurrentTexture"]);
    // The first frames' submissions drain too: their spans are when the GPU process finished each frame.
    const submit = queue?.submit;
    if (queue && submit)
        queue.submit = function (/** @type {any[]} */ ...args) {
            const event = note("submit", "submit");
            const result = submit.apply(this, args);
            if (event) {
                event.end = performance.now();
                if (frames.length <= 8) drain(`submit ${frames.length}`);
            }
            return result;
        };
    wrap(globalThis, "createImageBitmap", "decode", () => ["createImageBitmap"]);
    wrap(globalThis.BaseAudioContext?.prototype, "decodeAudioData", "decode", () => ["decodeAudioData"]);
    wrap(globalThis.HTMLImageElement?.prototype, "decode", "decode", () => ["image decode"]);
    const request = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = (callback) => request((stamp) => {
        if (stamp !== lastStamp && handoff() !== undefined) {
            lastStamp = stamp;
            frames.push(performance.now());
            stamps.push(stamp);
            ends.push(0);
        }
        try { callback(stamp); }
        finally { if (ends.length) ends[ends.length - 1] = performance.now(); }
    });
    Object.defineProperty(window, "__shallotHandoff", {
        value: {
            framesAfterHandoff: () => frames.length,
            read: () => ({
                handoff: handoff(), timeOrigin: performance.timeOrigin, frames, ends, stamps, events, drains,
                measures: performance.getEntriesByType("measure")
                    .filter((entry) => entry.name.startsWith("shallot:"))
                    .map(({ name, startTime, duration }) => ({ name, start: startTime, duration })),
            }),
        },
    });
}

/** Runs in the page before the probe. @param {typeof HANDOFF_TIMING & { primes: readonly string[] }} options */
export function installPrimes({ handoff, primes }) {
    const handed = () => performance.getEntriesByName(handoff, "mark").length > 0;
    const device = globalThis.GPUDevice?.prototype;
    const canvasContext = globalThis.GPUCanvasContext?.prototype;
    if (!device || !canvasContext) return;
    if (primes.includes("canvas")) {
        const configure = canvasContext.configure;
        canvasContext.configure = function (/** @type {any[]} */ ...args) {
            const canvas = /** @type {HTMLCanvasElement} */ (this.canvas);
            // The engine's automatic ratio, so its sizing after this configure finds nothing to change.
            const ratio = Math.min(Math.max(devicePixelRatio, 1), 2);
            if (canvas.clientWidth) {
                canvas.width = Math.round(canvas.clientWidth * ratio);
                canvas.height = Math.round(canvas.clientHeight * ratio);
            }
            return configure.apply(this, args);
        };
    }
    const createRenderPipeline = device.createRenderPipeline;
    let primedDevice = false;
    /** @param {GPUDevice} gpu */
    const primeDevice = (gpu) => {
        if (primedDevice || !primes.includes("device")) return;
        primedDevice = true;
        const module = gpu.createShaderModule({
            code: "@vertex fn v() -> @builtin(position) vec4f { return vec4f(0.0, 0.0, 0.0, 1.0); } @fragment fn f() -> @location(0) vec4f { return vec4f(0.0); }",
        });
        const format = navigator.gpu.getPreferredCanvasFormat();
        const pipeline = createRenderPipeline.call(gpu, {
            layout: "auto",
            vertex: { module },
            fragment: { module, targets: [{ format }] },
        });
        const index = gpu.createBuffer({ size: 64, usage: GPUBufferUsage.INDEX });
        const indirect = gpu.createBuffer({ size: 64, usage: GPUBufferUsage.INDIRECT });
        const bundles = [/** @type {const} */ ("uint32"), "uint16", undefined].map((indexFormat) => {
            const bundle = gpu.createRenderBundleEncoder({ colorFormats: [format] });
            bundle.setPipeline(pipeline);
            if (indexFormat) {
                bundle.setIndexBuffer(index, indexFormat);
                bundle.drawIndexedIndirect(indirect, 0);
                bundle.drawIndexed(0);
            } else {
                bundle.drawIndirect(indirect, 0);
                bundle.draw(0);
            }
            return bundle.finish();
        });
        const target = gpu.createTexture({ size: [4, 4], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
        const encoder = gpu.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store" }],
        });
        pass.executeBundles(bundles);
        pass.setPipeline(pipeline);
        pass.setIndexBuffer(index, "uint32");
        pass.drawIndexedIndirect(indirect, 0);
        pass.drawIndirect(indirect, 0);
        pass.draw(0);
        pass.end();
        gpu.queue.submit([encoder.finish()]);
    };
    /** @param {GPUDevice} gpu @param {GPURenderPipelineDescriptor} descriptor @param {GPURenderPipeline} pipeline */
    const prime = (gpu, descriptor, pipeline) => {
        if (handed()) return;
        gpu.pushErrorScope("validation");
        try {
            primeDevice(gpu);
            const colorFormats = [...(descriptor.fragment?.targets ?? [])].map((target) => target?.format ?? null);
            const depthStencilFormat = descriptor.depthStencil?.format;
            if (primes.includes("bundles") && (colorFormats.some(Boolean) || depthStencilFormat)) {
                const bundle = gpu.createRenderBundleEncoder({
                    colorFormats,
                    depthStencilFormat,
                    sampleCount: descriptor.multisample?.count ?? 1,
                });
                bundle.setPipeline(pipeline);
                bundle.finish();
            }
        } finally {
            gpu.popErrorScope().then((error) => error && console.warn(`prime: ${error.message}`));
        }
    };
    device.createRenderPipeline = function (/** @type {GPURenderPipelineDescriptor} */ descriptor) {
        const pipeline = createRenderPipeline.call(this, descriptor);
        prime(this, descriptor, pipeline);
        return pipeline;
    };
    const createRenderPipelineAsync = device.createRenderPipelineAsync;
    device.createRenderPipelineAsync = function (/** @type {GPURenderPipelineDescriptor} */ descriptor) {
        return createRenderPipelineAsync.call(this, descriptor).then((pipeline) => {
            prime(this, descriptor, pipeline);
            return pipeline;
        });
    };
}

/** @param {HandoffRecord} record @returns {HandoffReport} */
export function summarizeHandoff(record) {
    if (record.handoff === undefined) throw new Error("no shallot:handoff mark");
    const round = (/** @type {number} */ ms) => Math.round(ms * 10) / 10;
    const intervals = record.stamps.slice(1).map((stamp, i) => round(stamp - record.stamps[i]));
    const sorted = [...intervals].sort((a, b) => a - b);
    const typical = sorted[Math.floor(sorted.length / 2)] ?? 0;
    const settled = intervals.findIndex((_, i) =>
        intervals.slice(i, i + 10).every((interval) => interval <= typical * 1.5),
    );
    const gap = round((record.stamps[settled < 0 ? record.stamps.length - 1 : settled] ?? record.handoff) - record.handoff);
    return { gap, intervals };
}
