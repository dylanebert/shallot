export const HANDOFF_TIMING = {
    handoff: "shallot:handoff",
};

/** @typedef {{ handoff: number | undefined, frames: number[], ends: number[], stamps: number[], events: { kind: string, label: string, time: number, bytes?: number }[], measures: { name: string, start: number, duration: number }[] }} HandoffRecord */
/** @typedef {{ gap: number, intervals: number[] }} HandoffReport */

/**
 * Install the manual oracle's measurement probe before navigating.
 * @param {import("playwright/test").Page} page
 * @param {{ frames: number }} options
 */
export async function observeHandoff(page, { frames }) {
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
    /** @param {string} kind @param {string} label @param {number} [bytes] */
    const note = (kind, label, bytes) => {
        if (handoff() !== undefined) events.push({ kind, label, time: performance.now(), bytes });
    };
    /** @param {any} target @param {string} key @param {string} kind @param {(args: any[]) => [string, number?]} describe */
    const wrap = (target, key, kind, describe) => {
        const original = target?.[key];
        if (typeof original !== "function") return;
        target[key] = function (/** @type {any[]} */ ...args) {
            const [label, bytes] = describe(args);
            note(kind, label, bytes);
            return original.apply(this, args);
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
                handoff: handoff(), frames, ends, stamps, events,
                measures: performance.getEntriesByType("measure")
                    .filter((entry) => entry.name.startsWith("shallot:"))
                    .map(({ name, startTime, duration }) => ({ name, start: startTime, duration })),
            }),
        },
    });
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
