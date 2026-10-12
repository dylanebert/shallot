export type ScheduleMode = "gpu-bound" | "double-fire";

type CallbackRecord = {
    timestamp: number;
    elapsedMs: number;
    intervalMs: number | null;
    pending: number;
    doubleFire: boolean;
};

type PendingBucket = {
    second: number;
    samples: number;
    first: number;
    last: number;
    max: number;
};

type ScheduleResult = {
    mode: ScheduleMode;
    startedAt: string;
    userAgent: string;
    viewport: { width: number; height: number; pixelRatio: number };
    adapter: Record<string, unknown> | null;
    durationMs: number;
    complete: boolean;
    stopReason: string | null;
    callbacks: CallbackRecord[];
    closingCallback: CallbackRecord | null;
    pendingBuckets: PendingBucket[];
    doubleFireCount: number;
    gpuWork: {
        workgroups: number;
        iterations: number;
        samplesMs: number[];
    } | null;
};

declare global {
    interface Window {
        __shallotScheduleResult: ScheduleResult;
        __shallotScheduleFrame: (timestamp: number, pending: number) => boolean;
        __shallotScheduleGpuWork: (workgroups: number, iterations: number) => void;
        __shallotScheduleGpuSample: (durationMs: number) => void;
    }
}

const DURATION_MS = 30_000;
const PENDING_STOP = 64;

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[middle - 1]! + sorted[middle]!) / 2
        : sorted[middle]!;
}

function sustainedGrowth(buckets: readonly PendingBucket[]): boolean {
    const recent = buckets.slice(-5);
    if (recent.length < 5) return false;
    return (
        recent.every(
            (bucket, index) =>
                index === 0 ||
                (bucket.second === recent[index - 1]!.second + 1 &&
                    bucket.max > recent[index - 1]!.max),
        ) && recent[4]!.max - recent[0]!.max >= 8
    );
}

/** Install the fixed 30-second callback reader used by the stage-1 manual oracle. */
export function installScheduleRecorder(mode: ScheduleMode): ScheduleResult {
    const result: ScheduleResult = {
        mode,
        startedAt: new Date().toISOString(),
        userAgent: navigator.userAgent,
        viewport: {
            width: window.screen.width,
            height: window.screen.height,
            pixelRatio: window.devicePixelRatio,
        },
        adapter: null,
        durationMs: 0,
        complete: false,
        stopReason: null,
        callbacks: [],
        closingCallback: null,
        pendingBuckets: [],
        doubleFireCount: 0,
        gpuWork: null,
    };
    window.__shallotScheduleResult = result;

    const banner = document.createElement("pre");
    banner.setAttribute("aria-live", "polite");
    banner.style.cssText =
        "position:fixed;left:8px;bottom:8px;z-index:2147483647;margin:0;padding:8px 10px;max-width:80vw;overflow:hidden;background:#111e;color:#d9f99d;font:12px/1.4 ui-monospace,monospace;pointer-events:none";
    banner.textContent = `${mode}: waiting for first frame`;
    document.body.append(banner);

    void navigator.gpu?.requestAdapter().then((adapter) => {
        if (!adapter) return;
        const info = adapter.info;
        result.adapter = {
            vendor: info.vendor,
            architecture: info.architecture,
            device: info.device,
            description: info.description,
            features: [...adapter.features].sort(),
        };
    });

    const intervals: number[] = [];
    let start: number | undefined;
    let previous: number | undefined;
    let lastBanner = -Infinity;
    let maxPending = 0;

    window.__shallotScheduleGpuWork = (workgroups, iterations) => {
        result.gpuWork = { workgroups, iterations, samplesMs: [] };
    };
    window.__shallotScheduleGpuSample = (durationMs) => {
        result.gpuWork?.samplesMs.push(durationMs);
    };
    window.__shallotScheduleFrame = (timestamp, pending) => {
        if (result.complete) return true;
        start ??= timestamp;
        const elapsedMs = timestamp - start;
        const intervalMs = previous === undefined ? null : timestamp - previous;
        if (elapsedMs >= DURATION_MS) {
            result.closingCallback = { timestamp, elapsedMs, intervalMs, pending, doubleFire: false };
            result.durationMs = elapsedMs;
            result.complete = true;
            result.stopReason =
                mode === "gpu-bound" && pending >= PENDING_STOP
                    ? `pending reached safety limit ${PENDING_STOP} at the closing callback`
                    : "30-second window complete";
            banner.textContent =
                result.stopReason === "30-second window complete"
                    ? `${mode}: complete (${result.callbacks.length} callbacks)`
                    : `${mode}: STOP — ${result.stopReason}`;
            console.info("[settled-schedule] complete", result.mode, result.callbacks.length);
            return true;
        }

        previous = timestamp;
        let doubleFire = false;
        if (intervalMs !== null && intervalMs > 0) {
            if (intervals.length === 20) doubleFire = intervalMs < median(intervals) / 2;
            intervals.push(intervalMs);
            if (intervals.length > 20) intervals.shift();
        }
        if (doubleFire) result.doubleFireCount++;
        result.callbacks.push({ timestamp, elapsedMs, intervalMs, pending, doubleFire });
        maxPending = Math.max(maxPending, pending);

        const second = Math.floor(elapsedMs / 1000);
        let bucket = result.pendingBuckets.at(-1);
        if (!bucket || bucket.second !== second) {
            bucket = { second, samples: 0, first: pending, last: pending, max: pending };
            result.pendingBuckets.push(bucket);
        }
        bucket.samples++;
        bucket.last = pending;
        bucket.max = Math.max(bucket.max, pending);
        result.durationMs = elapsedMs;

        let stopReason: string | undefined;
        if (mode === "gpu-bound" && pending >= PENDING_STOP) {
            stopReason = `pending reached safety limit ${PENDING_STOP}`;
        } else if (mode === "gpu-bound" && sustainedGrowth(result.pendingBuckets)) {
            stopReason = "five successive one-second pending maxima rose by at least eight frames";
        }
        if (stopReason) {
            result.complete = true;
            result.stopReason = stopReason;
            banner.textContent = `${mode}: STOP — ${stopReason}`;
            console.warn("[settled-schedule] stopped", stopReason);
            return true;
        }

        if (elapsedMs - lastBanner >= 250) {
            lastBanner = elapsedMs;
            banner.textContent =
                `${mode}: ${(elapsedMs / 1000).toFixed(1)} s · ${result.callbacks.length} callbacks · ` +
                `pending ${pending} (max ${maxPending}) · ` +
                `double fires ${result.doubleFireCount}`;
        }
        return false;
    };

    return result;
}
