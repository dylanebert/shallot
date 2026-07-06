/**
 * median of the recent rAF-callback `intervals`, computed into the caller-owned `scratch` so the frame
 * loop allocates nothing per frame. The median tracks the live present cadence (~16.7ms under a 60Hz
 * throttle, ~4ms on a 240Hz desktop) and ignores the transient short intervals a double-fire injects,
 * unlike a moving average. Returns 0 for an empty window. Fills `scratch`; otherwise pure. Unit-tested.
 */
export function median(intervals: readonly number[], scratch: number[]): number {
    const n = intervals.length;
    if (n === 0) return 0;
    for (let i = 0; i < n; i++) scratch[i] = intervals[i];
    scratch.length = n;
    scratch.sort((a, b) => a - b);
    return scratch[n >> 1];
}

/**
 * decide whether to coalesce (skip) this `requestAnimationFrame` callback as a double-fire: it landed
 * under half the estimated present `cadence` since the last rendered frame. Chrome occasionally delivers
 * two callbacks a few ms apart that net to one display interval (notably under a fullscreen present
 * throttle on a high-refresh monitor); rendering both submits two frames for one present, which fills the
 * swapchain queue and surfaces as input latency. Skipping the early one keeps the loop at one submit per
 * present. The threshold scales with the {@link median} cadence, so a 240Hz desktop is never coalesced.
 * Pure. Unit-tested.
 */
export function coalesce(t: number, lastRender: number, cadence: number): boolean {
    return cadence > 0 && t - lastRender < cadence * 0.5;
}
