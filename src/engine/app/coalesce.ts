/**
 * median of the first `count` recent rAF-callback `intervals` (in any order), insertion-sorted into the
 * caller-owned `scratch` so the frame loop allocates nothing per frame. The median tracks the live present
 * cadence (~16.7ms under a 60Hz throttle, ~4ms on a 240Hz desktop) and ignores the transient short
 * intervals a double-fire injects, unlike a moving average. Returns 0 for an empty window. Fills
 * `scratch`; otherwise pure. Unit-tested.
 */
export function median(intervals: Float64Array, count: number, scratch: Float64Array): number {
    if (count === 0) return 0;
    for (let i = 0; i < count; i++) {
        const v = intervals[i];
        let j = i;
        while (j > 0 && scratch[j - 1] > v) {
            scratch[j] = scratch[j - 1];
            j--;
        }
        scratch[j] = v;
    }
    return scratch[count >> 1];
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

/**
 * the sim delta (seconds) for a frame at rAF timestamp `t` given the last rendered frame's `lastRender`,
 * or a negative `lastRender` on the first frame. The first frame steps 0: the loop cannot seed
 * `lastRender` from `now()` at boot, because a rAF timestamp is the frame's vsync-aligned *start*, which
 * can precede a wall-clock read taken just before the callback — measured as `step received
 * -0.000205` on the no-walls flow, one red in four runs. After that first frame rAF timestamps are
 * monotonic, so the clamp is a floor the loop never hits, kept so a timebase seam can never drive
 * the accumulator negative again. Pure. Unit-tested.
 */
export function frameDelta(t: number, lastRender: number): number {
    if (lastRender < 0) return 0;
    return Math.max(0, t - lastRender) / 1000;
}
