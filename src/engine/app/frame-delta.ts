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
