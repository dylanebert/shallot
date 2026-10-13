/**
 * The simulation delta in seconds for a frame at rAF timestamp `t` (milliseconds), given the last
 * rendered frame's timestamp `lastRender`, or a negative `lastRender` on the first frame. The first frame
 * steps 0, since a rAF timestamp marks the frame's vsync-aligned start and can precede a wall-clock seed.
 * The delta is never negative.
 */
export function frameDelta(t: number, lastRender: number): number {
    if (lastRender < 0) return 0;
    return Math.max(0, t - lastRender) / 1000;
}
