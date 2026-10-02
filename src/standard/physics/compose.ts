/** shortest-arc nlerp from `prev` to `curr` at `t`: flip `prev` into `curr`'s hemisphere, lerp,
 *  renormalize. Returns the identity quat if the blend degenerates. */
export function nlerpShortest(
    prev: readonly [number, number, number, number],
    curr: readonly [number, number, number, number],
    t: number,
): [number, number, number, number] {
    const out: [number, number, number, number] = [0, 0, 0, 1];
    nlerpShortestInto(prev, 0, curr, 0, t, out, 0);
    return out;
}

/** Write a normalized shortest-arc blend into a caller-owned record, without temporary vectors. */
export function nlerpShortestInto(
    prev: ArrayLike<number>,
    p: number,
    curr: ArrayLike<number>,
    q: number,
    t: number,
    out: { [index: number]: number },
    offset: number,
): void {
    const dot =
        prev[p] * curr[q] +
        prev[p + 1] * curr[q + 1] +
        prev[p + 2] * curr[q + 2] +
        prev[p + 3] * curr[q + 3];
    const flip = dot < 0 ? -1 : 1;
    const x = prev[p] * flip * (1 - t) + curr[q] * t;
    const y = prev[p + 1] * flip * (1 - t) + curr[q + 1] * t;
    const z = prev[p + 2] * flip * (1 - t) + curr[q + 2] * t;
    const w = prev[p + 3] * flip * (1 - t) + curr[q + 3] * t;
    const len = Math.sqrt(x * x + y * y + z * z + w * w);
    out[offset] = len > 1e-12 ? x / len : 0;
    out[offset + 1] = len > 1e-12 ? y / len : 0;
    out[offset + 2] = len > 1e-12 ? z / len : 0;
    out[offset + 3] = len > 1e-12 ? w / len : 1;
}
