// Stage 9's straightness instrument — device-free pure math, the pure half of the grazing-view
// discriminator (`grazingCapture.ts` is the GPU/DOM-bound half that supplies the real sampled data and
// the analytic anchors). The spec's own gate gap: stage 7's capture probe measures transition *width*
// along one ray perpendicular to the road, so it is structurally blind to boundary *straightness* — the
// defect axis the release-look screenshot actually showed. This module is the straightness criterion the
// stage owes: given a 1D sampler and a sequence of analytic (ground-truth) anchor points with their own
// local search axis, find where the real edge actually sits, and turn the deviations into one number.
//
// Deliberately dimension-agnostic: `detectEdgeOffset` doesn't know whether it's walking screen pixels or
// world metres, or whether it's sampling luminance or height — only that it's a 1D signal crossing a
// midpoint. Stage 9 fed it screen-space albedo luminance; stage 10 (`grazingCapture.ts`'s
// `heightSilhouette`) feeds it world-space mesh height instead, at the axis the defect actually lives on
// (screen-space albedo was blind to a height silhouette whose texture boundary stayed sharp). Same code,
// different sampler.

/**
 * walks `sampleAt` along the unit direction (`dirX`, `dirY`) from (`anchorX`, `anchorY`), over
 * `[-radiusPx, +radiusPx]`, and returns the signed offset (in whatever unit `dirX`/`dirY`/`radiusPx` are
 * expressed in — screen pixels for stage 9's albedo probe, world metres for stage 10's height probe) of
 * the first place the sampled value crosses the midpoint between `lo` and `hi` — linearly interpolated
 * between the two straddling samples for sub-step precision. `null` when no crossing is found in range
 * (the edge isn't where the anchor expects it, or the probe missed the frame/heightfield). Deliberately
 * reports only the first crossing scanning outward from the anchor: a boundary probe expects exactly one
 * transition near its own anchor, and the *offset itself* — not scanning past it for a second one — is the
 * raggedness measurement.
 */
export function detectEdgeOffset(
    sampleAt: (x: number, y: number) => number,
    anchorX: number,
    anchorY: number,
    dirX: number,
    dirY: number,
    lo: number,
    hi: number,
    radiusPx: number,
    stepsPerPx = 4,
): number | null {
    const mid = (lo + hi) / 2;
    const n = Math.max(2, Math.round(radiusPx * 2 * stepsPerPx));
    let prevT = -radiusPx;
    let prevV = sampleAt(anchorX + dirX * prevT, anchorY + dirY * prevT);
    for (let i = 1; i <= n; i++) {
        const t = -radiusPx + (i / n) * (2 * radiusPx);
        const v = sampleAt(anchorX + dirX * t, anchorY + dirY * t);
        const up = prevV < mid && v >= mid;
        const down = prevV > mid && v <= mid;
        if (up || down) {
            const frac = (mid - prevV) / (v - prevV);
            return prevT + frac * (t - prevT);
        }
        prevT = t;
        prevV = v;
    }
    return null;
}

/** raggedness summary over a sequence of {@link detectEdgeOffset} results (each already a *deviation from
 *  the analytic anchor*, since the anchor itself is the ground-truth boundary position): RMS and max of
 *  the non-null offsets, plus how many of the probes actually found an edge. RMS is the metric the
 *  two-resolution reading compares — a straight rendered edge has every offset near zero; a staircase has
 *  offsets that swing across the step width. */
export function raggedness(offsets: ReadonlyArray<number | null>): {
    rms: number;
    max: number;
    n: number;
} {
    const valid = offsets.filter((o): o is number => o !== null);
    if (valid.length === 0) return { rms: 0, max: 0, n: 0 };
    const meanSq = valid.reduce((sum, o) => sum + o * o, 0) / valid.length;
    return {
        rms: Math.sqrt(meanSq),
        max: Math.max(...valid.map(Math.abs)),
        n: valid.length,
    };
}
