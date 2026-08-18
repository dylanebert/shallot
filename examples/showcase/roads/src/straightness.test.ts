import { describe, expect, test } from "bun:test";
import { detectEdgeOffset, raggedness } from "./straightness";

// The device-free proof that the straightness instrument actually discriminates — the property stage 9's
// gate gap names: a measurement that can't go red on a deliberately-ragged input is not evidence. Both
// scenarios below share one synthetic "image": a boundary between a low (lo) and high (hi) plateau,
// running along the y axis, whose x-position at each row is either exactly the anchor's own x (straight)
// or offset by an alternating amount (a staircase, the exact shape a grid-quantized silhouette makes).

const LO = 20;
const HI = 200;
const RADIUS = 6;
const ROWS = 24;
// a hard-step synthetic sampler (no antialiasing) only resolves a crossing to the discrete sample grid's
// own step size (radius·2/n); this many steps per pixel keeps that quantization bias (half a step) an
// order of magnitude below the raggedness thresholds below, so the assertions measure the *signal*
// (straight vs staircase), not the synthetic sampler's own grid.
const STEPS_PER_PX = 20;

/** a synthetic image sampler: `edgeXAt(y)` gives the true boundary column for row `y`; `sampleAt` reads
 *  `hi` left of it and `lo` right of it (a hard step, not antialiased — the sub-pixel interpolation in
 *  {@link detectEdgeOffset} is what recovers a precise crossing from a small number of discrete samples). */
function stepSampler(edgeXAt: (y: number) => number): (x: number, y: number) => number {
    return (x, y) => (x < edgeXAt(Math.round(y)) ? HI : LO);
}

describe("detectEdgeOffset + raggedness discriminate straight from ragged", () => {
    test("a straight edge reads as ~0 raggedness", () => {
        const anchorX = 50;
        const sample = stepSampler(() => anchorX); // edge sits exactly at every anchor
        const offsets = Array.from({ length: ROWS }, (_, i) =>
            detectEdgeOffset(sample, anchorX, i, 1, 0, LO, HI, RADIUS, STEPS_PER_PX),
        );
        const r = raggedness(offsets);
        expect(r.n).toBe(ROWS); // every probe found its crossing
        expect(r.rms).toBeLessThan(0.05); // sub-pixel: exact given a hard step at an integer column
    });

    test("a staircase edge (grid-quantized silhouette) reads as ragged — the input the instrument must catch", () => {
        const anchorX = 50;
        // steps every 4 rows by ±2px, the shape a mesh quantized to a coarse grid casts: flat runs
        // separated by sudden jumps, not a smooth wobble.
        const stepAmplitude = 2;
        const edgeXAt = (y: number) =>
            anchorX + (Math.floor(y / 4) % 2 === 0 ? stepAmplitude : -stepAmplitude);
        const sample = stepSampler(edgeXAt);
        const offsets = Array.from({ length: ROWS }, (_, i) =>
            detectEdgeOffset(sample, anchorX, i, 1, 0, LO, HI, RADIUS, STEPS_PER_PX),
        );
        const r = raggedness(offsets);
        expect(r.n).toBe(ROWS);
        // the offsets alternate ±stepAmplitude around the anchor (not all on one side, the way a soft/
        // misregistered-but-straight edge would read) — rms lands near stepAmplitude itself.
        expect(r.rms).toBeGreaterThan(1.5);
        expect(r.max).toBeGreaterThanOrEqual(stepAmplitude - 0.1);
    });

    test("mutation: zeroing the staircase's amplitude collapses it back to the straight-edge reading — proves the metric tracks the input, not a fixed constant", () => {
        // this is the input-side mutation the spec's gate-gap calls for: the same code path, the same
        // probe count and radius, only the deliberately-ragged input's amplitude changed. If raggedness()
        // reported a fixed number regardless of the sampler, this would still be "ragged" — it isn't.
        const anchorX = 50;
        const edgeXAt = (y: number) => anchorX + (Math.floor(y / 4) % 2 === 0 ? 0 : 0); // amplitude zeroed
        const sample = stepSampler(edgeXAt);
        const offsets = Array.from({ length: ROWS }, (_, i) =>
            detectEdgeOffset(sample, anchorX, i, 1, 0, LO, HI, RADIUS, STEPS_PER_PX),
        );
        const r = raggedness(offsets);
        expect(r.rms).toBeLessThan(0.05);
    });

    test("no crossing in range reports null, not a fabricated offset", () => {
        const sample = () => LO; // never reaches hi — no edge anywhere in the probe's radius
        const offset = detectEdgeOffset(sample, 50, 0, 1, 0, LO, HI, RADIUS);
        expect(offset).toBeNull();
        expect(raggedness([offset, offset])).toEqual({ rms: 0, max: 0, n: 0 });
    });
});
