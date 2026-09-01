// CPU-only regression proof of the storage-seam pyramid claim's own math (`slope-seam.ts`) —
// self-contained, no GPU adapter needed. What this file CAN see: level >= 1's "published" pyramid
// is built here through `slope.ts`'s own `reduceSlopeMip` — a SEPARATE, independently-authored CPU
// mirror of the GPU `mipKernel` (plain double-precision `reduce`/`/4`, no per-step `Math.fround`
// discipline) — while the "expected" comparison below recomputes through `expectedFromPublished`,
// a DIFFERENT implementation of the same reduction (Math.fround-chained, matching the WGSL
// expression order exactly). Two independently-written functions computing the same mean/residual
// formula: a defect in either one shows up as a step-distance divergence between "published" and
// "expected", which the mutation table in `slope-seam.ts` now exercises directly on
// `reduceSlopeMip` (mutation 6). What this file CANNOT see: level 0 has no second CPU
// implementation anywhere in this package — `expectedLevel0` is the only CPU mirror of
// `slopePostKernel`'s own WGSL logic that exists, so this file necessarily builds AND checks level
// 0 with that SAME function, proving only that the f16 round-trip arithmetic is self-consistent,
// never that `slopePostKernel` itself is correct. That reach belongs entirely to the companion
// real-device arm in the `ocean-slope` gym scenario, which reads the ACTUAL published texture and
// buffers (real WGSL kernel output) rather than a JS-simulated storage rounding.
import { expect, test } from "bun:test";
import {
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeMipSize,
} from "../src/slope";
import {
    CHANNELS,
    expectedFromPublished,
    expectedLevel0,
    f16Neighbors,
    f16NextDown,
    f16NextUp,
    f16Round,
    f16StepDistance,
} from "../src/slope-seam";
import { generateH0 } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

function quantize(field: Float32Array): Float32Array {
    return Float32Array.from(new Float16Array(field));
}

/** Simulated published pyramid: level 0 quantized straight from the (unquantized) FFT output via
 *  `expectedLevel0` (the only CPU form level 0 has — see file header for what this cannot catch);
 *  every level >= 1 quantized from the PREVIOUS level's own quantized (published) values through
 *  `reduceSlopeMip`, the package's independently-authored CPU mirror of the GPU `mipKernel` —
 *  deliberately NOT `expectedFromPublished`, which is what the comparison below uses as `expected`,
 *  so the two sides of the level >= 1 comparison come from two different implementations. */
function publishedPyramid(): Float32Array[] {
    const h0 = generateH0(config, 0);
    const { xField, zField } = runSlopeCpuPipeline(h0, config, 0);
    const N = config.N;
    const xReal = new Float32Array(N * N);
    const zReal = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
        xReal[i] = xField[i * 2];
        zReal[i] = zField[i * 2];
    }
    const level0 = quantize(expectedLevel0(xReal, zReal));
    const levels = [level0];
    for (let level = 1; level < SLOPE_MIP_LEVELS; level++) {
        const parentSize = slopeMipSize(config, level - 1);
        levels.push(quantize(reduceSlopeMip(levels[level - 1], parentSize)));
    }
    return levels;
}

function allowedSteps(level: number, channel: (typeof CHANNELS)[number]): number {
    if (level === 0 && channel === "residual") return 0; // hardcoded literal, no rounding ambiguity
    return 1; // bare discrete claim — see the Ledger's I3g-r re-verdict for why no slack term sits here
}

test("f16 neighbour helpers round-trip and bracket correctly across binades", () => {
    expect(f16Round(0)).toBe(0);
    expect(f16Neighbors(0)).toEqual([0, 0]);
    // 1.0 sits at a binade boundary: the step above (in [1,2)) is 2^-10, the step below (in
    // [0.5,1)) is 2^-11 — the two are NOT equal, which is exactly why neighbours are derived from
    // the f16 bit pattern rather than from a single "step at this binade" formula.
    const [lo, hi] = f16Neighbors(1.0);
    expect(lo).toBe(1.0);
    expect(hi).toBe(1.0);
    const justAbove = 1.0 + 2 ** -12;
    const [loAbove, hiAbove] = f16Neighbors(justAbove);
    expect(loAbove).toBe(1.0);
    expect(hiAbove).toBeCloseTo(1.0 + 2 ** -10, 6);
    const justBelow = 1.0 - 2 ** -13;
    const [loBelow, hiBelow] = f16Neighbors(justBelow);
    expect(hiBelow).toBe(1.0);
    expect(loBelow).toBeCloseTo(1.0 - 2 ** -11, 6);
    expect(f16NextUp(f16NextDown(1.0))).toBe(1.0);
    expect(f16StepDistance(1.0, 1.0)).toBe(0);
    expect(f16StepDistance(1.0, hiAbove)).toBe(1);
});

test("every published texel is within its allowed f16 rounding distance of its own expected value, at every level and channel", () => {
    const published = publishedPyramid();
    const h0 = generateH0(config, 0);
    const { xField, zField } = runSlopeCpuPipeline(h0, config, 0);
    const N = config.N;
    const xReal = new Float32Array(N * N);
    const zReal = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
        xReal[i] = xField[i * 2];
        zReal[i] = zField[i * 2];
    }
    const histogram: Record<number, number> = { 0: 0, 1: 0 };
    let total = 0;
    let farCount = 0;
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        const expected =
            level === 0
                ? expectedLevel0(xReal, zReal)
                : expectedFromPublished(published[level - 1], slopeMipSize(config, level - 1));
        const size = slopeMipSize(config, level);
        for (let i = 0; i < size * size; i++) {
            CHANNELS.forEach((channel, c) => {
                total++;
                const e = expected[i * 4 + c];
                const p = published[level][i * 4 + c];
                const steps = f16StepDistance(e, p);
                const bucket = steps >= 2 ? 2 : steps;
                histogram[bucket] = (histogram[bucket] ?? 0) + 1;
                const limit = allowedSteps(level, channel);
                if (steps > limit) farCount++;
                expect(steps).toBeLessThanOrEqual(limit);
            });
        }
    }
    console.log(
        `seam claim total=${total}, histogram[0]=${histogram[0]}, histogram[1]=${histogram[1]}, ` +
            `histogram[>=2]=${histogram[2] ?? 0}, violations=${farCount}`,
    );
    expect(farCount).toBe(0);
});

test("a zeroed published level reds the same seam claim (vacuity witness)", () => {
    const published = publishedPyramid();
    const h0 = generateH0(config, 0);
    const { xField, zField } = runSlopeCpuPipeline(h0, config, 0);
    const N = config.N;
    const xReal = new Float32Array(N * N);
    const zReal = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
        xReal[i] = xField[i * 2];
        zReal[i] = zField[i * 2];
    }
    const expected = expectedLevel0(xReal, zReal);
    const zeroed = new Float32Array(published[0].length);
    let violations = 0;
    for (let i = 0; i < N * N; i++) {
        for (let c = 0; c < CHANNELS.length; c++) {
            const steps = f16StepDistance(expected[i * 4 + c], zeroed[i * 4 + c]);
            if (steps > allowedSteps(0, CHANNELS[c])) violations++;
        }
    }
    // slopeX/slopeZ/energy at level 0 are all nonzero over a real field, so zeroing the payload
    // must red on at least those three channels' worth of texels.
    expect(violations).toBeGreaterThan(0);
});
