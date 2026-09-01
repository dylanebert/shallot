// Trigger: changes to slope-cascade resolution, patch length, declared band, or seeded realization
// can break the N-independent mode placement and represented slope moment checked here.
import { expect, test } from "bun:test";
import {
    composedSlopePsd,
    rasterSlopeMoment,
    realizedSlopeMss,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    slopeMomentAgreementTolerance,
} from "../src/slope";
import { generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

/** Relative machine epsilon of the `Float32Array` storage the CPU pipeline reads and writes
 *  through (`ifft2`'s output, `slopeSpectra`'s input) — the quantization quantum of the
 *  representation being compared, not an authored constant. */
const FLOAT32_EPSILON = 2 ** -23;

/**
 * FFT round-off bound for two rasters carrying identical modes by construction (same seed, same
 * represented band, denser grid's shared-`k` modes bit-identical to the base grid's — asserted
 * below). The internal `ifft2` butterfly accumulates in `Float64Array` at every stage, so its own
 * rounding is ~1e-16 relative and negligible next to the single `Float32Array` store/read that
 * brackets it on both ends of the pipeline; that Float32 quantization quantum is the dominant,
 * and only material, source of a genuine per-side reading error here. `realizedSlopeMss` reads
 * one real component per sample and squares it (`d(x²)/x² ≈ 2·dx/x`), so a single rounded read
 * carries a relative error bound of `FLOAT32_EPSILON` in the squared term; comparing two
 * independently-rounded readings (one per grid size) bounds their worst-case relative gap by
 * twice that quantum — no averaging or cancellation assumed, no free safety multiplier.
 */
function fftRoundoffBound(): number {
    return 2 * FLOAT32_EPSILON;
}

const DECLARED_BANDS = [
    [config.kLo, 20],
    [20, 40],
    [40, 50],
    [50, 55],
    [55, config.kHi],
] as const;

/** Seeded h0 masked to the modes whose radial wavenumber falls inside `[lo, hi]`. */
function bandMask(h0: Float32Array, lo: number, hi: number): { band: Float32Array; modes: number } {
    const band = new Float32Array(h0.length);
    let modes = 0;
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const k = Math.hypot(
                kIndex(x, config.N) * ((2 * Math.PI) / config.L),
                kIndex(y, config.N) * ((2 * Math.PI) / config.L),
            );
            if (k < lo || k > hi) continue;
            const i = (y * config.N + x) * 2;
            band[i] = h0[i];
            band[i + 1] = h0[i + 1];
            if (h0[i] !== 0 || h0[i + 1] !== 0) modes++;
        }
    }
    return { band, modes };
}

/**
 * This by-path oracle compares seeded CPU realizations and their declared radial bands. It does
 * not claim an independent source-density check or implement a second restricted-PSD quadrature;
 * the production composed-PSD helper supplies the expected band moments.
 */
test("slope N-invariance compares the declared raster moments directly", () => {
    const denser = { ...config, N: config.N * 2 };
    const rasterMoment = rasterSlopeMoment(generateH0(config, 17), config);
    const denserRasterMoment = rasterSlopeMoment(generateH0(denser, 17), denser);
    const bound = fftRoundoffBound();
    const deviation = Math.abs(denserRasterMoment / rasterMoment - 1);
    console.log(
        `slope N-invariance: N=${config.N} moment=${rasterMoment}, ` +
            `2N=${denser.N} moment=${denserRasterMoment}, deviation=${deviation}, bound=${bound}`,
    );
    expect(rasterMoment).toBeGreaterThan(0);
    expect(denserRasterMoment).toBeGreaterThan(0);
    // The two rasters carry identical modes by construction (asserted below), not sampled ones,
    // so a seeded sampling-error bound is not this comparison's error model — the bound above is
    // the FFT/storage round-off scale instead.
    expect(deviation).toBeLessThan(bound);

    const base = generateH0(config, 17);
    const high = generateH0(denser, 17);
    const dk = (2 * Math.PI) / config.L;
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const kx = kIndex(x, config.N) * dk;
            const kz = kIndex(y, config.N) * dk;
            if (Math.hypot(kx, kz) < config.kLo || Math.hypot(kx, kz) > config.kHi) continue;
            const hiX = kIndex(x, config.N) >= 0 ? x : denser.N + kIndex(x, config.N);
            const hiY = kIndex(y, config.N) >= 0 ? y : denser.N + kIndex(y, config.N);
            const a = (y * config.N + x) * 2;
            const b = (hiY * denser.N + hiX) * 2;
            expect(high[b]).toBe(base[a]);
            expect(high[b + 1]).toBe(base[a + 1]);
        }
    }
});

/**
 * Red-witness for the round-off bound above: the guarded arm's own assertion
 * (`deviation < fftRoundoffBound()`) re-run with only the subject mutated — a per-`N` `H0` draw
 * (independent seeds at `N` and `2N`) in place of the shared-by-construction draw the guarded arm
 * uses. Two independently-drawn rasters carry different realized moments by physical variance, not
 * merely round-off, so the same tight bound must red here; if it didn't, the bound would be too
 * loose to mean anything.
 */
test("slope N-invariance's round-off bound reds on independently-drawn per-N rasters", () => {
    const denser = { ...config, N: config.N * 2 };
    const rasterMoment = rasterSlopeMoment(generateH0(config, 41), config);
    const independentDenserMoment = rasterSlopeMoment(generateH0(denser, 43), denser);
    const bound = fftRoundoffBound();
    const deviation = Math.abs(independentDenserMoment / rasterMoment - 1);
    console.log(
        `slope N-invariance round-off vacuity: independently-drawn deviation=${deviation}, ` +
            `bound=${bound}`,
    );
    expect(deviation).toBeGreaterThan(bound);
});

test("slope coverage checks every declared radial band against its composed moment", () => {
    const h0 = generateH0(config, 0);
    for (const [lo, hi] of DECLARED_BANDS) {
        const { band, modes } = bandMask(h0, lo, hi);
        expect(modes, `no seeded modes in ${lo}..${hi} rad/m`).toBeGreaterThan(0);
        const realized = realizedSlopeMss(runSlopeCpuPipeline(band, config));
        const expected = composedSlopePsd({ ...config, kLo: lo, kHi: hi });
        const ratio = realized / expected;
        const bound = slopeMomentAgreementTolerance({ ...config, kLo: lo, kHi: hi });
        console.log(`slope band ${lo}..${hi} ratio=${ratio}, bound=${bound}`);
        expect(realized).toBeGreaterThan(0);
        expect(Math.abs(ratio - 1)).toBeLessThan(bound);
    }
});

/**
 * The per-band twin of `slope.test.ts`'s "missing gradient k red-witnesses the composed-coverage
 * arm": the guarded arm above ("slope coverage checks every declared radial band...") re-run at
 * each declared sub-band with only the realized subject mutated. The expectation
 * (`composedSlopePsd`) and the bound (`slopeMomentAgreementTolerance`) stay the guarded arm's own,
 * unmutated, per band — no separation floor authored here.
 */
test("missing gradient k red-witnesses the per-band coverage arm at every declared sub-band", () => {
    const h0 = generateH0(config, 0);
    for (const [lo, hi] of DECLARED_BANDS) {
        const { band } = bandMask(h0, lo, hi);
        const bandConfig = { ...config, kLo: lo, kHi: hi };
        const expected = composedSlopePsd(bandConfig);
        const bound = slopeMomentAgreementTolerance(bandConfig);
        const greenRealized = realizedSlopeMss(runSlopeCpuPipeline(band, config));
        const mutatedRealized = realizedSlopeMss(
            runSlopeCpuPipeline(band, config, 0, { missingGradientK: true }),
        );
        const greenDeviation = Math.abs(greenRealized / expected - 1);
        const mutatedDeviation = Math.abs(mutatedRealized / expected - 1);
        console.log(
            `slope band ${lo}..${hi} dropped-gradient reach: green=${greenDeviation}, ` +
                `mutated=${mutatedDeviation}, bound=${bound}, ` +
                `reach=${(mutatedDeviation / bound).toFixed(1)}x`,
        );
        expect(greenDeviation).toBeLessThan(bound);
        expect(mutatedDeviation).toBeGreaterThan(bound);
    }
});
