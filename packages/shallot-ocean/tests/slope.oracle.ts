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
 *  through — the quantization quantum of the representation being compared, not an authored
 *  constant. */
const FLOAT32_EPSILON = 2 ** -23;

/**
 * Per-side accumulation depth: the count of independent `Float32Array` *store* points on the
 * path from a seeded draw to the realized moment `rasterSlopeMoment` reads (every intervening
 * arithmetic step — the time evolution's rotation, `slopeSpectra`'s `kx`/`kz` multiply, the
 * FFT's own butterfly — runs in plain JS `number`s or an explicit `Float64Array`, so it is the
 * *store*, not the arithmetic, that truncates to Float32):
 *   1. `generateH0`'s own output (`spectrum.ts:364-365`)
 *   2. `runSlopeCpuPipeline`'s time-evolved spectrum `h` (`slope.ts:211`, written `slope.ts:227-228`)
 *   3. `slopeSpectra`'s `x`/`z` outputs (`slope.ts:176-177`, written `slope.ts:187-190`)
 *   4. `ifft2`'s `out` (`fft.ts:121`, written `fft.ts:122-125`)
 * `ifft2`'s internal butterfly accumulates in `Float64Array` at every stage, so its own rounding
 * is ~1e-16 relative and negligible next to any one of the four Float32 stores above — it
 * contributes no fifth term here.
 */
const FLOAT32_STORE_DEPTH = 4;

/**
 * FFT/storage round-off bound for two rasters carrying identical modes by construction (same
 * seed, same represented band, denser grid's shared-`k` modes bit-identical to the base grid's —
 * asserted below). Each of the `FLOAT32_STORE_DEPTH` stores above introduces an independent
 * relative rounding of magnitude at most `FLOAT32_EPSILON`; treating the intervening linear steps
 * as non-amplifying (a real-scalar multiply, a norm-preserving FFT, no near-cancelling sum on
 * this path) bounds the compounded relative error in one side's realized field value, to first
 * order, by the additive worst case `FLOAT32_STORE_DEPTH · FLOAT32_EPSILON` — no averaging or
 * cancellation assumed. `realizedSlopeMss` then squares that field value
 * (`d(x²)/x² ≈ 2·dx/x`), doubling the bound, and the comparison sums two independently-rounded
 * sides (base `N` and denser `2N`), doubling it again. The combined multiplier is therefore
 * `FLOAT32_STORE_DEPTH · 2 (square) · 2 (two sides) = 16`.
 */
function fftRoundoffBound(): number {
    return FLOAT32_STORE_DEPTH * 2 * 2 * FLOAT32_EPSILON;
}

/**
 * A test-authored radial partition of the cascade's declared band, its outer edges (`config.kLo`,
 * `config.kHi`) read from production, its four interior cut points (20, 40, 50, 55) chosen here
 * for coverage and read from no production declaration — no drift arm is constructible against a
 * production source for those interior edges today, since none exists to drift against. The two
 * loops below iterate this array directly, so an emptied or shortened list would read green on
 * both; the cardinality assertion after each loop is what catches that.
 */
const SUB_BANDS = [
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
 * This by-path oracle compares seeded CPU realizations and their radial sub-bands. It does
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
            `2N=${denser.N} moment=${denserRasterMoment}, deviation=${deviation}, bound=${bound}, ` +
            `margin=${(bound / deviation).toFixed(1)}x`,
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
 * (`deviation < fftRoundoffBound()`) re-run with only the subject mutated. The base draw stays at
 * the guarded arm's own seed (17); only the denser side's draw changes, to an independent seed, in
 * place of the shared-by-construction draw the guarded arm uses. Two rasters sharing no drawn
 * modes carry different realized moments by physical variance, not merely round-off, so the same
 * tight bound must red here; if it didn't, the bound would be too loose to mean anything.
 */
test("slope N-invariance's round-off bound reds on an independently-drawn denser raster", () => {
    const denser = { ...config, N: config.N * 2 };
    const rasterMoment = rasterSlopeMoment(generateH0(config, 17), config);
    const independentDenserMoment = rasterSlopeMoment(generateH0(denser, 43), denser);
    const bound = fftRoundoffBound();
    const deviation = Math.abs(independentDenserMoment / rasterMoment - 1);
    console.log(
        `slope N-invariance round-off vacuity: independently-drawn deviation=${deviation}, ` +
            `bound=${bound}, overshoot=${(deviation / bound).toFixed(1)}x`,
    );
    expect(deviation).toBeGreaterThan(bound);
});

test("slope coverage checks every radial sub-band against its composed moment", () => {
    expect(SUB_BANDS.length).toBe(5);
    const h0 = generateH0(config, 0);
    let iterations = 0;
    for (const [lo, hi] of SUB_BANDS) {
        iterations++;
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
    expect(iterations).toBe(SUB_BANDS.length);
});

/**
 * The per-band twin of `slope.test.ts`'s "missing gradient k red-witnesses the composed-coverage
 * arm": the guarded arm above ("slope coverage checks every radial sub-band...") re-run at each
 * sub-band with only the realized subject mutated. The expectation (`composedSlopePsd`) and the
 * bound (`slopeMomentAgreementTolerance`) stay the guarded arm's own, unmutated, per band — no
 * separation floor authored here.
 */
test("missing gradient k red-witnesses the per-band coverage arm at every radial sub-band", () => {
    expect(SUB_BANDS.length).toBe(5);
    const h0 = generateH0(config, 0);
    let iterations = 0;
    for (const [lo, hi] of SUB_BANDS) {
        iterations++;
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
    expect(iterations).toBe(SUB_BANDS.length);
});
