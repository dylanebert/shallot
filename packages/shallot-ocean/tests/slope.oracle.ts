// Trigger: changes to slope-cascade resolution, patch length, declared band, or seeded realization
// can break the N-independent mode placement and represented slope moment checked here.
import { expect, test } from "bun:test";
import { ifft2 } from "../src/fft";
import {
    composedSlopePsd,
    rasterSlopeMoment,
    realizedSlopeMss,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    slopeMomentAgreementTolerance,
    slopeSpectra,
} from "../src/slope";
import { generateH0, kIndex } from "../src/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;

/** The `Float32` ULP spacing (machine epsilon) at 1.0 — twice the round-to-nearest relative
 *  rounding bound (`2 ** -24`), used here as a deliberately 2×-conservative stand-in for it. The
 *  safe direction, so kept rather than tightened. */
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
 * seed, same represented band; the base grid's Nyquist strictly covers `config.kHi` — asserted
 * below — so the denser grid's shared-`k` modes are bit-identical to the base grid's, itself
 * asserted in the guarded arm).
 *
 * The route is through Parseval, never through a single field sample. `realizedSlopeMss` reads
 * only each sample's real component, but the spectral coefficients here are Hermitian-symmetric
 * by construction (`slopeSpectra` applied to a real, Hermitian-symmetric `h`), so the discarded
 * imaginary components are themselves ~0 by construction and the quantity measured is, to that
 * same round-off, the field's total energy `(1/N²)·Σ_x|out(x)|²`. `ifft2` is UNNORMALIZED
 * (`fft.ts`'s own docblock), so by Parseval `Σ_x|out(x)|² = N²·Σ_k|in(k)|²` — verified numerically
 * below against the code's actual constant, not assumed. The `1/N²` in `realizedSlopeMss` cancels
 * that `N²`, so the realized moment is, up to round-off, exactly `Σ_k|in(k)|²`: a sum of
 * non-negative terms. That sum is immune to `ifft2`'s own near-cancelling butterfly sum
 * (`fft.ts:71`, `re[oddI] = evenR - tr`) because Parseval only needs the transform itself to be
 * exact — which it is, to float64 precision — never that any one output sample be well-conditioned
 * relative to the terms that produced it.
 *
 * Every one of the `FLOAT32_STORE_DEPTH` stores enters through that same route: a relative-
 * `FLOAT32_EPSILON` perturbation of one non-negative term in a sum of squares moves that term's
 * relative contribution, and so the sum's relative error, by at most `2·FLOAT32_EPSILON`
 * (`d(x²)/x² ≈ 2·dx/x`) regardless of the perturbations' signs — no cancellation needed on either
 * side of this argument. Stores 1–3 (`h0`, the time-evolved `h`, `slopeSpectra`'s `x`/`z`)
 * compound additively into the spectral coefficient `in(k)` the Parseval sum reads: bounded by
 * `3·FLOAT32_EPSILON` before squaring, `6·FLOAT32_EPSILON` after. Store 4 (`ifft2`'s own output
 * quantization) is a second, independent quantization applied directly to the stored field samples
 * summed on the left of the Parseval identity, contributing its own `2·FLOAT32_EPSILON`. One
 * side's worst-case relative error in the realized moment is therefore
 * `6·FLOAT32_EPSILON + 2·FLOAT32_EPSILON = 8·FLOAT32_EPSILON`; comparing two independently-rounded
 * sides (base `N` and denser `2N`) sums both sides' bound, giving the final multiplier
 * `FLOAT32_STORE_DEPTH · 2 (square) · 2 (two sides) = 16` — the same number a prior, incorrect
 * single-field-sample route also landed on, now reached without that route's two false premises
 * (an unnormalized FFT is not norm-preserving, and its butterfly sum is exactly the near-cancelling
 * sum that route disclaimed depending on).
 */
function fftRoundoffBound(): number {
    return FLOAT32_STORE_DEPTH * 2 * 2 * FLOAT32_EPSILON;
}

/**
 * Grounds the Parseval route above: `ifft2`'s own energy-scaling constant, derived from actual
 * code execution rather than assumed from `fft.ts`'s stated convention. A single unit impulse at
 * `k=(0,0)` produces an exactly-computable field — every sample equals `exp(i·0) = 1`, with no
 * floating-point rounding anywhere in that specific case (`1.0` stores and sums exactly) — so its
 * energy ratio pins the constant with no round-off to bound. That derived constant is then checked
 * against a real slope-cascade spectrum at the round-off tolerance store 4 alone contributes, the
 * same sub-claim `fftRoundoffBound`'s comment reasons from.
 */
test("ifft2's Parseval energy constant is derived from the code, not assumed", () => {
    const N = config.N;
    const impulse = new Float32Array(N * N * 2);
    impulse[0] = 1;
    const impulseField = ifft2(impulse, N);
    let impulseEnergy = 0;
    for (let i = 0; i < impulseField.length; i++) impulseEnergy += impulseField[i] ** 2;
    expect(impulseEnergy).toBe(N * N);
    const parsevalConstant = impulseEnergy;

    const { x: spectrum } = slopeSpectra(generateH0(config, 5), config);
    let inEnergy = 0;
    for (let i = 0; i < spectrum.length; i++) inEnergy += spectrum[i] ** 2;
    const field = ifft2(spectrum, N);
    let outEnergy = 0;
    for (let i = 0; i < field.length; i++) outEnergy += field[i] ** 2;
    const predicted = parsevalConstant * inEnergy;
    const bound = 2 * FLOAT32_EPSILON;
    const deviation = Math.abs(outEnergy / predicted - 1);
    console.log(
        `ifft2 Parseval: constant=${parsevalConstant} (N²=${N * N}), outEnergy=${outEnergy}, ` +
            `predicted=${predicted}, deviation=${deviation}, bound=${bound}`,
    );
    expect(deviation).toBeLessThan(bound);
});

test("slope cascade's declared band Nyquist headroom: N=256,L=13 covers kHi with margin", () => {
    const dk = (2 * Math.PI) / config.L;
    const kNyquist = (config.N / 2) * dk;
    expect(kNyquist).toBeGreaterThan(config.kHi);
});

/**
 * A test-authored radial partition of the cascade's declared band, its outer edges (`config.kLo`,
 * `config.kHi`) read from production, its four interior cut points (20, 40, 50, 55) chosen here
 * for coverage and read from no production declaration — no drift arm is constructible against a
 * production source for those interior edges today, since none exists to drift against. The two
 * loops below iterate this array directly, so an emptied or shortened list would read green on
 * both; the `expect(SUB_BANDS.length).toBe(5)` each test asserts before iterating is what catches
 * that. A per-loop iteration counter was tried and removed: it derives from the same array with no
 * early exit, so it is provably equal to `.length` by construction and cannot fail — a decorative
 * assertion that argues for coverage it does not check.
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
    for (const [lo, hi] of SUB_BANDS) {
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
 * arm": the guarded arm above ("slope coverage checks every radial sub-band...") re-run at each
 * sub-band with only the realized subject mutated. The expectation (`composedSlopePsd`) and the
 * bound (`slopeMomentAgreementTolerance`) stay the guarded arm's own, unmutated, per band — no
 * separation floor authored here.
 */
test("missing gradient k red-witnesses the per-band coverage arm at every radial sub-band", () => {
    expect(SUB_BANDS.length).toBe(5);
    const h0 = generateH0(config, 0);
    for (const [lo, hi] of SUB_BANDS) {
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
