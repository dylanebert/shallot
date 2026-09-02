// Reconstruction kernel claim — closed-form correctness proofs for the shipped bicubic Catmull-Rom
// kernel (`bicubicSample`), replacing `checkReconstructionContinuity`'s authored-tolerance gradient-
// jump gate (I3m re-verdict, 2026-09-01): a linear operator's fidelity is proven on a polynomial
// fixture in closed form, never a tail statistic with an authored or headroom-padded tolerance.
//
// Two fixtures, two claim classes:
//
//   - A POSITION-ENCODED BI-QUADRATIC field (every texel a distinct value: `biQuadratic`) proves the
//     structural claims — exact reproduction at texel centres and everywhere on the polynomial,
//     periodicity under translation by `N`, and `wrap`'s own literal-index behaviour — with bicubic
//     Catmull-Rom's cubic-exactness as the mechanism (a bi-quadratic is a special case of the cubics a
//     uniform Catmull-Rom spline reproduces exactly, since its tangents are exact central differences
//     of the same polynomial). Sample points sit >= 2 texels from the wrap seam so the fixture's own
//     non-periodicity (a plain polynomial, not a periodic one) never contaminates the reading.
//     `bilinearSample` cannot reproduce the quadratic term (its own red witness), and a LINEAR-ONLY
//     fixture isolates the weaker claim `nearestSample` also fails: raw texel-nearest cannot reproduce
//     even a straight line off-grid.
//
//   - GENERIC (pseudo-random, non-polynomial) data via `syntheticField` proves the convergence claim:
//     a truly C1 kernel's two one-sided finite-difference derivative estimates at a texel boundary
//     converge to the same value as the differencing step `h` -> 0, so the residual "jump"
//     `maxGradientJump` reads is pure O(h) truncation error that HALVES when `h` halves — never a real
//     discontinuity, which reads h-independent instead (bilinear's own red witness below). The
//     bi-quadratic fixture above is unusable for THIS claim: bicubic reproduces it exactly everywhere
//     (see above), so the "jump" it would read degenerates to pure round-off noise that does not scale
//     as O(h) at all, which would prove nothing about convergence.
//
// The only tolerance anywhere is f64 round-off, derived from `bicubicSample`'s own operation count and
// the fixture's own magnitude (never authored, never a headroom multiplier) — printed beside every
// reading below.
//
// Mutation table (each applied in place at this stage's own ref, run, reverted with
// `git show HEAD:<path>`, never shipped):
//   - `reconstruction.ts`'s `catmullRom1D` `c` coefficient (`0.5*p2` -> `0.35*p2`) -> RED: 8 pass / 3
//     fail (exact bi-quadratic reproduction, exact linear reproduction, and the O(h) convergence ratio
//     all break).
//   - `reconstruction.ts`'s `wrap` narrowed to `i % n` (drops the negative-index wraparound) -> RED:
//     10 pass / 1 fail (only the dedicated literal-index assertions catch it — every other test's
//     sample points stay >= 2 texels from the seam by construction, so `wrap` is never called
//     out-of-range there; this is the fixture doing its job, not a coverage gap).
import { describe, expect, test } from "bun:test";
import {
    bicubicSample,
    bilinearSample,
    type Field,
    maxGradientJump,
    nearestSample,
    syntheticField,
    wrap,
} from "../src/reconstruction";

// ── f64 round-off model — derived from `bicubicSample`'s own operation count, never authored ────────
// `catmullRom1D`'s coefficients, read directly off its source: `a`'s 4 terms sum |coeff| = 4, `b`'s = 6,
// `c`'s = 1; the Horner evaluation nests them under `t in [0,1]`, so the worst-case intermediate
// magnitude is bounded by |p1| + |c| + |b| + |a| <= 12 * (max |field value|). Flop count per
// `catmullRom1D` call, counted literally from its source: `a` (4 mul + 3 add/sub = 7), `b` (3 mul + 3
// add/sub = 6), `c` (2 mul + 1 add = 3), the Horner eval (3 mul + 3 add = 6) = 22. `bicubicSample` makes
// 5 such calls (4 row combines + 1 column combine) = 110.
const ROUNDING_UNIT = Number.EPSILON / 2; // f64 machine unit round-off
const CATMULLROM_FLOPS = 7 + 6 + 3 + 6; // 22 — see derivation above
const BICUBIC_FLOPS = 5 * CATMULLROM_FLOPS; // 110
const INTERMEDIATE_MAGNITUDE_BOUND = 12; // |p1|+|c|+|b|+|a| coefficient-sum bound, see derivation above

/** forward-error bound on one `bicubicSample` evaluation (Higham-style: `n` floating ops each
 *  contributing relative error <= the rounding unit, over intermediates bounded by
 *  `INTERMEDIATE_MAGNITUDE_BOUND * fieldMagnitude`). */
function evalRoundoff(fieldMagnitude: number): number {
    return BICUBIC_FLOPS * ROUNDING_UNIT * INTERMEDIATE_MAGNITUDE_BOUND * fieldMagnitude;
}

/** round-off floor on one `maxGradientJump` reading at step `h`: 4 kernel evaluations feed the jump
 *  (2 for each one-sided difference), each division by `h` amplifying its evaluation error by `1/h`. */
function jumpRoundoff(h: number, fieldMagnitude: number): number {
    return (4 * evalRoundoff(fieldMagnitude)) / h;
}

/**
 * Round-off-only floor on `|ratio - 2|` for `jump(h) / jump(h/2)`, where
 * `jump(h) = C*h + O(h^2) + roundoff(h)`. Halving `h` doubles the `1/h` round-off floor
 * (`jumpRoundoff(h/2, M) = 2 * jumpRoundoff(h, M)`), and estimating `C*h` by the measured `jumpAtH`
 * itself (the reading, never a separate authored constant), first-order error propagation gives
 * `|ratio - 2| <= 2/(C*h) * (roundoff(h) + 2*roundoff(h/2)) = 10 * jumpRoundoff(h, M) / jumpAtH`.
 * This is the FLOOR on the ratio measurement's own uncertainty — it is not, by itself, a bound on
 * `ratio - 2`: the true jump also carries a genuine O(h) Taylor correction term (from
 * `f'(k+1.5h)-f'(k-1.5h) = 3h*f''(k) + O(h^3)` plus each one-sided estimate's own O(h^2) truncation),
 * so `ratio - 2` itself is `O(h)`, not round-off-scale. `ratioDeviationHalvingTolerance` below is what
 * asserts the actual claim ("ratio deviation halving across three h") using this floor.
 */
function ratioDeviationTolerance(jumpAtH: number, h: number, fieldMagnitude: number): number {
    return (10 * jumpRoundoff(h, fieldMagnitude)) / jumpAtH;
}

/**
 * The C1 claim is that `ratio - 2` (a genuine O(h) truncation term, not noise — see
 * `ratioDeviationTolerance`'s docblock) itself HALVES when `h` halves, proving the underlying "jump" is
 * O(h) all the way down rather than settling on some nonzero floor (which a real discontinuity would).
 * `dev0 = ratio(h0,h1) - 2`, `dev1 = ratio(h1,h2) - 2`; a real O(h) term gives `dev0 / dev1 == 2` up to
 * each ratio's OWN round-off floor (`tol0`, `tol1`, from `ratioDeviationTolerance`) propagated through
 * the division — a difference-of-two-small-quantities amplifies relative round-off, which is exactly
 * what this propagation states rather than pads: `|dev0/dev1 - 2| <= 2 * (tol0/|dev0| + tol1/|dev1|)`
 * (triangle inequality over each side's own relative round-off contribution).
 */
function ratioDeviationHalvingTolerance(
    dev0: number,
    dev1: number,
    tol0: number,
    tol1: number,
): number {
    return 2 * (tol0 / Math.abs(dev0) + tol1 / Math.abs(dev1));
}

function fieldMagnitude(field: Field): number {
    let m = 0;
    for (const row of field) for (const v of row) if (Math.abs(v) > m) m = Math.abs(v);
    return m;
}

// ── position-encoded bi-quadratic fixture ────────────────────────────────────────────────────────
const N = 24;
// coefficients with no shared rational relationship, so `biQuadratic`/`linear` take a distinct value
// at every one of the N*N integer texel positions — verified below, not just assumed.
function biQuadratic(x: number, y: number): number {
    return 0.372 * x * x + 0.613 * y * y + 0.197 * x * y + 0.839 * x + 0.293 * y + 0.113;
}
function linear(x: number, y: number): number {
    return 0.839 * x + 0.293 * y + 0.113;
}
function fieldFrom(fn: (x: number, y: number) => number, n: number): Field {
    const field: Field = [];
    for (let y = 0; y < n; y++) {
        const row: number[] = new Array(n);
        for (let x = 0; x < n; x++) row[x] = fn(x, y);
        field.push(row);
    }
    return field;
}
const quadraticField = fieldFrom(biQuadratic, N);
const linearField = fieldFrom(linear, N);

// sample points >= 2 texels from the wrap seam (bicubic's own taps span ix-1..ix+2, so anywhere within
// [2, N-4] keeps every tap inside the non-periodic fixture's real domain).
const SAFE_UV: Array<[number, number]> = [
    [6.3, 8.7],
    [10.2, 14.9],
    [15.6, 5.4],
    [11.1, 11.1],
];

describe("position-encoded fixture — every texel a distinct value", () => {
    test("bi-quadratic fixture has no repeated texel value", () => {
        const values = new Set<number>();
        for (const row of quadraticField) for (const v of row) values.add(v);
        expect(values.size).toBe(N * N);
    });
    test("linear fixture has no repeated texel value", () => {
        const values = new Set<number>();
        for (const row of linearField) for (const v of row) values.add(v);
        expect(values.size).toBe(N * N);
    });
});

describe("exact at texel centres", () => {
    test("bicubic reads the exact stored value at every integer (u, v) >= 2 texels from the seam", () => {
        for (let iy = 2; iy < N - 2; iy++) {
            for (let ix = 2; ix < N - 2; ix++) {
                expect(bicubicSample(quadraticField, N, ix, iy)).toBe(quadraticField[iy][ix]);
            }
        }
    });
});

describe("exact on the bi-quadratic — bicubic reproduces it, bilinear does not", () => {
    const M = fieldMagnitude(quadraticField);
    const tol = evalRoundoff(M);

    test(`bicubic matches the closed form within the f64 round-off bound (${tol.toExponential(3)})`, () => {
        for (const [u, v] of SAFE_UV) {
            const got = bicubicSample(quadraticField, N, u, v);
            const want = biQuadratic(u, v);
            console.log(
                `bicubic(${u},${v}) = ${got} vs closed form ${want}, |diff|=${Math.abs(got - want).toExponential(3)} tol=${tol.toExponential(3)}`,
            );
            expect(Math.abs(got - want)).toBeLessThanOrEqual(tol);
        }
    });

    test("RED-WITNESS — bilinear breaks the same closed-form match at the same points and tolerance", () => {
        // re-runs the guarded arm's own comparison with only the subject (the kernel) mutated to
        // bilinearSample, against the SAME closed form and the SAME tolerance.
        for (const [u, v] of SAFE_UV) {
            const got = bilinearSample(quadraticField, N, u, v);
            const want = biQuadratic(u, v);
            const diff = Math.abs(got - want);
            console.log(
                `bilinear(${u},${v}) = ${got} vs closed form ${want}, |diff|=${diff.toExponential(3)} tol=${tol.toExponential(3)}`,
            );
            expect(diff).toBeGreaterThan(tol);
        }
    });
});

describe("exact on the linear — bicubic and bilinear reproduce it, nearest does not", () => {
    const M = fieldMagnitude(linearField);
    const tol = evalRoundoff(M);

    test(`bicubic and bilinear match the linear closed form within the f64 round-off bound (${tol.toExponential(3)})`, () => {
        for (const [u, v] of SAFE_UV) {
            const want = linear(u, v);
            const bicubicDiff = Math.abs(bicubicSample(linearField, N, u, v) - want);
            const bilinearDiff = Math.abs(bilinearSample(linearField, N, u, v) - want);
            console.log(
                `linear(${u},${v}): bicubic diff=${bicubicDiff.toExponential(3)} bilinear diff=${bilinearDiff.toExponential(3)} tol=${tol.toExponential(3)}`,
            );
            expect(bicubicDiff).toBeLessThanOrEqual(tol);
            expect(bilinearDiff).toBeLessThanOrEqual(tol);
        }
    });

    test("RED-WITNESS — nearest breaks the same closed-form match at the same points and tolerance", () => {
        // re-runs the guarded arm's own comparison with only the subject mutated to nearestSample.
        for (const [u, v] of SAFE_UV) {
            const want = linear(u, v);
            const got = nearestSample(linearField, N, u, v);
            const diff = Math.abs(got - want);
            console.log(
                `nearest(${u},${v}) = ${got} vs closed form ${want}, |diff|=${diff.toExponential(3)} tol=${tol.toExponential(3)}`,
            );
            expect(diff).toBeGreaterThan(tol);
        }
    });
});

describe("periodic under translation by N", () => {
    const M = fieldMagnitude(quadraticField);
    const tol = evalRoundoff(M) * 2; // two evaluations combined by subtraction

    test("bicubic(u + N, v + N) matches bicubic(u, v) within f64 round-off", () => {
        for (const [u, v] of SAFE_UV) {
            const a = bicubicSample(quadraticField, N, u, v);
            const b = bicubicSample(quadraticField, N, u + N, v + N);
            const diff = Math.abs(a - b);
            console.log(
                `translate(${u},${v}) diff=${diff.toExponential(3)} tol=${tol.toExponential(3)}`,
            );
            expect(diff).toBeLessThanOrEqual(tol);
        }
    });
});

describe("wrap — literal index assertions", () => {
    test("negative and over-range indices wrap to the literal expected index", () => {
        expect(wrap(-1, N)).toBe(N - 1);
        expect(wrap(N, N)).toBe(0);
        expect(wrap(N + 5, N)).toBe(5);
        expect(wrap(-N - 3, N)).toBe(N - 3);
        expect(wrap(0, N)).toBe(0);
        expect(wrap(N - 1, N)).toBe(N - 1);
    });
});

describe("C1 by order of convergence — generic (non-polynomial) data", () => {
    // the position-encoded fixtures above are unusable here: bicubic reproduces the bi-quadratic
    // EXACTLY everywhere (see this file's header), so the finite-difference "jump" it would read
    // degenerates to pure round-off noise that does not scale as O(h) at all. `syntheticField`'s
    // generic, non-polynomial data is what the original continuity gate exercised, and is what a real
    // O(h) truncation signal needs to be visible above round-off.
    const NGeneric = 128;
    const field = syntheticField(NGeneric, 1);
    const M = fieldMagnitude(field);
    const H0 = 2e-4;
    const H1 = H0 / 2;
    const H2 = H1 / 2;

    test("bicubic: the ratio's own deviation from 2 halves as h halves (ratio deviation halving, no headroom)", () => {
        const j0 = maxGradientJump(bicubicSample, field, NGeneric, H0);
        const j1 = maxGradientJump(bicubicSample, field, NGeneric, H1);
        const j2 = maxGradientJump(bicubicSample, field, NGeneric, H2);
        const ratio01 = j0 / j1;
        const ratio12 = j1 / j2;
        const dev0 = ratio01 - 2;
        const dev1 = ratio12 - 2;
        const tol0 = ratioDeviationTolerance(j0, H0, M);
        const tol1 = ratioDeviationTolerance(j1, H1, M);
        const devRatio = dev0 / dev1;
        const devTol = ratioDeviationHalvingTolerance(dev0, dev1, tol0, tol1);
        console.log(
            `bicubic jump(h)=${j0.toExponential(4)} jump(h/2)=${j1.toExponential(4)} jump(h/4)=${j2.toExponential(4)} ` +
                `ratio01=${ratio01.toFixed(6)} ratio12=${ratio12.toFixed(6)} dev0=${dev0.toExponential(3)} dev1=${dev1.toExponential(3)} ` +
                `devRatio=${devRatio.toFixed(4)} (tol ${devTol.toExponential(2)} around 2)`,
        );
        expect(Math.abs(devRatio - 2)).toBeLessThanOrEqual(devTol);
    });

    test("RED-WITNESS — bilinear's jump is h-independent (a real discontinuity, not O(h) truncation)", () => {
        // re-runs the guarded arm's own ratio-near-2 assertion with only the subject mutated: a real
        // C0 discontinuity does not shrink with h, so its ratio stays near 1, not 2.
        const j0 = maxGradientJump(bilinearSample, field, NGeneric, H0);
        const j1 = maxGradientJump(bilinearSample, field, NGeneric, H1);
        const ratio01 = j0 / j1;
        const tol01 = ratioDeviationTolerance(j0, H0, M);
        console.log(
            `bilinear RED-WITNESS: jump(h)=${j0.toExponential(4)} jump(h/2)=${j1.toExponential(4)} ratio01=${ratio01.toFixed(6)} (tol ${tol01.toExponential(2)})`,
        );
        expect(Math.abs(ratio01 - 2)).toBeGreaterThan(tol01);
    });
});
