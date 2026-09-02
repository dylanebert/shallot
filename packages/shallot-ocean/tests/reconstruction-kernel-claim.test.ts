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
// The only tolerance anywhere is f64 round-off, derived from EACH kernel's own operation count and the
// fixture's own magnitude (never authored, never a headroom multiplier) — printed beside every reading
// below. I3m-r correction round (2026-09-01): the round-off model used to be `bicubicSample`'s 110-flop
// count applied in front of `bilinearSample` and `nearestSample` evaluations too (the "exact on the
// linear" claim below graded bilinear's reading against bicubic's error model, and the C1 red witness
// graded bilinear's jump reading the same way) — each kernel now carries its own model, derived from its
// own source below, and a reading is only ever compared against the model for the kernel that produced
// it (never against another kernel's model, and never against `nearestSample`'s, since `nearestSample`
// performs no floating-point arithmetic on the payload at all and is only ever used as a RED-WITNESS
// subject, reusing the tolerance of the claim it is shown to break rather than a model of its own).
//
// Mutation table (each applied in place at this stage's own ref, run, reverted with
// `git show HEAD:<path>`, never shipped):
//   - `reconstruction.ts`'s `catmullRom1D` `c` coefficient (`0.5*p2` -> `0.35*p2`) -> RED: 9 pass / 3
//     fail (exact bi-quadratic reproduction, exact linear reproduction, and the O(h) convergence claim
//     all break).
//   - `reconstruction.ts`'s `wrap` narrowed to `i % n` (drops the negative-index wraparound) -> RED:
//     11 pass / 1 fail (only the dedicated literal-index assertions catch it — every other test's
//     sample points stay >= 2 texels from the seam by construction, so `wrap` is never called
//     out-of-range there; this is the fixture doing its job, not a coverage gap).
//   - `bicubicSample`'s row-loop `p1` tap (`wrap(ix, n)` -> `wrap(ix + 1, n)`, reading the wrong texel
//     at `t=0`) -> RED: 8 pass / 4 fail — reaches "exact at texel centres" (the arm neither of the two
//     mutations above can reach — `catmullRom1D`'s own `t` variable multiplies every one of `a`, `b`,
//     `c` at `t=0`, so the identity `bicubicSample(field, N, ix, iy) === field[iy][ix]` holds
//     regardless of any coefficient mutation by construction, and both recorded taps stay in-range so
//     `wrap` is inert there too — a mutation to WHICH texel is read is the only thing this claim can be
//     sensitive to, and this is that mutation) plus the bi-quadratic/linear bicubic-exactness and
//     bicubic C1-convergence claims, which also read through the shifted `p1` tap. Does not reach
//     "periodic under translation by N": the shift is itself periodic in `ix`
//     (`wrap(ix + N + 1, n) === wrap(ix + 1, n)`), so a mesh-relative offset stays offset by the same
//     wrong amount at `u` and `u + N` alike.
//   - `wrap`'s large-index branch clamped instead of reduced (`i >= n` returns `n - 1` rather than
//     `i % n`) -> RED: 10 pass / 2 fail — reaches "periodic under translation by N" (the shifted taps
//     at `u + N` land past `n` and collapse onto a single clamped texel, while the un-shifted taps at
//     `u` stay untouched, so the two readings diverge far past round-off) and the literal-index
//     assertions (`wrap(N, N)` and `wrap(N + 5, N)` both clamp to `N - 1` instead of reducing). Does
//     not reach "exact at texel centres": every tap
//     the texel-centre claim reads (`ix` in `[2, N-3]`, taps in `[1, N-1]`) stays `< n`, so the clamp
//     branch never triggers there.
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

// ── f64 round-off model — derived from each kernel's own operation count, never authored ────────────
const ROUNDING_UNIT = Number.EPSILON / 2; // f64 machine unit round-off

interface KernelRoundoffModel {
    readonly flops: number;
    readonly intermediateMagnitudeBound: number;
}

// bicubicSample: `catmullRom1D`'s coefficients, read directly off its source: `a`'s 4 terms sum
// |coeff| = 4, `b`'s = 6, `c`'s = 1; the Horner evaluation nests them under `t in [0,1]`, so the
// worst-case intermediate magnitude is bounded by |p1| + |c| + |b| + |a| <= 12 * (max |field value|).
// Flop count per `catmullRom1D` call, counted literally from its source: `a` (4 mul + 3 add/sub = 7),
// `b` (3 mul + 3 add/sub = 6), `c` (2 mul + 1 add = 3), the Horner eval (3 mul + 3 add = 6) = 22.
// `bicubicSample` makes 5 such calls (4 row combines + 1 column combine) = 110.
const CATMULLROM_FLOPS = 7 + 6 + 3 + 6; // 22
const BICUBIC_MODEL: KernelRoundoffModel = {
    flops: 5 * CATMULLROM_FLOPS, // 110
    intermediateMagnitudeBound: 12,
};

// bilinearSample: two subtracts (`1 - fx`, `1 - fy`), four corner terms each two multiplies
// (`t_ij * w1 * w2`, 4 * 2 = 8 muls), three adds summing the four terms — 2 + 8 + 3 = 13 flops, read
// literally off its source. Every weight (`1-fx`, `fx`, `1-fy`, `fy`) sits in `[0, 1]` and the four
// per-corner weights sum to exactly 1 (a convex combination), unlike Catmull-Rom's `a`/`b`/`c`
// coefficients above, which carry negative terms that amplify intermediates beyond the field
// magnitude (the 12x bound above). No bilinear intermediate product or partial sum can exceed the
// field magnitude itself, so its bound is 1x.
const BILINEAR_MODEL: KernelRoundoffModel = {
    flops: 2 + 8 + 3, // 13
    intermediateMagnitudeBound: 1,
};

/** forward-error bound on one kernel evaluation (Higham-style: `model.flops` floating ops each
 *  contributing relative error <= the rounding unit, over intermediates bounded by
 *  `model.intermediateMagnitudeBound * fieldMagnitude`). */
function evalRoundoff(fieldMagnitude: number, model: KernelRoundoffModel): number {
    return model.flops * ROUNDING_UNIT * model.intermediateMagnitudeBound * fieldMagnitude;
}

/** round-off floor on one `maxGradientJump` reading at step `h` for kernel `model`: 4 kernel
 *  evaluations feed the jump (2 for each one-sided difference), each division by `h` amplifying its
 *  evaluation error by `1/h`. */
function jumpRoundoff(h: number, fieldMagnitude: number, model: KernelRoundoffModel): number {
    return (4 * evalRoundoff(fieldMagnitude, model)) / h;
}

/**
 * Round-off-only floor on one `ratio = jump(h) / jump(h/2)` measurement's own uncertainty, where
 * `jump(h) = C*h + O(h^2) + roundoff(h)`. Halving `h` doubles the `1/h` round-off floor
 * (`jumpRoundoff(h/2, M) = 2 * jumpRoundoff(h, M)`), and estimating `C*h` by the measured `jumpAtH`
 * itself (the raw jump reading — a well-resolved, non-vanishing quantity at these `h`, never the
 * small `ratio - 2` deviation the claim below tests FOR smallness; see `deviationHalvingTolerance`'s
 * docblock for why THAT quantity may never sit in a denominator), first-order error propagation gives
 * `|ratio - 2| <= 2/(C*h) * (roundoff(h) + 2*roundoff(h/2)) = 10 * jumpRoundoff(h, M) / jumpAtH`.
 */
function ratioDeviationTolerance(
    jumpAtH: number,
    h: number,
    fieldMagnitude: number,
    model: KernelRoundoffModel,
): number {
    return (10 * jumpRoundoff(h, fieldMagnitude, model)) / jumpAtH;
}

/**
 * I3m-r correction round (2026-09-01): the C1 claim's tolerance used to be
 * `2 * (tol0/|dev0| + tol1/|dev1|)` — dividing by `dev0`/`dev1`, the very "ratio - 2" deviations the
 * claim tests for a halving relationship. A kernel converging BETTER than claimed (a smaller genuine
 * `dev0`/`dev1`) auto-widened its own tolerance under that form, so the assertion was structurally
 * unfailable in the limit regardless of how small the deviations happened to be (finite at today's
 * numbers, `devTol` ~= 0.37, but not derived from anything that keeps it finite).
 *
 * The claim "`dev0 / dev1 == 2`" is restated as "`dev0 - 2*dev1 == 0`" instead: each side's round-off
 * floor (`tol0` on `dev0`, `tol1` on `dev1` — themselves already relative-to-`jumpAtH` bounds from
 * `ratioDeviationTolerance`, not relative to `dev0`/`dev1`) propagates ADDITIVELY through the
 * subtraction (triangle inequality: `|dev0 - 2*dev1| <= tol0 + 2*tol1`), so the tolerance never
 * depends on how small `dev0`/`dev1` measure — it depends only on `tol0`/`tol1`, which are read off
 * `h`, the fixture magnitude and the raw jump readings.
 */
function deviationHalvingTolerance(tol0: number, tol1: number): number {
    return tol0 + 2 * tol1;
}

/** ASSERTED resolution floor (not merely printed): `dev`'s own round-off floor `tol` must sit
 *  strictly below `|dev|` itself, or the reading carries no signal above its own noise floor and the
 *  halving claim above is not yet a meaningful measurement of anything. */
function assertResolved(label: string, dev: number, tol: number): void {
    expect(
        Math.abs(dev),
        `${label}: |dev|=${Math.abs(dev).toExponential(3)} must clear its round-off floor ${tol.toExponential(3)} to be a resolved reading`,
    ).toBeGreaterThan(tol);
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

// I3m-r correction round (2026-09-01): neither recorded mutation above reached this claim — at
// `t=0` (an exact integer sample point), `catmullRom1D`'s return collapses to `p1` regardless of the
// `a`/`b`/`c` coefficients (each is multiplied by `t`), so no coefficient mutation can ever reach it,
// and both sample-point taps stay in-range so `wrap`'s narrowing is inert here too. This file's
// mutation table now records the mutation this claim IS sensitive to: which texel `p1` reads (the
// `bicubicSample` row-loop `p1` tap, `wrap(ix, n)` -> `wrap(ix + 1, n)`) — see this file's header.
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
    const tol = evalRoundoff(M, BICUBIC_MODEL);

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
    // I3m-r correction round (2026-09-01): bilinear's own reading used to be graded against
    // bicubic's 110-flop model (`tolBicubic` below) — each kernel now carries its own tolerance,
    // derived from its own operation count (see this file's header).
    const tolBicubic = evalRoundoff(M, BICUBIC_MODEL);
    const tolBilinear = evalRoundoff(M, BILINEAR_MODEL);

    test(`bicubic matches the linear closed form within its own f64 round-off bound (${tolBicubic.toExponential(3)})`, () => {
        for (const [u, v] of SAFE_UV) {
            const want = linear(u, v);
            const bicubicDiff = Math.abs(bicubicSample(linearField, N, u, v) - want);
            console.log(
                `linear(${u},${v}): bicubic diff=${bicubicDiff.toExponential(3)} tol=${tolBicubic.toExponential(3)}`,
            );
            expect(bicubicDiff).toBeLessThanOrEqual(tolBicubic);
        }
    });

    test(`bilinear matches the linear closed form within its own f64 round-off bound (${tolBilinear.toExponential(3)})`, () => {
        for (const [u, v] of SAFE_UV) {
            const want = linear(u, v);
            const bilinearDiff = Math.abs(bilinearSample(linearField, N, u, v) - want);
            console.log(
                `linear(${u},${v}): bilinear diff=${bilinearDiff.toExponential(3)} tol=${tolBilinear.toExponential(3)}`,
            );
            expect(bilinearDiff).toBeLessThanOrEqual(tolBilinear);
        }
    });

    test("RED-WITNESS — nearest breaks the same closed-form match at the same points, exceeding both kernels' tolerances", () => {
        // re-runs the guarded arm's own comparison with only the subject mutated to nearestSample,
        // against the more generous of the two tolerances above (bicubic's) — nearestSample performs
        // no floating-point arithmetic on the payload at all, so its diff is order-of-magnitude larger
        // than either model and clearing the looser bound implies clearing the tighter one too.
        for (const [u, v] of SAFE_UV) {
            const want = linear(u, v);
            const got = nearestSample(linearField, N, u, v);
            const diff = Math.abs(got - want);
            console.log(
                `nearest(${u},${v}) = ${got} vs closed form ${want}, |diff|=${diff.toExponential(3)} tol=${tolBicubic.toExponential(3)}`,
            );
            expect(diff).toBeGreaterThan(tolBicubic);
        }
    });
});

// I3m-r correction round (2026-09-01): neither recorded mutation above reached this claim either —
// the `c`-coefficient mutation changes VALUES, not which texel is read, so it does not distinguish
// `f(u+N,v+N)` from `f(u,v)` (both sides read the same, still-consistently-wrong texels), and the
// `wrap` mutation only drops NEGATIVE-index wraparound, which `u+N` (always positive here) never
// reaches. This file's mutation table now records the mutation this claim IS sensitive to: `wrap`'s
// large-index branch clamped instead of reduced — see this file's header.
describe("periodic under translation by N", () => {
    const M = fieldMagnitude(quadraticField);
    const tol = evalRoundoff(M, BICUBIC_MODEL) * 2; // two evaluations combined by subtraction

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

    /** shared by the green bicubic reading and its bilinear red witness — Gate law: "a red witness is
     *  the guarded arm's own assertion re-run with only the subject mutated" — so both call sites run
     *  the exact same measurement, only `kernel`/`model` differ. Computes `dev0`, `dev1` and their
     *  round-off floors for `kernel`, asserts the resolution floor (item 2's asserted floor, not a
     *  printed one), and returns `dev0 - 2*dev1` plus its additive tolerance for the caller to gate. */
    function measureDeviationHalving(
        kernel: (field: Field, n: number, u: number, v: number) => number,
        model: KernelRoundoffModel,
        label: string,
    ): { diff: number; tol: number } {
        const j0 = maxGradientJump(kernel, field, NGeneric, H0);
        const j1 = maxGradientJump(kernel, field, NGeneric, H1);
        const j2 = maxGradientJump(kernel, field, NGeneric, H2);
        const ratio01 = j0 / j1;
        const ratio12 = j1 / j2;
        const dev0 = ratio01 - 2;
        const dev1 = ratio12 - 2;
        const tol0 = ratioDeviationTolerance(j0, H0, M, model);
        const tol1 = ratioDeviationTolerance(j1, H1, M, model);
        assertResolved(`${label} dev0`, dev0, tol0);
        assertResolved(`${label} dev1`, dev1, tol1);
        const diff = dev0 - 2 * dev1;
        const tol = deviationHalvingTolerance(tol0, tol1);
        console.log(
            `${label} jump(h)=${j0.toExponential(4)} jump(h/2)=${j1.toExponential(4)} jump(h/4)=${j2.toExponential(4)} ` +
                `ratio01=${ratio01.toFixed(6)} ratio12=${ratio12.toFixed(6)} dev0=${dev0.toExponential(3)} dev1=${dev1.toExponential(3)} ` +
                `dev0-2*dev1=${diff.toExponential(3)} (tol ${tol.toExponential(3)})`,
        );
        return { diff, tol };
    }

    test("bicubic: dev0 - 2*dev1 is within its round-off floor of 0 (order-2 halving, no headroom)", () => {
        const { diff, tol } = measureDeviationHalving(bicubicSample, BICUBIC_MODEL, "bicubic");
        expect(Math.abs(diff)).toBeLessThanOrEqual(tol);
    });

    test("RED-WITNESS — bilinear's jump is h-independent (a real discontinuity, not O(h) truncation)", () => {
        // re-runs the guarded arm's own assertion ("dev0 - 2*dev1 within its round-off floor of 0")
        // with only the subject mutated to bilinearSample — I3m-r correction round (2026-09-01): this
        // used to test a DIFFERENT, weaker two-h claim ("ratio01 stays near 1, not 2") than the
        // guarded arm's own three-h halving claim, so the docblock ("re-runs the guarded arm's own
        // ratio-near-2 assertion") named an assertion the test didn't make. A real C0 discontinuity
        // does not shrink with h, so its ratio stays near 1 at every h — dev0 and dev1 both sit near
        // -1, and `dev0 - 2*dev1` sits near +1, far outside round-off.
        const { diff, tol } = measureDeviationHalving(
            bilinearSample,
            BILINEAR_MODEL,
            "bilinear RED-WITNESS",
        );
        expect(Math.abs(diff)).toBeGreaterThan(tol);
    });
});
