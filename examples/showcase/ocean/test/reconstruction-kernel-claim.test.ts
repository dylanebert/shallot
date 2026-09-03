// Reconstruction kernel claim — closed-form correctness proofs for the shipped bicubic Catmull-Rom
// kernel (`bicubicSample`). A linear operator's fidelity is proven on a polynomial fixture in closed
// form, never a tail statistic with an authored or headroom-padded tolerance, and continuity is the
// discrete identity a piecewise cubic licenses — never an order-of-convergence limit (I3m-r correction
// round 2, 2026-09-01: `checkReconstructionContinuity` and `maxGradientJump` are retired; this file's
// seam-identity arm below is their heir).
//
// Three fixtures, three claim classes:
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
//   - GENERIC (pseudo-random, non-polynomial) `syntheticField` data proves the CONTINUITY claim, as the
//     seam identity a piecewise cubic licenses rather than a limit: for fixed `v`, `bicubicSample` is a
//     degree-<=3 polynomial in `u` on each cell (the row taps `p0..p3` are constant within a cell, and
//     `catmullRom1D` is cubic in its own `t` argument), so four IN-CELL samples on each side of a seam
//     determine that side's cubic exactly — a closed-form Lagrange fit through those four samples
//     recovers the cell's seam value and seam derivative in closed form, with no `h`, no limit, and no
//     truncation term. C0 asserts the two sides' seam values agree; C1 asserts the two sides' seam
//     derivatives agree — both within a bound derived from the Lagrange weights' own absolute sums, the
//     kernel's per-call round-off model, and the fixture's magnitude (never authored). The fit's own
//     premise — that the four in-cell samples really do determine a single cubic spanning the whole
//     cell, not just the four points fitted — is its own leg: the same fit must predict an unseen FIFTH
//     in-cell sample within the same bound. A seam-STRADDLING stencil (two samples from each side) is
//     the leg's control: since the two cells carry different cubics beyond first order (C1 only, not
//     C3), a fit built across the seam must MISS its own fifth sample by more than the bound — proving
//     the premise leg is discriminating rather than vacuous.
//
// The only tolerance anywhere is f64 round-off, derived from EACH kernel's own operation count and the
// fixture's own magnitude (never authored, never a headroom multiplier) — printed beside every reading
// below. Every seam-identity reading is graded at bicubic's own round-off bound, never a per-kernel one
// (Gate law: "a red witness is the guarded arm's own assertion re-run with only the subject mutated") —
// `GuardedModel` (= `BICUBIC_MODEL`) is resolved exactly once, and every witness call site below
// passes only the swapped kernel, never a second model argument alongside it.
//
// Mutation table (each applied in place at this stage's own ref, run, reverted with
// `git show HEAD:<path>`, never shipped — the exact counts below are measured, one mutation at a time,
// against this file's own suite of 26 tests, re-measured for the I3m-r bound re-verdict (round 3) after
// every kernel coefficient and weight the bound consumes was switched from a transcribed literal to a
// basis-vector evaluation of the shipped kernel (finding 1), so the coefficient-sums assertion and the
// closed-form identity pointwise check are now sensitive to the same source mutations the readings
// below already were):
//   - `reconstruction.ts`'s `catmullRom1D` `c` coefficient (`0.5*p2` -> `0.35*p2`) -> RED: 18 pass / 8
//     fail (the recovered coefficient-sums assertion itself — `CATMULLROM_C_ABS_SUM` reads `0.85`
//     against the closed-form `1`, since the mutation is read back live rather than through a
//     transcribed literal — the Σ|w_i(t)| = 1+t(1-t) pointwise identity check, exact bi-quadratic
//     reproduction, exact linear reproduction, both BICUBIC_MODEL before/after comparisons — which
//     re-run the bi-quadratic and seam-leg readings respectively — and BOTH seam legs (C0 and C1):
//     mutating `c` breaks Catmull-Rom's `a+b+c === p2-p1` identity, so a cell's true value AND
//     derivative at its far (`t=1`) boundary stop matching the next cell's exact `t=0` value of `p1`, a
//     real discontinuity the closed-form fit on each side correctly recovers and compares). The premise
//     and straddle-control legs do NOT break: each cell is still some cubic (just a different one), and
//     a closed-form fit through four points of any cubic reproduces every other point of that same cubic
//     exactly, so the mutation is invisible to a same-cell reading by construction.
//   - `reconstruction.ts`'s `wrap` narrowed to `i % n` (drops the negative-index wraparound) -> RED: 12
//     pass / 14 fail — reached at seam `k=0` on both axes, where the left-side fit's samples land at
//     `ix`/`iy = -1` and every tap read goes through `wrap(ix-1, n) = wrap(-2, n)`; the narrowed form
//     returns a negative row/column index there, and indexing `field[negativeIndex]` (the OUTER array,
//     not a texel) is `undefined`, so the next index into it throws — every one of the 13 seam-identity
//     tests (C0, C1, premise, straddle-control, the seam-leg before/after test, and all 3 red-witness
//     describes, the Lagrange-4 one carrying its own straddle-control leg) shares this same `k=0` case in
//     its own sweep and throws independently (each computes its own sweep rather than a shared one for
//     exactly this reason — see this file's "continuity" describe), plus the dedicated `wrap`
//     literal-index assertions break on their own terms. Every other seam (`k = 1..N-1`) stays in-range
//     and is unaffected — this is the fixture doing its job (the periodic seam is swept, not skipped),
//     not a coverage gap. The recovered coefficient-sums and pointwise-identity checks do NOT reach this
//     mutation: neither ever calls `wrap`, `catmullRom1D` alone having no texel indices to wrap.
//   - `bicubicSample`'s row-loop `p1` tap (`wrap(ix, n)` -> `wrap(ix + 1, n)`, reading the wrong texel
//     at `t=0`) -> RED: 19 pass / 7 fail — reaches "exact at texel centres" (the arm neither of the two
//     mutations above can reach — `catmullRom1D`'s own `t` variable multiplies every one of `a`, `b`,
//     `c` at `t=0`, so the identity `bicubicSample(field, N, ix, iy) === field[iy][ix]` holds regardless
//     of any coefficient mutation by construction, and both recorded taps stay in-range so `wrap` is
//     inert there too — a mutation to WHICH texel is read is the only thing this claim can be sensitive
//     to, and this is that mutation), the bi-quadratic/linear bicubic-exactness claims, both
//     BICUBIC_MODEL before/after comparisons, and BOTH seam legs (shifting `p1` breaks the
//     same-grid-point identity each side's fit relies on: the left cell's true `t=1` value is no longer
//     `p2 = field[k]`). Does not reach "periodic under translation by N": the shift is itself periodic in
//     `ix` (`wrap(ix + N + 1, n) === wrap(ix + 1, n)`), so a mesh-relative offset stays offset by the
//     same wrong amount at `u` and `u + N` alike. Does not reach premise, straddle-control, or the
//     recovered coefficient-sums/pointwise-identity checks: `catmullRom1D` itself is untouched, so each
//     cell is still cubic and a same-cell in-cell fit still reproduces it exactly.
//   - `wrap`'s large-index branch clamped instead of reduced (`i >= n` returns `n - 1` rather than
//     `i % n`) -> RED: 23 pass / 3 fail — reaches "periodic under translation by N" (the shifted taps at
//     `u + N` land past `n` and collapse onto a single clamped texel, while the un-shifted taps at `u`
//     stay untouched, so the two readings diverge far past round-off), the literal-index assertions
//     (`wrap(N, N)` and `wrap(N + 5, N)` both clamp to `N - 1` instead of reducing), and — unexpectedly —
//     the `nearest` red-witness's premise leg: `nearestSample` rounds with `Math.round`, and at the last
//     seam (`k = N-1 = 127`) an offset of `0.9` rounds `127.9` up to exactly `128 = n`, tripping the
//     clamp branch and changing that one reading enough to (coincidentally) clear the "must exceed the
//     bound" assertion at that single seam. Does not reach "exact at texel centres", any other
//     seam-identity leg (including the seam-leg before/after test and the Lagrange-4 straddle-control
//     leg), or the recovered coefficient-sums/pointwise-identity checks: `catmullRom1D` itself is
//     untouched and every other in-range tap stays `< n`, so the clamp branch is otherwise inert.
import { describe, expect, test } from "bun:test";
import {
    bicubicSample,
    bilinearSample,
    catmullRom1D,
    type Field,
    nearestSample,
    type ReconstructionKernel,
    syntheticField,
    wrap,
} from "../src/ocean/reconstruction";

// ── f64 round-off model — derived from each kernel's own operation count and coefficient structure,
// never authored ─────────────────────────────────────────────────────────────────────────────────
const ROUNDING_UNIT = Number.EPSILON / 2; // f64 machine unit round-off

interface KernelRoundoffModel {
    readonly flops: number;
    readonly intermediateMagnitudeBound: number;
}

function absSum(values: readonly number[]): number {
    let s = 0;
    for (const v of values) s += Math.abs(v);
    return s;
}

/** max_{t in [0,1]} sum_i |w_i(t)| for a cubic blending-function set, sampled on a grid — used only as
 *  a CHECK against the closed form below (`MAX_ROW_WEIGHT_SUM_CATMULLROM`), never as the bound itself:
 *  an odd `samples` count never lands exactly on `t=0.5` (the retired 20001-sample sweep computed
 *  `i / 20001` for `i` in `[0, 20001]`, 20002 points, none of them `0.5`), which is exactly the point
 *  the maximum is attained, so a dense-sample search silently under-reports it. `samples` here is even
 *  so `t = samples/2 / samples = 0.5` is hit exactly. */
function maxAbsWeightSum(weights: (t: number) => readonly number[], samples: number): number {
    let max = 0;
    for (let i = 0; i <= samples; i++) {
        const t = i / samples;
        const sum = absSum(weights(t));
        if (sum > max) max = sum;
    }
    return max;
}

// `catmullRom1D`'s blending functions, RECOVERED from the shipped kernel by basis-vector evaluation
// (I3m-r bound re-verdict, item 1) — never a transcription of its `a`/`b`/`c` source expressions.
// Passing `e_i` (all-zero except `p_i = 1`) isolates `w_i(t) = catmullRom1D(e_i, t)` exactly, since
// the kernel is linear in `(p0..p3)` for any fixed `t`. A mutation to any of `catmullRom1D`'s own
// coefficients changes what this function reads back, unlike a hand-copied literal array — the
// difference the round-3 correction exists to make.
function catmullRomWeights(t: number): [number, number, number, number] {
    return [
        catmullRom1D(1, 0, 0, 0, t),
        catmullRom1D(0, 1, 0, 0, t),
        catmullRom1D(0, 0, 1, 0, t),
        catmullRom1D(0, 0, 0, 1, t),
    ];
}

// max_{t in [0,1]} sum_i |w_i(t)| for `catmullRom1D`'s blending functions, in closed form:
// `Σ_i w_i(t) = 1` always (Catmull-Rom reproduces a constant field exactly — a partition of unity),
// and on `[0,1]` the two side weights `w0`, `w3` are the only ones that go negative (`w0(t) =
// -0.5*t*(1-t)^2`, `w3(t) = 0.5*t^2*(t-1)`, both <= 0 there), so `Σ_i |w_i(t)| = (w1+w2) - (w0+w3) = 1
// - 2*(w0(t)+w3(t)) = 1 + t*(1-t)` — a ONE-LINE identity, verified below against a fine grid rather
// than searched, whose maximum on `[0,1]` lands at `t=0.5` (`1 + 0.5*0.5 = 1.25`).
const MAX_ROW_WEIGHT_SUM_CATMULLROM = absSum(catmullRomWeights(0.5));

// Flop count per `catmullRom1D` call, counted literally from its source: `a` (4 mul + 3 add/sub = 7),
// `b` (3 mul + 3 add/sub = 6), `c` (2 mul + 1 add = 3), the Horner eval (3 mul + 3 add = 6) = 22.
const CATMULLROM_FLOPS = 7 + 6 + 3 + 6; // 22

/** recover `catmullRom1D`'s per-`p_i` Horner-collected coefficients (`a_i`, `b_i`, `c_i` — the
 *  coefficient of `p_i` within the collected `a`, `b`, `c` the kernel's Horner chain evaluates) by
 *  basis-vector evaluation, never a transcription of the source's `a`/`b`/`c` expressions.
 *  `w_i(t) = [i===1] + t*c_i + t^2*b_i + t^3*a_i` is an exact cubic in `t` (the Horner form `p1 +
 *  t*(c + t*(b + t*a))` expanded), so its three non-constant coefficients are recovered exactly from
 *  four basis-vector evaluations at integer `t = 0,1,2,3` via finite differences — the third
 *  difference divided by `3!` gives the cubic coefficient (`a_i`), the reduced second difference
 *  gives the quadratic one (`b_i`), and the remainder gives the linear one (`c_i`). A mutated kernel's
 *  own coefficients are recovered just as faithfully — the recovery is generic over any cubic in
 *  `t`, never asserted equal to the source by construction; the two literal-index evaluations at
 *  `t=2`, `t=3` (beyond the `t=0,1` a cubic already needs) are what catch a non-cubic kernel's
 *  divergence from this fit. */
function catmullRomHornerCoeffs(i: 0 | 1 | 2 | 3): { a: number; b: number; c: number } {
    const p: [number, number, number, number] = [0, 0, 0, 0];
    p[i] = 1;
    const f = (t: number) => catmullRom1D(p[0], p[1], p[2], p[3], t);
    const f0 = f(0);
    const f1 = f(1);
    const f2 = f(2);
    const f3 = f(3);
    const d1 = f1 - f0;
    const d2 = f2 - f1;
    const d3 = f3 - f2;
    const dd1 = d2 - d1;
    const dd2 = d3 - d2;
    const a = (dd2 - dd1) / 6;
    const b = (dd1 - 6 * a) / 2;
    const c = d1 - a - b;
    return { a, b, c };
}

function catmullRomAbsSum(component: "a" | "b" | "c"): number {
    let sum = 0;
    for (let i = 0 as 0 | 1 | 2 | 3; i < 4; i++)
        sum += Math.abs(catmullRomHornerCoeffs(i)[component]);
    return sum;
}

const CATMULLROM_A_ABS_SUM = catmullRomAbsSum("a"); // recovered, expected 4
const CATMULLROM_B_ABS_SUM = catmullRomAbsSum("b"); // recovered, expected 6
const CATMULLROM_C_ABS_SUM = catmullRomAbsSum("c"); // recovered, expected 1

// `catmullRom1D` evaluates `p1 + t*(c + t*(b + t*a))` (Horner form), `t in [0,1]`. With every input
// bounded by M (`|p_i| <= M`), each collected coefficient is bounded by its own absolute-sum times M
// (`|a| <= 4M`, `|b| <= 6M`, `|c| <= 1M`), and each Horner step multiplies by `t <= 1` and adds the
// next term, so the running bound only grows:
const CATMULLROM_HORNER_STEP1_BOUND = CATMULLROM_B_ABS_SUM + CATMULLROM_A_ABS_SUM; // |b + t*a| <= 6M + 4M = 10M
const CATMULLROM_HORNER_STEP2_BOUND = CATMULLROM_C_ABS_SUM + CATMULLROM_HORNER_STEP1_BOUND; // |c + t*(...)| <= 1M + 10M = 11M
// |p1 + t*(...)| <= 1M + 11M = 12M — the per-call intermediate-magnitude bound, in units of the call's
// own input bound M (never a bare literal: derived from the coefficient sums above).
const CATMULLROM_INTERMEDIATE_MULTIPLIER = 1 + CATMULLROM_HORNER_STEP2_BOUND;

// `bicubicSample` makes 5 `catmullRom1D` calls: 4 ROW calls over raw texels (inputs bounded by the
// field magnitude M, so — by the Horner derivation above — intermediates bounded by
// `CATMULLROM_INTERMEDIATE_MULTIPLIER * M` = 12M) and 1 COLUMN call over the 4 ROW OUTPUTS. A row
// call's OUTPUT (not its intermediates) is itself bounded by `W * M`, where `W =
// MAX_ROW_WEIGHT_SUM_CATMULLROM` (output = sum_i w_i(t)*p_i, so |output| <= W * max|p_i|), so the
// column call's INPUTS are bounded by W*M, and its own intermediates — same Horner chain, same 12x
// multiplier — are bounded by `12 * W * M`. The column call's bound dominates the row calls' (`W > 1`),
// so it is the per-evaluation bound `evalRoundoff` consumes for the whole kernel — stated per call,
// never as a single hand-derived literal.
// both constants below are MULTIPLIERS over the field magnitude M (row inputs are bounded by M itself,
// column inputs by W*M), never a bound in the field's own units.
const BICUBIC_ROW_INTERMEDIATE_BOUND = CATMULLROM_INTERMEDIATE_MULTIPLIER; // 12x over M, row-call inputs
const BICUBIC_COLUMN_INTERMEDIATE_BOUND =
    CATMULLROM_INTERMEDIATE_MULTIPLIER * MAX_ROW_WEIGHT_SUM_CATMULLROM; // 12*W x over M, column-call inputs
const BICUBIC_MODEL: KernelRoundoffModel = {
    flops: 5 * CATMULLROM_FLOPS, // 110
    intermediateMagnitudeBound: Math.max(
        BICUBIC_ROW_INTERMEDIATE_BOUND,
        BICUBIC_COLUMN_INTERMEDIATE_BOUND,
    ),
};

// bilinearSample: two subtracts (`1 - fx`, `1 - fy`), four corner terms each two multiplies
// (`t_ij * w1 * w2`, 4 * 2 = 8 muls), three adds summing the four terms — 2 + 8 + 3 = 13 flops, read
// literally off its source. Every weight (`1-fx`, `fx`, `1-fy`, `fy`) sits in `[0, 1]` and the four
// per-corner weights sum to exactly 1 (a convex combination), unlike Catmull-Rom's `a`/`b`/`c`
// coefficients above, whose absolute sums (4, 6, 1) exceed 1 and so amplify intermediates to multiples
// of the field magnitude through the Horner chain. No bilinear intermediate product or partial sum can
// exceed the field magnitude itself, so its bound is 1x — a single combine step, no row/column
// compounding.
const BILINEAR_MODEL: KernelRoundoffModel = {
    flops: 2 + 8 + 3, // 13
    intermediateMagnitudeBound: 1,
};

// ── test-side four-point Lagrange kernel (item 2's witness) ──────────────────────────────────────
// Plain cubic Lagrange interpolation through the SAME 4 taps `bicubicSample` reads (nodes at local
// coordinates -1, 0, 1, 2), as opposed to Catmull-Rom's Hermite-style blend. Exact on cubics (any
// degree-<=3 source is reproduced everywhere by a 4-point Lagrange fit) — including at t=0 and t=1,
// where it reproduces p1/p2 exactly, same as Catmull-Rom — but its DERIVATIVE at those shared
// endpoints is whatever that cell's own 4-point cubic implies, with no cross-cell tangent-matching
// construction the way Catmull-Rom's `a`/`b`/`c` provide, so it is generically only C0, not C1.
function lagrangeCubic1D(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const tm1 = t - 1;
    const tm2 = t - 2;
    const tp1 = t + 1;
    const l0 = (t * tm1 * tm2) / -6;
    const l1 = (tp1 * tm1 * tm2) / 2;
    const l2 = (tp1 * t * tm2) / -2;
    const l3 = (tp1 * t * tm1) / 6;
    return p0 * l0 + p1 * l1 + p2 * l2 + p3 * l3;
}

/** the test-side witness kernel — same 16-tap structure as `bicubicSample`, `lagrangeCubic1D` in place
 *  of `catmullRom1D`. Committed here (never in `src/`), the one witness separating cubic-exactness
 *  from C1. */
function lagrange4Sample(field: Field, n: number, u: number, v: number): number {
    const ix = Math.floor(u);
    const iy = Math.floor(v);
    const fx = u - ix;
    const fy = v - iy;
    const rows: number[] = [];
    for (let j = -1; j <= 2; j++) {
        const yy = wrap(iy + j, n);
        const p0 = field[yy][wrap(ix - 1, n)];
        const p1 = field[yy][wrap(ix, n)];
        const p2 = field[yy][wrap(ix + 1, n)];
        const p3 = field[yy][wrap(ix + 2, n)];
        rows.push(lagrangeCubic1D(p0, p1, p2, p3, fx));
    }
    return lagrangeCubic1D(rows[0], rows[1], rows[2], rows[3], fy);
}

// I3m-r correction round 2, item 6: `lagrangeCubic1D`'s own blending-function weight sum and round-off
// model (`lagrangeWeights`, `MAX_ROW_WEIGHT_SUM_LAGRANGE`, `LAGRANGE_FLOPS`, `LAGRANGE_MODEL`) were
// dead — `lagrange4Sample` is a witness kernel whose seam-identity readings are graded at bicubic's
// OWN bound (`GuardedModel = BICUBIC_MODEL` below, per the Gate law: "a red witness is the guarded
// arm's own assertion re-run with only the subject mutated" — the bound stays fixed, only the kernel
// varies), so it never needs a round-off model of its own. Deleted rather than left dormant.
//
// I3m-r bound re-verdict, item 6: the identity-dispatching `modelFor(kernel)` lookup this file used to
// resolve `GuardedModel` was deleted — its `bilinearSample` branch and fallback `throw` were
// unreachable, since the one call site always passed `bicubicSample` literally (`BILINEAR_MODEL` is
// read directly everywhere else in this file, never through the lookup). `GuardedModel` now reads
// `BICUBIC_MODEL` directly.

/** forward-error bound on one kernel evaluation (Higham-style: `model.flops` floating ops each
 *  contributing relative error <= the rounding unit, over intermediates bounded by
 *  `model.intermediateMagnitudeBound * fieldMagnitude`). */
function evalRoundoff(fieldMagnitude: number, model: KernelRoundoffModel): number {
    return model.flops * ROUNDING_UNIT * model.intermediateMagnitudeBound * fieldMagnitude;
}

function fieldMagnitude(field: Field): number {
    let m = 0;
    for (const row of field) for (const v of row) if (Math.abs(v) > m) m = Math.abs(v);
    return m;
}

// ── closed-form Lagrange fit — no `h`, no limit, no finite difference ────────────────────────────

/** cubic Lagrange basis weight `L_i(x0)` and its EXACT derivative `L_i'(x0)` for 4 nodes `xs`,
 *  evaluated at query point `x0` — a rational function of the nodes (product rule on the numerator),
 *  never a finite difference. */
function lagrangeWeightsAndDerivatives(
    xs: readonly [number, number, number, number],
    x0: number,
): { weights: [number, number, number, number]; derivatives: [number, number, number, number] } {
    const weights: number[] = [];
    const derivatives: number[] = [];
    for (let i = 0; i < 4; i++) {
        let denom = 1;
        for (let j = 0; j < 4; j++) if (j !== i) denom *= xs[i] - xs[j];
        let num = 1;
        for (let j = 0; j < 4; j++) if (j !== i) num *= x0 - xs[j];
        let dNum = 0;
        for (let m = 0; m < 4; m++) {
            if (m === i) continue;
            let term = 1;
            for (let j = 0; j < 4; j++) {
                if (j === i || j === m) continue;
                term *= x0 - xs[j];
            }
            dNum += term;
        }
        weights.push(num / denom);
        derivatives.push(dNum / denom);
    }
    return {
        weights: weights as [number, number, number, number],
        derivatives: derivatives as [number, number, number, number],
    };
}

interface LagrangeFitResult {
    value: number;
    derivative: number;
    valueTol: number;
    derivativeTol: number;
}

/** fit the degree-<=3 polynomial through 4 samples `(xs[i], values[i])` and evaluate its value and
 *  derivative at `x0` in closed form — exact for any degree-<=3 source, no truncation term. The fit's
 *  own round-off floor is derived from the weights' absolute sums (never authored): each sample
 *  carries independent round-off error bounded by `sampleErrorBound` (one kernel evaluation's own
 *  forward-error bound), propagated linearly through the fixed weights. */
function lagrangeFit(
    xs: readonly [number, number, number, number],
    values: readonly [number, number, number, number],
    x0: number,
    sampleErrorBound: number,
): LagrangeFitResult {
    const { weights, derivatives } = lagrangeWeightsAndDerivatives(xs, x0);
    let value = 0;
    let derivative = 0;
    let absWeightSum = 0;
    let absDerivSum = 0;
    for (let i = 0; i < 4; i++) {
        value += weights[i] * values[i];
        derivative += derivatives[i] * values[i];
        absWeightSum += Math.abs(weights[i]);
        absDerivSum += Math.abs(derivatives[i]);
    }
    return {
        value,
        derivative,
        valueTol: absWeightSum * sampleErrorBound,
        derivativeTol: absDerivSum * sampleErrorBound,
    };
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

describe("BICUBIC_MODEL intermediate bound — per-call derivation, before/after", () => {
    test("catmullRom1D's coefficient sums, Horner intermediate multiplier, and W are derived, not literals", () => {
        expect(CATMULLROM_A_ABS_SUM).toBe(4);
        expect(CATMULLROM_B_ABS_SUM).toBe(6);
        expect(CATMULLROM_C_ABS_SUM).toBe(1);
        expect(CATMULLROM_INTERMEDIATE_MULTIPLIER).toBe(12);
        expect(MAX_ROW_WEIGHT_SUM_CATMULLROM).toBe(1.25);
        // check the closed form against a fine grid that includes t=0.5 exactly (an EVEN sample count,
        // unlike the retired 20001-point odd sweep, which computed i/20001 for i in [0, 20001] — 20002
        // points, none of them 0.5, the exact point the maximum is attained) — the grid must never
        // exceed the closed form, confirming t=0.5 really is the maximum and not merely a value.
        const gridMax = maxAbsWeightSum(catmullRomWeights, 20000);
        console.log(
            `catmullRom weight-sum: closed form at t=0.5 = ${MAX_ROW_WEIGHT_SUM_CATMULLROM}, ` +
                `20000-sample grid max (includes t=0.5) = ${gridMax}`,
        );
        expect(gridMax).toBeLessThanOrEqual(MAX_ROW_WEIGHT_SUM_CATMULLROM);
    });

    test("Sigma|w_i(t)| equals the closed form 1 + t*(1-t) pointwise, not just at its t=0.5 maximum", () => {
        // I3m-r bound re-verdict, item 1: the closed form is asserted against the RECOVERED weights
        // (basis-vector evaluation of the shipped kernel) at every sampled t, not only checked as an
        // upper bound on a grid maximum — a mutation that moved the identity anywhere on [0,1] would
        // otherwise be invisible to a max-only check whose own maximum happened to still clear.
        const samples = 2000;
        let maxDeviation = 0;
        for (let i = 0; i <= samples; i++) {
            const t = i / samples;
            const closed = 1 + t * (1 - t);
            const recovered = absSum(catmullRomWeights(t));
            maxDeviation = Math.max(maxDeviation, Math.abs(recovered - closed));
        }
        console.log(
            `Sigma|w_i(t)| vs closed form 1+t(1-t): max pointwise deviation over [0,1] at ${samples} samples = ${maxDeviation.toExponential(3)}`,
        );
        // Derived, not authored: `recovered` is `absSum` (3 adds) over 4 `catmullRom1D` calls (each
        // CATMULLROM_FLOPS=22 flops) on unit-magnitude basis vectors, so its forward-error bound is
        // `evalRoundoff` at flops = 4*CATMULLROM_FLOPS + 3 with the same per-call Horner intermediate
        // multiplier (M=1) every other catmullRom1D reading in this file uses; `closed`'s three
        // exact-`t` ops carry negligible error beside it.
        const identityTolerance = evalRoundoff(1, {
            flops: 4 * CATMULLROM_FLOPS + 3,
            intermediateMagnitudeBound: CATMULLROM_INTERMEDIATE_MULTIPLIER,
        });
        console.log(`  derived identity tolerance = ${identityTolerance.toExponential(3)}`);
        expect(maxDeviation).toBeLessThanOrEqual(identityTolerance);
    });

    test("the correctly-derived per-call bound (row 12x, column 12x*W) replaces `9beeef3b`'s W^2 output-amplification bound; before/after moves no verdict", () => {
        // `9beeef3b` shipped `intermediateMagnitudeBound = W^2` (~1.5625) — an OUTPUT-amplification
        // bound (the row call's OUTPUT is bounded by `W*M`, so squaring it looked plausible for a
        // "compounded" column bound) rather than the Horner chain's own compounded INTERMEDIATE bound
        // this file derives above (12x for row calls, 12*W for column calls, the greater of the two).
        // Recovered here from the SAME `MAX_ROW_WEIGHT_SUM_CATMULLROM` this file derives everywhere
        // else — never a re-typed `1.5625` literal — so a mutation moving the shipped kernel's own W
        // moves this "before" reading too, exactly as it moves the "after" one. `12` (the uniform bound
        // a9223fce shipped) is not this comparison's "before": it was retired one round earlier, when
        // `9beeef3b` replaced it with W^2, and `9beeef3b`'s W^2 is what this file's own 12*W replaced.
        const beforeBound = MAX_ROW_WEIGHT_SUM_CATMULLROM ** 2; // `9beeef3b`'s shipped bound, ~1.5625
        const afterBound = BICUBIC_MODEL.intermediateMagnitudeBound; // 12*W ~= 15, this file's own bound today
        console.log(
            `BICUBIC_MODEL intermediate bound (both a multiplier over the field magnitude M): ` +
                `9beeef3b (W^2)=${beforeBound.toFixed(4)}xM today (row=${BICUBIC_ROW_INTERMEDIATE_BOUND}, column=12*W)=${afterBound.toFixed(4)}xM`,
        );
        // `9beeef3b`'s bound is TIGHTER (smaller), not looser: an output-amplification bound undercounts
        // the Horner chain's own compounded intermediate bound, which is exactly why the reviewer
        // convicted it. Every measured reading below still clears the tighter bound too: the derivation
        // moves no verdict, only the margin.
        expect(beforeBound).toBeLessThan(afterBound);

        const beforeModel: KernelRoundoffModel = {
            flops: BICUBIC_MODEL.flops,
            intermediateMagnitudeBound: beforeBound,
        };
        const M = fieldMagnitude(quadraticField);
        const beforeTol = evalRoundoff(M, beforeModel);
        const afterTol = evalRoundoff(M, BICUBIC_MODEL);
        for (const [u, v] of SAFE_UV) {
            const diff = Math.abs(bicubicSample(quadraticField, N, u, v) - biQuadratic(u, v));
            console.log(
                `  quadratic (${u},${v}) |diff|=${diff.toExponential(3)} beforeTol(9beeef3b)=${beforeTol.toExponential(3)} afterTol(today)=${afterTol.toExponential(3)}`,
            );
            expect(diff).toBeLessThanOrEqual(beforeTol);
            expect(diff).toBeLessThanOrEqual(afterTol);
        }
    });
});

// I3m-r correction round (2026-09-01): neither recorded mutation above reached this claim — at
// `t=0` (an exact integer sample point), `catmullRom1D`'s return collapses to `p1` regardless of the
// `a`/`b`/`c` coefficients (each is multiplied by `t`), so no coefficient mutation can ever reach it,
// and both sample-point taps stay in-range so `wrap`'s narrowing is inert here too. This file's
// mutation table records the mutation this claim IS sensitive to: which texel `p1` reads (the
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
// reaches. This file's mutation table records the mutation this claim IS sensitive to: `wrap`'s
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

// ── continuity: the seam identity a piecewise cubic licenses ────────────────────────────────────
//
// For fixed `v`, `bicubicSample(field, N, u, v)` is a degree-<=3 polynomial in `u` on each cell
// `[k, k+1)`: the row taps `p0..p3` are constant within the cell (they depend only on `floor(u)`), and
// `catmullRom1D` is cubic in its own `t = u - floor(u)` argument. Four samples strictly INSIDE a cell
// therefore determine that cell's cubic exactly via a closed-form Lagrange fit (`lagrangeFit` above),
// and extrapolating that fit to the cell's boundary (`x0 = k`, the seam) recovers the cell's exact seam
// value and seam derivative — no `h`, no limit, no truncation term. Symmetric for `v`-seams (fixed `u`,
// sweeping `v`).
//
// C0 asserts the two sides' extrapolated seam values agree; C1 asserts the two sides' extrapolated
// seam derivatives agree — each within a bound derived purely from the Lagrange weights' own absolute
// sums (`lagrangeFit`'s `valueTol`/`derivativeTol`) and bicubic's own per-call round-off bound
// (`GuardedModel` = `BICUBIC_MODEL`), added across both sides (triangle inequality). No ratio, no `h`,
// and no reading of the subject kernel appears in any denominator anywhere below.
//
// The fit's PREMISE — that four in-cell samples really do pin down the WHOLE cell's cubic, not just
// those four points — is its own leg: the same fit (same nodes, same values) is re-evaluated at a
// FIFTH in-cell point, distinct from the four fitted ones, and must match the kernel's own reading
// there within the same `valueTol`. The STRADDLING-STENCIL control (two samples from each side of the
// seam) proves that leg is discriminating rather than vacuous: since the two cells carry different
// cubics beyond C1, a fit straddling the seam does not correspond to either cell's true cubic, and must
// MISS its own fifth in-cell prediction by MORE than that fit's own `valueTol`.
//
// Every seam `k = 0..NSeam-1` on both axes is swept (the periodic seam at `k=0` included — its
// left-side samples land at negative `u`/`v`, exercising `wrap`'s negative branch), each at several
// off-grid perpendicular offsets, on `syntheticField` (generic, non-polynomial data: the bi-quadratic
// fixture above is unusable here, since bicubic reproduces it exactly everywhere, degenerating any
// "jump" to pure round-off noise carrying no continuity signal).
describe("continuity — the seam identity a piecewise cubic licenses", () => {
    const SeamN = 128;
    const seamField = syntheticField(SeamN, 1);
    const seamFieldMagnitude = fieldMagnitude(seamField);
    const GuardedModel = BICUBIC_MODEL; // every seam-identity reading below is graded here
    const seamSampleTol = evalRoundoff(seamFieldMagnitude, GuardedModel);

    const FitOffsets: [number, number, number, number] = [0.9, 0.7, 0.5, 0.3]; // distances from the seam, all strictly inside one cell
    const PremiseOffset = 0.1; // a fifth in-cell point, distinct from every fit node above
    const StraddleOffsets: [number, number, number, number] = [-0.3, -0.1, 0.1, 0.3]; // two nodes each side of the seam
    const StraddleTargetOffset = 0.5; // an in-cell point on the right side, distinct from every straddle node
    const PerpOffsets = [3.17, 47.44, 90.71]; // several off-grid perpendicular offsets, swept per seam
    const Axes = ["u", "v"] as const;
    type Axis = (typeof Axes)[number];

    function sampleAt(
        kernel: ReconstructionKernel,
        axis: Axis,
        along: number,
        perp: number,
    ): number {
        return axis === "u"
            ? kernel(seamField, SeamN, along, perp)
            : kernel(seamField, SeamN, perp, along);
    }

    interface SeamMeasurement {
        axis: Axis;
        seam: number;
        perp: number;
        c0Diff: number;
        c0Tol: number;
        c1Diff: number;
        c1Tol: number;
        leftPremiseDiff: number;
        leftPremiseTol: number;
        rightPremiseDiff: number;
        rightPremiseTol: number;
        straddleDiff: number;
        straddleTol: number;
    }

    function measureSeam(
        kernel: ReconstructionKernel,
        axis: Axis,
        seam: number,
        perp: number,
    ): SeamMeasurement {
        const leftXs = FitOffsets.map((d) => seam - d) as [number, number, number, number];
        const leftValues = leftXs.map((x) => sampleAt(kernel, axis, x, perp)) as [
            number,
            number,
            number,
            number,
        ];
        const rightXs = FitOffsets.map((d) => seam + d) as [number, number, number, number];
        const rightValues = rightXs.map((x) => sampleAt(kernel, axis, x, perp)) as [
            number,
            number,
            number,
            number,
        ];

        const leftAtSeam = lagrangeFit(leftXs, leftValues, seam, seamSampleTol);
        const rightAtSeam = lagrangeFit(rightXs, rightValues, seam, seamSampleTol);

        const leftPremiseX = seam - PremiseOffset;
        const rightPremiseX = seam + PremiseOffset;
        const leftPremiseFit = lagrangeFit(leftXs, leftValues, leftPremiseX, seamSampleTol);
        const rightPremiseFit = lagrangeFit(rightXs, rightValues, rightPremiseX, seamSampleTol);
        const leftPremiseActual = sampleAt(kernel, axis, leftPremiseX, perp);
        const rightPremiseActual = sampleAt(kernel, axis, rightPremiseX, perp);

        const straddleXs = StraddleOffsets.map((d) => seam + d) as [number, number, number, number];
        const straddleValues = straddleXs.map((x) => sampleAt(kernel, axis, x, perp)) as [
            number,
            number,
            number,
            number,
        ];
        const straddleTargetX = seam + StraddleTargetOffset;
        const straddleFit = lagrangeFit(straddleXs, straddleValues, straddleTargetX, seamSampleTol);
        const straddleActual = sampleAt(kernel, axis, straddleTargetX, perp);

        return {
            axis,
            seam,
            perp,
            c0Diff: Math.abs(leftAtSeam.value - rightAtSeam.value),
            c0Tol: leftAtSeam.valueTol + rightAtSeam.valueTol,
            c1Diff: Math.abs(leftAtSeam.derivative - rightAtSeam.derivative),
            c1Tol: leftAtSeam.derivativeTol + rightAtSeam.derivativeTol,
            leftPremiseDiff: Math.abs(leftPremiseFit.value - leftPremiseActual),
            leftPremiseTol: leftPremiseFit.valueTol,
            rightPremiseDiff: Math.abs(rightPremiseFit.value - rightPremiseActual),
            rightPremiseTol: rightPremiseFit.valueTol,
            straddleDiff: Math.abs(straddleFit.value - straddleActual),
            straddleTol: straddleFit.valueTol,
        };
    }

    /** every seam on both axes, at every declared perpendicular offset — the population every describe
     *  block below reads from, so a red witness's "only the kernel mutated" claim is literal: same
     *  seams, same offsets, same fixture, same bound. */
    function sweepSeams(kernel: ReconstructionKernel): SeamMeasurement[] {
        const out: SeamMeasurement[] = [];
        for (const axis of Axes) {
            for (let seam = 0; seam < SeamN; seam++) {
                for (const perp of PerpOffsets) out.push(measureSeam(kernel, axis, seam, perp));
            }
        }
        return out;
    }

    /** the measurement with the SMALLEST margin — for a leg required to CLEAR its bound (`diff <=
     *  tol`, `mode: "mustClear"`), that is the smallest `tol - diff`, the reading nearest to breaking
     *  the bound. For a leg required to EXCEED its bound (`diff > tol`, `mode: "mustExceed"`, the
     *  straddle control alone), minimizing `tol - diff` instead finds the reading with the LARGEST
     *  margin past the bound — the opposite of "nearest to vacuity" the control needs, so that mode
     *  minimizes `diff - tol` instead, the reading closest to failing to exceed its own bound. */
    function worstSlack(
        results: readonly SeamMeasurement[],
        diffKey: "c0Diff" | "c1Diff" | "leftPremiseDiff" | "rightPremiseDiff" | "straddleDiff",
        tolKey: "c0Tol" | "c1Tol" | "leftPremiseTol" | "rightPremiseTol" | "straddleTol",
        mode: "mustClear" | "mustExceed" = "mustClear",
    ): SeamMeasurement {
        let worst = results[0];
        let worstMargin = Number.POSITIVE_INFINITY;
        for (const r of results) {
            const margin = mode === "mustClear" ? r[tolKey] - r[diffKey] : r[diffKey] - r[tolKey];
            if (margin < worstMargin) {
                worstMargin = margin;
                worst = r;
            }
        }
        return worst;
    }

    // Each test below computes its own sweep rather than sharing one eagerly-computed array: a
    // mutation reached only at one seam (e.g. `wrap`'s negative-index branch, reached at k=0) can throw
    // rather than merely misread, and a throw during a SHARED describe-scope computation would abort
    // registration of every sibling test in this describe at once, hiding their own pass/fail status
    // (measured while red-witnessing the `wrap` mutation below). Computing per-test costs a few extra
    // cheap sweeps (each ~11k trivial kernel calls) and keeps every leg's own reach independently
    // reportable.

    test("C0: bicubic's two one-sided in-cell fits agree at the seam within round-off", () => {
        const results = sweepSeams(bicubicSample);
        const worst = worstSlack(results, "c0Diff", "c0Tol");
        console.log(
            `C0 worst case, closest to BREAKING its clear-bound (axis=${worst.axis} seam=${worst.seam} perp=${worst.perp}): ` +
                `diff=${worst.c0Diff.toExponential(3)} tol=${worst.c0Tol.toExponential(3)}`,
        );
        for (const r of results) expect(r.c0Diff).toBeLessThanOrEqual(r.c0Tol);
    });

    test("C1: bicubic's two one-sided in-cell fits agree on the seam derivative within round-off", () => {
        const results = sweepSeams(bicubicSample);
        const worst = worstSlack(results, "c1Diff", "c1Tol");
        console.log(
            `C1 worst case, closest to BREAKING its clear-bound (axis=${worst.axis} seam=${worst.seam} perp=${worst.perp}): ` +
                `diff=${worst.c1Diff.toExponential(3)} tol=${worst.c1Tol.toExponential(3)}`,
        );
        for (const r of results) expect(r.c1Diff).toBeLessThanOrEqual(r.c1Tol);
    });

    test("premise: each side's fit predicts an unseen fifth in-cell sample within its own bound", () => {
        const results = sweepSeams(bicubicSample);
        const worstLeft = worstSlack(results, "leftPremiseDiff", "leftPremiseTol");
        const worstRight = worstSlack(results, "rightPremiseDiff", "rightPremiseTol");
        console.log(
            `premise-left worst case, closest to BREAKING its clear-bound (axis=${worstLeft.axis} seam=${worstLeft.seam} perp=${worstLeft.perp}): ` +
                `diff=${worstLeft.leftPremiseDiff.toExponential(3)} tol=${worstLeft.leftPremiseTol.toExponential(3)}`,
        );
        console.log(
            `premise-right worst case, closest to BREAKING its clear-bound (axis=${worstRight.axis} seam=${worstRight.seam} perp=${worstRight.perp}): ` +
                `diff=${worstRight.rightPremiseDiff.toExponential(3)} tol=${worstRight.rightPremiseTol.toExponential(3)}`,
        );
        for (const r of results) {
            expect(r.leftPremiseDiff).toBeLessThanOrEqual(r.leftPremiseTol);
            expect(r.rightPremiseDiff).toBeLessThanOrEqual(r.rightPremiseTol);
        }
    });

    test("CONTROL — a seam-straddling stencil misses its own fifth in-cell prediction by more than its bound", () => {
        const results = sweepSeams(bicubicSample);
        // this leg must EXCEED its bound, so its worst case is the reading NEAREST VACUITY — the
        // smallest margin by which it still exceeds, not (as C0/C1/premise above) the smallest margin
        // by which it still clears.
        const worst = worstSlack(results, "straddleDiff", "straddleTol", "mustExceed");
        console.log(
            `straddle-control worst case, nearest VACUITY — smallest margin by which it still exceeds its must-exceed bound (axis=${worst.axis} seam=${worst.seam} perp=${worst.perp}): ` +
                `diff=${worst.straddleDiff.toExponential(3)} tol=${worst.straddleTol.toExponential(3)}`,
        );
        for (const r of results) expect(r.straddleDiff).toBeGreaterThan(r.straddleTol);
    });

    test("seam-identity bound before/after — EVERY seam-leg reading of the sweep clears both `9beeef3b`'s and today's bound", () => {
        // `9beeef3b` shipped `intermediateMagnitudeBound = W^2` (~1.5625, the same output-amplification
        // bound as the bi-quadratic-only describe above) for the WHOLE seam-identity reading below,
        // replaced by this file's own row/column-derived `12*W` (~15). Recovered from the same
        // `MAX_ROW_WEIGHT_SUM_CATMULLROM` this file derives everywhere else, never a re-typed `1.5625`
        // literal — `12` (a9223fce's uniform guess) is not the "before" here either: it was retired one
        // round before `9beeef3b` shipped, so it is not what this file's own bound replaced.
        const beforeBound = MAX_ROW_WEIGHT_SUM_CATMULLROM ** 2; // `9beeef3b`'s shipped bound, ~1.5625
        const afterBound = GuardedModel.intermediateMagnitudeBound; // 12*W ~= 15, this file's own bound today
        const beforeModel: KernelRoundoffModel = {
            flops: GuardedModel.flops,
            intermediateMagnitudeBound: beforeBound,
        };
        const beforeSeamTol = evalRoundoff(seamFieldMagnitude, beforeModel);
        console.log(
            `seam-identity intermediate bound (both a multiplier over the field magnitude M): ` +
                `9beeef3b (W^2)=${beforeBound.toFixed(4)}xM today (12*W)=${afterBound.toFixed(4)}xM`,
        );
        console.log(
            `seam-identity tolerance: 9beeef3b=${beforeSeamTol.toExponential(3)} today=${seamSampleTol.toExponential(3)}`,
        );

        // `9beeef3b`'s bound is TIGHTER (a SMALLER multiplier), not looser, than today's — the reviewer
        // convicted W^2 for UNDER-counting the Horner chain's own compounded intermediate bound, not for
        // over-counting it. A tighter (smaller) `mustClear` tolerance is the HARDER side to clear for
        // C0/C1/premise, and a tighter `mustExceed` bound is the EASIER side to exceed for the straddle
        // control — both directions are checked against BOTH bounds below rather than assumed.
        const scaleToBefore = beforeSeamTol / seamSampleTol;
        expect(scaleToBefore).toBeLessThan(1);

        // Every seam-leg reading of the sweep, not one argmin: a single worst-case selection can hide a
        // reading that clears the tighter bound only by coincidence at its own worst point while failing
        // elsewhere, so every one of the sweep's ~768 readings is checked against both bounds directly.
        const results = sweepSeams(bicubicSample);
        for (const r of results) {
            expect(r.c0Diff).toBeLessThanOrEqual(r.c0Tol);
            expect(r.c0Diff).toBeLessThanOrEqual(r.c0Tol * scaleToBefore);
            expect(r.c1Diff).toBeLessThanOrEqual(r.c1Tol);
            expect(r.c1Diff).toBeLessThanOrEqual(r.c1Tol * scaleToBefore);
            expect(r.leftPremiseDiff).toBeLessThanOrEqual(r.leftPremiseTol);
            expect(r.leftPremiseDiff).toBeLessThanOrEqual(r.leftPremiseTol * scaleToBefore);
            expect(r.rightPremiseDiff).toBeLessThanOrEqual(r.rightPremiseTol);
            expect(r.rightPremiseDiff).toBeLessThanOrEqual(r.rightPremiseTol * scaleToBefore);
            // straddle control: a must-EXCEED leg, so the tighter (smaller) before-bound is the EASIER
            // side to clear. Since `scaleToBefore < 1` is already asserted above, clearing today's
            // tighter `mustExceed` bound implies clearing the looser `9beeef3b`-scaled one too, so only
            // the single leg is asserted here.
            expect(r.straddleDiff).toBeGreaterThan(r.straddleTol);
        }

        const worst = worstSlack(results, "c1Diff", "c1Tol");
        console.log(
            `  representative (C1 worst-margin) reading (axis=${worst.axis} seam=${worst.seam} perp=${worst.perp}): ` +
                `diff=${worst.c1Diff.toExponential(3)} tolToday=${worst.c1Tol.toExponential(3)} tol9beeef3b=${(worst.c1Tol * scaleToBefore).toExponential(3)}`,
        );
    });

    describe("RED-WITNESS — bilinear reds C1, passes C0 (guarded bound, only the kernel mutated)", () => {
        // reaches: C0 (passes — bilinear is piecewise linear in each cell for fixed perp, and its two
        // adjacent cells already share the exact corner value the seam sits on, so both one-sided fits
        // recover that same value exactly), C1 (reds — bilinear's per-cell slope depends on that cell's
        // own two corner values and is generically discontinuous across the seam).
        test("C0 still agrees within the guarded bound", () => {
            const results = sweepSeams(bilinearSample);
            for (const r of results) expect(r.c0Diff).toBeLessThanOrEqual(r.c0Tol);
        });

        test("C1 breaks the guarded bound", () => {
            const results = sweepSeams(bilinearSample);
            for (const r of results) expect(r.c1Diff).toBeGreaterThan(r.c1Tol);
        });
    });

    describe("RED-WITNESS — nearest reds C0 and the premise leg (guarded bound, only the kernel mutated)", () => {
        // reaches: C0 (reds — `nearestSample` rounds to the nearest INTEGER texel, so among the four
        // fit offsets [0.9, 0.7, 0.5, 0.3] the rounding split is ASYMMETRIC across the seam, not the
        // even 2/2 it might look like: on the LEFT side (samples at `seam - offset`, e.g. seam=5 gives
        // 4.1/4.3/4.5/4.7), `Math.round` breaks the `.5` tie upward, splitting 2 round to `seam-1` and
        // 2 to `seam`; on the RIGHT side (samples at `seam + offset`, e.g. 5.3/5.5/5.7/5.9), the SAME
        // upward tie-break pulls the `.5` case toward `seam+1` too, splitting 3 round to `seam+1` and
        // only 1 (the `0.3` offset) to `seam` — either way the "in-cell fit" reads a step function
        // rather than a cubic and its seam extrapolation is meaningless), premise (reds — the same
        // step-function data cannot predict its own unseen sample either).
        test("C0 breaks the guarded bound", () => {
            const results = sweepSeams(nearestSample);
            for (const r of results) expect(r.c0Diff).toBeGreaterThan(r.c0Tol);
        });

        test("the premise leg breaks too", () => {
            const results = sweepSeams(nearestSample);
            for (const r of results) {
                expect(r.leftPremiseDiff).toBeGreaterThan(r.leftPremiseTol);
                expect(r.rightPremiseDiff).toBeGreaterThan(r.rightPremiseTol);
            }
        });
    });

    describe("RED-WITNESS — a test-side four-point Lagrange kernel: C0, the premise leg and the straddle control stay green, C1 alone reds (guarded bound, only the kernel mutated)", () => {
        // exact on cubics (see this file's header), so it is run on exactly the four seam-identity legs
        // this describe's own titled tests below name — C0, premise, and the straddle control, all of
        // which stay green, plus C1, which reds — to make that pairing an assertion rather than a claim
        // resting on the two arms it happened to be run against before: C0 (passes — both cells' Lagrange cubics already agree at the shared
        // grid point p1/p2), premise (passes — a fit through 4 points of a cubic reproduces every other
        // point of that SAME cubic exactly), straddle control (passes, i.e. still MISSES its fifth
        // in-cell prediction by more than the bound — a straddling fit spans two different per-cell
        // Lagrange cubics same as it spans two different per-cell Catmull-Rom cubics, so the same
        // cross-seam mismatch this file's straddle control proves for bicubic holds generically for any
        // cubic-exact 4-tap scheme), C1 (reds — no cross-cell tangent construction ties the two cells'
        // independently-fitted cubics' derivatives together).
        test("C0 stays green", () => {
            const results = sweepSeams(lagrange4Sample);
            for (const r of results) expect(r.c0Diff).toBeLessThanOrEqual(r.c0Tol);
        });

        test("the premise leg stays green", () => {
            const results = sweepSeams(lagrange4Sample);
            for (const r of results) {
                expect(r.leftPremiseDiff).toBeLessThanOrEqual(r.leftPremiseTol);
                expect(r.rightPremiseDiff).toBeLessThanOrEqual(r.rightPremiseTol);
            }
        });

        test("the straddle control stays green (still misses its fifth in-cell prediction by more than its bound)", () => {
            const results = sweepSeams(lagrange4Sample);
            for (const r of results) expect(r.straddleDiff).toBeGreaterThan(r.straddleTol);
        });

        test("C1 breaks the guarded bound", () => {
            const results = sweepSeams(lagrange4Sample);
            for (const r of results) expect(r.c1Diff).toBeGreaterThan(r.c1Tol);
        });
    });
});
