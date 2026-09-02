// By-path I2r-b choppiness oracle. Solves λ (`spectrum.ts`'s `LAMBDA`) by bisection against a
// sourced whitecap anchor over a numerically-realized (seeded, IFFT'd) seed×phase ensemble — never a
// closed form. The prior round's closed-form fold arm (`origin/shallot-water-surface/I2r`) derived λ
// algebraically from the SAME published spectrum the anchor comparison was meant to check against,
// so it was self-certifying by construction; this oracle's whole point is that the composed fold
// fraction is a number nothing here derives in closed form, only measures.
//
// COMPOSED WORLD GRID: `composed-fold.ts` superposes both displacement cascades' unit-λ gradients
// at each point of one shared world grid (spacing resolves cascade 1's own texel, extent spans
// cascade 0's own period — that module's own header states and derives both), then reads
// det(I + λ(G0+G1)) < 0 as a fraction of world AREA. This replaces the prior round's per-cascade
// fold pooled by raw texel count (cascade 1, 13% of the represented area, carried 80% of that
// pooling's weight — an authored free scalar the solve was monotone in). λ is one physical scalar
// shared by both cascades (Tessendorf's own convention — one choppiness for the whole sea);
// `spectrum.ts` already ships both `CASCADE_CONFIGS` entries with the same `LAMBDA`.
//
// PHASE: a normalization (or a solve) fitted at a single instant is a fit, not a normalization —
// counter-rotating ±k terms carry a coherent cross term at t=0 that later phases do not, and the
// declared band's small effective mode count makes that error order-one (spec Residue). Every fold
// reading below is a mean over SEED and PHASE: `PHASE_PERIODS` mirrors `realization.oracle.ts`'s own
// shape (nine consecutive integer multiples of the dominant period, spanning periods 0..8 — the
// same ≥8-period coverage that file's `phaseAveragedGridVariance` uses), computed here independently
// off this file's own `dominantPeriod` rather than imported, since the two files' formulas coincide
// by definition (same sea state) and re-deriving is cheaper than a cross-file dependency for one
// number. The t=0-only reading is kept and printed as the RED CONTROL this invariant names — the
// single-point formulation the phase mean corrects for — never as a gate.
//
// COST: this file is `.oracle.ts` (exempt from the default per-file test-duration cap,
// `packages/shallot/tests/test-cap.ts`), but `fold-anchor-oracle-reach.test.ts` spawns the WHOLE of
// it synchronously inside a capped `.test.ts` file to prove `bun run test` actually executes it — so
// this file's own wall-clock is bounded by that cap (5000ms) minus the reach sentinel's own overhead,
// not by the exemption. The composed-world-grid build (330×330 points, both cascades) is the cost:
// ~20ms/realization for the FFT pipeline (`updateH`/`chop`/`spectralGradient`/`idft2`, all four
// production primitives, unmodified) plus ~6ms for the nearest-neighbor composition, measured. Seed
// windows are 4 seeds each (not 40): the measured per-seed-mean stderr at n=40 was already two orders
// of magnitude below the spec's own ±30%/10% tolerances (~0.1-0.3% against 10-30%), so n=4 spends the
// same margin at 1/10 the realization count — the seed×phase product (4 seeds × 9 phases × 3 windows =
// 108 realizations, ~2.5s) is what a by-path file spawned from inside a capped file can afford; the
// full-fidelity 40-seed ensemble stays the shape a future by-path-only widening could restore without
// touching this file's structure.
//
// SEED WINDOWS: SOLVE (seeds 0..3) is declared before any reading. HELD_A (1000..1003) and HELD_B
// (2000..2003) are disjoint from SOLVE and from each other. Every window uses a per-cascade seed
// offset (cascade 1 draws `seed + CASCADE1_OFFSET`) so one window index doesn't correlate the two
// cascades' realizations.
//
// TOLERANCES: the spec's own ±30% (held-out fold vs anchor) and 10% (held-out-solved λ vs the
// declared-set λ) are printed beside this run's own measured standard error of the pooled-fold
// ensemble mean (empirical stderr across seeds, and separately across phases — both printed, neither
// substituted as a tolerance anywhere: Gate law, "every finite-sample tolerance is derived from and
// printed beside the estimator's predicted sampling error", and the guarded arm's tolerance stays the
// spec-named constant, never the estimator's own noise).
import { describe, expect, test } from "bun:test";
import {
    type CascadeGradientField,
    composeWorldGrid,
    foldFractionAt,
    realPart,
    worldGridSpec,
} from "../src/composed-fold";
import { chop, idft2, spectralGradient, updateH } from "../src/cpu-reference";
import {
    CASCADE_CONFIGS,
    type CascadeConfig,
    FOLD_REGIME,
    G,
    generateH0,
    LAMBDA,
    SEA_STATE,
    whitecapFraction,
} from "../src/spectrum";

const [CFG0, CFG1] = CASCADE_CONFIGS;
const U10 = CFG0.windSpeed;
const ANCHOR = whitecapFraction(U10);
const GRID = worldGridSpec(CASCADE_CONFIGS);

const SOLVE_SEEDS = [0, 1, 2, 3];
const HELD_A_SEEDS = [1000, 1001, 1002, 1003];
const HELD_B_SEEDS = [2000, 2001, 2002, 2003];
const CASCADE1_OFFSET = 500_000; // decorrelates cascade 0 / cascade 1 draws within one window index

// dominant period — same closed form as `realization.oracle.ts`'s own (re-derived, not imported;
// see this file's header). Nine phases, periods 0..8: >= 8 dominant periods of coverage.
const dominantK = (G * SEA_STATE.omegaC ** 2) / SEA_STATE.windSpeed ** 2;
const dominantPeriod = (2 * Math.PI) / Math.sqrt(G * dominantK);
const PHASE_PERIODS = Array.from({ length: 9 }, (_, period) => period);
const PHASES = PHASE_PERIODS.map((period) => period * dominantPeriod);

const ANCHOR_BAND_REL = 0.3; // spec Validation: held-out fold must stay within ±30% of the anchor
const LAMBDA_AGREEMENT_REL = 0.1; // spec Validation: held-out-solved λ must agree within 10%
const LAMBDA_MUTATION_REL = 0.2; // spec Validation: the red-witness sweep is ±20% on λ
const BISECT_LO = 0.1;
const BISECT_HI = 30;
const BISECT_ITERS = 24;

function mean(values: number[]): number {
    return values.reduce((a, b) => a + b, 0) / values.length;
}

function sampleStderr(values: number[]): number {
    const m = mean(values);
    const variance =
        values.reduce((sum, v) => sum + (v - m) ** 2, 0) / Math.max(values.length - 1, 1);
    return Math.sqrt(variance / values.length);
}

/** Unit-λ real-space gradient field for one cascade realization at one time, via the exact
 *  production primitives `cpu-reference.ts` transcribes from `ocean.ts`'s WGSL. */
function cascadeGradientField(
    h0: Float32Array,
    config: CascadeConfig,
    time: number,
): CascadeGradientField {
    const { N, L } = config;
    const h = updateH(h0, N, L, time);
    const { dxSpec, dzSpec } = chop(h, N, L);
    const { gxxSpec, gxzSpec, gzzSpec } = spectralGradient(dxSpec, dzSpec, N, L);
    return {
        N,
        L,
        gxx: realPart(idft2(gxxSpec, N), N),
        gxz: realPart(idft2(gxzSpec, N), N),
        gzz: realPart(idft2(gzzSpec, N), N),
    };
}

interface SeedRealization {
    seed: number;
    /** one composed (unit-λ) world-grid field per `PHASES` entry, index-aligned. */
    phases: ReturnType<typeof composeWorldGrid>[];
}

function buildSeedRealization(seed: number): SeedRealization {
    const h0Cfg0 = generateH0(CFG0, seed);
    const h0Cfg1 = generateH0(CFG1, seed + CASCADE1_OFFSET);
    const phases = PHASES.map((time) =>
        composeWorldGrid(
            [cascadeGradientField(h0Cfg0, CFG0, time), cascadeGradientField(h0Cfg1, CFG1, time)],
            GRID,
        ),
    );
    return { seed, phases };
}

interface SeedWindow {
    label: string;
    realizations: SeedRealization[];
}

function buildWindow(label: string, seeds: number[]): SeedWindow {
    return { label, realizations: seeds.map(buildSeedRealization) };
}

interface EnsembleReading {
    mean: number;
    /** empirical stderr ACROSS SEEDS of the per-seed, phase-averaged fold reading. */
    seedStderr: number;
    /** empirical stderr ACROSS PHASES of the per-phase, seed-averaged fold reading — the spread the
     *  phase mean itself is estimated from (Residue: "the invariant is a mean over phase and seed
     *  with both spreads reported"). */
    phaseStderr: number;
    /** the spec-named 95% Bernoulli interval on the pooled proportion, printed for comparison only. */
    bernoulli: number;
}

const Z95 = 1.96;

function bernoulliInterval(p: number, n: number): number {
    const clampedP = Math.max(0, Math.min(1, p));
    return Z95 * Math.sqrt((clampedP * (1 - clampedP)) / n);
}

/** Composed world-grid fold fraction at `lambda` for one seed's realization, averaged over every
 *  declared phase — a fold reading is never taken at a single instant (this file's header). */
function seedPhaseMeanFold(realization: SeedRealization, lambda: number): number[] {
    return realization.phases.map((field) => foldFractionAt(field, lambda));
}

function ensembleFold(lambda: number, window: SeedWindow): EnsembleReading {
    const perSeedPhase = window.realizations.map((r) => seedPhaseMeanFold(r, lambda));
    const seedMeans = perSeedPhase.map((phases) => mean(phases));
    const numPhases = perSeedPhase[0].length;
    const phaseMeans = PHASE_PERIODS.map((_, p) => mean(perSeedPhase.map((phases) => phases[p])));

    const totalArea = window.realizations.length * numPhases * GRID.gridN * GRID.gridN;
    return {
        mean: mean(seedMeans),
        seedStderr: sampleStderr(seedMeans),
        phaseStderr: sampleStderr(phaseMeans),
        bernoulli: bernoulliInterval(mean(seedMeans), totalArea),
    };
}

interface BisectResult {
    lambda: number;
    loFinal: number;
    hiFinal: number;
    loFold: number;
    hiFold: number;
}

/** Bisects λ so `ensembleFold(λ)` lands on `ANCHOR` over one window's cached composed fields. The
 *  fold fraction is monotone increasing in λ over the searched range (measured; every reading in
 *  this file's derivation run confirms it), so ordinary bisection applies. Returns the final bracket
 *  AND its own fold readings — this file's convergence arm asserts both a structural property (the
 *  bracket halved geometrically `BISECT_ITERS` times) and a semantic one (ANCHOR lies between the
 *  bracket endpoints' own measured fold values), never a tolerance re-run against itself. */
function bisectLambda(
    window: SeedWindow,
    lo = BISECT_LO,
    hi = BISECT_HI,
    iters = BISECT_ITERS,
): BisectResult {
    for (let i = 0; i < iters; i++) {
        const mid = (lo + hi) / 2;
        const { mean: m } = ensembleFold(mid, window);
        if (m < ANCHOR) lo = mid;
        else hi = mid;
    }
    return {
        lambda: (lo + hi) / 2,
        loFinal: lo,
        hiFinal: hi,
        loFold: ensembleFold(lo, window).mean,
        hiFold: ensembleFold(hi, window).mean,
    };
}

function withinAnchorBand(pooled: number): boolean {
    return Math.abs(pooled - ANCHOR) / ANCHOR <= ANCHOR_BAND_REL;
}

const solveWindow = buildWindow("solve[0..3]", SOLVE_SEEDS);
const heldA = buildWindow("heldA[1000..1003]", HELD_A_SEEDS);
const heldB = buildWindow("heldB[2000..2003]", HELD_B_SEEDS);

describe("whitecap anchor (Monahan & O'Muircheartaigh 1980)", () => {
    test(`W(U10=${U10}) = min(0.5, 3.84e-6 · U10^3.41)`, () => {
        console.log(`whitecapFraction(${U10}) = ${(ANCHOR * 100).toFixed(4)}%`);
        expect(ANCHOR).toBeGreaterThan(0);
        expect(ANCHOR).toBeLessThanOrEqual(0.5);
    });
});

describe("composed-world-grid fold ensemble solves λ against the anchor (declared solve set)", () => {
    test(`declared world grid: spacing=${GRID.spacing.toFixed(6)}m resolves cascade 1's own texel (${(CFG1.L / CFG1.N).toFixed(6)}m); extent=${GRID.extent.toFixed(3)}m spans cascade 0's own period (${CFG0.L}m)`, () => {
        console.log(
            `world grid: ${GRID.gridN}x${GRID.gridN} points, spacing=${GRID.spacing.toFixed(6)}m, extent=${GRID.extent.toFixed(3)}m`,
        );
        expect(GRID.spacing).toBeLessThanOrEqual(CFG1.L / CFG1.N);
        expect(GRID.extent).toBeGreaterThanOrEqual(CFG0.L - GRID.spacing);
    });

    const solveBisect = bisectLambda(solveWindow);
    const lambdaSolve = solveBisect.lambda;
    const solveReading = ensembleFold(lambdaSolve, solveWindow);

    test("bisection converges: the final bracket halved geometrically over the declared iteration count, and ANCHOR lies inside the bracket's own fold readings", () => {
        const expectedWidth = (BISECT_HI - BISECT_LO) / 2 ** BISECT_ITERS;
        const actualWidth = solveBisect.hiFinal - solveBisect.loFinal;
        const relWidthError = Math.abs(actualWidth - expectedWidth) / expectedWidth;
        console.log(
            `bisection bracket: lo=${solveBisect.loFinal.toFixed(9)} (fold=${(solveBisect.loFold * 100).toFixed(4)}%) ` +
                `hi=${solveBisect.hiFinal.toFixed(9)} (fold=${(solveBisect.hiFold * 100).toFixed(4)}%) ` +
                `width=${actualWidth.toExponential(4)} expectedWidth=${expectedWidth.toExponential(4)} ` +
                `ANCHOR=${(ANCHOR * 100).toFixed(4)}%`,
        );
        expect(relWidthError).toBeLessThan(1e-9);
        expect(solveBisect.loFold).toBeLessThanOrEqual(ANCHOR);
        expect(solveBisect.hiFold).toBeGreaterThanOrEqual(ANCHOR);
    });

    test("RED CONTROL — the single-instant (t=0 only) reading, printed beside the phase-averaged one, never gated (Residue: a normalization fitted at one instant is a fit, not a normalization)", () => {
        const t0OnlyMean = mean(
            solveWindow.realizations.map((r) => foldFractionAt(r.phases[0], lambdaSolve)),
        );
        console.log(
            `t=0 ONLY (red control): pooledFold=${(t0OnlyMean * 100).toFixed(4)}% vs phase-averaged=${(solveReading.mean * 100).toFixed(4)}% ` +
                `anchor=${(ANCHOR * 100).toFixed(4)}% — printed only, never gated`,
        );
    });

    test("the shipped LAMBDA (spectrum.ts) matches this run's live solve within the declared 10% agreement bound", () => {
        const relDiff = Math.abs(LAMBDA - lambdaSolve) / lambdaSolve;
        console.log(
            `lambdaSolve=${lambdaSolve.toFixed(6)} seedStderr=${(solveReading.seedStderr * 100).toFixed(4)}% ` +
                `phaseStderr=${(solveReading.phaseStderr * 100).toFixed(4)}% bernoulli95=±${(solveReading.bernoulli * 100).toFixed(4)}% ` +
                `shipped LAMBDA=${LAMBDA} relDiff=${(relDiff * 100).toFixed(3)}%`,
        );
        expect(CASCADE_CONFIGS[0].lambda).toBe(LAMBDA);
        expect(CASCADE_CONFIGS[1].lambda).toBe(LAMBDA);
        expect(relDiff).toBeLessThanOrEqual(LAMBDA_AGREEMENT_REL);
    });

    for (const held of [heldA, heldB]) {
        test(`${held.label}: pooled fold at the solved λ stays within ±${ANCHOR_BAND_REL * 100}% of the anchor`, () => {
            const reading = ensembleFold(lambdaSolve, held);
            const relDiff = (reading.mean - ANCHOR) / ANCHOR;
            console.log(
                `${held.label} pooledFold=${(reading.mean * 100).toFixed(4)}% anchor=${(ANCHOR * 100).toFixed(4)}% ` +
                    `relDiff=${(relDiff * 100).toFixed(2)}% seedStderr=${(reading.seedStderr * 100).toFixed(4)}% ` +
                    `phaseStderr=${(reading.phaseStderr * 100).toFixed(4)}% bernoulli95=±${(reading.bernoulli * 100).toFixed(4)}%`,
            );
            expect(withinAnchorBand(reading.mean)).toBe(true);
        });

        test(`${held.label}: λ solved independently on this window agrees with the declared-set solve within ${LAMBDA_AGREEMENT_REL * 100}%`, () => {
            const heldBisect = bisectLambda(held);
            const lambdaHeld = heldBisect.lambda;
            const relDiff = Math.abs(lambdaHeld - lambdaSolve) / lambdaSolve;
            console.log(
                `${held.label} lambdaHeld=${lambdaHeld.toFixed(6)} lambdaSolve=${lambdaSolve.toFixed(6)} relDiff=${(relDiff * 100).toFixed(3)}%`,
            );
            expect(relDiff).toBeLessThanOrEqual(LAMBDA_AGREEMENT_REL);
        });
    }

    test("RED-WITNESS — a ±20% λ mutation breaks the held-out anchor-band agreement (Gate law: guarded arm re-run with only λ mutated)", () => {
        const mutatedUp = lambdaSolve * (1 + LAMBDA_MUTATION_REL);
        const mutatedDown = lambdaSolve * (1 - LAMBDA_MUTATION_REL);
        const readingUp = ensembleFold(mutatedUp, heldA);
        const readingDown = ensembleFold(mutatedDown, heldA);
        console.log(
            `mutated +20% λ=${mutatedUp.toFixed(4)} pooledFold=${(readingUp.mean * 100).toFixed(3)}% withinBand=${withinAnchorBand(readingUp.mean)}`,
        );
        console.log(
            `mutated -20% λ=${mutatedDown.toFixed(4)} pooledFold=${(readingDown.mean * 100).toFixed(3)}% withinBand=${withinAnchorBand(readingDown.mean)}`,
        );
        // the guarded arm above asserts withinAnchorBand(...) === true on the unmutated λ; this
        // red-witness re-runs the identical predicate on the mutated subject and requires it to flip.
        expect(withinAnchorBand(readingUp.mean)).toBe(false);
        expect(withinAnchorBand(readingDown.mean)).toBe(false);
    });

    describe("Gaussian/erfc corroboration — prints its model error, never gates (spec Validation)", () => {
        // σ: pooled RMS of the composed unit-λ Jacobian TRACE (∂Dx/∂x + ∂Dz/∂z) over the solve
        // window's own seed×phase ensemble — no isotropic-Gaussian projection factor (the rejected
        // round's √(5/3), algebraically re-imported and then re-derived to √(8/3) under an isotropic
        // trace-variance model, deleted rather than corrected: this file's whole point is measuring,
        // never projecting).
        let sumSquares = 0;
        let count = 0;
        for (const r of solveWindow.realizations) {
            for (const field of r.phases) {
                const n = field.gridN * field.gridN;
                for (let i = 0; i < n; i++) {
                    const trace = field.gxx[i] + field.gzz[i];
                    sumSquares += trace * trace;
                }
                count += n;
            }
        }
        const effectiveSlopeSigma = Math.sqrt(sumSquares / count);
        const ceilingLambda = 1 / effectiveSlopeSigma;

        test("erfc(1/(√2·λ·σ)) against the measured composed fold at the solved λ", () => {
            const predicted = foldProbability(lambdaSolve, effectiveSlopeSigma);
            const modelError = Math.abs(predicted - solveReading.mean) / solveReading.mean;
            console.log(
                `erfc model: sigma=${effectiveSlopeSigma.toFixed(6)} predicted=${(predicted * 100).toFixed(4)}% ` +
                    `measured=${(solveReading.mean * 100).toFixed(4)}% modelError=${(modelError * 100).toFixed(2)}% ` +
                    "— reading only, never gated",
            );
        });

        test("FOLD_REGIME.effectiveSlopeSigma and .ceilingLambda match this run's live recomputation", () => {
            console.log(
                `effectiveSlopeSigma=${effectiveSlopeSigma.toFixed(6)} ceilingLambda=${ceilingLambda.toFixed(6)}`,
            );
            expect(FOLD_REGIME.effectiveSlopeSigma).toBeCloseTo(effectiveSlopeSigma, 6);
            expect(FOLD_REGIME.ceilingLambda).toBeCloseTo(ceilingLambda, 6);
        });

        test("fold band, printed anchor→ceiling (never ceiling→λ): the composed field's own regime for shading's foam seeding", () => {
            const foldAtCeiling = ensembleFold(ceilingLambda, solveWindow);
            console.log(
                `fold band: anchor(λ=${lambdaSolve.toFixed(3)}, fold=${(solveReading.mean * 100).toFixed(2)}%, ` +
                    `whitecapAnchor=${(ANCHOR * 100).toFixed(2)}%) → ceiling(λ=${ceilingLambda.toFixed(3)}, ` +
                    `fold=${(foldAtCeiling.mean * 100).toFixed(2)}%) — reading only, never gated`,
            );
        });
    });
});

function erfc(x: number): number {
    const sign = x < 0 ? -1 : 1;
    const a = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * a);
    const p =
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)));
    const value = p * t * Math.exp(-a * a);
    return sign > 0 ? value : 2 - value;
}

function foldProbability(lambda: number, sigma: number): number {
    return lambda > 0 && sigma > 0 ? 0.5 * erfc(1 / (Math.SQRT2 * lambda * sigma)) : 0;
}
