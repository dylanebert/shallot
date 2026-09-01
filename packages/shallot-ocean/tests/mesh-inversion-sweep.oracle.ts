// Trigger: changes to the clipmap's spacing schedule, the reconstruction kernel a mesh vertex reads
// through, or the displacement operator can change how a displaced mesh's triangles fold relative to
// the field they sample.
//
// The render mesh is a discretization of a continuous displacement field: a mesh vertex reads that
// field through a reconstruction kernel and moves. Two things can go wrong independently — the
// kernel a vertex reads through, and the spacing between vertices — so this file holds the mesh's
// own instrument (does a displaced triangle invert its winding), separate from the field-only
// instrument (`field-mesh-agreement.test.ts`) that never touches a mesh at all. A defect that shows
// up only when the field is read through a discretization is invisible to a field-only reading, no
// matter how precise.
//
// Two sweeps, each holding one variable fixed while moving the other, over the shipped displacement
// cascades at t=0 (the fold fraction is time-invariant, so a single fixed time is sufficient):
//
// Leg (a) — reconstruction swap at the shipped mesh (`OCEAN_CLIP_LEVELS`, unmodified): bicubic
//   Catmull-Rom (shipped, C1) vs bilinear (C0) vs raw texel-nearest (C^-1, no interpolation — brackets
//   bicubic from the opposite side bilinear does, since it has no negative lobes to overshoot with).
//
// Leg (b) — near-spacing sweep at the shipped reconstruction (bicubic): the shipped near spacing,
//   cascade 1's own texel, and twice that texel (below / at / above the current boundary), holding
//   the same ring structure (5 levels, doubling spacing/radius each ring) so ring-for-ring
//   comparisons are meaningful.
//
// Both legs read every ring's own triangle population (never vacuously passing a ring with zero
// triangles) and print the per-ring winding-flip fraction for a human reading — reconstruction and
// spacing choices are already locked (`clipmap.ts`, `reconstruction.ts`), so these two legs are a
// standing regression instrument on the shipped configuration, not a live decision sweep.
//
// Negative control: a flat plane (forced-zero displacement) cannot invert any triangle, so every
// ring must read EXACTLY 0% flips — asserted, not just printed, and the one bound in this file that
// is not derived from the measurement under test: zero is ground truth for a flat plane regardless
// of what the displaced sweeps read. Its committed red witness (below) re-runs the exact same
// assertion with only the subject mutated: `ignoreZeroControl` forces real displacement through
// even when the zero control is requested, mirroring the hand mutation ("comment out the early
// return") this instrument now runs automatically.
//
// Beyond population, ring 0 (the near field — the one ring `OCEAN_CLIP_CONFIG.nearSpacing` is
// derived to resolve at the finest cascade's Nyquist limit) is asserted against a composed CONTINUUM
// reference: `continuumFoldFraction` re-runs the SAME central-difference-on-reconstruction comparison
// `field-mesh-agreement.test.ts` uses, generalized to sum both cascades' contributions (matching
// `measureMeshFlips`'s own `displacedXZ`) and sampled at random continuous points over ring 0's own
// world extent — never through the mesh's discrete vertices. This is the "field measured through a
// mesh is two instruments" split named in this spec's own residue: one instrument (the field-only
// arm) reads the field alone, a second (this one) reads the geometry it is applied to, at the one
// ring the mesh's own near spacing is derived to keep resolved.
//
// Rings 1-4 (leg a) and the two Nyquist-violating spacings (leg b) stay population-only, on purpose:
// coarsening past the Nyquist bound is EXPECTED to under-count folds (a fold that fits between two
// widely-spaced vertices cannot invert a triangle there), so a coarser ring's flip fraction has no
// reason to track the continuum rate and asserting it as if it should would defend the wrong claim
// (`checks.md`'s "pins the status quo at a position nobody has questioned"). Measured: ring 0 at the
// shipped near spacing (0.12m, below the 0.1211m Nyquist bound `checkContinuity` derives) reads
// 2.56% against a 2.52% continuum reference; leg (b)'s at-boundary (0.2422m) and above-boundary
// (0.4844m) spacings read 2.06%/1.11% against continuum references of 2.70%/2.18% at their own
// extents — a real, monotone undercount, not noise.
//
// Mutation table (each applied in place at this stage's own ref, run, reverted with
// `git show HEAD:<path>`, never shipped):
//   - `reconstruction.ts`'s `catmullRom1D` `c` coefficient (`0.5*p2` → `0.35*p2`) → RED (leg (a)'s
//     ring 0 comparison and its red witness both fail: 9 pass / 4 fail).
//   - `reconstruction.ts`'s `wrap` widened to `i % n` → RED (11 of 13 tests fail, several by thrown
//     error on an out-of-range array read).
import { describe, expect, test } from "bun:test";
import {
    buildClipLevels,
    buildClipmapMesh,
    type ClipLevel,
    OCEAN_CLIP_CONFIG,
    OCEAN_CLIP_LEVELS,
} from "../src/clipmap";
import { runCpuPipeline } from "../src/cpu-reference";
import {
    bicubicSample,
    bilinearSample,
    type Field,
    nearestSample,
    type ReconstructionKernel,
} from "../src/reconstruction";
import { CASCADE_CONFIGS, generateH0 } from "../src/spectrum";

const T = 0; // the fold fraction is time-invariant — see this file's header.
const VF = 8; // px py pz u | nx ny nz v — clipmap.ts's authoring layout.

// ── the field: one CPU-reference pipeline run per cascade, shared by every leg ───────────────────
interface CascadeField {
    N: number;
    L: number;
    dx: Field;
    dz: Field;
}
function toField(flat: Float64Array, N: number): Field {
    const f: Field = [];
    for (let y = 0; y < N; y++) {
        const row: number[] = new Array(N);
        for (let x = 0; x < N; x++) row[x] = flat[y * N + x];
        f.push(row);
    }
    return f;
}
const cascadeFields: CascadeField[] = CASCADE_CONFIGS.map((cfg) => {
    const h0 = generateH0(cfg);
    const cpu = runCpuPipeline(h0, cfg, T);
    return {
        N: cfg.N,
        L: cfg.L,
        dx: toField(cpu.jacobian.dxRaw, cfg.N),
        dz: toField(cpu.jacobian.dzRaw, cfg.N),
    };
});

function texelUV(world: number, L: number, N: number): number {
    return (world / L + 0.5) * N - 0.5;
}

// ── frozen, test-local reference kernel — the continuum reference's inputs, never the mutable
//    imported `bicubicSample`/`catmullRom1D` this file's leg (a) also grades ─────────────────────
// Written from the textbook uniform Catmull-Rom (tau=0.5) formula, independently of
// `reconstruction.ts` (same derivation `field-mesh-agreement.test.ts` uses, transcribed fresh here
// rather than imported — a shared import would make this file's reference move with a mutation to
// that OTHER test file's copy, not just with a mutation to the mutable subject). A defect in the
// shipped `reconstruction.ts` moves the READING (`measureMeshFlips`, driven by the imported kernel)
// away from this fixed reference, which is what lets leg (a)'s mutation reach this file (Gate law).
function refWrap(i: number, n: number): number {
    return ((i % n) + n) % n;
}
function refCatmullRom1D(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
    const c = -0.5 * p0 + 0.5 * p2;
    return p1 + t * (c + t * (b + t * a));
}
function refBicubicSample(field: Field, n: number, u: number, v: number): number {
    const ix = Math.floor(u);
    const iy = Math.floor(v);
    const fx = u - ix;
    const fy = v - iy;
    const rows: number[] = [];
    for (let j = -1; j <= 2; j++) {
        const yy = refWrap(iy + j, n);
        const p0 = field[yy][refWrap(ix - 1, n)];
        const p1 = field[yy][refWrap(ix, n)];
        const p2 = field[yy][refWrap(ix + 1, n)];
        const p3 = field[yy][refWrap(ix + 2, n)];
        rows.push(refCatmullRom1D(p0, p1, p2, p3, fx));
    }
    return refCatmullRom1D(rows[0], rows[1], rows[2], rows[3], fy);
}

/** deterministic PRNG (mulberry32) — reproducible random world points, matches
 *  `field-mesh-agreement.test.ts`'s own generator (no external dependency, no shared import). */
function mulberry32(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const CONTINUUM_H_STEP_FRAC = 1 / 8; // matches field-mesh-agreement.test.ts's own central-diff step.
const CONTINUUM_POINTS = 20000; // see derivation in `samplingRelativeError`'s call sites below.

/** composed Jacobian determinant of the SUMMED cascades' displacement field, through the FROZEN
 *  reference kernel, at world point (x, z) — the continuum-side counterpart to `measureMeshFlips`'s
 *  own per-vertex `displacedXZ` sum, generalized from `field-mesh-agreement.test.ts`'s single-cascade
 *  `centralDiffDet`. Each cascade uses its OWN central-difference step (its own texel × H_STEP_FRAC),
 *  matching that file's per-cascade convention. */
function composedContinuumDet(fields: readonly CascadeField[], x: number, z: number): number {
    let dDxdx = 0;
    let dDxdz = 0;
    let dDzdx = 0;
    let dDzdz = 0;
    for (const c of fields) {
        const h = (c.L / c.N) * CONTINUUM_H_STEP_FRAC;
        const sample = (field: Field, wx: number, wz: number) =>
            refBicubicSample(field, c.N, texelUV(wx, c.L, c.N), texelUV(wz, c.L, c.N));
        dDxdx += (sample(c.dx, x + h, z) - sample(c.dx, x - h, z)) / (2 * h);
        dDxdz += (sample(c.dx, x, z + h) - sample(c.dx, x, z - h)) / (2 * h);
        dDzdx += (sample(c.dz, x + h, z) - sample(c.dz, x - h, z)) / (2 * h);
        dDzdz += (sample(c.dz, x, z + h) - sample(c.dz, x, z - h)) / (2 * h);
    }
    const Jxx = 1 + dDxdx;
    const Jzz = 1 + dDzdz;
    const Jxz = (dDxdz + dDzdx) / 2;
    return Jxx * Jzz - Jxz * Jxz;
}

/** composed continuum fold fraction over `[-rOuter, rOuter]²` — `CONTINUUM_POINTS` random points,
 *  the frozen reference kernel, never a mesh vertex. The "field's own fold fraction" ring 0's flip
 *  fraction is compared against. */
function continuumFoldFraction(
    fields: readonly CascadeField[],
    rOuter: number,
    seed: number,
): number {
    const rand = mulberry32(seed);
    let foldCount = 0;
    for (let i = 0; i < CONTINUUM_POINTS; i++) {
        const x = (rand() * 2 - 1) * rOuter;
        const z = (rand() * 2 - 1) * rOuter;
        if (composedContinuumDet(fields, x, z) < 0) foldCount++;
    }
    return foldCount / CONTINUUM_POINTS;
}

/** 95% normal-approximation relative-error bound on an `n`-sample Bernoulli-proportion estimator of
 *  population value `p` — `slope.ts`'s `slopeMomentAgreementTolerance` z-quantile convention, same
 *  formula `field-mesh-agreement.test.ts` uses. `Math.max(p, 1e-6)` matches this file's own relDiff
 *  denominator floor below. */
function samplingRelativeError(p: number, n: number): number {
    const clampedP = Math.max(0, Math.min(1, p));
    const se = Math.sqrt((clampedP * (1 - clampedP)) / n);
    return (1.96 * se) / Math.max(clampedP, 1e-6);
}

/** relative difference between a ring's flip fraction and the continuum fold fraction over the SAME
 *  world extent, denominator-floored the same way `field-mesh-agreement.test.ts`'s `relDiff` is. */
function ring0RelDiff(ring0Flip: number, continuumFold: number): number {
    return Math.abs(ring0Flip - continuumFold) / Math.max(ring0Flip, continuumFold, 1e-6);
}

/** the frozen reference kernel's own central-difference truncation error against a synthetic
 *  single-mode field at cascade 1's `kHi` (the shipped bands' worst case — the highest frequency
 *  either cascade admits) — verbatim `field-mesh-agreement.test.ts`'s `referenceTruncationError`,
 *  transcribed fresh here for the same reason `refBicubicSample` is (see this file's header). Leg (a)
 *  VARIES the reconstruction kernel (spacing held fixed, fine), so kernel truncation error is the
 *  relevant, varying error source there and this term is included in ring 0's bound. Leg (b) holds
 *  the kernel fixed at `bicubicSample` throughout and varies spacing instead — the SAME kernel drives
 *  both the continuum reference and the mesh reading in that leg, so kernel truncation error is a
 *  common-mode term that largely cancels out of their DIFFERENCE, and leg (b)'s bound omits it
 *  (composed only of the two sides' own finite-sample error — see `leg (b)`'s tests below). */
function worstCaseTruncationError(): number {
    const cfg = CASCADE_CONFIGS[1];
    const texel = cfg.L / cfg.N;
    const h = texel * CONTINUUM_H_STEP_FRAC;
    const field: Field = [];
    for (let y = 0; y < cfg.N; y++) {
        const row: number[] = new Array(cfg.N);
        for (let x = 0; x < cfg.N; x++) {
            const world = ((x + 0.5) / cfg.N - 0.5) * cfg.L;
            row[x] = Math.cos(cfg.kHi * world);
        }
        field.push(row);
    }
    const rand = mulberry32(0xbeef);
    let sumSq = 0;
    let sumAnalyticSq = 0;
    const samples = 64;
    for (let i = 0; i < samples; i++) {
        const world = (rand() - 0.5) * cfg.L;
        const sampleAt = (w: number) =>
            refBicubicSample(field, cfg.N, texelUV(w, cfg.L, cfg.N), 3.0);
        const estD = (sampleAt(world + h) - sampleAt(world - h)) / (2 * h);
        const analyticD = -cfg.kHi * Math.sin(cfg.kHi * world);
        sumSq += (estD - analyticD) ** 2;
        sumAnalyticSq += analyticD * analyticD;
    }
    return Math.sqrt(sumSq / samples) / Math.sqrt(Math.max(sumAnalyticSq / samples, 1e-9));
}

function signedArea2D(
    ax: number,
    az: number,
    bx: number,
    bz: number,
    cx: number,
    cz: number,
): number {
    return 0.5 * ((bx - ax) * (cz - az) - (cx - ax) * (bz - az));
}

function ringIndexForRadius(r: number, levels: readonly ClipLevel[]): number {
    for (let i = 0; i < levels.length; i++) {
        if (r <= levels[i].rOuter + 1e-6) return i;
    }
    return levels.length - 1;
}

interface RingResult {
    ring: number;
    spacing: number;
    triCount: number;
    flipCount: number;
    flipFraction: number;
}
interface SweepResult {
    perRing: RingResult[];
    totalTriCount: number;
    totalFlips: number;
    totalFlipFraction: number;
}

/** the one structural mutation this file's red witness needs: forces real displacement through even
 *  when the caller asked for the zero control, so the guarded assertion's own subject is mutated
 *  without touching the assertion or its bound (Gate law). */
interface MeshFlipMutation {
    ignoreZeroControl?: boolean;
}

/** builds `levels`'s mesh, displaces every vertex through `kernel` over `fields` (or forces zero
 *  displacement when `zeroControl` and `mutation.ignoreZeroControl` is not set), and reads the
 *  per-ring XZ-projected winding-flip fraction. */
function measureMeshFlips(
    levels: readonly ClipLevel[],
    kernel: ReconstructionKernel,
    fields: readonly CascadeField[],
    zeroControl: boolean,
    mutation: MeshFlipMutation = {},
): SweepResult {
    const { vertices, indices } = buildClipmapMesh(levels);
    const vertCount = vertices.length / VF;
    const triCount = indices.length / 3;
    const vx = (i: number) => vertices[i * VF];
    const vz = (i: number) => vertices[i * VF + 2];

    function displacedXZ(x: number, z: number): [number, number] {
        if (zeroControl && !mutation.ignoreZeroControl) return [x, z];
        let dx = 0;
        let dz = 0;
        for (const c of fields) {
            const u = texelUV(x, c.L, c.N);
            const v = texelUV(z, c.L, c.N);
            dx += kernel(c.dx, c.N, u, v);
            dz += kernel(c.dz, c.N, u, v);
        }
        return [x + dx, z + dz];
    }

    const dispX = new Float64Array(vertCount);
    const dispZ = new Float64Array(vertCount);
    for (let i = 0; i < vertCount; i++) {
        const [X, Z] = displacedXZ(vx(i), vz(i));
        dispX[i] = X;
        dispZ[i] = Z;
    }

    const ringTriCount = new Array(levels.length).fill(0);
    const ringFlips = new Array(levels.length).fill(0);
    let totalFlips = 0;
    for (let t = 0; t < triCount; t++) {
        const i0 = indices[t * 3];
        const i1 = indices[t * 3 + 1];
        const i2 = indices[t * 3 + 2];
        const preSign = signedArea2D(vx(i0), vz(i0), vx(i1), vz(i1), vx(i2), vz(i2)) >= 0 ? 1 : -1;
        const postSign =
            signedArea2D(dispX[i0], dispZ[i0], dispX[i1], dispZ[i1], dispX[i2], dispZ[i2]) >= 0
                ? 1
                : -1;
        const centroidR = Math.max(
            Math.abs((vx(i0) + vx(i1) + vx(i2)) / 3),
            Math.abs((vz(i0) + vz(i1) + vz(i2)) / 3),
        );
        const ring = ringIndexForRadius(centroidR, levels);
        ringTriCount[ring]++;
        if (preSign !== postSign) {
            ringFlips[ring]++;
            totalFlips++;
        }
    }

    const perRing: RingResult[] = levels.map((lvl, r) => ({
        ring: r,
        spacing: lvl.spacing,
        triCount: ringTriCount[r],
        flipCount: ringFlips[r],
        flipFraction: ringTriCount[r] > 0 ? ringFlips[r] / ringTriCount[r] : Number.NaN,
    }));
    return {
        perRing,
        totalTriCount: triCount,
        totalFlips,
        totalFlipFraction: totalFlips / triCount,
    };
}

function printSweep(label: string, res: SweepResult): void {
    console.log(
        `[${label}] total: ${res.totalFlips} / ${res.totalTriCount} triangles flipped (${(res.totalFlipFraction * 100).toFixed(2)}%)`,
    );
    for (const r of res.perRing) {
        console.log(
            `    ring ${r.ring} (spacing ${r.spacing}m): ${r.triCount} tris, flip ${(r.flipFraction * 100).toFixed(2)}%`,
        );
    }
}

/** every ring in `res` must have a nonzero triangle population — a ring reading 0/0 flips would
 *  otherwise pass vacuously. */
function assertPopulated(res: SweepResult): void {
    for (const r of res.perRing) {
        expect(r.triCount, `ring ${r.ring} has 0 triangles`).toBeGreaterThan(0);
    }
}

describe("zero-displacement negative control (shipped mesh, shipped kernel)", () => {
    const zeroRes = measureMeshFlips(OCEAN_CLIP_LEVELS, bicubicSample, cascadeFields, true);

    test("every ring has a nonzero triangle population", () => {
        printSweep("zero-control", zeroRes);
        assertPopulated(zeroRes);
    });

    test("every ring reads exactly 0% flips under forced-zero displacement", () => {
        for (const r of zeroRes.perRing) {
            expect(r.flipFraction, `ring ${r.ring}`).toBe(0);
        }
    });

    test("RED-WITNESS — ignoreZeroControl forces real displacement through and breaks the 0% reading", () => {
        // re-runs the guarded arm's own assertion ("every ring reads exactly 0%") with only the
        // subject mutated: the zero-control early return is bypassed, so the mesh displaces exactly
        // as it would under the real field.
        const mutatedRes = measureMeshFlips(OCEAN_CLIP_LEVELS, bicubicSample, cascadeFields, true, {
            ignoreZeroControl: true,
        });
        printSweep("zero-control RED-WITNESS (ignoreZeroControl)", mutatedRes);
        const anyNonzero = mutatedRes.perRing.some((r) => r.flipFraction > 0);
        expect(
            anyNonzero,
            "mutation must break the 0%-everywhere reading on at least one ring",
        ).toBe(true);
    });
});

describe("leg (a): reconstruction swap at the shipped mesh", () => {
    const reconLegs: [string, ReconstructionKernel][] = [
        ["bicubic Catmull-Rom (shipped, C1)", bicubicSample],
        ["bilinear (C0)", bilinearSample],
        ["nearest (C^-1, no interpolation)", nearestSample],
    ];
    for (const [label, kernel] of reconLegs) {
        test(`${label}: every ring has a nonzero triangle population`, () => {
            const res = measureMeshFlips(OCEAN_CLIP_LEVELS, kernel, cascadeFields, false);
            printSweep(label, res);
            assertPopulated(res);
        });
    }

    const ring0 = OCEAN_CLIP_LEVELS[0];
    const continuumFold = continuumFoldFraction(cascadeFields, ring0.rOuter, 0xc0de);
    const bound =
        worstCaseTruncationError() + samplingRelativeError(continuumFold, CONTINUUM_POINTS);

    test("bicubic (shipped): ring 0's flip fraction agrees with the composed continuum fold fraction within the reconstruction-error-derived tolerance", () => {
        const res = measureMeshFlips(OCEAN_CLIP_LEVELS, bicubicSample, cascadeFields, false);
        const ring0Flip = res.perRing[0].flipFraction;
        const relDiff = ring0RelDiff(ring0Flip, continuumFold);
        console.log(
            `leg (a) ring 0: mesh flip=${(ring0Flip * 100).toFixed(3)}% continuum=${(continuumFold * 100).toFixed(3)}% ` +
                `relDiff=${(relDiff * 100).toFixed(2)}% tolerance=${(bound * 100).toFixed(2)}%`,
        );
        expect(relDiff).toBeLessThanOrEqual(bound);
    });

    test("RED-WITNESS — nearest-texel sampling on ring 0 breaks the same tolerance", () => {
        // re-runs the guarded arm's own comparison with only the subject (the reconstruction kernel)
        // mutated to `nearestSample`, at the SAME ring 0 and the SAME continuum-reference-derived
        // tolerance — never an overridden extent, never a separately authored bound. `bound` is
        // computed once above from `continuumFold` alone (never from a mesh reading), so it is
        // identical in this test and the green one above — only the kernel producing `ring0Flip`
        // differs.
        const res = measureMeshFlips(OCEAN_CLIP_LEVELS, nearestSample, cascadeFields, false);
        const ring0Flip = res.perRing[0].flipFraction;
        const relDiff = ring0RelDiff(ring0Flip, continuumFold);
        console.log(
            `leg (a) ring 0 RED-WITNESS (nearest): mesh flip=${(ring0Flip * 100).toFixed(3)}% continuum=${(continuumFold * 100).toFixed(3)}% ` +
                `relDiff=${(relDiff * 100).toFixed(2)}% tolerance=${(bound * 100).toFixed(2)}%`,
        );
        expect(relDiff).toBeGreaterThan(bound);
    });
});

describe("leg (b): near-spacing sweep at the shipped reconstruction (bicubic)", () => {
    // coreHalfExtent = nearSpacing * 100 keeps the core-grid-is-integral check exact for all three
    // spacings and totalHalfExtent = coreHalfExtent * 16 keeps the same 5-level, doubling-per-ring
    // structure OCEAN_CLIP_LEVELS uses, so ring r's spacing is always nearSpacing * 2^r.
    const finestTexel = CASCADE_CONFIGS[1].L / CASCADE_CONFIGS[1].N;
    const spacingLegs = [OCEAN_CLIP_CONFIG.nearSpacing, finestTexel, finestTexel * 2];
    for (const nearSpacing of spacingLegs) {
        test(`near spacing ${nearSpacing}m: every ring has a nonzero triangle population`, () => {
            const levels = buildClipLevels({
                coreHalfExtent: nearSpacing * 100,
                nearSpacing,
                totalHalfExtent: nearSpacing * 100 * 16,
            });
            const res = measureMeshFlips(levels, bicubicSample, cascadeFields, false);
            printSweep(`near spacing ${nearSpacing}m`, res);
            assertPopulated(res);
        });
    }

    // Ring 0's flip fraction against the composed continuum fold fraction, at EACH spacing's own
    // ring-0 world extent. The kernel is held fixed at `bicubicSample` throughout this leg, so the
    // SAME kernel drives both the continuum reference and the mesh reading — kernel truncation error
    // is common-mode and this bound omits it (see `worstCaseTruncationError`'s own docblock), leaving
    // only the two sides' finite-sample error. Only `nearSpacing` (below the Nyquist bound
    // `checkContinuity` derives) is asserted to AGREE: `finestTexel` sits at the boundary and
    // `finestTexel * 2` sits above it, and coarsening past that bound is expected to under-count
    // folds (this file's header) — not a defect, so nothing here defends that under-count as
    // correctness. `finestTexel * 2` is this leg's own red witness: the SAME comparison, re-run with
    // only the subject (near spacing) mutated to a spacing coarse enough to break it.
    test(`near spacing ${OCEAN_CLIP_CONFIG.nearSpacing}m (below the Nyquist bound): ring 0's flip fraction agrees with the composed continuum fold fraction`, () => {
        const nearSpacing = OCEAN_CLIP_CONFIG.nearSpacing;
        const levels = buildClipLevels({
            coreHalfExtent: nearSpacing * 100,
            nearSpacing,
            totalHalfExtent: nearSpacing * 100 * 16,
        });
        const res = measureMeshFlips(levels, bicubicSample, cascadeFields, false);
        const ring0Flip = res.perRing[0].flipFraction;
        const continuumFold = continuumFoldFraction(cascadeFields, levels[0].rOuter, 0xc0de);
        const bound = samplingRelativeError(continuumFold, CONTINUUM_POINTS);
        const relDiff = ring0RelDiff(ring0Flip, continuumFold);
        console.log(
            `leg (b) near spacing ${nearSpacing}m ring 0: mesh flip=${(ring0Flip * 100).toFixed(3)}% continuum=${(continuumFold * 100).toFixed(3)}% ` +
                `relDiff=${(relDiff * 100).toFixed(2)}% tolerance=${(bound * 100).toFixed(2)}%`,
        );
        expect(relDiff).toBeLessThanOrEqual(bound);
    });

    test(`RED-WITNESS — near spacing ${finestTexel * 2}m (above the Nyquist bound) breaks the same comparison`, () => {
        // re-runs the guarded arm's own comparison with only the subject (near spacing) mutated,
        // using the SAME formula (no separately authored bound) at this spacing's own ring 0 extent.
        const nearSpacing = finestTexel * 2;
        const levels = buildClipLevels({
            coreHalfExtent: nearSpacing * 100,
            nearSpacing,
            totalHalfExtent: nearSpacing * 100 * 16,
        });
        const res = measureMeshFlips(levels, bicubicSample, cascadeFields, false);
        const ring0Flip = res.perRing[0].flipFraction;
        const continuumFold = continuumFoldFraction(cascadeFields, levels[0].rOuter, 0xc0de);
        const bound = samplingRelativeError(continuumFold, CONTINUUM_POINTS);
        const relDiff = ring0RelDiff(ring0Flip, continuumFold);
        console.log(
            `leg (b) RED-WITNESS near spacing ${nearSpacing}m ring 0: mesh flip=${(ring0Flip * 100).toFixed(3)}% continuum=${(continuumFold * 100).toFixed(3)}% ` +
                `relDiff=${(relDiff * 100).toFixed(2)}% tolerance=${(bound * 100).toFixed(2)}%`,
        );
        expect(relDiff).toBeGreaterThan(bound);
    });
});
