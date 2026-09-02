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
// Leg (b) — near-spacing sweep at the shipped reconstruction (bicubic): three near spacings relative
//   to `finestCascadeTexel/2` (the Nyquist-sampling bound `OCEAN_CLIP_CONFIG.nearSpacing` targets) —
//   the shipped spacing (0.99x), cascade 1's own texel (2x), and twice that texel (4x) — none of them
//   sitting exactly "at" the bound; the ratio to `finestCascadeTexel/2` is printed beside each leg.
//   Holds the same ring structure (5 levels, doubling spacing/radius each ring) so ring-for-ring
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
// I3m re-verdict (2026-09-01): this file used to also assert ring 0 (the near field) against a
// composed CONTINUUM fold-fraction reference built from a frozen, test-local Catmull-Rom
// transcription, sampled at random continuous world points. The consult ruled that comparison's
// *kind* wrong: a fold fraction is a tail statistic, not a quantity a reconstruction kernel's
// derivative error bounds, so a "frozen reference" transcribed from the very kernel under test is a
// regression pin, not an independent check. That comparison, its red witness, and the frozen
// transcription are deleted rather than re-sized — this oracle now imports the shipped kernel on
// BOTH sides of every assertion and asserts only the mesh: population (both legs) and the
// zero-displacement negative control above. Leg (a)'s ring-0-vs-continuum reading becomes a
// print-only sweep (leg (b)'s own ring-0 reading follows the same shape — see the I3m-r correction
// round paragraph below); `field-mesh-agreement.test.ts` is where a field-only fold-fraction reading
// is printed with its own Bernoulli interval, never gated.
//
// Rings 1-4 (leg a) stay population-only, on purpose: coarsening past the Nyquist bound is EXPECTED
// to under-count folds (a fold that fits between two widely-spaced vertices cannot invert a triangle
// there), so a coarser ring's flip fraction has no reason to track any reference rate and asserting
// it as if it should would defend the wrong claim (`checks.md`'s "pins the status quo at a position
// nobody has questioned").
//
// I3m-r correction round (2026-09-01): leg (b)'s ring-0 flip fraction used to be ASSERTED
// non-increasing across the three near-spacing legs. The consult ruled that ordering claim wrong too
// (not just the continuum comparison the I3m re-verdict already retired): a fold fraction is a tail
// statistic (this file's own header, above) with no mutation in this file's table that reaches an
// ordering over it — the two recorded mutations below leave the direction of the trend unexamined,
// so the assertion was never red-witnessed by anything this file actually runs. Leg (b) is now a
// print-only sweep, exactly like leg (a): ring 0's flip fraction at each of the three near spacings,
// printed with its 95% Bernoulli sampling interval, never ordered or gated. The near-spacing-above-
// Nyquist claim these three spacings were chosen to bracket is the closed-form inequality
// `clipmap-continuity.test.ts` already asserts on `OCEAN_CLIP_CONFIG` against `CASCADE_CONFIGS`
// (`checkContinuity`'s near-field finding), with its own red witness there — not re-asserted through
// a mesh reading here. Measured: 4.286% / 2.058% / 1.113% across the three legs (unchanged by this
// correction; only the assertion around the reading moved).
//
// Mutation table (each applied in place at this stage's own ref, run, reverted with
// `git show HEAD:<path>`, never shipped):
//   - `reconstruction.ts`'s `catmullRom1D` `c` coefficient (`0.5*p2` → `0.35*p2`) → GREEN, 11 pass / 0
//     fail. This oracle asserts only the mesh, so a reconstruction-kernel-fidelity defect does not
//     reach it by design. This mutation is caught by `reconstruction-kernel-claim.test.ts`'s
//     closed-form claims instead, the correct instrument for a reconstruction-kernel defect per the
//     Gate law — recorded here as a negative result so a reader doesn't mistake this oracle's silence
//     on it for a coverage gap.
//   - `reconstruction.ts`'s `wrap` narrowed to `i % n` (drops the negative-index wraparound) → RED:
//     9 fail / 2 pass, every failure a thrown out-of-range array read propagating out of
//     `bicubicSample` — population and both print-only sweeps all reach it (a thrown error inside a
//     `console.log` argument still fails the enclosing `test`).
import { describe, expect, test } from "bun:test";
import {
    buildClipLevels,
    buildClipmapMesh,
    type ClipLevel,
    finestCascadeTexel,
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

const Z95 = 1.96; // matches `field-mesh-agreement.test.ts`'s own Bernoulli interval z-quantile.

/** 95% normal-approximation Bernoulli interval half-width, in the same units as `flipFraction`, for
 *  a proportion measured over `triCount` triangles — a tail statistic's reading gets its sampling
 *  uncertainty stated beside it, never a gate (Gate law; `field-mesh-agreement.test.ts`'s own reading
 *  is the field-only sibling this print sweep matches in shape). Returns 0 (never NaN) at `p=0`. */
function bernoulliInterval(flipFraction: number, triCount: number): number {
    if (triCount === 0) return Number.NaN;
    const p = Math.max(0, Math.min(1, flipFraction));
    return Z95 * Math.sqrt((p * (1 - p)) / triCount);
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

    // I3m re-verdict (2026-09-01): ring 0's flip fraction used to be asserted against a composed
    // continuum fold-fraction reference here (see this file's header) — retired outright, not
    // re-derived. This is now a print-only sweep: every reconstruction leg's ring 0 flip fraction,
    // for a human reading, asserting nothing about agreement with any field reference.
    test("print-only: ring 0's flip fraction across every reconstruction leg", () => {
        for (const [label, kernel] of reconLegs) {
            const res = measureMeshFlips(OCEAN_CLIP_LEVELS, kernel, cascadeFields, false);
            const ring0 = res.perRing[0];
            const interval = bernoulliInterval(ring0.flipFraction, ring0.triCount);
            console.log(
                `leg (a) ring 0 (${label}): mesh flip=${(ring0.flipFraction * 100).toFixed(3)}% ± ${(interval * 100).toFixed(3)}% (95%, n=${ring0.triCount}) — reading only, not gated`,
            );
        }
    });
});

describe("leg (b): near-spacing sweep at the shipped reconstruction (bicubic)", () => {
    // coreHalfExtent = nearSpacing * 100 keeps the core-grid-is-integral check exact for all three
    // spacings and totalHalfExtent = coreHalfExtent * 16 keeps the same 5-level, doubling-per-ring
    // structure OCEAN_CLIP_LEVELS uses, so ring r's spacing is always nearSpacing * 2^r.
    //
    // The three legs are 0.99x / 2x / 4x of `finestCascadeTexel(CASCADE_CONFIGS)/2` (the Nyquist-
    // sampling bound `OCEAN_CLIP_CONFIG.nearSpacing` is derived to sit just below) — none of them "at"
    // that bound. The ratio to the reference is printed beside each leg below.
    const nyquistReference = finestCascadeTexel(CASCADE_CONFIGS) / 2;
    const spacingLegs = [OCEAN_CLIP_CONFIG.nearSpacing, nyquistReference * 2, nyquistReference * 4];
    for (const nearSpacing of spacingLegs) {
        test(`near spacing ${nearSpacing}m (${(nearSpacing / nyquistReference).toFixed(2)}x reference): every ring has a nonzero triangle population`, () => {
            const levels = buildClipLevels({
                coreHalfExtent: nearSpacing * 100,
                nearSpacing,
                totalHalfExtent: nearSpacing * 100 * 16,
            });
            const res = measureMeshFlips(levels, bicubicSample, cascadeFields, false);
            printSweep(
                `near spacing ${nearSpacing}m (${(nearSpacing / nyquistReference).toFixed(2)}x of finestCascadeTexel/2=${nyquistReference.toFixed(4)}m)`,
                res,
            );
            assertPopulated(res);
        });
    }

    // I3m-r correction round (2026-09-01): ring 0's flip fraction used to be ASSERTED non-increasing
    // across these three near-spacing legs (0.99x / 2x / 4x of the Nyquist reference
    // `finestCascadeTexel(CASCADE_CONFIGS)/2` — none of the three legs sits exactly "at" that
    // reference). The consult ruled the ordering claim wrong, same as leg (a)'s continuum comparison
    // before it: a fold fraction is a tail statistic, and this file's own mutation table has no entry
    // that reaches an ordering over it — a `clipmap.ts` ring-doubling mutation this comment used to
    // cite as the ordering's red witness does not exist anywhere in this file's mutation table above,
    // and no mutation was ever run to check the ordering assertion's own reach. This is now a
    // print-only sweep, matching leg (a)'s shape: ring 0's flip fraction at each near spacing, printed
    // with its Bernoulli interval and its ratio to the reference, never ordered or gated. The
    // near-spacing-above-Nyquist claim these spacings were chosen to bracket lives at
    // `clipmap-continuity.test.ts`'s closed-form inequality on `OCEAN_CLIP_CONFIG` against
    // `CASCADE_CONFIGS`, with its own red witness there (see this file's header) — not re-derived
    // through a mesh reading here.
    test("print-only: ring 0's flip fraction across the three near-spacing legs", () => {
        for (const nearSpacing of spacingLegs) {
            const levels = buildClipLevels({
                coreHalfExtent: nearSpacing * 100,
                nearSpacing,
                totalHalfExtent: nearSpacing * 100 * 16,
            });
            const ring0 = measureMeshFlips(levels, bicubicSample, cascadeFields, false).perRing[0];
            const interval = bernoulliInterval(ring0.flipFraction, ring0.triCount);
            console.log(
                `leg (b) ring 0 (spacing ${nearSpacing}m, ${(nearSpacing / nyquistReference).toFixed(2)}x reference): ` +
                    `mesh flip=${(ring0.flipFraction * 100).toFixed(3)}% ± ${(interval * 100).toFixed(3)}% (95%, n=${ring0.triCount}) — reading only, not gated`,
            );
        }
    });
});
