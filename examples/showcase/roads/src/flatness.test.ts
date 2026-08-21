import { describe, expect, test } from "bun:test";
import { meshHeightAt } from "./capture";
import { applyEdit } from "./editPure";
import {
    buildBandedLatticeVertices,
    buildDeviceFreeVertices,
    buildLatticeVertices,
    CROSS_SECTION_TOL,
    checkSurfaceFlatness,
    EDGE_EPSILON,
    flattenFieldAt,
    gradeBound,
    RECONSTRUCTION_AGREEMENT_TOL,
    reconstructionAgreement,
    SAMPLE_STEP,
} from "./flatness";
import type { Polyline, StrokeDocument } from "./overlay/document";
import { generateNetwork } from "./overlay/network";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";
import { CELLS, SPACING } from "./terrain/grid";
import { GROUND_LEVEL, makePermutation } from "./terrain/noise";
import { heightAtCpu, MAX_GRADE } from "./terrain/profile";
import { SEED } from "./terrain/terrain";

// Stage 15c's pin legitimacy — the criterion 15b's numbers are read against (spec Validation, "Surface
// flatness along the road", 2026-08-19 second consult). No code under test changes behaviour; the pins
// are replaced with the two-leg criterion (Leg A: the field, Leg B: the mesh's convergence). Numbers are
// measured, not predicted — every bound below is read off an actual run, with margin for run-to-run
// float noise, never fitted so tight a legitimate reading could flip the assertion.

// Stage 1 (`roads-interactive.md`) deleted route selection: `generateNetwork` takes no seed and always
// returns the one fixed standard chord. The sample population these two constants pin was re-measured
// against that chord (not carried over from the seed-1337 route-selected network they replace).
const SAMPLE_COUNT_SPACING = 546;
const NO_CUT_LONGITUDINAL = 57;

describe("surface flatness — sanity (the oracle can read flat)", () => {
    test("a manufactured target === natural profile reads flat on both axes", () => {
        // when the flatten target height exactly equals the natural height everywhere, flattenHeight
        // returns that one value regardless of coreDist (`terrain/flatten.ts`'s own definition) — so this
        // is flat by construction, the mutation-style proof the checker isn't just always red
        // (`coding.md`'s "survives its own mutations").
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-50, 0],
                        [50, 0],
                    ],
                    halfWidth: 4,
                },
            ],
        };
        const segments = [
            { ax: -50, az: 0, bx: 50, bz: 0, halfWidth: 4, aHeight: 5, bHeight: 5, road: 0 },
        ];
        const raw = buildLatticeVertices(SPACING, CELLS, segments, 16, () => 5);
        const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);
        expect(result.longitudinal.length).toBe(0);
        expect(result.crossSection.length).toBe(0);
    });

    test("flat natural terrain with no cut at all reads within tolerance — the RELIEF=0 analogue (arm ii)", () => {
        // undeformed (GROUND_LEVEL everywhere) natural terrain, no flatten target to blend toward
        // (empty segments) — the device-free analogue of a zeroed-RELIEF, no-cut build over the real
        // network's own footprint geometry.
        const doc = generateNetwork();
        const raw = buildLatticeVertices(SPACING, CELLS, [], 16, () => GROUND_LEVEL);
        const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);
        expect(result.longitudinal.length).toBe(0);
        expect(result.crossSection.length).toBe(0);
    });
});

describe("surface flatness — Leg A: the continuous field, no mesh (arm v)", () => {
    // Leg A (spec Validation, 2026-08-19 second consult): the continuous flattened field —
    // `networkCoreCpu`'s blended target eased toward natural via `flattenHeight`, with no mesh at all.
    // The field is exactly flat by construction (non-overlapping primitives never contend, so exactly
    // one weight survives at every sampled point), so a non-zero reading is the blend's design, not
    // the mesh's discretization. This is the leg that gates the blend's design and it carries no
    // fitted number.
    //
    // Stage 20: the relative-depth suppression factor and its `suppressionBand` parameter are deleted
    // (non-overlapping primitives never contend, so the suppression had nothing left to suppress). The
    // band-`= falloff` red-first arm whose subject was the deleted mechanism is removed with it.
    //
    // Stage 1 (`roads-interactive.md`): the "stage 18 arm (c)" overlapping-pair null control — the
    // discriminating proof that non-overlap is what buys exactness — is deleted along with the
    // multi-road generator's own non-overlap-by-construction guarantee it was proving (one road cannot
    // overlap, so there is no clearance/non-overlap machinery left for that arm to be a foil against;
    // `checks.md`'s hollowed-foil rule). The "stage 17 arm (a)" synthetic non-overlapping network below
    // still exercises the surviving multi-road blend machinery directly.
    const perm = makePermutation(SEED);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("the continuous field reads exactly zero on the real (non-overlapping) generator", () => {
        // on the real non-overlapping generator exactly one primitive's weight survives at every
        // sampled point (the others are past their own falloff), so the continuous field is exactly
        // flat — zero violations, zero amplitude on both axes. This is the structural claim 15b's
        // consult made and the leg that gates the blend's design; it carries no fitted number, only
        // the exact zero.
        const realDoc = generateNetwork();
        const { segments: realSegs, cutDepth: realCutDepth } = buildNetworkGeometry(realDoc, SEED);
        const realFalloff = computeFalloff(realCutDepth);
        const sample = (x: number, z: number) =>
            flattenFieldAt(x, z, realSegs, realFalloff, natural);
        const result = checkSurfaceFlatness(sample, realDoc);
        console.log(
            `LEG_A_GREEN longitudinal=${result.longitudinal.length} crossSection=${result.crossSection.length} maxCrossSectionExcess=${result.maxCrossSectionExcess} maxLongitudinalExcess=${result.maxLongitudinalExcess}`,
        );
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
    });
});

describe("surface flatness — shipped pipeline at SEED=1337 (arm i, stage 15b)", () => {
    const doc = generateNetwork();
    const raw = buildDeviceFreeVertices(doc, SEED);
    const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);

    test("the readings are reported (no fitted bound — Leg B gates the mesh, Leg A gates the field)", () => {
        // 15c: the fitted bounds (`longitudinal.length < 20`, `maxCrossSectionExcess < 0.3`) are deleted
        // — they were written to 15b's own reading and would defend the miss once the blend improves. The
        // mesh residual is gated by Leg B's convergence assertion (amplitude ratio +
        // count decrease), not by an absolute bound (Blocker 3: any honest bound is ≥ MAX_GRADE·SPACING =
        // 0.48 m and reads stage 12's founding 0.471 m defect green). The readings are logged as evidence.
        console.log(
            `SURFACE_FLATNESS_SHIPPED longitudinal=${result.longitudinal.length} crossSection=${result.crossSection.length} sampleCount=${result.sampleCount} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(4)} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(4)}`,
        );
        // no absolute amplitude bound is admissible (Blocker 3) — the readings are reported,
        // not gated here. Leg A gates the field (exactly zero), Leg B gates the mesh (convergence).
    });

    test("every violation sits inside the road footprint the document defines", () => {
        // the oracle only ever walks centreline/edge lines derived from `doc`'s own halfWidth — a
        // violation reported outside the document's footprint would mean the sampler drifted off the road
        // it claims to be checking.
        for (const v of [...result.longitudinal, ...result.crossSection]) {
            expect(v.roadIndex).toBeGreaterThanOrEqual(0);
            expect(v.roadIndex).toBeLessThan(doc.polylines.length);
        }
    });
});

describe("surface flatness — stage 18 arm (b): real generator reads exactly zero at both resolutions", () => {
    // The real-generator exactness arm (spec Validation, "Surface flatness in the corridor — exactly zero,
    // unconditional"): `checkSurfaceFlatness` over the banded lattice on the real `generateNetwork()`
    // reads exactly 0 violations / 0.0000 m on both axes, at `SPACING` and at `SPACING/2`. This reading
    // landed at stage 18 and is what licenses the exactness claim on the shipped pipeline — it is not an
    // owed reading. The non-overlapping generator guarantees no two primitives' falloff bands overlap at
    // any sampled station, so the composite target is affine everywhere the oracle samples, and
    // barycentric interpolation reproduces an affine field exactly at any cell size and any road angle.
    // The population is pinned `toBe(SAMPLE_COUNT_SPACING)` at both resolutions, so an emptied population
    // reds instead of passing on empty arrays — do not reintroduce a `sampleCount > 1000` inequality
    // beside the exact pin (the spec forbids it; the exact pin is strictly stronger). Stage 1
    // (`roads-interactive.md`) re-pinned the population at the standard chord's own reading (546, not
    // carried over from the seed-1337 route-selected network's 1197).
    const doc = generateNetwork();
    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(doc, SEED);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("exactly 0 violations and 0.0000 m on both axes at SPACING", () => {
        const coarseRaw = buildBandedLatticeVertices(SPACING, CELLS, segments, falloff, natural);
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(coarseRaw, x, z, SPACING, CELLS),
            doc,
        );
        console.log(
            `REAL_EXACTNESS_SPACING crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} sampleCount=${result.sampleCount} cutDepth=${cutDepth.toFixed(4)} falloff=${falloff.toFixed(4)}`,
        );
        // Stage-18 repair: pin the sampleCount at the widened endpointMargin (halfWidth + √2·SPACING)
        // so a future change that silently shrinks the population reds instead of passing on a
        // thinner array. Stage 1 (`roads-interactive.md`) reduced the network to one road: the
        // population re-pinned at the one-road document's own reading (re-measured, not carried over
        // from the five-road network).
        expect(result.sampleCount).toBe(SAMPLE_COUNT_SPACING);
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
        expect(result.maxLongitudinalExcess).toBe(0);
    });

    test("exactly 0 violations and 0.0000 m on both axes at SPACING/2", () => {
        const fineSpacing = SPACING / 2;
        const fineCells = CELLS * 2;
        const fineRaw = buildBandedLatticeVertices(
            fineSpacing,
            fineCells,
            segments,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(fineRaw, x, z, fineSpacing, fineCells),
            doc,
        );
        console.log(
            `REAL_EXACTNESS_SPACING_HALF crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} sampleCount=${result.sampleCount}`,
        );
        // Stage-18 repair: pin the sampleCount at the widened endpointMargin (halfWidth + √2·SPACING).
        // Stage 1 (`roads-interactive.md`): re-pinned at the one-road document's own reading.
        expect(result.sampleCount).toBe(SAMPLE_COUNT_SPACING);
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
        expect(result.maxLongitudinalExcess).toBe(0);
    });

    // R2 (stage-18 repair) deleted at stage 1 (`roads-interactive.md`): the multi-seed scan existed to
    // witness "by construction" holding across the route-selected generator's seed-dependent placements.
    // `generateNetwork` no longer takes a seed — it always returns the one fixed standard chord, so a
    // seed loop here would just re-check the same document N times. The arm's subject (seed-varying
    // placement) is gone; the both-resolutions reading above is the whole population now.
});

describe("surface flatness — null control: no cut, real relief (arm iii)", () => {
    test("still finds a real signal — proves the instrument detects steps, not just 'nothing is flat'", () => {
        // the device-free analogue of a no-cut build on real relief: the flatten kernel is skipped
        // entirely (empty segments, `flattenHeight`'s own
        // documented empty-network fallback degrades to plain `heightAt`), but the *document* used to
        // define the sampled footprint is still the real network — so the sampled lines run through
        // genuinely undeformed, rolling terrain (RELIEF=40) that was never asked to be flat. This is a
        // *different* mechanism from the shipped defect (raw noise steepness, not a reconstruction crease)
        // — it's the null control that still has something to find (`checks.md`'s witness-distinguishable
        // clause): a broken instrument that always reads "flat" would fail this arm the same way it would
        // pass arm ii, so the two arms together are what makes either one meaningful.
        const doc = generateNetwork();
        const perm = makePermutation(SEED);
        const raw = buildLatticeVertices(SPACING, CELLS, [], 16, (x, z) => heightAtCpu(x, z, perm));
        const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);
        console.log(
            `SURFACE_FLATNESS_NO_CUT longitudinal=${result.longitudinal.length} crossSection=${result.crossSection.length} sampleCount=${result.sampleCount}`,
        );
        // Stage 1 (`roads-interactive.md`) reduced the network to one road, re-pinning the population and
        // the failing count at the one-road document's own reading (re-measured, not carried over from
        // the five-road network) — the population is pinned, not just the failing count: a shrinking
        // sample array would otherwise buy this arm a green the same way it would buy the exactness arms
        // one (this unit's own residue — an amputated instrument degrades the quantifier, not just the
        // population).
        expect(result.sampleCount).toBe(SAMPLE_COUNT_SPACING);
        expect(result.longitudinal.length).toBe(NO_CUT_LONGITUDINAL);
    });
});

describe("surface flatness — the banded lattice reads what the full lattice reads (stage 12)", () => {
    // The null control that makes `buildBandedLatticeVertices`'s narrowing safe (spec Validation, "The
    // banded lattice reads what the full lattice reads"): for one drag at both resolutions,
    // `checkSurfaceFlatness` over the banded builder returns results *identical* to the full builder.
    // A zeroed vertex the oracle reaches decodes to a height nowhere near the corridor, so an
    // over-narrowed band reds loudly rather than passing silently.
    //
    // RED-FIRST WITNESS: `buildBandedLatticeVertices`'s band margin shrunk from the derived
    // `seg.halfWidth + √2 · spacing` (one cell *diagonal* past the footprint) to `seg.halfWidth +
    // spacing` (one cell) reds both arms in this block. Whole-file reading under that mutation: 11 pass /
    // 4 fail — these two, plus the two synthetic-network exactness arms below (five 30° chords). The
    // real-generator exactness pair stays green under it, which is the derivation showing itself: that
    // chord is axis-aligned, where one cell *is* the cell's reach along the normal. Verbatim, at SPACING:
    //   error: expect(received).toEqual(expected)
    //
    //     {
    //   -   "crossSection": [],
    //   -   "longitudinal": [],
    //   -   "maxCrossSectionExcess": 0,
    //   -   "maxLongitudinalExcess": 0,
    //   +   "crossSection": [
    //   +     {
    //   +       "bound": 0.0024414435034714275,
    //   +       "deltaFromCentre": 5.609365561403816,
    //   +       "line": "edgePos",
    //   +       "roadIndex": 0,
    //   +       "t": 0.03130588209152568,
    // — a 5.6 m cross-section step where the full lattice reads exactly zero, which is a dropped
    // corner decoding out of a zeroed vertex. That reading is also what corrected the band's
    // derivation: `halfWidth + spacing` is one cell short of the cell *diagonal* the 45° chord's
    // normal reaches.
    //
    // The drag is a 45°-ish chord on purpose: the band is tightest where the lattice's cell axes are
    // furthest from the chord's own frame, so an axis-aligned chord would not discriminate a margin
    // one cell-diagonal short.
    const dragged = applyEdit(generateNetwork(), 1, 300, 300);
    const perm = makePermutation(SEED);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);
    const { segments, cutDepth } = buildNetworkGeometry(dragged, SEED);
    const falloff = computeFalloff(cutDepth);

    const nonZeroVertices = (raw: Uint32Array): number => {
        let n = 0;
        for (let i = 0; i < raw.length; i += 4) if (raw[i] !== 0 || raw[i + 1] !== 0) n++;
        return n;
    };

    for (const [label, spacing, cells] of [
        ["SPACING", SPACING, CELLS],
        ["SPACING/2", SPACING / 2, CELLS * 2],
    ] as const) {
        test(`identical results at ${label}`, () => {
            const full = buildLatticeVertices(spacing, cells, segments, falloff, natural);
            const banded = buildBandedLatticeVertices(spacing, cells, segments, falloff, natural);

            // the banded buffer must really be sparse — otherwise this arm passes over a builder that
            // silently filled everything, and the identity below would be a tautology.
            const bandedFilled = nonZeroVertices(banded);
            const fullFilled = nonZeroVertices(full);
            console.log(
                `BANDED_FILL ${label} banded=${bandedFilled} full=${fullFilled} ratio=${(bandedFilled / fullFilled).toFixed(4)}`,
            );
            expect(bandedFilled).toBeGreaterThan(0);
            expect(bandedFilled).toBeLessThan(fullFilled / 10);

            const fullResult = checkSurfaceFlatness(
                (x, z) => meshHeightAt(full, x, z, spacing, cells),
                dragged,
            );
            const bandedResult = checkSurfaceFlatness(
                (x, z) => meshHeightAt(banded, x, z, spacing, cells),
                dragged,
            );
            expect(bandedResult).toEqual(fullResult);
            // and the reading the corpus arms depend on: exactly zero on both axes
            expect(fullResult.crossSection.length).toBe(0);
            expect(fullResult.longitudinal.length).toBe(0);
            expect(fullResult.sampleCount).toBeGreaterThan(0);
        });
    }
});

describe("checkSurfaceFlatness — window/threshold derivation, no candidate treatment", () => {
    test("gradeBound scales with Δs and the road-design grade limit alone", () => {
        // MAX_GRADE is the only slope-scaling input — no falloff term appears.
        expect(gradeBound(SAMPLE_STEP)).toBeGreaterThan(MAX_GRADE * SAMPLE_STEP);
        expect(gradeBound(2 * SAMPLE_STEP)).toBeGreaterThan(gradeBound(SAMPLE_STEP));
    });

    test("CROSS_SECTION_TOL and EDGE_EPSILON are small, fixed fractions of the mesh's own scale", () => {
        expect(CROSS_SECTION_TOL).toBeGreaterThan(0);
        expect(CROSS_SECTION_TOL).toBeLessThan(0.01); // quantization-scale, not a road-scale slack
        expect(EDGE_EPSILON).toBeGreaterThan(0);
        expect(EDGE_EPSILON).toBeLessThan(SPACING / 10);
    });
});

describe("reconstructionAgreement — the device arm's own fidelity pin", () => {
    test("the CPU device-free lattice agrees with an identically-parameterized CPU-built 'device' buffer", () => {
        // `bun test` has no real device, so this exercises the differential's own logic against a second,
        // independently-called `buildDeviceFreeVertices` run standing in for `readVertices()` — the real
        // device arm lives in `gate.ts` (`bun run gate`), which calls this same function against the real
        // GPU. Same seed in, so the two builds should be exactly reproducible (this project's
        // height kernel is itself deterministic-in-seed, `gate.ts`'s own `deterministic` check) — this
        // pins that {@link reconstructionAgreement}'s own comparison logic reads zero when there's nothing
        // to disagree about, before it's trusted as the real device's fidelity gate.
        const doc = generateNetwork();
        const deviceStandIn = buildDeviceFreeVertices(doc, SEED);
        const agreement = reconstructionAgreement(deviceStandIn, doc, SEED);
        expect(agreement.sampleCount).toBeGreaterThan(0);
        expect(agreement.maxDiffM).toBeLessThanOrEqual(RECONSTRUCTION_AGREEMENT_TOL);
    });
});

// Stage 17's exactness arm — the instrument that decides a round ships with the chord profile
// (spec Validation, "Surface flatness in the corridor"). The chord's affine target makes the in-corridor
// reconstruction error vanish identically on a non-overlapping network (barycentric interpolation
// reproduces an affine field exactly at any cell size and any road angle). Stage 1
// (`roads-interactive.md`) deleted "stage 18 arm (c)"'s hand-built OVERLAPPING pair — the discriminating
// proof that non-overlap is what buys exactness — along with the multi-road generator's own
// non-overlap-by-construction guarantee it was a foil against: one road cannot overlap, so there is no
// clearance/non-overlap machinery left in `overlay/network.ts` for that arm to prove anything about
// (`checks.md`'s hollowed-foil rule). This arm still exercises the surviving multi-road blend machinery
// in `networkCoreCpu` directly, over a hand-built non-overlapping synthetic network — that machinery
// generalizes over `NetworkSegment.road` even though the real generator only ever emits one road.

describe("surface flatness — stage 17 arm (a): synthetic non-overlapping network reads exactly zero", () => {
    // a hand-built NON-OVERLAPPING network: five straight roads at 30° heading (not 0°/45°/90°),
    // 200 m apart perpendicular to the heading, 200 m long. Chord endpoint heights are the natural
    // terrain heights (`heightAtCpu`). The 200 m spacing is well above the ~63 m clearance the chord's
    // own falloff demands (`computeFalloff(cutDepth) + halfWidth + FLAT_CORE_MARGIN`), so no two
    // primitives' falloff bands overlap, and the affine-exactness argument holds at every sampled
    // station.
    //
    // Stage 12: banded lattice, for the same reason as the real-generator pair above — the oracle samples
    // only inside the five footprints, and the arms' exact-zero assertion is what makes an over-narrow
    // band red rather than green. The multi-road geometry also exercises the band's per-segment capsule
    // union, which a single chord cannot.
    const Heading = Math.PI / 6; // 30° — non-axis- and non-45°-aligned
    const RoadSpacing = 200; // metres, perpendicular separation between adjacent roads
    const RoadLen = 200; // metres
    const RoadHw = 4; // halfWidth, matching the generator's ROAD_HALF_WIDTH
    const ux = Math.cos(Heading);
    const uz = Math.sin(Heading);
    const nx = -uz;
    const nz = ux;

    const polylines: Polyline[] = [];
    for (let i = -2; i <= 2; i++) {
        const cx = nx * i * RoadSpacing;
        const cz = nz * i * RoadSpacing;
        polylines.push({
            points: [
                [cx - ux * (RoadLen / 2), cz - uz * (RoadLen / 2)],
                [cx + ux * (RoadLen / 2), cz + uz * (RoadLen / 2)],
            ],
            halfWidth: RoadHw,
        });
    }
    const syntheticDoc: StrokeDocument = { polylines };

    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(syntheticDoc, SEED);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("exactly 0 violations and 0.0000 m on both axes at SPACING", () => {
        const coarseRaw = buildBandedLatticeVertices(SPACING, CELLS, segments, falloff, natural);
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(coarseRaw, x, z, SPACING, CELLS),
            syntheticDoc,
        );
        console.log(
            `SYNTH_SPACING crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} sampleCount=${result.sampleCount} cutDepth=${cutDepth.toFixed(4)} falloff=${falloff.toFixed(4)}`,
        );
        expect(result.sampleCount).toBeGreaterThan(1000);
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
        expect(result.maxLongitudinalExcess).toBe(0);
    });

    test("exactly 0 violations and 0.0000 m on both axes at SPACING/2", () => {
        const fineSpacing = SPACING / 2;
        const fineCells = CELLS * 2;
        const fineRaw = buildBandedLatticeVertices(
            fineSpacing,
            fineCells,
            segments,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(fineRaw, x, z, fineSpacing, fineCells),
            syntheticDoc,
        );
        console.log(
            `SYNTH_SPACING_HALF crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} sampleCount=${result.sampleCount}`,
        );
        expect(result.sampleCount).toBeGreaterThan(1000);
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
        expect(result.maxLongitudinalExcess).toBe(0);
    });
});
