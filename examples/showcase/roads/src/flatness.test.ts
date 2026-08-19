import { describe, expect, test } from "bun:test";
import { meshHeightAt } from "./capture";
import {
    buildDeviceFreeVertices,
    buildLatticeVertices,
    CROSS_SECTION_TOL,
    checkSurfaceFlatness,
    EDGE_EPSILON,
    gradeBound,
    RECONSTRUCTION_AGREEMENT_TOL,
    reconstructionAgreement,
    SAMPLE_STEP,
} from "./flatness";
import type { StrokeDocument } from "./overlay/document";
import { generateNetwork } from "./overlay/network";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";
import { CELLS, SPACING } from "./terrain/grid";
import { GROUND_LEVEL, makePermutation } from "./terrain/noise";
import { DEFAULT_SMOOTH_RADIUS, heightAtCpu, MAX_GRADE } from "./terrain/profile";
import { SEED } from "./terrain/terrain";

// Stage 12's proof procedure (spec Validation, "Surface flatness along the road"): all four arms run
// device-free against the real seeded network, printed as evidence for the coordinator's report as well as
// asserted here. Numbers are measured, not predicted — every bound below is read off an actual run
// (`__explore.test.ts`, deleted before commit) rather than picked in advance, with margin for run-to-run
// float noise, never fitted so tight a legitimate reading could flip the assertion.

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
            polygons: [],
        };
        const segments = [{ ax: -50, az: 0, bx: 50, bz: 0, halfWidth: 4, aHeight: 5, bHeight: 5 }];
        const raw = buildLatticeVertices(SPACING, CELLS, segments, [], 16, () => 5);
        const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);
        expect(result.longitudinal.length).toBe(0);
        expect(result.crossSection.length).toBe(0);
    });

    test("flat natural terrain with no cut at all reads within tolerance — the RELIEF=0 analogue (arm ii)", () => {
        // undeformed (GROUND_LEVEL everywhere) natural terrain, no flatten target to blend toward
        // (empty segments/polygons) — the device-side control is `VITE_ROADS_RELIEF=0` (env-gated, so it
        // can only be exercised through the real Vite/browser build, `env.ts`'s own module comment); this
        // is its device-free analogue over the real network's own footprint geometry.
        const doc = generateNetwork(SEED);
        const raw = buildLatticeVertices(SPACING, CELLS, [], [], 16, () => GROUND_LEVEL);
        const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);
        expect(result.longitudinal.length).toBe(0);
        expect(result.crossSection.length).toBe(0);
    });
});

describe("surface flatness — shipped pipeline at SEED=1337 (arm i, the unit's live defect)", () => {
    const doc = generateNetwork(SEED);
    const raw = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS);
    const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);

    test("reads red on both axes — cross-section and longitudinal both violate", () => {
        // measured (2026-08-18, this worktree, after the endpoint-margin fix — arc-length halfWidth, not
        // a fraction of segment length): 34 longitudinal violations (max delta ~0.344 m against a ~0.14 m
        // bound) and 1117 cross-section violations (max delta ~0.471 m against a ~0.0024 m bound) over
        // 2052 total footprint samples across the network's 5 roads. Both axes read red — the
        // reconstruction-axis defect isn't confined to the longitudinal profile the way stages 6-7's own
        // (different, vertex-only) flattening oracle stayed green on.
        console.log(
            `SURFACE_FLATNESS_SHIPPED longitudinal=${result.longitudinal.length} crossSection=${result.crossSection.length} sampleCount=${result.sampleCount} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(4)} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(4)}`,
        );
        expect(result.longitudinal.length).toBeGreaterThan(0);
        expect(result.crossSection.length).toBeGreaterThan(0);
    });

    test("violation magnitude sits an order of magnitude over tolerance, not at the noise floor", () => {
        const maxLongitudinalDelta = result.longitudinal.reduce((m, v) => Math.max(m, v.delta), 0);
        const maxCrossSectionDelta = result.crossSection.reduce(
            (m, v) => Math.max(m, v.deltaFromCentre),
            0,
        );
        // a loose floor (measured ~0.33 m / ~0.49 m) — not fitted tight to today's exact reading, just
        // ruling out "this is quantization noise wearing a violation's clothes." CROSS_SECTION_TOL is
        // ~2 mm and the longitudinal step bound ~0.1-0.15 m, so either reading landing here is already
        // 10-30x its own tolerance.
        expect(maxLongitudinalDelta).toBeGreaterThan(0.15);
        expect(maxCrossSectionDelta).toBeGreaterThan(0.15);
    });

    test("every violation sits inside the road footprint the document itself defines", () => {
        // the oracle only ever walks centreline/edge lines derived from `doc`'s own halfWidth — a
        // violation reported outside the document's footprint would mean the sampler drifted off the road
        // it claims to be checking.
        for (const v of [...result.longitudinal, ...result.crossSection]) {
            expect(v.roadIndex).toBeGreaterThanOrEqual(0);
            expect(v.roadIndex).toBeLessThan(doc.polylines.length);
        }
    });
});

describe("surface flatness — null control: no cut, real relief (arm iii)", () => {
    test("still finds a real signal — proves the instrument detects steps, not just 'nothing is flat'", () => {
        // the device-side control is `VITE_ROADS_NO_CUT=1` on real relief (`terrain.ts`'s own `NO_CUT`
        // env flag) — the flatten kernel is skipped entirely (empty segments/polygons, `flattenHeight`'s
        // own documented empty-network fallback degrades to plain `heightAt`), but the *document* used to
        // define the sampled footprint is still the real network — so the sampled lines run through
        // genuinely undeformed, rolling terrain (RELIEF=40) that was never asked to be flat. This is a
        // *different* mechanism from the shipped defect (raw noise steepness, not a reconstruction crease)
        // — it's the null control that still has something to find (`checks.md`'s witness-distinguishable
        // clause): a broken instrument that always reads "flat" would fail this arm the same way it would
        // pass arm ii, so the two arms together are what makes either one meaningful.
        const doc = generateNetwork(SEED);
        const perm = makePermutation(SEED);
        const raw = buildLatticeVertices(SPACING, CELLS, [], [], 16, (x, z) =>
            heightAtCpu(x, z, perm),
        );
        const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);
        console.log(
            `SURFACE_FLATNESS_NO_CUT longitudinal=${result.longitudinal.length} crossSection=${result.crossSection.length} sampleCount=${result.sampleCount}`,
        );
        // measured (2026-08-18, after the endpoint-margin fix): 283 longitudinal violations of 2052
        // samples, well above the shipped pipeline's own 34 — raw undeformed relief violates the grade
        // bound far more often than the flattened corridor does, exactly the "real signal" this control is
        // meant to prove exists.
        expect(result.longitudinal.length).toBeGreaterThan(100);
    });
});

describe("surface flatness — discrimination (arm iv)", () => {
    // The spec's own Validation criterion (`shallot-roads.md` stage 12) names *amplitude*, not count: "at
    // SPACING/2 the violation amplitude moves in the mechanism's predicted direction, and across
    // falloff-scale 1 vs 3 it moves ≈ not at all." An earlier version of these tests asserted on
    // violation *count* instead — a population statistic that moves whenever any small geometric
    // perturbation pushes samples across the bound, so it responds to perturbation-in-general rather than
    // to the specific mechanism this arm exists to isolate. Measured directly (below): falloff-scale moves
    // count by 23.5%/34.0% (longitudinal/cross-section) against mesh resolution's 38.2%/38.8% — comparable
    // orders, no clean separation. Amplitude (maxDelta) is what actually discriminates, and cross-section
    // is the axis where it does so cleanly; longitudinal amplitude moves the same *direction* but weakly,
    // so it's corroboration, not a second independent discriminator.
    //
    // "≈ not at all" vs "moves" is read here as *at least one order of magnitude apart* — a falloff-scale
    // effect within 10x of the mesh-resolution effect on the same statistic isn't negligible relative to
    // it, whatever its absolute size. That threshold is chosen from the spec's own contrastive language,
    // not fitted to today's readings (which come in at ~90x separation on cross-section amplitude, far
    // inside the 10x bar).

    test("cross-section amplitude: falloff-scale ~inert, mesh resolution moves — the primary discriminator", () => {
        const doc = generateNetwork(SEED);
        const raw1 = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS, 1);
        const raw3 = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS, 3);
        const r1 = checkSurfaceFlatness((x, z) => meshHeightAt(raw1, x, z), doc);
        const r3 = checkSurfaceFlatness((x, z) => meshHeightAt(raw3, x, z), doc);

        const crossDelta1 = r1.crossSection.reduce((m, v) => Math.max(m, v.deltaFromCentre), 0);
        const crossDelta3 = r3.crossSection.reduce((m, v) => Math.max(m, v.deltaFromCentre), 0);
        const longDelta1 = r1.longitudinal.reduce((m, v) => Math.max(m, v.delta), 0);
        const longDelta3 = r3.longitudinal.reduce((m, v) => Math.max(m, v.delta), 0);

        console.log(
            `SURFACE_FLATNESS_FALLOFF_SCALE scale1_longCount=${r1.longitudinal.length} scale1_longMaxDelta=${longDelta1.toFixed(4)} scale1_crossCount=${r1.crossSection.length} scale1_crossMaxDelta=${crossDelta1.toFixed(4)} scale3_longCount=${r3.longitudinal.length} scale3_longMaxDelta=${longDelta3.toFixed(4)} scale3_crossCount=${r3.crossSection.length} scale3_crossMaxDelta=${crossDelta3.toFixed(4)}`,
        );

        // measured (2026-08-18, after the endpoint-margin fix): cross-section maxDelta moves ~0.2%
        // (0.4737 -> 0.4730 m), longitudinal maxDelta ~1.1% (0.3436 -> 0.3399 m) across a 3x falloff
        // widening — this is the arm's primary discriminator; see the paired mesh-resolution test below
        // for the comparison that makes "~inert" meaningful (a percentage alone means nothing without the
        // other leg to compare against).
        expect(Math.abs(crossDelta3 - crossDelta1) / crossDelta1).toBeLessThan(0.1);
    });

    test("mesh resolution (SPACING/2) moves cross-section amplitude an order of magnitude more than falloff-scale — and violation count does NOT discriminate (negative result, pinned)", () => {
        const doc = generateNetwork(SEED);
        const perm = makePermutation(SEED);

        // falloff-scale leg, recomputed here (cheap, same SPACING) so this test stands on its own —
        // both legs need to sit side by side for the comparison and the count-negative-result pin.
        const raw1 = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS, 1);
        const raw3 = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS, 3);
        const r1 = checkSurfaceFlatness((x, z) => meshHeightAt(raw1, x, z), doc);
        const r3 = checkSurfaceFlatness((x, z) => meshHeightAt(raw3, x, z), doc);
        const falloffCrossDelta1 = r1.crossSection.reduce(
            (m, v) => Math.max(m, v.deltaFromCentre),
            0,
        );
        const falloffCrossDelta3 = r3.crossSection.reduce(
            (m, v) => Math.max(m, v.deltaFromCentre),
            0,
        );
        const falloffLongDelta1 = r1.longitudinal.reduce((m, v) => Math.max(m, v.delta), 0);
        const falloffLongDelta3 = r3.longitudinal.reduce((m, v) => Math.max(m, v.delta), 0);
        const falloffCrossCountChange =
            Math.abs(r3.crossSection.length - r1.crossSection.length) / r1.crossSection.length;
        const falloffLongCountChange =
            Math.abs(r3.longitudinal.length - r1.longitudinal.length) / r1.longitudinal.length;

        // mesh-resolution leg: segments/falloff held fixed at production values (falloffScale = 1),
        // only the reconstruction lattice's own spacing/cells vary — isolates the mesh-resolution axis
        // from any falloff-width change (the same split the original arm used).
        const { segments, cutDepth } = buildNetworkGeometry(doc, SEED, DEFAULT_SMOOTH_RADIUS);
        const falloff = computeFalloff(cutDepth, 1);
        const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

        const rawCoarse = buildLatticeVertices(
            SPACING,
            CELLS,
            segments,
            doc.polygons,
            falloff,
            natural,
        );
        const coarse = checkSurfaceFlatness(
            (x, z) => meshHeightAt(rawCoarse, x, z, SPACING, CELLS),
            doc,
        );

        const fineSpacing = SPACING / 2;
        const fineCells = CELLS * 2;
        const rawFine = buildLatticeVertices(
            fineSpacing,
            fineCells,
            segments,
            doc.polygons,
            falloff,
            natural,
        );
        const fine = checkSurfaceFlatness(
            (x, z) => meshHeightAt(rawFine, x, z, fineSpacing, fineCells),
            doc,
        );

        const spacingCrossDeltaCoarse = coarse.crossSection.reduce(
            (m, v) => Math.max(m, v.deltaFromCentre),
            0,
        );
        const spacingCrossDeltaFine = fine.crossSection.reduce(
            (m, v) => Math.max(m, v.deltaFromCentre),
            0,
        );
        const spacingLongDeltaCoarse = coarse.longitudinal.reduce(
            (m, v) => Math.max(m, v.delta),
            0,
        );
        const spacingLongDeltaFine = fine.longitudinal.reduce((m, v) => Math.max(m, v.delta), 0);
        const spacingCrossCountChange =
            Math.abs(fine.crossSection.length - coarse.crossSection.length) /
            coarse.crossSection.length;
        const spacingLongCountChange =
            Math.abs(fine.longitudinal.length - coarse.longitudinal.length) /
            coarse.longitudinal.length;

        console.log(
            `SURFACE_FLATNESS_SPACING coarse_longCount=${coarse.longitudinal.length} coarse_longMaxDelta=${spacingLongDeltaCoarse.toFixed(4)} coarse_crossCount=${coarse.crossSection.length} coarse_crossMaxDelta=${spacingCrossDeltaCoarse.toFixed(4)} fine_longCount=${fine.longitudinal.length} fine_longMaxDelta=${spacingLongDeltaFine.toFixed(4)} fine_crossCount=${fine.crossSection.length} fine_crossMaxDelta=${spacingCrossDeltaFine.toFixed(4)}`,
        );

        const falloffCrossAmpChange =
            Math.abs(falloffCrossDelta3 - falloffCrossDelta1) / falloffCrossDelta1;
        const spacingCrossAmpChange =
            Math.abs(spacingCrossDeltaFine - spacingCrossDeltaCoarse) / spacingCrossDeltaCoarse;
        const falloffLongAmpChange =
            Math.abs(falloffLongDelta3 - falloffLongDelta1) / falloffLongDelta1;
        const spacingLongAmpChange =
            Math.abs(spacingLongDeltaFine - spacingLongDeltaCoarse) / spacingLongDeltaCoarse;

        // measured (2026-08-18, after the endpoint-margin fix): cross-section amplitude moves ~19.2%
        // under SPACING/2 against falloff-scale's own ~0.2% on the same statistic — an order of
        // magnitude the "≈ not at all" vs "moves" bar (defined above) reads as a real discrimination,
        // not noise on either side.
        expect(falloffCrossAmpChange * 10).toBeLessThan(spacingCrossAmpChange);

        // longitudinal amplitude moves the same *direction* (falloff-scale smaller than mesh
        // resolution) but weakly — ~1.1% vs ~3.1%, roughly 3x apart, not the order-of-magnitude split
        // cross-section shows. Asserted only as a directional ordering, per the coordinator's own
        // read: "1.1% vs 3.1% is not a clean split" — this is corroboration that the direction is
        // right, not a second independent discriminator standing on its own.
        expect(falloffLongAmpChange).toBeLessThan(spacingLongAmpChange);

        // Negative result, pinned so a later stage (13's removal gate, 15's fix gate) can't cite
        // violation *count* as a comparable-across-treatments reading the way this arm's own history
        // briefly did: on both axes, falloff-scale's count change sits within ~15 percentage points of
        // mesh-resolution's own count change (23.5% vs 38.2% longitudinal, 34.0% vs 38.8%
        // cross-section) — the same population-noise ballpark as the treatment's own effect, not an
        // order of magnitude apart the way amplitude is. 20 points is a coarse "same ballpark" bound,
        // chosen because it is far short of the multiplicative order-of-magnitude gap amplitude shows
        // above (a 20-point *absolute* gap on a same-order base is nowhere near a 10x *relative* one) —
        // if this ever tightens to a real order-of-magnitude gap, count would have become a
        // discriminator after all and this assertion should fail loudly, not be widened to keep it
        // green.
        expect(Math.abs(falloffLongCountChange - spacingLongCountChange)).toBeLessThan(0.2);
        expect(Math.abs(falloffCrossCountChange - spacingCrossCountChange)).toBeLessThan(0.2);
    }, 20_000);
});

describe("checkSurfaceFlatness — window/threshold derivation, no candidate treatment", () => {
    test("gradeBound scales with Δs and the road-design grade limit alone", () => {
        // MAX_GRADE is the only slope-scaling input — no falloff/smoothRadius/falloffScale term appears.
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
        // GPU. Same seed/smoothRadius/falloffScale in, so the two builds should be exactly reproducible
        // (this project's height kernel is itself deterministic-in-seed, `gate.ts`'s own `deterministic`
        // check) — this pins that {@link reconstructionAgreement}'s own comparison logic reads zero
        // when there's nothing to disagree about, before it's trusted as the real device's fidelity gate.
        const doc = generateNetwork(SEED);
        const deviceStandIn = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS, 1);
        const agreement = reconstructionAgreement(
            deviceStandIn,
            doc,
            SEED,
            DEFAULT_SMOOTH_RADIUS,
            1,
        );
        expect(agreement.sampleCount).toBeGreaterThan(0);
        expect(agreement.maxDiffM).toBeLessThanOrEqual(RECONSTRUCTION_AGREEMENT_TOL);
    });
});
