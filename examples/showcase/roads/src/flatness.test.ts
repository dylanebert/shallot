import { describe, expect, test } from "bun:test";
import { meshHeightAt } from "./capture";
import {
    buildDeviceFreeVertices,
    buildLatticeVertices,
    CROSS_SECTION_TOL,
    checkSurfaceFlatness,
    EDGE_EPSILON,
    flattenFieldAt,
    gradeBound,
    inJunctionZone,
    JUNCTION_ZONE,
    RECONSTRUCTION_AGREEMENT_TOL,
    reconstructionAgreement,
    SAMPLE_STEP,
} from "./flatness";
import type { StrokeDocument } from "./overlay/document";
import { generateNetwork } from "./overlay/network";
import { buildNetworkGeometry, computeFalloff, FLAT_CORE_MARGIN } from "./terrain/flatten";
import { CELLS, SPACING } from "./terrain/grid";
import { GROUND_LEVEL, makePermutation } from "./terrain/noise";
import { DEFAULT_SMOOTH_RADIUS, heightAtCpu, MAX_GRADE } from "./terrain/profile";
import { SEED } from "./terrain/terrain";

// Stage 15c's pin legitimacy — the criterion 15b's numbers are read against (spec Validation, "Surface
// flatness along the road", 2026-08-19 second consult). No code under test changes behaviour; the pins
// are replaced with the two-leg criterion (Leg A: the field, Leg B: the mesh's convergence), the
// partition invariant, and the measured exclusion extent. Numbers are measured, not predicted — every
// bound below is read off an actual run, with margin for run-to-run float noise, never fitted so tight a
// legitimate reading could flip the assertion.

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
        const segments = [
            { ax: -50, az: 0, bx: 50, bz: 0, halfWidth: 4, aHeight: 5, bHeight: 5, road: 0 },
        ];
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

describe("surface flatness — Leg A: the continuous field, no mesh (arm v)", () => {
    // Leg A (spec Validation, 2026-08-19 second consult): the continuous flattened field —
    // `networkCoreCpu`'s blended target eased toward natural via `flattenHeight`, with no mesh at all.
    // Outside the designed junction zone the field is exactly flat by construction (15b's relative-depth
    // suppression makes the host primitive's target win outright), so a non-zero reading outside the zone
    // is the blend's design, not the mesh's discretization. This is the leg that gates the blend's design
    // and it carries no fitted number.

    const doc = generateNetwork(SEED);
    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(doc, SEED, DEFAULT_SMOOTH_RADIUS);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("red-first: band = falloff reads 417 / 0.493 m (the consult's refuted candidate)", () => {
        // the suppression band set to the falloff distance is too slow — primitives farther away still
        // contribute, so the blend contamination leaks outside the junction zone. The consult measured
        // 417 total cross-section violations / 0.493 m at the analytic limit; this reproduces that red
        // reading before the green arm, witnessing the instrument discriminates (a pin that reads green on
        // a broken blend is worth nothing — `coding.md`'s "a check is evidence only if you've seen it
        // fail").
        const sample = (x: number, z: number) =>
            flattenFieldAt(x, z, segments, doc.polygons, falloff, natural, falloff);
        const result = checkSurfaceFlatness(sample, doc);
        const total = result.crossSection.length + result.crossSectionInZone.length;
        console.log(
            `LEG_A_RED_FALLOFF crossSection=${result.crossSection.length} crossSectionInZone=${result.crossSectionInZone.length} total=${total} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(4)}`,
        );
        expect(result.crossSection.length).toBeGreaterThan(0);
        expect(result.maxCrossSectionExcess).toBeGreaterThan(0.4);
    });

    test("the shipped band reads exactly zero outside the junction zone on both axes", () => {
        // the shipped suppression band (`FLAT_CORE_MARGIN`): outside the junction zone the host
        // primitive's target wins outright by construction, so the continuous field is exactly flat —
        // zero violations, zero amplitude on both axes. This is the structural claim 15b's consult made
        // and the leg that gates the blend's design; it carries no fitted number, only the exact zero.
        const sample = (x: number, z: number) =>
            flattenFieldAt(x, z, segments, doc.polygons, falloff, natural, FLAT_CORE_MARGIN);
        const result = checkSurfaceFlatness(sample, doc);
        console.log(
            `LEG_A_GREEN longitudinal=${result.longitudinal.length} crossSection=${result.crossSection.length} maxCrossSectionExcess=${result.maxCrossSectionExcess} maxLongitudinalExcess=${result.maxLongitudinalExcess} crossSectionInZone=${result.crossSectionInZone.length}`,
        );
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
    });
});

describe("surface flatness — shipped pipeline at SEED=1337 (arm i, stage 15b)", () => {
    const doc = generateNetwork(SEED);
    const raw = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS);
    const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);

    test("the readings are reported (no fitted bound — Leg B gates the mesh, Leg A gates the field)", () => {
        // 15c: the fitted bounds (`longitudinal.length < 20`, `maxCrossSectionExcess < 0.3`) are deleted
        // — they were written to 15b's own reading and would defend the miss once the blend improves. The
        // mesh residual outside the zone is gated by Leg B's convergence assertion (amplitude ratio +
        // count decrease), not by an absolute bound (Blocker 3: any honest bound is ≥ MAX_GRADE·SPACING =
        // 0.48 m and reads stage 12's founding 0.471 m defect green). The readings are logged as evidence.
        console.log(
            `SURFACE_FLATNESS_SHIPPED longitudinal=${result.longitudinal.length} crossSection=${result.crossSection.length} crossSectionInZone=${result.crossSectionInZone.length} sampleCount=${result.sampleCount} excludedStationCount=${result.excludedStationCount} excludedStationFraction=${result.excludedStationFraction.toFixed(4)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(4)} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(4)} maxCrossSectionExcessInZone=${result.maxCrossSectionExcessInZone.toFixed(4)}`,
        );
        // no absolute out-of-zone amplitude bound is admissible (Blocker 3) — the readings are reported,
        // not gated here. Leg A gates the field (exactly zero), Leg B gates the mesh (convergence).
    });

    test("the partition invariant: zone membership matches the carve-out's classification", () => {
        // the carve-out partitions cross-section violations into out-of-zone (`crossSection`) and in-zone
        // (`crossSectionInZone`). The partition invariant: every in-zone member satisfies `inJunctionZone`,
        // every out-of-zone member does not, and the union equals the un-partitioned run (the oracle
        // checks every station and pushes each violation to exactly one bucket, so the union is complete by
        // construction). This stays true when the junction compromise improves — a dirt-demand pin
        // (`crossSectionInZone.length > 0` / `maxInZone > 0.1`) would break if the compromise improved to
        // zero, but the partition invariant holds either way.
        for (const v of result.crossSectionInZone) {
            expect(inJunctionZone(v.x, v.z, doc)).toBe(true);
        }
        for (const v of result.crossSection) {
            expect(inJunctionZone(v.x, v.z, doc)).toBe(false);
        }
        // the union is complete: every cross-section violation is in exactly one bucket (by construction
        // — the oracle's loop checks every station and partitions). The total is non-zero: the shipped
        // mesh reads real violations on both sides of the carve-out.
        expect(result.crossSection.length + result.crossSectionInZone.length).toBeGreaterThan(0);
    });

    test("the exclusion extent is a document-only property, invariant under FLAT_CORE_MARGIN", () => {
        // the carve-out's extent (excluded-station count + fraction) is reported beside `sampleCount` —
        // an exclusion is a deletion primitive (`checks.md`), measured in the diff that adds it. The
        // extent is a document-only property: `inJunctionZone` reads `doc`'s geometry and `JUNCTION_ZONE`
        // alone (never `FLAT_CORE_MARGIN` — the margin cancels in `second − nearest` and is deleted from
        // the core terms), so the exclusion fraction is invariant under a future treatment widening the
        // margin. The document-only prediction is ≈15% of sampled stations (read off the null control
        // falling 283 → 241 under the carve-out alone, ≈14.8% of longitudinal comparisons).
        expect(result.excludedStationCount).toBeGreaterThan(0);
        expect(result.excludedStationFraction).toBeGreaterThan(0.1);
        expect(result.excludedStationFraction).toBeLessThan(0.2);
    });

    test("every violation (including in-zone) sits inside the road footprint the document defines", () => {
        // the oracle only ever walks centreline/edge lines derived from `doc`'s own halfWidth — a
        // violation reported outside the document's footprint would mean the sampler drifted off the road
        // it claims to be checking. The containment check covers `crossSectionInZone` too, not just the
        // out-of-zone violations.
        for (const v of [
            ...result.longitudinal,
            ...result.crossSection,
            ...result.crossSectionInZone,
        ]) {
            expect(v.roadIndex).toBeGreaterThanOrEqual(0);
            expect(v.roadIndex).toBeLessThan(doc.polylines.length);
        }
    });
});

describe("surface flatness — Leg B: convergence (the real mesh outside the zone)", () => {
    // Leg B (spec Validation, 2026-08-19 second consult): the real mesh's residual outside the junction
    // zone is first-order reconstruction error of a C⁰ field over a `SPACING` cell, so the assertion is
    // convergence, never a bound — amplitude at `SPACING/2` falls to [0.25, 0.75]× its `SPACING` value and
    // both counts strictly decrease. No absolute out-of-zone amplitude bound is admissible (Blocker 3):
    // any bound derived from the reconstruction term is ≥ MAX_GRADE·SPACING = 0.48 m and reads stage 12's
    // own 0.471 m founding defect green, and anything under it is fitted to today's residual. The
    // mesh-resolution leg's fitted percentage floor (19.2% → 2% → 10%, three floors each set just under
    // its own round's fresh reading) is merged into this ratio band — derived from the mechanism's
    // convergence order and un-refittable.

    const doc = generateNetwork(SEED);
    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(doc, SEED, DEFAULT_SMOOTH_RADIUS);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    const coarseRaw = buildLatticeVertices(
        SPACING,
        CELLS,
        segments,
        doc.polygons,
        falloff,
        natural,
    );
    const coarse = checkSurfaceFlatness(
        (x, z) => meshHeightAt(coarseRaw, x, z, SPACING, CELLS),
        doc,
    );

    const fineSpacing = SPACING / 2;
    const fineCells = CELLS * 2;
    const fineRaw = buildLatticeVertices(
        fineSpacing,
        fineCells,
        segments,
        doc.polygons,
        falloff,
        natural,
    );
    const fine = checkSurfaceFlatness(
        (x, z) => meshHeightAt(fineRaw, x, z, fineSpacing, fineCells),
        doc,
    );

    test("amplitude at SPACING/2 is within [0.25, 0.75]× its SPACING value (convergence ratio band)", () => {
        // the residual is first-order in cell size: amplitude halves (roughly) when the cell halves. The
        // ratio band [0.25, 0.75] is derived from the mechanism's convergence order (a first-order term
        // over a C⁰ field), not fitted to today's reading — stage 15's blend-driven residual sat at ratio
        // ≈ 0.96 and fails it; today's 0.432 passes. The `Math.abs` is dropped: a finer mesh reading
        // *worse* (ratio > 1) would pass the old `Math.abs` form, hiding a divergence the convergence leg
        // exists to catch.
        const ratio = fine.maxCrossSectionExcess / coarse.maxCrossSectionExcess;
        console.log(
            `LEG_B_RATIO coarse_max=${coarse.maxCrossSectionExcess.toFixed(4)} fine_max=${fine.maxCrossSectionExcess.toFixed(4)} ratio=${ratio.toFixed(4)}`,
        );
        expect(ratio).toBeGreaterThanOrEqual(0.25);
        expect(ratio).toBeLessThanOrEqual(0.75);
    });

    test("both counts strictly decrease from SPACING to SPACING/2", () => {
        // convergence is monotone: halving the cell halves the violations on both axes. The counts are
        // pinned as strictly decreasing (not as absolute numbers — a count bound is a non-discriminating
        // statistic, `checks.md`'s amplitude-never-count clause, and `longitudinal.length < 20` violated
        // that at 15b).
        console.log(
            `LEG_B_COUNTS coarse_crossSection=${coarse.crossSection.length} fine_crossSection=${fine.crossSection.length} coarse_longitudinal=${coarse.longitudinal.length} fine_longitudinal=${fine.longitudinal.length}`,
        );
        expect(fine.crossSection.length).toBeLessThan(coarse.crossSection.length);
        expect(fine.longitudinal.length).toBeLessThan(coarse.longitudinal.length);
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
        // measured (2026-08-19, post-carve-out): 241 longitudinal violations of 2052 samples, well above
        // the shipped pipeline's own 11 — raw undeformed relief violates the grade bound far more often
        // than the flattened corridor does, exactly the "real signal" this control is meant to prove
        // exists. (Pre-carve-out the reading was 283; the junction-zone exclusion cut it to 241, also
        // cutting the sensitivity of the arm that polices the carve-out — a disclosed trade.)
        expect(result.longitudinal.length).toBeGreaterThan(100);
    });
});

describe("checkSurfaceFlatness — window/threshold derivation, no candidate treatment", () => {
    test("gradeBound scales with Δs and the road-design grade limit alone", () => {
        // MAX_GRADE is the only slope-scaling input — no falloff/smoothRadius term appears.
        expect(gradeBound(SAMPLE_STEP)).toBeGreaterThan(MAX_GRADE * SAMPLE_STEP);
        expect(gradeBound(2 * SAMPLE_STEP)).toBeGreaterThan(gradeBound(SAMPLE_STEP));
    });

    test("CROSS_SECTION_TOL and EDGE_EPSILON are small, fixed fractions of the mesh's own scale", () => {
        expect(CROSS_SECTION_TOL).toBeGreaterThan(0);
        expect(CROSS_SECTION_TOL).toBeLessThan(0.01); // quantization-scale, not a road-scale slack
        expect(EDGE_EPSILON).toBeGreaterThan(0);
        expect(EDGE_EPSILON).toBeLessThan(SPACING / 10);
    });

    test("JUNCTION_ZONE is the mesh cell diagonal, not a treatment quantity", () => {
        // the carve-out band reads `√2·SPACING` (the oracle's own mesh constant), never `FLAT_CORE_MARGIN`
        // — a future treatment widening the margin cannot silently widen the carve-out. The two coincide
        // by shared derivation, not by reading the same variable.
        expect(JUNCTION_ZONE).toBe(Math.SQRT2 * SPACING);
        expect(JUNCTION_ZONE).toBe(FLAT_CORE_MARGIN);
    });
});

describe("reconstructionAgreement — the device arm's own fidelity pin", () => {
    test("the CPU device-free lattice agrees with an identically-parameterized CPU-built 'device' buffer", () => {
        // `bun test` has no real device, so this exercises the differential's own logic against a second,
        // independently-called `buildDeviceFreeVertices` run standing in for `readVertices()` — the real
        // device arm lives in `gate.ts` (`bun run gate`), which calls this same function against the real
        // GPU. Same seed/smoothRadius in, so the two builds should be exactly reproducible (this project's
        // height kernel is itself deterministic-in-seed, `gate.ts`'s own `deterministic` check) — this
        // pins that {@link reconstructionAgreement}'s own comparison logic reads zero when there's nothing
        // to disagree about, before it's trusted as the real device's fidelity gate.
        const doc = generateNetwork(SEED);
        const deviceStandIn = buildDeviceFreeVertices(doc, SEED, DEFAULT_SMOOTH_RADIUS);
        const agreement = reconstructionAgreement(deviceStandIn, doc, SEED, DEFAULT_SMOOTH_RADIUS);
        expect(agreement.sampleCount).toBeGreaterThan(0);
        expect(agreement.maxDiffM).toBeLessThanOrEqual(RECONSTRUCTION_AGREEMENT_TOL);
    });
});
