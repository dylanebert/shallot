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
import type { PolygonStamp, Polyline, StrokeDocument } from "./overlay/document";
import { generateNetwork } from "./overlay/network";
import { buildNetworkGeometry, computeFalloff, FLAT_CORE_MARGIN } from "./terrain/flatten";
import { CELLS, SPACING } from "./terrain/grid";
import { GROUND_LEVEL, makePermutation } from "./terrain/noise";
import { heightAtCpu, MAX_GRADE } from "./terrain/profile";
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
    //
    // Stage 18: the real generator no longer produces overlaps, so the red-first arm uses a hand-built
    // overlapping pair (two parallel roads 15 m apart at 30° heading — close enough that their falloff
    // bands overlap and the edge+ is outside the junction zone, so violations are not excluded,
    // but far enough that the suppression with band = falloff doesn't fully kill the distant road,
    // so the blend is a position-weighted combination of two affine fields and is not affine). The
    // green arm uses the real non-overlapping `generateNetwork(SEED)`, which reads exactly zero.
    const overlapHeading = Math.PI / 6; // 30° — non-axis- and non-45°-aligned
    const overlapSep = 15; // metres, perpendicular separation (centreline-to-centreline)
    const oux = Math.cos(overlapHeading);
    const ouz = Math.sin(overlapHeading);
    const onx = -ouz;
    const onz = oux;
    const ocx2 = onx * overlapSep;
    const ocz2 = onz * overlapSep;
    const overlapDoc: StrokeDocument = {
        polylines: [
            {
                points: [
                    [-oux * 100, -ouz * 100],
                    [oux * 100, ouz * 100],
                ],
                halfWidth: 4,
            },
            {
                points: [
                    [ocx2 - oux * 100, ocz2 - ouz * 100],
                    [ocx2 + oux * 100, ocz2 + ouz * 100],
                ],
                halfWidth: 4,
            },
        ],
        polygons: [],
    };
    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(overlapDoc, SEED);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("red-first: band = falloff still reads non-zero on a hand-built overlapping pair", () => {
        // the suppression band set to the falloff distance is too slow — primitives farther away still
        // contribute, so the blend contamination leaks outside the junction zone. The hand-built
        // overlapping pair (10 m apart) guarantees the falloff bands overlap, so the composite target
        // is a position-weighted combination of two affine fields and is not affine — exactness
        // dies there and nowhere else. This arm MUST stay red — it is the discriminator that proves
        // non-overlap is what buys exactness (a pin that reads green on a broken blend is worth
        // nothing — `coding.md`'s "a check is evidence only if you've seen it fail").
        const sample = (x: number, z: number) =>
            flattenFieldAt(x, z, segments, overlapDoc.polygons, falloff, natural, falloff);
        const result = checkSurfaceFlatness(sample, overlapDoc);
        console.log(
            `LEG_A_FALLOFF_CHORD crossSection=${result.crossSection.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)}`,
        );
        expect(result.crossSection.length).toBeGreaterThan(100);
        expect(result.maxCrossSectionExcess).toBeGreaterThan(0.1);
    });

    test("the shipped band reads exactly zero on the real (non-overlapping) generator", () => {
        // the shipped suppression band (`FLAT_CORE_MARGIN`): on the real non-overlapping generator
        // the host primitive's target wins outright by construction, so the continuous field is
        // exactly flat — zero violations, zero amplitude on both axes. This is the structural claim
        // 15b's consult made and the leg that gates the blend's design; it carries no fitted
        // number, only the exact zero.
        const realDoc = generateNetwork(SEED);
        const { segments: realSegs, cutDepth: realCutDepth } = buildNetworkGeometry(realDoc, SEED);
        const realFalloff = computeFalloff(realCutDepth);
        const sample = (x: number, z: number) =>
            flattenFieldAt(
                x,
                z,
                realSegs,
                realDoc.polygons,
                realFalloff,
                natural,
                FLAT_CORE_MARGIN,
            );
        const result = checkSurfaceFlatness(sample, realDoc);
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
    const raw = buildDeviceFreeVertices(doc, SEED);
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
        // construction). This stays true when the junction compromise improves to zero — a dirt-demand pin
        // (`crossSectionInZone.length > 0` / `maxInZone > 0.1`) would break if the compromise improved to
        // zero, but the partition invariant holds either way.
        for (const v of result.crossSectionInZone) {
            expect(inJunctionZone(v.x, v.z, doc)).toBe(true);
        }
        for (const v of result.crossSection) {
            expect(inJunctionZone(v.x, v.z, doc)).toBe(false);
        }
        // Stage 18: the non-overlapping generator eliminates all junction zones, so both buckets
        // are empty — the union is vacuously complete (0 + 0 = 0). The invariant holds trivially.
        expect(result.crossSection.length + result.crossSectionInZone.length).toBe(0);
    });

    test("the exclusion extent is a document-only property, invariant under FLAT_CORE_MARGIN", () => {
        // the carve-out's extent (excluded-station count + fraction) is reported beside `sampleCount` —
        // an exclusion is a deletion primitive (`checks.md`), measured in the diff that adds it. The
        // extent is a document-only property: `inJunctionZone` reads `doc`'s geometry and `JUNCTION_ZONE`
        // alone (never `FLAT_CORE_MARGIN` — the margin cancels in `second − nearest` and is deleted from
        // the core terms), so the exclusion fraction is invariant under a future treatment widening the
        // margin. The document-only prediction is ≈15% of sampled stations (read off the null control
        // falling 283 → 241 under the carve-out alone, ≈14.8% of longitudinal comparisons).
        // Stage 18: the non-overlapping generator eliminates all junction zones, so the exclusion
        // extent is exactly zero — `excludedStationCount === 0` is the structural precondition for
        // the exact-zero claim (the carve-out is a deletion primitive, `checks.md`).
        expect(result.excludedStationCount).toBe(0);
        expect(result.excludedStationFraction).toBe(0);
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

describe("surface flatness — stage 18 arm (b): real generator reads exactly zero at both resolutions", () => {
    // The real-generator exactness arm (spec Validation, "Surface flatness in the corridor — exactly zero,
    // unconditional"): `checkSurfaceFlatness` over `buildLatticeVertices` on the real `generateNetwork(SEED)`
    // reads exactly 0 violations / 0.0000 m on both axes, at `SPACING` and at `SPACING/2`. This is the
    // reading the unit still owes and which no green so far licenses. The non-overlapping generator
    // guarantees no two primitives' falloff bands overlap at any sampled station, so the composite
    // target is affine everywhere the oracle samples, and barycentric interpolation reproduces an affine
    // field exactly at any cell size and any road angle. Both a `sampleCount > 1000` vacuity guard and
    // the `excludedStationCount === 0` pin are asserted at both resolutions, so an emptied population
    // reds instead of passing on empty arrays.
    const doc = generateNetwork(SEED);
    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(doc, SEED);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("exactly 0 violations and 0.0000 m on both axes at SPACING", () => {
        const coarseRaw = buildLatticeVertices(
            SPACING,
            CELLS,
            segments,
            doc.polygons,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(coarseRaw, x, z, SPACING, CELLS),
            doc,
        );
        console.log(
            `REAL_EXACTNESS_SPACING crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} sampleCount=${result.sampleCount} excludedStationCount=${result.excludedStationCount} cutDepth=${cutDepth.toFixed(4)} falloff=${falloff.toFixed(4)}`,
        );
        expect(result.sampleCount).toBeGreaterThan(1000);
        // Stage-18 repair: pin the sampleCount at the widened endpointMargin (halfWidth + √2·SPACING)
        // so a future change that silently shrinks the population reds instead of passing on a
        // thinner array. Old margin (halfWidth): 2031; new margin: 1860 at both resolutions.
        expect(result.sampleCount).toBe(1860);
        expect(result.excludedStationCount).toBe(0);
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
        expect(result.maxLongitudinalExcess).toBe(0);
    });

    test("exactly 0 violations and 0.0000 m on both axes at SPACING/2", () => {
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
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(fineRaw, x, z, fineSpacing, fineCells),
            doc,
        );
        console.log(
            `REAL_EXACTNESS_SPACING_HALF crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} sampleCount=${result.sampleCount} excludedStationCount=${result.excludedStationCount}`,
        );
        expect(result.sampleCount).toBeGreaterThan(1000);
        // Stage-18 repair: pin the sampleCount at the widened endpointMargin (halfWidth + √2·SPACING).
        // Old margin (halfWidth): 2031; new margin: 1860 at both resolutions.
        expect(result.sampleCount).toBe(1860);
        expect(result.excludedStationCount).toBe(0);
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
        expect(result.maxLongitudinalExcess).toBe(0);
    });

    // R2 (stage-18 repair): the exactness arm ran at one seed (SEED = 1337), so "by construction"
    // was untested across seeds. The boot seed keeps its full both-resolutions reading above; this
    // scan exercises additional seeds at SPACING, asserting exactly 0 violations and 0 max excess on
    // both axes. The seed count is the largest that keeps the suite comfortably fast — measured at
    // ~360 ms/seed (building the full lattice + checkSurfaceFlatness), 8 seeds adds ~2.9 s to the
    // suite's ~8.8 s baseline. The seeds are a fixed spread chosen to include the 5 seeds the pre-fix
    // scan found violated (1522, 1596, 1818, 2558, 3076 — the endpoint-clamping defect) so the fix is
    // exercised, not just the clean seeds.
    test("R2: a multi-seed scan at SPACING reads exactly 0 on both axes across 8 additional seeds", () => {
        const scanSeeds = [42, 1522, 1596, 1818, 2558, 3076, 5000, 7000];
        for (const seed of scanSeeds) {
            const scanDoc = generateNetwork(seed);
            const scanPerm = makePermutation(seed);
            const { segments: scanSegs, cutDepth: scanCutDepth } = buildNetworkGeometry(
                scanDoc,
                seed,
            );
            const scanFalloff = computeFalloff(scanCutDepth);
            const scanNatural = (x: number, z: number) => heightAtCpu(x, z, scanPerm);
            const scanRaw = buildLatticeVertices(
                SPACING,
                CELLS,
                scanSegs,
                scanDoc.polygons,
                scanFalloff,
                scanNatural,
            );
            const result = checkSurfaceFlatness(
                (x, z) => meshHeightAt(scanRaw, x, z, SPACING, CELLS),
                scanDoc,
            );
            expect(
                result.crossSection.length,
                `seed ${seed}: crossSection=${result.crossSection.length} (expected 0)`,
            ).toBe(0);
            expect(
                result.maxCrossSectionExcess,
                `seed ${seed}: maxCrossSectionExcess=${result.maxCrossSectionExcess} (expected 0)`,
            ).toBe(0);
            expect(
                result.longitudinal.length,
                `seed ${seed}: longitudinal=${result.longitudinal.length} (expected 0)`,
            ).toBe(0);
            expect(
                result.maxLongitudinalExcess,
                `seed ${seed}: maxLongitudinalExcess=${result.maxLongitudinalExcess} (expected 0)`,
            ).toBe(0);
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
        // GPU. Same seed in, so the two builds should be exactly reproducible (this project's
        // height kernel is itself deterministic-in-seed, `gate.ts`'s own `deterministic` check) — this
        // pins that {@link reconstructionAgreement}'s own comparison logic reads zero when there's nothing
        // to disagree about, before it's trusted as the real device's fidelity gate.
        const doc = generateNetwork(SEED);
        const deviceStandIn = buildDeviceFreeVertices(doc, SEED);
        const agreement = reconstructionAgreement(deviceStandIn, doc, SEED);
        expect(agreement.sampleCount).toBeGreaterThan(0);
        expect(agreement.maxDiffM).toBeLessThanOrEqual(RECONSTRUCTION_AGREEMENT_TOL);
    });
});

// Stage 17's two exactness arms — the instrument that decides a round ships with the chord profile
// (spec Validation, "Surface flatness in the corridor"). The chord's affine target makes the in-corridor
// reconstruction error vanish identically on a non-overlapping network (barycentric interpolation
// reproduces an affine field exactly at any cell size and any road angle), and the overlap arm is the
// discriminating proof that non-overlap is what buys exactness: where two primitives' falloffs overlap,
// the composite target is a position-weighted combination of two affine fields and is not affine, so
// exactness dies there and nowhere else.

describe("surface flatness — stage 17 arm (a): synthetic non-overlapping network reads exactly zero", () => {
    // a hand-built NON-OVERLAPPING network: five straight roads at 30° heading (not 0°/45°/90°),
    // 200 m apart perpendicular to the heading, 200 m long, plus one 32 m carpark square clear of all
    // roads. Chord endpoint heights are the natural terrain heights (`heightAtCpu`). The 200 m spacing
    // is well above the ~63 m clearance the chord's own falloff demands (`computeFalloff(cutDepth) +
    // halfWidth + FLAT_CORE_MARGIN`), so no two primitives' falloff bands overlap — the junction zone is
    // empty by construction, and the affine-exactness argument holds at every sampled station.
    const Heading = Math.PI / 6; // 30° — non-axis- and non-45°-aligned
    const RoadSpacing = 200; // metres, perpendicular separation between adjacent roads
    const RoadLen = 200; // metres
    const RoadHw = 4; // halfWidth, matching the generator's ROAD_HALF_WIDTH
    const CarparkHalf = 16; // 32 m carpark square, half-extent
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
    // carpark at (450, 450) — far from every road (nearest road centre is ~516 m away)
    const polygons: PolygonStamp[] = [
        {
            points: [
                [450 - CarparkHalf, 450 - CarparkHalf],
                [450 + CarparkHalf, 450 - CarparkHalf],
                [450 + CarparkHalf, 450 + CarparkHalf],
                [450 - CarparkHalf, 450 + CarparkHalf],
            ],
        },
    ];
    const syntheticDoc: StrokeDocument = { polylines, polygons };

    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(syntheticDoc, SEED);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("the junction zone is empty — no two primitives' cores come within JUNCTION_ZONE", () => {
        // with 200 m spacing and the carpark 516 m from the nearest road, no sampled station falls
        // inside a junction zone — `excludedStationCount === 0` is the structural precondition for the
        // exact-zero claim (the carve-out is a deletion primitive, `checks.md`).
        const coarseRaw = buildLatticeVertices(
            SPACING,
            CELLS,
            segments,
            polygons,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(coarseRaw, x, z, SPACING, CELLS),
            syntheticDoc,
        );
        console.log(
            `SYNTH_SPACING crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} excludedStationCount=${result.excludedStationCount} sampleCount=${result.sampleCount} cutDepth=${cutDepth.toFixed(4)} falloff=${falloff.toFixed(4)}`,
        );
        expect(result.excludedStationCount).toBe(0);
    });

    test("exactly 0 violations and 0.0000 m on both axes at SPACING", () => {
        const coarseRaw = buildLatticeVertices(
            SPACING,
            CELLS,
            segments,
            polygons,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(coarseRaw, x, z, SPACING, CELLS),
            syntheticDoc,
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
        const fineRaw = buildLatticeVertices(
            fineSpacing,
            fineCells,
            segments,
            polygons,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(fineRaw, x, z, fineSpacing, fineCells),
            syntheticDoc,
        );
        console.log(
            `SYNTH_SPACING_HALF crossSection=${result.crossSection.length} longitudinal=${result.longitudinal.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)} excludedStationCount=${result.excludedStationCount} sampleCount=${result.sampleCount}`,
        );
        expect(result.sampleCount).toBeGreaterThan(1000);
        expect(result.excludedStationCount).toBe(0);
        expect(result.crossSection.length).toBe(0);
        expect(result.maxCrossSectionExcess).toBe(0);
        expect(result.longitudinal.length).toBe(0);
        expect(result.maxLongitudinalExcess).toBe(0);
    });
});

describe("surface flatness — stage 18 arm (c): hand-built overlapping pair stays non-zero", () => {
    // The red-first overlap arm, rewritten for a generator that no longer produces overlaps: a hand-built
    // OVERLAPPING pair (two parallel roads 15 m apart at 30° heading — close enough that their falloff
    // bands overlap and the edge+ is outside the junction zone, so violations are not excluded).
    // This arm is the discriminating proof that non-overlap is what buys exactness: where two
    // primitives' falloffs overlap, `networkCoreCpu` blends their targets by a position-dependent
    // weight, so the composite target is a position-weighted combination of two affine fields and is
    // NOT affine. Barycentric interpolation cannot reproduce a non-affine field, so the reconstruction
    // error is non-zero exactly there and nowhere else. This arm MUST stay red — it is the evidence
    // that the exactness arm's zero is a property of non-overlap, not of the chord profile alone.
    const overlapHeading = Math.PI / 6; // 30° — non-axis- and non-45°-aligned
    const overlapSep = 15; // metres, perpendicular separation (centreline-to-centreline)
    const oux = Math.cos(overlapHeading);
    const ouz = Math.sin(overlapHeading);
    const onx = -ouz;
    const onz = oux;
    const ocx2 = onx * overlapSep;
    const ocz2 = onz * overlapSep;
    const overlapDoc: StrokeDocument = {
        polylines: [
            {
                points: [
                    [-oux * 100, -ouz * 100],
                    [oux * 100, ouz * 100],
                ],
                halfWidth: 4,
            },
            {
                points: [
                    [ocx2 - oux * 100, ocz2 - ouz * 100],
                    [ocx2 + oux * 100, ocz2 + ouz * 100],
                ],
                halfWidth: 4,
            },
        ],
        polygons: [],
    };
    const perm = makePermutation(SEED);
    const { segments, cutDepth } = buildNetworkGeometry(overlapDoc, SEED);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    test("non-zero at SPACING — cross-section violations and amplitude both clearly above zero", () => {
        const raw = buildLatticeVertices(
            SPACING,
            CELLS,
            segments,
            overlapDoc.polygons,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(raw, x, z, SPACING, CELLS),
            overlapDoc,
        );
        console.log(
            `OVERLAP_ARM_SPACING crossSection=${result.crossSection.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} longitudinal=${result.longitudinal.length} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)}`,
        );
        // band: well above zero, not fitted to the exact reading. The amplitude is the discriminating
        // statistic — it sat still at 0.2373 → 0.2356 m across the pre-chord → chord profile swap on the
        // old overlapping generator, proving the chord removed the non-junction reconstruction error
        // (count fell 362 → 99) without touching the overlap contamination (amplitude stayed).
        expect(result.crossSection.length).toBeGreaterThan(50);
        expect(result.maxCrossSectionExcess).toBeGreaterThan(0.1);
    });

    test("non-zero at SPACING/2 — amplitude roughly half of SPACING (convergence), still clearly above zero", () => {
        const fineSpacing = SPACING / 2;
        const fineCells = CELLS * 2;
        const raw = buildLatticeVertices(
            fineSpacing,
            fineCells,
            segments,
            overlapDoc.polygons,
            falloff,
            natural,
        );
        const result = checkSurfaceFlatness(
            (x, z) => meshHeightAt(raw, x, z, fineSpacing, fineCells),
            overlapDoc,
        );
        console.log(
            `OVERLAP_ARM_SPACING_HALF crossSection=${result.crossSection.length} maxCrossSectionExcess=${result.maxCrossSectionExcess.toFixed(5)} longitudinal=${result.longitudinal.length} maxLongitudinalExcess=${result.maxLongitudinalExcess.toFixed(5)}`,
        );
        expect(result.crossSection.length).toBeGreaterThan(20);
        expect(result.maxCrossSectionExcess).toBeGreaterThan(0.05);
    });
});
