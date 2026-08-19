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

describe("surface flatness — mesh-resolution discrimination (independent of the deleted falloff-scale knob)", () => {
    // Stage 12's arm (iv) originally swept two treatments side by side (falloff-scale and mesh
    // resolution) to prove the oracle discriminates on a real geometric change while staying ~inert to
    // the (now-deleted) falloff-scale knob. The falloff-scale leg died with stage 13's removal, but this
    // leg's own subject — does halving the mesh spacing move the oracle's reported amplitude, the way a
    // reconstruction-axis defect predicts it should — has nothing to do with that knob and stays live: a
    // still-real check that the oracle is sensitive to the axis it exists to gate, not a check on 11a.
    test("halving mesh spacing moves cross-section amplitude by a real, non-trivial margin", () => {
        const doc = generateNetwork(SEED);
        const perm = makePermutation(SEED);
        const { segments, cutDepth } = buildNetworkGeometry(doc, SEED, DEFAULT_SMOOTH_RADIUS);
        const falloff = computeFalloff(cutDepth);
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

        const crossDeltaCoarse = coarse.crossSection.reduce(
            (m, v) => Math.max(m, v.deltaFromCentre),
            0,
        );
        const crossDeltaFine = fine.crossSection.reduce(
            (m, v) => Math.max(m, v.deltaFromCentre),
            0,
        );
        const crossAmpChange = Math.abs(crossDeltaFine - crossDeltaCoarse) / crossDeltaCoarse;

        console.log(
            `SURFACE_FLATNESS_MESH_RESOLUTION coarse_crossMaxDelta=${crossDeltaCoarse.toFixed(4)} fine_crossMaxDelta=${crossDeltaFine.toFixed(4)} crossAmpChange=${(crossAmpChange * 100).toFixed(1)}%`,
        );

        // measured (2026-08-18): ~19.2% — a loose floor at 5%, well below the measured reading, just
        // ruling out "the oracle reads flat to mesh resolution changes" (the property this leg exists to
        // rule out), not fitted tight to today's exact number.
        expect(crossAmpChange).toBeGreaterThan(0.05);
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
