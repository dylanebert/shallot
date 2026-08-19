import { describe, expect, test } from "bun:test";
import { buildNetworkGeometry, computeFalloff, FLAT_CORE_MARGIN } from "../terrain/flatten";
import { gridX, gridZ, SPACING, WORLD_HALF, worldX, worldZ } from "../terrain/grid";
import { DEFAULT_SMOOTH_RADIUS, PROFILE_STEP } from "../terrain/profile";
import { documentDirtyTiles, documentDistance, flattenSegments } from "./document";
import {
    captureProbePoints,
    generateNetwork,
    ROAD_COUNT,
    ROAD_HALF_WIDTH,
    roadFootprintDistance,
    roadPolygonFootprintDistance,
} from "./network";
import { tileId, tileOf } from "./tiles";

// this module's own copy of `terrain/terrain.ts`'s boot SEED — kept a plain literal rather than an
// import so this file's tests stay in the pure-data tier (no pull-in of terrain.ts's device-bound module
// graph); `capture.test.ts` and the device gate exercise the real import.
const BOOT_SEED = 1337;

// The network generator's determinism gate — the spec's own Validation criterion ("network determinism
// in its seed"). No device needed: `generateNetwork` is pure XZ placement (module header).

describe("generateNetwork — determinism in its seed", () => {
    test("the same seed returns an identical document, field for field", () => {
        const a = generateNetwork(42);
        const b = generateNetwork(42);
        expect(a).toEqual(b);
    });

    test("a different seed returns a different document", () => {
        const a = generateNetwork(1);
        const b = generateNetwork(2);
        expect(a).not.toEqual(b);
    });

    test("a run of ten distinct seeds never collides pairwise", () => {
        const docs = Array.from({ length: 10 }, (_, i) => generateNetwork(i + 1));
        for (let i = 0; i < docs.length; i++) {
            for (let j = i + 1; j < docs.length; j++) {
                expect(docs[i]).not.toEqual(docs[j]);
            }
        }
    });
});

describe("generateNetwork — shape", () => {
    test("a handful of roads, each a single straight segment, plus one carpark", () => {
        const doc = generateNetwork(7);
        expect(doc.polylines.length).toBe(ROAD_COUNT);
        for (const line of doc.polylines) {
            expect(line.points.length).toBe(2);
            expect(line.halfWidth).toBe(ROAD_HALF_WIDTH);
        }
        expect(doc.polygons.length).toBe(1);
        expect(doc.polygons[0].points.length).toBe(4);
    });

    test("every primitive stays within the world footprint over a wide seed scan (no clamped-AABB edge case)", () => {
        // a single hardcoded seed is a fixture, not a corpus — an earlier version of this
        // arm ran only seed 99, which happens not to trigger the escape a WORLD_MARGIN bounding only the
        // segment's *start* point allows: the endpoint is `x0 + cos(heading) * length` with `length` up
        // to ROAD_MAX_LENGTH in either direction, not half of it. Scan a real range, and pin the exact
        // seed (615) the adversarial pass demonstrated the escape with, so a regression names itself.
        const SeedScan = 5000;
        let worst = 0;
        let worstSeed = -1;
        for (let seed = 0; seed <= SeedScan; seed++) {
            const doc = generateNetwork(seed);
            for (const seg of flattenSegments(doc)) {
                for (const x of [seg.ax, seg.bx, seg.az, seg.bz]) {
                    if (Math.abs(x) > worst) {
                        worst = Math.abs(x);
                        worstSeed = seed;
                    }
                }
            }
            for (const [x, z] of doc.polygons[0].points) {
                for (const v of [x, z]) {
                    if (Math.abs(v) > worst) {
                        worst = Math.abs(v);
                        worstSeed = seed;
                    }
                }
            }
        }
        expect(
            worst,
            `worst abs coord: ${worst} WORLD_HALF: ${WORLD_HALF} seed: ${worstSeed}`,
        ).toBeLessThan(WORLD_HALF);
    });

    test("seed 615 specifically — the adversarial pass's own witness for the escape", () => {
        const doc = generateNetwork(615);
        for (const seg of flattenSegments(doc)) {
            for (const x of [seg.ax, seg.bx]) expect(Math.abs(x)).toBeLessThan(WORLD_HALF);
            for (const z of [seg.az, seg.bz]) expect(Math.abs(z)).toBeLessThan(WORLD_HALF);
        }
        for (const [x, z] of doc.polygons[0].points) {
            expect(Math.abs(x)).toBeLessThan(WORLD_HALF);
            expect(Math.abs(z)).toBeLessThan(WORLD_HALF);
        }
    });

    test("touches a real, non-empty dirty-tile set (documentDirtyTiles doesn't throw on it)", () => {
        const doc = generateNetwork(7);
        expect(() => documentDirtyTiles(doc)).not.toThrow();
        expect(documentDirtyTiles(doc).length).toBeGreaterThan(0);
    });
});

describe("captureProbePoints — the device gate's on/off-road pair over the procedural network", () => {
    test("at the boot seed: both points are grid-aligned, and classify as their names say", () => {
        const doc = generateNetwork(BOOT_SEED);
        const { onRoad, offRoad } = captureProbePoints(BOOT_SEED);

        // grid-aligned: worldX(gridX(x)) round-trips exactly for a point already sitting on the grid.
        expect(worldX(gridX(onRoad[0]))).toBe(onRoad[0]);
        expect(worldZ(gridZ(onRoad[1]))).toBe(onRoad[1]);
        expect(worldX(gridX(offRoad[0]))).toBe(offRoad[0]);
        expect(worldZ(gridZ(offRoad[1]))).toBe(offRoad[1]);

        expect(documentDistance(onRoad[0], onRoad[1], doc)).toBeLessThan(0);
        expect(documentDistance(offRoad[0], offRoad[1], doc)).toBeGreaterThan(0);
    });

    test("pinned witness at the boot seed — a regression names itself against these exact coordinates", () => {
        // computed once against generateNetwork(1337) and checked by hand against documentDistance
        // (on-road ≈ -2.84 m inside the road's 4 m half-width; off-road ≈ +3.34 m outside it, one grid
        // step further out — the nearest-to-origin road at this seed, not necessarily road 0).
        const { onRoad, offRoad } = captureProbePoints(BOOT_SEED);
        expect(onRoad).toEqual([-60, 84]);
        expect(offRoad).toEqual([-52, 80]);
    });

    test("is deterministic in the seed, like generateNetwork itself", () => {
        expect(captureProbePoints(99)).toEqual(captureProbePoints(99));
    });

    test("classifies correctly over a scan of seeds, not just the boot seed", () => {
        // ROAD_HALF_WIDTH/ROAD_MIN_LENGTH are fixed across every seed (network.ts), so the derivation's
        // inside/outside read shouldn't be a boot-seed coincidence — scan a real range rather than trust
        // one witness, the same lesson the network's own edge-case test above learned at seed 615: a
        // single hardcoded seed is a fixture, not a corpus.
        for (let seed = 0; seed < 200; seed++) {
            const doc = generateNetwork(seed);
            const { onRoad, offRoad } = captureProbePoints(seed);
            expect(documentDistance(onRoad[0], onRoad[1], doc)).toBeLessThan(0);
            expect(documentDistance(offRoad[0], offRoad[1], doc)).toBeGreaterThan(0);
        }
    });
});

describe("stage 14's device-arm reseed seeds — neither touches the boot network's on-road tile", () => {
    // `test/roads.spec.ts`'s reseed-integrity check (F9 twice, then read the boot network's own on-road
    // point) needs two fixed reseed seeds that don't coincidentally re-touch that exact tile — a real
    // road there after the swap would make a still-stale read pass for the wrong reason. Pinned here,
    // device-free, so a network.ts change that breaks the disjointness fails loud at this tier instead of
    // silently flaking the device gate.
    test("RESEED_SEED_A (111111) and RESEED_SEED_B (222222) both miss the boot on-road tile", () => {
        const { onRoad } = captureProbePoints(BOOT_SEED);
        const [tx, tz] = tileOf(onRoad[0], onRoad[1]);
        const bootOnRoadTile = tileId(tx, tz);

        for (const seed of [111111, 222222]) {
            const ids = documentDirtyTiles(generateNetwork(seed));
            expect(ids).not.toContain(bootOnRoadTile);
        }
    });
});

describe("stage 18 — non-overlap by construction: footprint disjointness over a seed scan", () => {
    // The spec's "Non-overlap by construction" criterion (Validation): over a seed scan 0–5000,
    // no two primitives' footprints come within the derived non-contamination bound, and every
    // primitive stays inside `WORLD_HALF`. A single seed is inadmissible — stage 6's own live
    // defect (a road 60 m outside the world at seed 615) was invisible to a single-seed footprint
    // arm. The clearance is derived from the network's own cut depth (measured between chord
    // endpoints), never a hardcoded constant — a deeper cut widens both the falloff and the required
    // separation.
    //
    // R1 (stage-18 repair): the arm asserts the derived non-contamination BOUND, not the generator's
    // own enforcement constant. A road contributes to the blend while its centreline distance is
    // < F + h + FCM (`networkCore`: coreDist = dist_to_centreline − h − FCM, active while coreDist < F).
    // `checkSurfaceFlatness` samples the centreline and both edge-offset lines, and the piecewise-linear
    // reconstruction at a sample point interpolates the enclosing triangle's vertices, which can sit up
    // to one cell diagonal √2·SPACING outside the footprint edge. So a contributing vertex can sit at
    // centreline distance ≥ D − h − √2·SPACING (D = centreline separation). Non-contamination needs
    // D − h − √2·SPACING ≥ F + h + FCM, i.e. the footprint-edge clearance R = D − 2h ≥
    // F + FCM + √2·SPACING — the bound. The generator enforces R = F + h + FCM + √2·SPACING (the
    // bound plus h = 4 m of slack); the arm asserts the bound itself, so the check is a claim about
    // the mechanism rather than a restatement of the constant the generator also uses. √2·SPACING is
    // the COARSE lattice's diagonal (the coarser lattice is the weaker case, so it bounds both
    // SPACING and SPACING/2).
    //
    // R3 (stage-18 repair): the seed loop runs inside a named `test()` so a failing seed is
    // attributable, appears in the test count, and is selectable with `-t`.
    //
    // R4 (stage-18 repair): `worstContainment` tracks the minimum containment SLACK
    // (WORLD_HALF − max|coord|), not the minimum |coord| (which reads 0.00 for a road through the
    // world centre and is meaningless).
    //
    // R5 (stage-18 repair): the diagnostics print the minimum per-seed SLACK (dist − required_seed,
    // minimised over seeds and pairs) with its seed and pair, so the line reads as a margin rather
    // than an incomparable global-minimum clearance against the boot seed's requirement.
    //
    // R6 (stage-18 repair): the road–carpark assertion accounts for the PROFILE_STEP sampling error.
    // `segmentToPolygonDistance` samples the segment at PROFILE_STEP intervals, so it overestimates
    // the true distance by up to PROFILE_STEP/2 (`polygonDistance` is 1-Lipschitz, so PROFILE_STEP/2
    // is a bound, not a guess). The arm asserts `sampled ≥ bound + PROFILE_STEP/2`, which implies
    // `true ≥ bound`. The generator enforces `bound + h + PROFILE_STEP` (tighter by h − PROFILE_STEP/2).
    test("over seeds 0–5000, every pair clears the derived non-contamination bound and stays in-world", () => {
        const SeedScan = 5000;
        let minSlack = Number.POSITIVE_INFINITY; // R5: min (dist − required) over all seeds and pairs
        let minSlackSeed = -1;
        let minSlackPair = "";
        let minContainmentSlack = Number.POSITIVE_INFINITY; // R4: min (WORLD_HALF − max|coord|)
        let minContainmentSeed = -1;

        for (let seed = 0; seed <= SeedScan; seed++) {
            const doc = generateNetwork(seed);
            const { cutDepth } = buildNetworkGeometry(doc, seed, DEFAULT_SMOOTH_RADIUS);
            const falloff = computeFalloff(cutDepth);
            // R1: the derived non-contamination bound — F + FCM + √2·SPACING. The generator enforces
            // this plus h = ROAD_HALF_WIDTH of slack; the arm asserts the bound itself.
            const bound = falloff + FLAT_CORE_MARGIN + Math.SQRT2 * SPACING;
            // R6: the road–carpark bound widens by PROFILE_STEP/2 to account for the sampling error in
            // segmentToPolygonDistance (polygonDistance is 1-Lipschitz, so PROFILE_STEP/2 is a bound).
            const carparkBound = bound + PROFILE_STEP / 2;

            // road–road pairs
            for (let i = 0; i < doc.polylines.length; i++) {
                for (let j = i + 1; j < doc.polylines.length; j++) {
                    const [a1, a2] = doc.polylines[i].points;
                    const [b1, b2] = doc.polylines[j].points;
                    const dist = roadFootprintDistance(a1, a2, b1, b2);
                    const slack = dist - bound;
                    if (slack < minSlack) {
                        minSlack = slack;
                        minSlackSeed = seed;
                        minSlackPair = `road ${i}–road ${j}`;
                    }
                    expect(
                        dist,
                        `seed ${seed}: road ${i}–road ${j} footprint dist ${dist.toFixed(2)} < bound ${bound.toFixed(2)}`,
                    ).toBeGreaterThanOrEqual(bound);
                }
            }

            // road–carpark pairs (R6: assert against bound + PROFILE_STEP/2)
            for (const [i, line] of doc.polylines.entries()) {
                const [a1, a2] = line.points;
                const dist = roadPolygonFootprintDistance(a1, a2, doc.polygons[0]);
                const slack = dist - carparkBound;
                if (slack < minSlack) {
                    minSlack = slack;
                    minSlackSeed = seed;
                    minSlackPair = `road ${i}–carpark`;
                }
                expect(
                    dist,
                    `seed ${seed}: road ${i}–carpark footprint dist ${dist.toFixed(2)} < bound ${carparkBound.toFixed(2)}`,
                ).toBeGreaterThanOrEqual(carparkBound);
            }

            // world containment (R4: track min slack = WORLD_HALF − max|coord|, not min |coord|)
            let maxAbsCoord = 0;
            for (const seg of flattenSegments(doc)) {
                for (const x of [seg.ax, seg.bx, seg.az, seg.bz]) {
                    maxAbsCoord = Math.max(maxAbsCoord, Math.abs(x));
                }
            }
            for (const [x, z] of doc.polygons[0].points) {
                for (const v of [x, z]) {
                    maxAbsCoord = Math.max(maxAbsCoord, Math.abs(v));
                }
            }
            const containmentSlack = WORLD_HALF - maxAbsCoord;
            if (containmentSlack < minContainmentSlack) {
                minContainmentSlack = containmentSlack;
                minContainmentSeed = seed;
            }
            for (const seg of flattenSegments(doc)) {
                for (const x of [seg.ax, seg.bx]) expect(Math.abs(x)).toBeLessThan(WORLD_HALF);
                for (const z of [seg.az, seg.bz]) expect(Math.abs(z)).toBeLessThan(WORLD_HALF);
            }
            for (const [x, z] of doc.polygons[0].points) {
                expect(Math.abs(x)).toBeLessThan(WORLD_HALF);
                expect(Math.abs(z)).toBeLessThan(WORLD_HALF);
            }
        }

        // R5: diagnostics — minimum per-seed slack (a readable margin, not an incomparable clearance)
        console.log(
            `DISJOINTNESS_SCAN minSlack=${minSlack.toFixed(2)} (seed ${minSlackSeed}, ${minSlackPair}) minContainmentSlack=${minContainmentSlack.toFixed(2)} (seed ${minContainmentSeed})`,
        );
        expect(minSlack).toBeFinite();
        expect(minContainmentSlack).toBeFinite();
    });
});
