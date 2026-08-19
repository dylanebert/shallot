import { describe, expect, test } from "bun:test";
import { buildNetworkGeometry, computeFalloff, FLAT_CORE_MARGIN } from "../terrain/flatten";
import { gridX, gridZ, WORLD_HALF, worldX, worldZ } from "../terrain/grid";
import { DEFAULT_SMOOTH_RADIUS } from "../terrain/profile";
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
    // no two primitives' footprints come within `computeFalloff(cutDepth) + ROAD_HALF_WIDTH +
    // FLAT_CORE_MARGIN`, and every primitive stays inside `WORLD_HALF`. A single seed is
    // inadmissible — stage 6's own live defect (a road 60 m outside the world at seed 615) was
    // invisible to a single-seed footprint arm. The clearance is derived from the network's own cut
    // depth (measured between chord endpoints), never a hardcoded constant — a deeper cut widens
    // both the falloff and the required separation.
    const SeedScan = 5000;
    let minClearance = Number.POSITIVE_INFINITY;
    let minClearanceSeed = -1;
    let minClearancePair = "";
    let worstContainment = Number.POSITIVE_INFINITY;
    let worstContainmentSeed = -1;

    for (let seed = 0; seed <= SeedScan; seed++) {
        const doc = generateNetwork(seed);
        const { cutDepth } = buildNetworkGeometry(doc, seed, DEFAULT_SMOOTH_RADIUS);
        const falloff = computeFalloff(cutDepth);
        const required = falloff + ROAD_HALF_WIDTH + FLAT_CORE_MARGIN;

        // road–road pairs
        for (let i = 0; i < doc.polylines.length; i++) {
            for (let j = i + 1; j < doc.polylines.length; j++) {
                const [a1, a2] = doc.polylines[i].points;
                const [b1, b2] = doc.polylines[j].points;
                const dist = roadFootprintDistance(a1, a2, b1, b2);
                if (dist < minClearance) {
                    minClearance = dist;
                    minClearanceSeed = seed;
                    minClearancePair = `road ${i}–road ${j}`;
                }
                expect(
                    dist,
                    `seed ${seed}: road ${i}–road ${j} footprint dist ${dist.toFixed(2)} < required ${required.toFixed(2)}`,
                ).toBeGreaterThanOrEqual(required);
            }
        }

        // road–carpark pairs
        for (const [i, line] of doc.polylines.entries()) {
            const [a1, a2] = line.points;
            const dist = roadPolygonFootprintDistance(a1, a2, doc.polygons[0]);
            if (dist < minClearance) {
                minClearance = dist;
                minClearanceSeed = seed;
                minClearancePair = `road ${i}–carpark`;
            }
            expect(
                dist,
                `seed ${seed}: road ${i}–carpark footprint dist ${dist.toFixed(2)} < required ${required.toFixed(2)}`,
            ).toBeGreaterThanOrEqual(required);
        }

        // world containment
        for (const seg of flattenSegments(doc)) {
            for (const x of [seg.ax, seg.bx, seg.az, seg.bz]) {
                if (Math.abs(x) < worstContainment) {
                    worstContainment = Math.abs(x);
                    worstContainmentSeed = seed;
                }
                expect(Math.abs(x)).toBeLessThan(WORLD_HALF);
            }
        }
        for (const [x, z] of doc.polygons[0].points) {
            for (const v of [x, z]) {
                if (Math.abs(v) < worstContainment) {
                    worstContainment = Math.abs(v);
                    worstContainmentSeed = seed;
                }
                expect(Math.abs(v)).toBeLessThan(WORLD_HALF);
            }
        }
    }

    test(`the scan reports the tightest clearance and worst containment (not a gate — diagnostics)`, () => {
        const { cutDepth } = buildNetworkGeometry(
            generateNetwork(BOOT_SEED),
            BOOT_SEED,
            DEFAULT_SMOOTH_RADIUS,
        );
        const falloff = computeFalloff(cutDepth);
        const required = falloff + ROAD_HALF_WIDTH + FLAT_CORE_MARGIN;
        console.log(
            `DISJOINTNESS_SCAN minClearance=${minClearance.toFixed(2)} (seed ${minClearanceSeed}, ${minClearancePair}) required=${required.toFixed(2)} worstContainment=${worstContainment.toFixed(2)} (seed ${worstContainmentSeed})`,
        );
        expect(minClearance).toBeFinite();
        expect(worstContainment).toBeFinite();
    });
});
