import { describe, expect, test } from "bun:test";
import { gridX, gridZ, WORLD_HALF, worldX, worldZ } from "../terrain/grid";
import { documentDirtyTiles, documentDistance, flattenSegments } from "./document";
import { captureProbePoints, generateNetwork, ROAD_COUNT, ROAD_HALF_WIDTH } from "./network";
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
        // (on-road ≈ -3.20 m inside the road's 4 m half-width; off-road ≈ +4.10 m outside it, one grid
        // step further out — the nearest-to-origin road at this seed, not necessarily road 0).
        const { onRoad, offRoad } = captureProbePoints(BOOT_SEED);
        expect(onRoad).toEqual([-8, -24]);
        expect(offRoad).toEqual([0, -28]);
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
