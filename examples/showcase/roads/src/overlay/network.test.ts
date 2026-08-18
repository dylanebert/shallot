import { describe, expect, test } from "bun:test";
import { WORLD_HALF } from "../terrain/grid";
import { documentDirtyTiles, flattenSegments } from "./document";
import { generateNetwork, ROAD_COUNT, ROAD_HALF_WIDTH } from "./network";

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
        // a single hardcoded seed is a fixture, not a corpus (checks.md) — an earlier version of this
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
