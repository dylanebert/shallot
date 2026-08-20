import { describe, expect, test } from "bun:test";
import { gridX, gridZ, WORLD_HALF, worldX, worldZ } from "../terrain/grid";
import { documentDirtyTiles, documentDistance, flattenSegments } from "./document";
import { captureProbePoints, generateNetwork, ROAD_COUNT, ROAD_HALF_WIDTH } from "./network";

// The network generator's shape gate — the standard road generateNetwork always returns.
// `roads-interactive.md` stage 1 deleted route selection: `generateNetwork` takes no seed, is not
// deterministic-in-a-seed (there is no seed), and always returns the same document, so the seed-scan
// arms this file used to run (determinism-in-seed, a 0–5000 seed world-footprint scan, a 200-seed
// captureProbePoints classification scan, the route-selection differential and its structural/aggregate
// arms, and the reseed-disjointness pin) are deleted with their subject rather than adapted — none of
// them have anything left to scan over.

describe("generateNetwork — shape", () => {
    test("one road, a single straight segment, within the world footprint", () => {
        const doc = generateNetwork();
        expect(doc.polylines.length).toBe(ROAD_COUNT);
        for (const line of doc.polylines) {
            expect(line.points.length).toBe(2);
            expect(line.halfWidth).toBe(ROAD_HALF_WIDTH);
        }
    });

    test("touches a real, non-empty dirty-tile set (documentDirtyTiles doesn't throw on it)", () => {
        const doc = generateNetwork();
        expect(() => documentDirtyTiles(doc)).not.toThrow();
        expect(documentDirtyTiles(doc).length).toBeGreaterThan(0);
    });

    test("is the same document every call — no seed, nothing left to vary it", () => {
        expect(generateNetwork()).toEqual(generateNetwork());
    });
});

describe("captureProbePoints — the device gate's on/off-road pair over the standard chord", () => {
    test("both points are grid-aligned, and classify as their names say", () => {
        const doc = generateNetwork();
        const { onRoad, offRoad } = captureProbePoints();

        // grid-aligned: worldX(gridX(x)) round-trips exactly for a point already sitting on the grid.
        expect(worldX(gridX(onRoad[0]))).toBe(onRoad[0]);
        expect(worldZ(gridZ(onRoad[1]))).toBe(onRoad[1]);
        expect(worldX(gridX(offRoad[0]))).toBe(offRoad[0]);
        expect(worldZ(gridZ(offRoad[1]))).toBe(offRoad[1]);

        expect(documentDistance(onRoad[0], onRoad[1], doc)).toBeLessThan(0);
        expect(documentDistance(offRoad[0], offRoad[1], doc)).toBeGreaterThan(0);
    });

    test("pinned witness — a regression names itself against these exact coordinates", () => {
        // computed once against the standard chord and checked by hand against documentDistance
        // (on-road = -4 m, exactly on the centreline, inside the road's 4 m half-width; off-road = +4 m
        // outside it, one grid step further out).
        const { onRoad, offRoad } = captureProbePoints();
        expect(onRoad).toEqual([0, 0]);
        expect(offRoad).toEqual([0, 8]);
    });

    test("every segment stays within the world footprint", () => {
        const doc = generateNetwork();
        for (const seg of flattenSegments(doc)) {
            for (const x of [seg.ax, seg.bx]) expect(Math.abs(x)).toBeLessThan(WORLD_HALF);
            for (const z of [seg.az, seg.bz]) expect(Math.abs(z)).toBeLessThan(WORLD_HALF);
        }
    });
});
