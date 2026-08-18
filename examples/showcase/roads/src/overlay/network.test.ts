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

    test("every primitive stays within the world footprint (no clamped-AABB edge case)", () => {
        const doc = generateNetwork(99);
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
