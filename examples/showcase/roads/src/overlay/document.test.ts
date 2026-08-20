import { describe, expect, test } from "bun:test";
import { WORLD_HALF } from "../terrain/grid";
import {
    documentDirtyTiles,
    documentDistance,
    drivable,
    flattenSegments,
    markingDistance,
    markingDistanceForSegment,
    type Segment,
    type StrokeDocument,
    segmentDistance,
    segmentRectDistance,
} from "./document";
import { ROAD_HALF_WIDTH } from "./network";
import { strokeDistance, strokeDocument } from "./stroke";
import {
    dirtyTiles,
    EDGE_INSET,
    LINE_HALF_WIDTH,
    type Rect,
    TEXEL_SIZE,
    TILE_SIZE,
    TILES_PER_SIDE,
    tileCoordOf,
    tileOrigin,
} from "./tiles";

describe("flattenSegments", () => {
    test("one polyline with N points yields N-1 segments, in order", () => {
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [0, 0],
                        [10, 0],
                        [10, 10],
                    ],
                    halfWidth: 2,
                },
            ],
        };
        const segs = flattenSegments(doc);
        expect(segs.length).toBe(2);
        expect(segs[0]).toEqual({ ax: 0, az: 0, bx: 10, bz: 0, halfWidth: 2 });
        expect(segs[1]).toEqual({ ax: 10, az: 0, bx: 10, bz: 10, halfWidth: 2 });
    });

    test("an empty document flattens to no segments", () => {
        expect(flattenSegments({ polylines: [] })).toEqual([]);
    });
});

describe("segmentDistance", () => {
    const seg = { ax: -10, az: 0, bx: 10, bz: 0, halfWidth: 2 };

    test("negative on the centreline, zero at the edge, positive beyond it", () => {
        expect(segmentDistance(0, 0, seg)).toBeLessThan(0);
        expect(segmentDistance(0, 2, seg)).toBeCloseTo(0, 10);
        expect(segmentDistance(0, 5, seg)).toBeCloseTo(3, 10);
    });

    test("beyond either endpoint, distance is to the endpoint, not the infinite line", () => {
        // straight out past B=(10,0): the nearest point on the segment is B itself
        expect(segmentDistance(15, 0, seg)).toBeCloseTo(5 - 2, 10);
        // diagonally past A=(-10,0)
        expect(segmentDistance(-13, 4, seg)).toBeCloseTo(5 - 2, 10);
    });

    test("a zero-length segment falls back to point distance", () => {
        const point = { ax: 0, az: 0, bx: 0, bz: 0, halfWidth: 1 };
        expect(segmentDistance(3, 4, point)).toBeCloseTo(5 - 1, 10);
    });
});

describe("documentDistance", () => {
    test("is the minimum over every segment", () => {
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-100, 0],
                        [100, 0],
                    ],
                    halfWidth: 2,
                },
            ],
        };
        expect(documentDistance(0, 0, doc)).toBeCloseTo(-2, 10);
    });

    test("cross-checks stroke.ts's hand-rolled strokeDistance over strokeDocument()", () => {
        // stroke.ts's strokeDistance is the stage-4 clamped-x formula (only valid for its own fixed
        // horizontal pattern); documentDistance is the stage-5 general form. They should agree exactly
        // over strokeDocument()'s one-segment shape, which is the point of keeping both alive.
        const doc = strokeDocument();
        const samples: Array<[number, number]> = [
            [0, 0],
            [0, 4],
            [0, 5],
            [149, 0],
            [151, 0],
            [-151, 3],
            [200, 200],
        ];
        for (const [x, z] of samples) {
            expect(documentDistance(x, z, doc)).toBeCloseTo(strokeDistance(x, z), 10);
        }
    });
});

describe("documentDirtyTiles", () => {
    test("an empty document throws rather than producing a degenerate set", () => {
        expect(() => documentDirtyTiles({ polylines: [] })).toThrow();
    });

    test("matches strokeRect/dirtyTiles' original row/column set for strokeDocument()", () => {
        // the exact set stroke.test.ts's own "strokeRect → dirtyTiles" describe block pins for the
        // single-arm pattern — documentDirtyTiles must agree now that it's the live path.
        const ids = new Set(documentDirtyTiles(strokeDocument()));
        const rows = new Set([...ids].map((id) => Math.floor(id / 16)));
        expect(rows).toEqual(new Set([7, 8]));
        const cols = [...ids].map((id) => id % 16);
        expect(Math.min(...cols)).toBe(5);
        expect(Math.max(...cols)).toBe(10);
    });
});

describe("only-touched-tiles oracle", () => {
    // a cross of two thin polylines through the origin: a short arm along X (tz row 7/8, the same
    // straddle strokeDocument's own pattern hits) and a longer arm along Z (tx col 7/8, but spanning many
    // more tile rows) — the redrawn tile set must be exactly their union: strictly wider than either arm
    // alone, so this genuinely tests a union, not one arm subsuming the other, and strictly *not* the
    // single bounding rect over both arms together (which would also mark the empty tiles between them —
    // documentDirtyTiles.ts's own header comment names this as the bug it avoids).
    const cross: StrokeDocument = {
        polylines: [
            {
                points: [
                    [-96, 0],
                    [96, 0],
                ],
                halfWidth: 1,
            }, // x in [-97,97] -> tx [6,9], tz {7,8}
            {
                points: [
                    [0, -160],
                    [0, 160],
                ],
                halfWidth: 1,
            }, // z in [-161,161] -> tz [5,10], tx {7,8}
        ],
    };

    test("the redrawn tile set is exactly the per-arm union, nothing more", () => {
        const ids = new Set(documentDirtyTiles(cross));
        const expected = new Set<number>();
        for (let tx = 6; tx <= 9; tx++) for (const tz of [7, 8]) expected.add(tz * 16 + tx);
        for (let tz = 5; tz <= 10; tz++) for (const tx of [7, 8]) expected.add(tz * 16 + tx);
        expect(expected.size).toBeGreaterThan(8); // a real union, wider than the horizontal arm alone
        expect(expected.size).toBeGreaterThan(12); // and wider than the vertical arm alone
        expect(ids).toEqual(expected);
    });

    test("a tile far from either arm is not in the set", () => {
        const ids = new Set(documentDirtyTiles(cross));
        expect(ids.has(0)).toBe(false); // corner tile, nowhere near the cross
        expect(ids.has(15)).toBe(false);
    });

    test("a bounding rect over both arms would over-mark a tile neither arm touches — the property the\n     per-primitive union avoids", () => {
        // tile (tx=9, tz=5): inside the single AABB spanning both arms (x up to 97, z down to -161) but
        // outside both arms' own footprints (horizontal arm never reaches tz=5; vertical arm never
        // reaches tx=9). A single-bounding-rect implementation would incorrectly include it.
        const id = 5 * 16 + 9;
        expect(documentDirtyTiles(cross)).not.toContain(id);
    });
});

describe("drivable", () => {
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
    };

    test("on the centreline and well inside the half-width — true", () => {
        expect(drivable(0, 0, doc)).toBe(true);
        expect(drivable(0, 3, doc)).toBe(true);
    });

    test("just past the half-width — false; the same point mirrored inward — true", () => {
        expect(drivable(0, 4.5, doc)).toBe(false);
        expect(drivable(0, 3.5, doc)).toBe(true);
    });

    test("agrees with documentDistance's own sign — the query it wraps, not a second definition", () => {
        for (const [x, z] of [
            [0, 0],
            [0, 10],
            [500, 500],
        ] as const) {
            expect(drivable(x, z, doc)).toBe(documentDistance(x, z, doc) <= 0);
        }
    });
});

describe("markingDistanceForSegment", () => {
    // the standard chord's shape: a horizontal road from (-100, 0) to (100, 0), halfWidth 4
    const seg: Segment = { ax: -100, az: 0, bx: 100, bz: 0, halfWidth: 4 };

    test("negative inside an edge line (centred at d = -EDGE_INSET, width LINE_WIDTH)", () => {
        // the edge line is at |z| = halfWidth - EDGE_INSET = 3.7 m from the centreline.
        // Derived from the segment's own halfWidth, not ROAD_HALF_WIDTH, so the probe stays on the
        // edge line whatever the road width is — one fixture, one source of truth.
        const edgeLineZ = seg.halfWidth - EDGE_INSET; // 3.7
        const result = markingDistanceForSegment(0, edgeLineZ, seg);
        expect(result).toBeLessThan(0); // inside the edge line
    });

    test("zero at the edge line boundary (LINE_HALF_WIDTH from the centre)", () => {
        const edgeLineZ = seg.halfWidth - EDGE_INSET; // 3.7
        const boundary = edgeLineZ + LINE_HALF_WIDTH; // 3.75
        expect(markingDistanceForSegment(0, boundary, seg)).toBeCloseTo(0, 6);
    });

    test("negative on the centreline inside a dash (station 120, phase < DASH_DUTY with offset)", () => {
        // with DASH_OFFSET, station 120: phase = fract((120 + DASH_OFFSET) / DASH_PERIOD) ≈ 0.09 < 0.25 → in a dash
        // world point at station 120 from A=(-100,0) along +X is (20, 0)
        const result = markingDistanceForSegment(20, 0, seg);
        expect(result).toBeLessThan(0); // inside the centreline dash
    });

    test("positive on the centreline in a gap (station 100/midpoint, phase >= DASH_DUTY with offset)", () => {
        // with DASH_OFFSET, station 100 (midpoint): phase = fract((100 + DASH_OFFSET) / DASH_PERIOD) ≈ 0.45 >= 0.25 → in a gap
        // world point at station 100 from A=(-100,0) along +X is (0, 0) — the on-road probe point
        const result = markingDistanceForSegment(0, 0, seg);
        expect(result).toBeGreaterThan(0); // in a gap, outside the marking
    });

    test("positive on the asphalt between the edge line and the centreline", () => {
        // |z| = 2.0 m: between the edge line (3.7 m) and the centreline (0 m)
        const result = markingDistanceForSegment(0, 2, seg);
        expect(result).toBeGreaterThan(0); // on the asphalt, no marking here
    });
});

describe("markingDistance", () => {
    test("is the minimum over every segment, same as documentDistance's shape", () => {
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-100, 0],
                        [100, 0],
                    ],
                    halfWidth: 4,
                },
            ],
        };
        // on the centreline, in a dash (station 120 with offset) → negative
        expect(markingDistance(20, 0, doc)).toBeLessThan(0);
        // on the centreline, in a gap (station 100/midpoint with offset) → positive
        expect(markingDistance(0, 0, doc)).toBeGreaterThan(0);
        // on the edge line → negative. Derived from the doc's own halfWidth, not ROAD_HALF_WIDTH.
        expect(markingDistance(0, doc.polylines[0].halfWidth - EDGE_INSET, doc)).toBeLessThan(0);
    });
});

describe("dirty set is the swath (stage 4d)", () => {
    // The capsule (swath) dirty-set test replaces the segment's axis-aligned bounding box. Three arms:
    // (1) the axis-aligned null control — the AABB was already exact there, so the count is unchanged at
    // 32; (2) the diagonal's drop — the AABB read 256 (the whole grid), the capsule reads the true swath;
    // (3) the one that matters — every tile the narrowing drops is verified to contain no point within
    // halfWidth + margin of the chord, so the narrowing produces no unbaked hole.
    //
    // RED-FIRST against 4c's shape (the AABB): a corner-to-corner diagonal read 256 under the AABB —
    // `git show 3bc5d55:examples/showcase/roads/src/overlay/document.ts` is the pre-image. The capsule
    // test narrows it to 46, the measured worst case over a scan of orientations.

    const halfWidth = ROAD_HALF_WIDTH;
    const margin = TEXEL_SIZE;
    const bound = WORLD_HALF - ROAD_HALF_WIDTH; // 508 — the admissible endpoint bound

    /** the old AABB dirty set for one segment — the pre-4d shape, reconstructed in the test so the
     *  narrowing can be verified against it. */
    function aabbDirtySet(seg: Segment): Set<number> {
        const rect: Rect = {
            minX: Math.min(seg.ax, seg.bx) - seg.halfWidth - margin,
            maxX: Math.max(seg.ax, seg.bx) + seg.halfWidth + margin,
            minZ: Math.min(seg.az, seg.bz) - seg.halfWidth - margin,
            maxZ: Math.max(seg.az, seg.bz) + seg.halfWidth + margin,
        };
        return new Set(dirtyTiles(rect));
    }

    /** a tile column's world-space rect, computed from exported tiles.ts helpers. */
    function tileRect(tx: number, tz: number): Rect {
        const [ox, oz] = tileOrigin(tx, tz);
        return { minX: ox, minZ: oz, maxX: ox + TILE_SIZE, maxZ: oz + TILE_SIZE };
    }

    test("an axis-aligned full-width chord reads exactly 32 (the AABB was already exact there)", () => {
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-bound, 0],
                        [bound, 0],
                    ],
                    halfWidth,
                },
            ],
        };
        const count = documentDirtyTiles(doc).length;
        expect(count).toBe(32); // the null control — the capsule test did not narrow the common case
    });

    test("a corner-to-corner diagonal drops from 256 (AABB) to the swath count", () => {
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-bound, -bound],
                        [bound, bound],
                    ],
                    halfWidth,
                },
            ],
        };
        const seg = flattenSegments(doc)[0];
        const aabbCount = aabbDirtySet(seg).size;
        const swathCount = documentDirtyTiles(doc).length;

        // the AABB read 256 — the whole grid (the pre-4d shape)
        expect(aabbCount).toBe(256);
        // the capsule reads the true swath — 46, the measured worst case
        expect(swathCount).toBe(46);
        // the narrowing dropped the count
        expect(swathCount).toBeLessThan(aabbCount);
        // a straight chord crosses at most 2 * TILES_PER_SIDE - 1 = 31 tiles before width, so 46 is
        // 1.48× the bound (the extra tiles are the halfWidth + margin band), not ~8× — the instrument
        // is measuring the real footprint, not the AABB
        expect(swathCount).toBeLessThanOrEqual(
            2 * TILES_PER_SIDE -
                1 +
                2 * Math.ceil((halfWidth + margin) / TILE_SIZE) * TILES_PER_SIDE,
        );
    });

    test("every tile the narrowing drops contains no point within halfWidth + margin of the chord", () => {
        const doc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-bound, -bound],
                        [bound, bound],
                    ],
                    halfWidth,
                },
            ],
        };
        const seg = flattenSegments(doc)[0];
        const aabbSet = aabbDirtySet(seg);
        const swathSet = new Set(documentDirtyTiles(doc));

        // the dropped tiles: in the AABB set but not in the capsule set
        const dropped: number[] = [];
        for (const id of aabbSet) {
            if (!swathSet.has(id)) dropped.push(id);
        }
        expect(dropped.length).toBeGreaterThan(0); // the narrowing actually dropped tiles

        // the one that matters: every dropped tile must contain no point within halfWidth + margin of
        // the segment. If a dropped tile's rect has a point within that distance, the capsule test
        // incorrectly excluded a tile the road actually touches — an unbaked hole.
        for (const id of dropped) {
            const [tx, tz] = tileCoordOf(id);
            const rect = tileRect(tx, tz);
            const dist = segmentRectDistance(seg, rect);
            expect(dist).toBeGreaterThan(halfWidth + margin);
        }
    });
});
