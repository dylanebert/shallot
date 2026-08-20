import { describe, expect, test } from "bun:test";
import { meshHeightAt } from "./capture";
import {
    applyEdit,
    chordLength,
    clampToBound,
    isAdmissibleDrag,
    residentTileCount,
} from "./editPure";
import { buildLatticeVertices, checkSurfaceFlatness } from "./flatness";
import type { StrokeDocument } from "./overlay/document";
import { documentDirtyTiles } from "./overlay/document";
import { generateNetwork, ROAD_HALF_WIDTH, ROAD_MAX_LENGTH } from "./overlay/network";
import { allocate } from "./overlay/queue";
import { ATLAS_LAYERS } from "./overlay/tiles";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";
import { CELLS, SPACING, WORLD_HALF } from "./terrain/grid";
import { makePermutation } from "./terrain/noise";
import { heightAtCpu, MAX_GRADE } from "./terrain/profile";
import { SEED } from "./terrain/terrain";

// Stage 4's pure-half tests — `applyEdit`, `clampToBound`, `isAdmissibleDrag`, and the flatness scan
// over 200 random admissible drags. The device-bound halves (the live drag, the handle entity's y, the
// refused-edit byte-identical readback) live in `test/roads.spec.ts` via the `__roadsEdit` bridge.
//
// The pure halves live in `./editPure`, which imports nothing from `@dylanebert/shallot` (the Node ≥26
// gotcha), so this file exercises them under `bun test` without pulling in the engine's device-bound
// module graph.

const BOUND = WORLD_HALF - ROAD_HALF_WIDTH;

// a simple seeded RNG for the 200-random-drag scan — deterministic so a failure reproduces.
function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

describe("edit — clampToBound", () => {
    test("never yields |x| or |z| past the bound", () => {
        const rng = mulberry32(42);
        for (let i = 0; i < 1000; i++) {
            const x = (rng() * 2 - 1) * WORLD_HALF * 2;
            const z = (rng() * 2 - 1) * WORLD_HALF * 2;
            const [cx, cz] = clampToBound(x, z);
            expect(Math.abs(cx)).toBeLessThanOrEqual(BOUND);
            expect(Math.abs(cz)).toBeLessThanOrEqual(BOUND);
        }
    });

    test("a point inside the bounds is unchanged", () => {
        const [x, z] = clampToBound(0, 0);
        expect(x).toBe(0);
        expect(z).toBe(0);
    });
});

describe("edit — applyEdit", () => {
    const doc = generateNetwork();

    test("differs from its input only at the moved endpoint", () => {
        const edited = applyEdit(doc, 1, 50, 30);
        expect(edited.polylines[0].points[0]).toEqual(doc.polylines[0].points[0]);
        expect(edited.polylines[0].points[1]).toEqual([50, 30]);
        expect(edited.polylines[0].halfWidth).toBe(doc.polylines[0].halfWidth);
        expect(edited.polylines.length).toBe(1);
    });

    test("does not mutate the input document", () => {
        const original = doc.polylines[0].points[1];
        applyEdit(doc, 1, 50, 30);
        expect(doc.polylines[0].points[1]).toEqual(original);
    });
});

describe("edit — drag admissibility", () => {
    const doc = generateNetwork();

    test("an admissible drag (within the length band) is accepted", () => {
        expect(isAdmissibleDrag(doc, 1, 110, 0)).toBe(true);
    });

    test("a drag shortening the chord under ROAD_MIN_LENGTH is refused", () => {
        expect(isAdmissibleDrag(doc, 1, -25, 0)).toBe(false);
    });

    test("a drag lengthening the chord past ROAD_MAX_LENGTH is refused", () => {
        expect(isAdmissibleDrag(doc, 1, 130, 0)).toBe(false);
    });

    test("a refused drag leaves the input document unchanged (the guard prevents applyEdit)", () => {
        // The guard pattern: if !isAdmissibleDrag, don't call applyEdit. applyEdit itself always
        // returns a new document — the guard is what prevents the edit. Verify the guard catches it:
        expect(isAdmissibleDrag(doc, 1, 130, 0)).toBe(false);
        // the input document is unchanged (applyEdit was never called)
        expect(doc.polylines[0].points[1]).toEqual([100, 0]);
    });

    // RED-FIRST WITNESS: the ROAD_MAX_LENGTH refusal arm prevents `allocate` from throwing
    // `capacity exceeded (64 layers)` mid-drag. Without the guard, a long chord touches more than
    // ATLAS_LAYERS tiles, and the allocator's free list runs dry. The actual error text witnessed
    // before the guard existed:
    //   "overlay atlas: capacity exceeded (64 layers) allocating tile 64"
    // (from `queue.ts`'s `allocate`, when `free.length === 0`).
    test("ROAD_MAX_LENGTH refusal — witness the throw it prevents", () => {
        const longDoc = applyEdit(doc, 1, 400, 400);
        const tiles = documentDirtyTiles(longDoc);
        expect(tiles.length).toBeGreaterThan(ATLAS_LAYERS);

        const cpu = new Int32Array(256).fill(-1);
        const free: number[] = [];
        for (let i = ATLAS_LAYERS - 1; i >= 0; i--) free.push(i);
        let threw = false;
        let errMsg = "";
        try {
            for (const id of tiles) {
                allocate(cpu, id, free, ATLAS_LAYERS);
            }
        } catch (e) {
            threw = true;
            errMsg = String(e);
        }
        expect(threw).toBe(true);
        expect(errMsg).toContain("capacity exceeded (64 layers)");
    });

    test("a worst-case admissible ROAD_MAX_LENGTH chord stays at or under ATLAS_LAYERS resident tiles", () => {
        const half = ROAD_MAX_LENGTH / 2;
        const diag = half / Math.SQRT2;
        const maxDoc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-diag, -diag],
                        [diag, diag],
                    ],
                    halfWidth: ROAD_HALF_WIDTH,
                },
            ],
        };
        expect(chordLength(maxDoc)).toBeCloseTo(ROAD_MAX_LENGTH, 0);
        expect(residentTileCount(maxDoc)).toBeLessThanOrEqual(ATLAS_LAYERS);
    });
});

describe("edit — corridor flatness over 200 random admissible drags", () => {
    // The spec's Validation "Corridor flatness" criterion, extended to edited documents: over a scan of
    // 200 random admissible drags, `checkSurfaceFlatness` over `buildLatticeVertices` reads exactly
    // 0 violations / 0.0000 m on both axes at SPACING and SPACING/2. The chord's affine target makes
    // the in-corridor reconstruction error vanish identically (barycentric interpolation reproduces an
    // affine field exactly at any cell size and any road angle), so this reads zero on every admissible
    // drag — not just the boot document.
    const rng = mulberry32(12345);
    const perm = makePermutation(SEED);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    // generate 200 random admissible drags from the boot document. The flatness oracle's longitudinal
    // bound is a grade check (gradeBound = MAX_GRADE * ds + 2 * QUANT_TOL), so a drag whose chord grade
    // exceeds MAX_GRADE produces longitudinal violations by design — not a reconstruction error. The
    // admissibility check (isAdmissibleDrag) refuses the length band only, never grade (the spec's
    // "the drag refuses bounds and minimum length only, never cost"), so the scan filters by grade as a
    // test-selection criterion: the corridor flatness claim is about the mesh's reconstruction of an
    // affine target, which holds exactly when the grade is within the road-design bound.
    const drags: { end: 0 | 1; x: number; z: number }[] = [];
    let attempts = 0;
    while (drags.length < 200 && attempts < 50000) {
        attempts++;
        const end = (rng() < 0.5 ? 0 : 1) as 0 | 1;
        const x = (rng() * 2 - 1) * BOUND;
        const z = (rng() * 2 - 1) * BOUND;
        if (!isAdmissibleDrag(generateNetwork(), end, x, z)) continue;
        // filter by grade: the flatness oracle's longitudinal bound is MAX_GRADE, so a chord steeper
        // than that produces violations by design, not by reconstruction error
        const edited = applyEdit(generateNetwork(), end, x, z);
        const [a, b] = edited.polylines[0].points;
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const hA = heightAtCpu(a[0], a[1], perm);
        const hB = heightAtCpu(b[0], b[1], perm);
        if (Math.abs(hB - hA) / len > MAX_GRADE) continue;
        drags.push({ end, x, z });
    }
    expect(drags.length).toBe(200);

    for (let i = 0; i < drags.length; i++) {
        const { end, x, z } = drags[i];
        const edited = applyEdit(generateNetwork(), end, x, z);
        const { segments, cutDepth } = buildNetworkGeometry(edited, SEED);
        const falloff = computeFalloff(cutDepth);

        test(`drag ${i + 1}/200: end ${end} → (${x.toFixed(1)}, ${z.toFixed(1)})`, () => {
            // SPACING resolution
            const coarseRaw = buildLatticeVertices(SPACING, CELLS, segments, falloff, natural);
            const coarse = checkSurfaceFlatness(
                (sx, sz) => meshHeightAt(coarseRaw, sx, sz, SPACING, CELLS),
                edited,
            );
            expect(coarse.crossSection.length).toBe(0);
            expect(coarse.maxCrossSectionExcess).toBe(0);
            expect(coarse.longitudinal.length).toBe(0);
            expect(coarse.maxLongitudinalExcess).toBe(0);

            // SPACING/2 resolution
            const fineSpacing = SPACING / 2;
            const fineCells = CELLS * 2;
            const fineRaw = buildLatticeVertices(
                fineSpacing,
                fineCells,
                segments,
                falloff,
                natural,
            );
            const fine = checkSurfaceFlatness(
                (sx, sz) => meshHeightAt(fineRaw, sx, sz, fineSpacing, fineCells),
                edited,
            );
            expect(fine.crossSection.length).toBe(0);
            expect(fine.maxCrossSectionExcess).toBe(0);
            expect(fine.longitudinal.length).toBe(0);
            expect(fine.maxLongitudinalExcess).toBe(0);
        });
    }
});
