import { describe, expect, test } from "bun:test";
import { meshHeightAt } from "./capture";
import { createGrabState, stepGrab } from "./edit";
import {
    applyEdit,
    chordLength,
    clampDragTarget,
    clampToBound,
    residentTileCount,
} from "./editPure";
import { buildLatticeVertices, checkSurfaceFlatness } from "./flatness";
import type { StrokeDocument } from "./overlay/document";
import { generateNetwork, ROAD_HALF_WIDTH, ROAD_MIN_LENGTH } from "./overlay/network";
import { ATLAS_LAYERS, TILE_COUNT } from "./overlay/tiles";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";
import { CELLS, SPACING, WORLD_HALF } from "./terrain/grid";
import { makePermutation } from "./terrain/noise";
import { heightAtCpu, MAX_GRADE } from "./terrain/profile";
import { SEED } from "./terrain/terrain";

// Stage 4's pure-half tests — `applyEdit`, `clampToBound`, `clampDragTarget`, the worst-case capacity
// arm, and the flatness scan over 200 random clamped drags. The device-bound halves (the live drag,
// the handle entity's y) live in `test/roads.spec.ts` via the `__roadsEdit` bridge.
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

describe("edit — every drag constraint clamps (stage 4c)", () => {
    // REGRESSION GUARD (not a red-first witness): this arm calls `clampDragTarget`, which did not
    // exist at the pre-image `3ea0417` (the pre-image had `isAdmissibleDrag`, a refusal predicate), so
    // it cannot compile against the old shape, let alone fail against it. It is a regression guard over
    // the new clamp shape: over a scan of drag targets, `applyEdit` (after `clampToBound` +
    // `clampDragTarget`) always moves the dragged endpoint for every input whose target differs from
    // the current position, and the resulting chord holds both the bound and the floor.
    //
    // The real red-first evidence for the ceiling deletion lives in two other arms: (1) the worst-case
    // chord capacity arm below (`edit.test.ts`, "the worst-case corner-to-corner chord stays at or under
    // ATLAS_LAYERS"), which fails at the pre-image because a corner-to-corner chord touches >64 tiles
    // against ATLAS_LAYERS=64; and (2) the device corner arm (`test/roads.spec.ts`, `cornerApplied ===
    // false` at the pre-image), where the old `__roadsEdit` bridge refused a target past ROAD_MAX_LENGTH
    // and returned false instead of applying the clamped edit.
    //
    // Historical note: the old arms this replaces ("a drag shortening under ROAD_MIN_LENGTH is refused",
    // "a drag lengthening past ROAD_MAX_LENGTH is refused") were green while pinning the freeze — they
    // encoded the refusal as the contract, so the suite passed over the defect the person rejected.
    //
    // The clamp law (`roads-interactive.md`'s Locked decision): a constraint on a dragged quantity is a
    // projection onto the nearest admissible value, never a no-op. The one admissible no-op is a target
    // equal to the current position.

    const doc = generateNetwork();
    const [ax, az] = doc.polylines[0].points[0] as [number, number];
    const [bx, bz] = doc.polylines[0].points[1] as [number, number];

    // a scan of drag targets for endpoint 1: normal, past the old 220 m ceiling (inside world),
    // past the world bound, under floor, and the no-op
    const targets: { x: number; z: number; label: string }[] = [
        { x: 50, z: 30, label: "normal (within bounds and floor)" },
        { x: 200, z: 200, label: "past old 220 m ceiling (inside world bound)" },
        { x: -200, z: -200, label: "past old 220 m ceiling (inside world bound, negative)" },
        { x: 10000, z: 10000, label: "far past world bound" },
        { x: -90, z: 0, label: "would shorten under floor (chord ~10 m)" },
        { x: -99, z: 0, label: "would shorten under floor (chord ~1 m)" },
        { x: bx, z: bz, label: "target equals current position (no-op)" },
    ];

    for (const { x, z, label } of targets) {
        test(`clamp + applyEdit: ${label} → target (${x}, ${z})`, () => {
            const [cx, cz] = clampToBound(x, z);
            const [fx, fz] = clampDragTarget(doc, 1, cx, cz);
            const edited = applyEdit(doc, 1, fx, fz);

            // the resulting endpoint is within bounds
            expect(Math.abs(fx)).toBeLessThanOrEqual(BOUND);
            expect(Math.abs(fz)).toBeLessThanOrEqual(BOUND);

            // the resulting chord holds the floor
            expect(chordLength(edited)).toBeGreaterThanOrEqual(ROAD_MIN_LENGTH - 1e-6);

            // the other endpoint is unchanged
            expect(edited.polylines[0].points[0]).toEqual([ax, az]);

            // if the clamped target differs from the current position, the output differs at endpoint 1
            const moved = fx !== bx || fz !== bz;
            const outputDiffers =
                edited.polylines[0].points[1][0] !== bx || edited.polylines[0].points[1][1] !== bz;
            expect(outputDiffers).toBe(moved);

            // the one admissible no-op: target equal to current position
            if (x === bx && z === bz) {
                expect(fx).toBe(bx);
                expect(fz).toBe(bz);
            }
        });
    }

    // property: over 500 random targets (some past bounds, some under floor), the clamp always
    // produces a valid endpoint and applyEdit always moves it
    test("500 random targets: clamp always yields a valid endpoint, applyEdit always moves it", () => {
        const rng = mulberry32(999);
        for (let i = 0; i < 500; i++) {
            // sample from a range wider than the world to include past-bound targets
            const x = (rng() * 2 - 1) * WORLD_HALF * 3;
            const z = (rng() * 2 - 1) * WORLD_HALF * 3;
            const [cx, cz] = clampToBound(x, z);
            const [fx, fz] = clampDragTarget(doc, 1, cx, cz);
            const edited = applyEdit(doc, 1, fx, fz);

            expect(Math.abs(fx)).toBeLessThanOrEqual(BOUND);
            expect(Math.abs(fz)).toBeLessThanOrEqual(BOUND);
            expect(chordLength(edited)).toBeGreaterThanOrEqual(ROAD_MIN_LENGTH - 1e-6);

            // the other endpoint is unchanged
            expect(edited.polylines[0].points[0]).toEqual([ax, az]);

            // if the clamped target differs from the current position, the output differs
            const moved = fx !== bx || fz !== bz;
            const outputDiffers =
                edited.polylines[0].points[1][0] !== bx || edited.polylines[0].points[1][1] !== bz;
            expect(outputDiffers).toBe(moved);
        }
    });
});

describe("edit — worst-case chord capacity (stage 4c)", () => {
    // The arm that survived the ROAD_MAX_LENGTH deletion, re-anchored on the invariant that mattered:
    // "no admissible drag can throw out of `allocate`". The worst case the world allows is a
    // corner-to-corner chord across the bounded 1024 m world. Its `documentDirtyTiles` count must be
    // at or under `ATLAS_LAYERS` (now `TILE_COUNT = 256`, sized by this measurement).
    test("the worst-case corner-to-corner chord stays at or under ATLAS_LAYERS resident tiles", () => {
        const bound = WORLD_HALF - ROAD_HALF_WIDTH;
        const worstDoc: StrokeDocument = {
            polylines: [
                {
                    points: [
                        [-bound, -bound],
                        [bound, bound],
                    ],
                    halfWidth: ROAD_HALF_WIDTH,
                },
            ],
        };
        // the chord length is the full diagonal of the bounded region (~1437 m)
        expect(chordLength(worstDoc)).toBeGreaterThan(1400);
        // the dirty-tile count is at or under ATLAS_LAYERS
        const count = residentTileCount(worstDoc);
        expect(count).toBeLessThanOrEqual(ATLAS_LAYERS);
        // ATLAS_LAYERS is TILE_COUNT (256) — fully resident
        expect(ATLAS_LAYERS).toBe(TILE_COUNT);
    });
});

describe("edit — corridor flatness over 200 random clamped drags", () => {
    // The spec's Validation "Corridor flatness" criterion, extended to edited documents: over a scan of
    // 200 random clamped drags, `checkSurfaceFlatness` over `buildLatticeVertices` reads exactly
    // 0 violations / 0.0000 m on both axes at SPACING and SPACING/2. The chord's affine target makes
    // the in-corridor reconstruction error vanish identically (barycentric interpolation reproduces an
    // affine field exactly at any cell size and any road angle), so this reads zero on every clamped
    // drag — not just the boot document.
    //
    // Stage 4c: the scan now draws from the *unbounded* band — targets past the world bound and past
    // the old ROAD_MAX_LENGTH ceiling — and clamps via `clampToBound` + `clampDragTarget` before
    // `applyEdit`. The old scan filtered by `isAdmissibleDrag` (the deleted refusal); the new scan
    // filters by grade only (the flatness oracle's longitudinal bound is MAX_GRADE, so a chord steeper
    // than that produces violations by design, not by reconstruction error).
    const rng = mulberry32(12345);
    const perm = makePermutation(SEED);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    const drags: { end: 0 | 1; x: number; z: number }[] = [];
    let attempts = 0;
    while (drags.length < 200 && attempts < 50000) {
        attempts++;
        const end = (rng() < 0.5 ? 0 : 1) as 0 | 1;
        // sample from the unbounded band — wider than the world to include past-bound targets
        const x = (rng() * 2 - 1) * WORLD_HALF * 3;
        const z = (rng() * 2 - 1) * WORLD_HALF * 3;
        const [cx, cz] = clampToBound(x, z);
        const [fx, fz] = clampDragTarget(generateNetwork(), end, cx, cz);
        // filter by grade: the flatness oracle's longitudinal bound is MAX_GRADE, so a chord steeper
        // than that produces violations by design, not by reconstruction error
        const edited = applyEdit(generateNetwork(), end, fx, fz);
        const [a, b] = edited.polylines[0].points;
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const hA = heightAtCpu(a[0], a[1], perm);
        const hB = heightAtCpu(b[0], b[1], perm);
        if (Math.abs(hB - hA) / len > MAX_GRADE) continue;
        drags.push({ end, x: fx, z: fz });
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

describe("edit — sticky grab latch (stage 4b)", () => {
    // RED-FIRST WITNESS: the shipped shape uses `left && !dragging && hovered >= 0` to start the drag —
    // not a rising-edge latch. A press with no handle under it latches nothing, but moving the cursor
    // over a handle while the button is still held *does* start a drag, because `!dragging` is true and
    // `hovered >= 0` is now satisfied. The failure text witnessed before the rising-edge fix:
    //   "expected { dragging: true, dragEnd: 0, prevLeft: true } to equal { dragging: false, dragEnd: 0, prevLeft: true }"
    // (frame 3 of sequence 1 — the old inline gate in `update()` starts a drag when the cursor reaches
    // a handle mid-hold, the new one does not because the rising edge already passed). There was no
    // `stepGrab` before this stage: the old logic was inline in `update()`, and the pre-image is
    // `git show a3827a7:examples/showcase/roads/src/edit.ts`.
    //
    // Only THIS first arm discriminates the new shape from the old. The other three produce identical
    // traces under both shapes, because the old gate's `dragging` already survived hover misses (once
    // true, only `!left` cleared it) — the real shipped defect was the mid-hold *start* plus an
    // `OrbitPick.claim` that re-ran a fresh hover test every frame and so released orbit suppression
    // mid-drag, which is what the person saw as "grab/ungrab mid movement". The remaining three arms
    // are regression guards over the new shape, not red-first witnesses; keeping that distinction
    // explicit is why this note exists.
    //
    // The latch is device-free state, so this is a unit arm over a synthetic press → move-off →
    // move-back → release sequence — not a device probe.

    test("a press with no handle under it latches nothing", () => {
        let s = createGrabState();
        // frame 1: idle
        s = stepGrab(s, false, -1);
        expect(s).toEqual({ dragging: false, dragEnd: 0, prevLeft: false });
        // frame 2: press with no handle under it
        s = stepGrab(s, true, -1);
        expect(s).toEqual({ dragging: false, dragEnd: 0, prevLeft: true });
        // frame 3: move over handle 0 while still held — must NOT latch
        s = stepGrab(s, true, 0);
        expect(s).toEqual({ dragging: false, dragEnd: 0, prevLeft: true });
        // frame 4: release
        s = stepGrab(s, false, 0);
        expect(s).toEqual({ dragging: false, dragEnd: 0, prevLeft: false });
    });

    test("latched index is unchanged through press → move-off → move-back → release", () => {
        let s = createGrabState();
        // frame 1: idle
        s = stepGrab(s, false, -1);
        expect(s).toEqual({ dragging: false, dragEnd: 0, prevLeft: false });
        // frame 2: press over handle 1 — latches dragEnd=1
        s = stepGrab(s, true, 1);
        expect(s).toEqual({ dragging: true, dragEnd: 1, prevLeft: true });
        // frame 3: move off the handle (hover miss) — dragging stays, dragEnd unchanged
        s = stepGrab(s, true, -1);
        expect(s).toEqual({ dragging: true, dragEnd: 1, prevLeft: true });
        // frame 4: move back over handle 0 — dragEnd stays 1, not 0
        s = stepGrab(s, true, 0);
        expect(s).toEqual({ dragging: true, dragEnd: 1, prevLeft: true });
        // frame 5: release — clears dragging, dragEnd stays at last value
        s = stepGrab(s, false, 0);
        expect(s).toEqual({ dragging: false, dragEnd: 1, prevLeft: false });
    });

    test("the release edge, and only the release edge, clears it", () => {
        let s = createGrabState();
        // press over handle 0
        s = stepGrab(s, true, 0);
        expect(s.dragging).toBe(true);
        // move off — still dragging
        s = stepGrab(s, true, -1);
        expect(s.dragging).toBe(true);
        // move over handle 1 — still dragging, still dragEnd 0
        s = stepGrab(s, true, 1);
        expect(s.dragging).toBe(true);
        expect(s.dragEnd).toBe(0);
        // release — clears
        s = stepGrab(s, false, -1);
        expect(s.dragging).toBe(false);
    });

    test("a second press after release latches fresh", () => {
        let s = createGrabState();
        // first press over handle 0
        s = stepGrab(s, true, 0);
        expect(s).toEqual({ dragging: true, dragEnd: 0, prevLeft: true });
        // release
        s = stepGrab(s, false, -1);
        expect(s.dragging).toBe(false);
        // second press over handle 1
        s = stepGrab(s, true, 1);
        expect(s).toEqual({ dragging: true, dragEnd: 1, prevLeft: true });
    });
});
