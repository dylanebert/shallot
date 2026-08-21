import { describe, expect, test } from "bun:test";
import { CORPUS_DRAGS, dragCorpus, mulberry32, SENTINEL_DRAGS, scanDrag } from "./dragCorpus";
import { createGrabState, stepGrab } from "./edit";
import {
    applyEdit,
    chordLength,
    clampDragTarget,
    clampToBound,
    projectRayToBound,
} from "./editPure";
import type { StrokeDocument } from "./overlay/document";
import { documentDirtyTiles } from "./overlay/document";
import { generateNetwork, ROAD_HALF_WIDTH, ROAD_MIN_LENGTH } from "./overlay/network";
import { ATLAS_LAYERS, TILE_COUNT } from "./overlay/tiles";
import { WORLD_HALF } from "./terrain/grid";

// Stage 4's pure-half tests — `applyEdit`, `clampToBound`, `clampDragTarget`, the worst-case capacity
// arm, and the corridor-flatness sentinel over the first few drags of `dragCorpus.ts`'s frozen corpus
// (the whole 200-drag corpus is `editCorridor.tier.ts`, run by path). The device-bound halves (the live
// drag, the handle entity's y) live in `test/roads.spec.ts` via the `__roadsEdit` bridge.
//
// The pure halves live in `./editPure`, which imports nothing from `@dylanebert/shallot` (the Node ≥26
// gotcha), so this file exercises them under `bun test` without pulling in the engine's device-bound
// module graph.

const BOUND = WORLD_HALF - ROAD_HALF_WIDTH;

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

describe("edit — worst-case chord capacity (stage 4d)", () => {
    // The arm that survived the ROAD_MAX_LENGTH deletion, re-anchored on the invariant that mattered:
    // "no admissible drag can throw out of `allocate`". The worst case the world allows is a
    // corner-to-corner chord across the bounded 1024 m world. Its `documentDirtyTiles` count must be
    // at or under `ATLAS_LAYERS` (now 64, sized by the stage 4d capsule-test measurement of 46 + headroom).
    //
    // Stage 4d narrowed the dirty set from the segment's AABB to its capsule (swath): the diagonal's
    // count dropped from 256 (the whole grid under the AABB) to 46 (the true swath). ATLAS_LAYERS fell
    // from TILE_COUNT (256, full residency) back to 64 — the original pre-4c value — with ~39% headroom
    // over the measured 46. The capacity arms in `queue.test.ts` are witnesses again: with capacity
    // below full residency, `allocate` can throw if `release` breaks.
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
        // the dirty-tile count (capsule swath) is at or under ATLAS_LAYERS. Read straight off
        // `documentDirtyTiles` — the production export the overlay itself calls — rather than through a
        // wrapper whose only reader was this arm (close sweep, B2).
        const count = documentDirtyTiles(worstDoc).length;
        expect(count).toBeLessThanOrEqual(ATLAS_LAYERS);
        // the measured worst-case swath is 46 (stage 4d), ATLAS_LAYERS is 64 with headroom
        expect(count).toBe(46);
        expect(ATLAS_LAYERS).toBe(64);
        expect(ATLAS_LAYERS).toBeLessThan(TILE_COUNT);
    });
});

describe(`edit — corridor flatness sentinel (${SENTINEL_DRAGS} of the ${CORPUS_DRAGS}-drag corpus)`, () => {
    // The default-suite sentinel for the corridor-flatness criterion (spec Validation "Corridor
    // flatness", extended to edited documents). The full corpus lives in `editCorridor.tier.ts`, run by
    // path; this arm scans the *same* frozen fixture's first `SENTINEL_DRAGS` drags, so the coverage
    // moved tiers rather than shrinking (`coding.md` Suite speed). The reading is the one the tier
    // asserts: exactly 0 violations / 0.0000 m on both axes at SPACING and SPACING/2, over the banded
    // lattice — whose equivalence to the full lattice for this oracle is `flatness.test.ts`'s null
    // control ("the banded lattice reads what the full lattice reads").
    const drags = dragCorpus(SENTINEL_DRAGS);

    test(`the sentinel slice is the corpus's own first ${SENTINEL_DRAGS} drags`, () => {
        expect(drags.length).toBe(SENTINEL_DRAGS);
        // prefix stability is what makes this a slice of the tier's population rather than a second
        // fixture: a corpus whose ordering depended on its own length would hand the two readers
        // different drags while both read green.
        expect(dragCorpus(SENTINEL_DRAGS + 3).slice(0, SENTINEL_DRAGS)).toEqual(drags);
    });

    for (let i = 0; i < drags.length; i++) {
        const drag = drags[i];
        test(`drag ${i + 1}/${SENTINEL_DRAGS}: end ${drag.end} → (${drag.x.toFixed(1)}, ${drag.z.toFixed(1)})`, () => {
            const { coarse, fine } = scanDrag(drag);

            expect(coarse.crossSection.length).toBe(0);
            expect(coarse.maxCrossSectionExcess).toBe(0);
            expect(coarse.longitudinal.length).toBe(0);
            expect(coarse.maxLongitudinalExcess).toBe(0);
            expect(coarse.sampleCount).toBeGreaterThan(0);

            expect(fine.crossSection.length).toBe(0);
            expect(fine.maxCrossSectionExcess).toBe(0);
            expect(fine.longitudinal.length).toBe(0);
            expect(fine.maxLongitudinalExcess).toBe(0);
            expect(fine.sampleCount).toBeGreaterThan(0);
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

describe("edit — no no-op frames in the drag's target derivation (stage 9)", () => {
    // The clamp-never-refuse law applied to the drag's whole derivation chain, not just the constraints
    // at its end. Three no-op paths shipped: stale cursor coordinates (the engine seam, fixed in
    // `input/index.ts`), a missed march holding `lastValidTarget` (fixed: the miss now projects onto
    // the world bound), and a null ray skipping the frame outright (fixed: the `if (dragging && ray)`
    // guard is gone). The contract: while a drag is held, every frame produces a target.
    //
    // RED-FIRST WITNESS: the mutation that produces the red is removing `clampToBound` from every
    // return path of `projectRayToBound`, so a shallow ray (e.g. dir [0.999, -0.001, 0.001] from
    // origin [0, 200, 0]) yields a target far outside the world bound. The arm then fails its `|x| ≤
    // Bound` assertion. The failure text witnessed before the fix:
    //   "shallow past MARCH_MAX (x-axis): |x|=199800 past bound 508" / "Expected: <= 508" / "Received: 199800"
    //
    // History (not this arm's output): against the shipped shape, the old `marchFlattenField` returned
    // null on a miss and the caller held `lastValidTarget` — the handle froze under a moving cursor.
    // That defect motivated replacing the null return with `projectRayToBound`, but this arm does not
    // call `marchFlattenField`; it discriminates the clamp on `projectRayToBound`'s return paths.
    //
    // The companion arm asserts `lastValidTarget` has no readers — a hold path that still exists
    // anywhere in `edit.ts` after this stage is the finding.

    const Bound = WORLD_HALF - ROAD_HALF_WIDTH;

    test("projectRayToBound returns a target inside the world bound for every ray direction", () => {
        // a camera position above the world centre, looking outward
        const origin: [number, number, number] = [0, 200, 0];
        // a scan of ray directions: aimed at the sky, aimed past MARCH_MAX, and normal downward rays
        const directions: { dir: [number, number, number]; label: string }[] = [
            // aimed at the sky (dy > 0) — the ray goes up, never crosses the ground
            { dir: [0.3, 0.5, 0.4], label: "skyward (up-right)" },
            { dir: [-0.2, 0.8, 0.1], label: "skyward (up-left)" },
            { dir: [0, 1, 0], label: "straight up" },
            // aimed so the crossing sits past MARCH_MAX — very shallow downward angle
            { dir: [0.999, -0.001, 0.001], label: "shallow past MARCH_MAX (x-axis)" },
            { dir: [0.001, -0.001, 0.999], label: "shallow past MARCH_MAX (z-axis)" },
            // normal downward rays
            { dir: [0.3, -0.7, 0.4], label: "downward (right-forward)" },
            { dir: [-0.5, -0.5, -0.5], label: "downward (left-back)" },
            { dir: [0, -1, 0], label: "straight down" },
            // horizontal rays
            { dir: [1, 0, 0], label: "horizontal (x-axis)" },
            { dir: [0, 0, 1], label: "horizontal (z-axis)" },
        ];

        for (const { dir, label } of directions) {
            const [x, z] = projectRayToBound(origin, dir);
            expect(
                Math.abs(x),
                `${label}: |x|=${Math.abs(x)} past bound ${Bound}`,
            ).toBeLessThanOrEqual(Bound);
            expect(
                Math.abs(z),
                `${label}: |z|=${Math.abs(z)} past bound ${Bound}`,
            ).toBeLessThanOrEqual(Bound);
        }
    });

    // Named for what it checks and not for the caller's property: `projectRayToBound` is stateless, so
    // pairwise distinctness over a ray scan discriminates a constant return and nothing about frame
    // history. "Every frame of a held drag moves the endpoint" is carried by the clamp arms above plus
    // the no-hold-path arm below, not here (close sweep, N4).
    test("projectRayToBound is injective over a ray scan — distinct rays yield distinct targets", () => {
        const origin: [number, number, number] = [0, 200, 0];
        // a scan of distinct ray directions — each should produce a distinct target
        const directions: [number, number, number][] = [
            [0.3, 0.5, 0.4],
            [-0.2, 0.8, 0.1],
            [0.999, -0.001, 0.001],
            [0.3, -0.7, 0.4],
            [-0.5, -0.5, -0.5],
            [1, 0, 0],
            [0, 0, 1],
            [0.7, 0.1, -0.7],
        ];

        const targets = directions.map((dir) => projectRayToBound(origin, dir));
        // every pair of distinct directions should yield a distinct target (not the same hold point)
        for (let i = 0; i < targets.length; i++) {
            for (let j = i + 1; j < targets.length; j++) {
                const [xi, zi] = targets[i];
                const [xj, zj] = targets[j];
                const same = xi === xj && zi === zj;
                expect(
                    same,
                    `directions ${i} and ${j} both yielded (${xi}, ${zi}) — target did not change`,
                ).toBe(false);
            }
        }
    });

    test("projectRayToBound from an off-centre camera still lands inside the bound", () => {
        // camera at a corner, looking inward and upward
        const origin: [number, number, number] = [400, 100, -400];
        const directions: [number, number, number][] = [
            [-0.5, 0.5, 0.5], // up and inward
            [-0.3, -0.3, 0.3], // down and inward
            [0, 1, 0], // straight up
            [-1, 0, 0], // horizontal inward
        ];
        for (const dir of directions) {
            const [x, z] = projectRayToBound(origin, dir);
            expect(Math.abs(x)).toBeLessThanOrEqual(Bound);
            expect(Math.abs(z)).toBeLessThanOrEqual(Bound);
        }
    });

    test("the identifier lastValidTarget appears nowhere in edit.ts", async () => {
        // Extent, stated because the old name implied more: this greps **one identifier**. A hold path
        // under any other name passes it, so it witnesses that *this* variable is gone and not that no
        // hold path exists — that is a reviewer's read of the drag block, not an arm (close sweep, N4).
        const source = await Bun.file(`${import.meta.dir}/edit.ts`).text();
        expect(
            source.includes("lastValidTarget"),
            "edit.ts still references lastValidTarget — the hold path was not fully removed",
        ).toBe(false);
    });
});
