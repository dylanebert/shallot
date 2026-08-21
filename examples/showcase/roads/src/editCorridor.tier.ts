import { describe, expect, test } from "bun:test";
import { CORPUS_DRAGS, dragCorpus, scanDrag } from "./dragCorpus";

// By-path tier (`coding.md` Suite speed): the whole 200-drag corridor-flatness corpus, moved out of the
// default suite because it costs ~12 s there against a whole-suite budget of ~30 s, while the property
// it pins is one corpus-scale oracle over one kernel. Coverage moved tiers and did not shrink — the
// default suite keeps `edit.test.ts`'s sentinel over the *same* frozen fixture's first
// `SENTINEL_DRAGS` entries (`dragCorpus.ts`).
//
// Run by path, one file per invocation:
//   bun test ./examples/showcase/roads/src/editCorridor.tier.ts     (from the shallot root)
//
// Trigger — every module this scan's readings are a function of, so a touch to any of them re-runs it:
// `src/flatness.ts` (the banded builder + the oracle), `src/dragCorpus.ts` (the fixture and the scan),
// `src/editPure.ts` (the clamps the corpus is drawn through), `src/capture.ts` (`meshHeightAt`),
// `src/overlay/network.ts` (the chord and its halfWidth), `src/terrain/flatten.ts` (the flatten field),
// `src/terrain/grid.ts` (`SPACING`/`CELLS`), `src/terrain/profile.ts` (`heightAtCpu`, `MAX_GRADE`) and
// `src/terrain/generate.ts` (`TERRAIN_QUANT`).

describe(`edit — corridor flatness over ${CORPUS_DRAGS} random clamped drags`, () => {
    // The spec's Validation "Corridor flatness" criterion, extended to edited documents: over the whole
    // corpus, `checkSurfaceFlatness` reads exactly 0 violations / 0.0000 m on both axes at SPACING and
    // SPACING/2. The chord's affine target makes the in-corridor reconstruction error vanish
    // identically (barycentric interpolation reproduces an affine field exactly at any cell size and
    // any road angle), so this reads zero on every clamped drag — not just the boot document.
    //
    // The corpus draws from the *unbounded* band (stage 4c) and clamps via `clampToBound` +
    // `clampDragTarget` before `applyEdit`; the only filter is grade (`dragCorpus.ts`).
    //
    // The lattice is the banded builder, whose equivalence to the full builder for this oracle is the
    // null control in `flatness.test.ts` ("the banded lattice reads what the full lattice reads").
    const drags = dragCorpus(CORPUS_DRAGS);

    test(`the corpus is fully populated (${CORPUS_DRAGS} admissible drags)`, () => {
        expect(drags.length).toBe(CORPUS_DRAGS);
    });

    for (let i = 0; i < drags.length; i++) {
        const drag = drags[i];
        test(`drag ${i + 1}/${CORPUS_DRAGS}: end ${drag.end} → (${drag.x.toFixed(1)}, ${drag.z.toFixed(1)})`, () => {
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
