import { describe, expect, test } from "bun:test";
import { CORPUS_DRAGS, dragCorpus, scanDrag } from "./dragCorpus";

// By-path tier (`coding.md` Suite speed): the whole 200-drag corridor-flatness corpus, run by path
// rather than in the default suite because it costs ~12 s there against a whole-suite budget of ~30 s,
// while the property it pins is one corpus-scale oracle over one kernel. The default suite keeps
// `edit.test.ts`'s sentinel over the *same* frozen fixture's first `SENTINEL_DRAGS` entries
// (`dragCorpus.ts`).
//
// Run by path, one file per invocation:
//   bun test ./examples/showcase/roads/src/editCorridor.tier.ts     (from the shallot root)
//
// Which edits are the cue to run it is this file's own transitive import cone — every `.ts` file reachable
// from this file's `./`-relative imports, walked mechanically. The tier imports `dragCorpus.ts` alone;
// `dragCorpus.ts` imports `capture.ts`, `editPure.ts`, `flatness.ts`, `overlay/network.ts` and
// `terrain/{flatten,grid,noise,profile,terrain}.ts`; `capture.ts` adds `overlay/document.ts`,
// `overlay/tiles.ts` and `terrain/generate.ts`; `terrain/terrain.ts` adds
// `overlay/{atlas,queue,rasterize,stroke}.ts` and `posts.ts`, and `posts.ts`
// adds `harness.ts` — 19 files beyond the tier itself. The cone is wider than the set of modules the
// scan's readings are a *function* of (`overlay/queue.ts` and `posts.ts` cannot move a flatness
// reading, they arrive because `terrain/terrain.ts` exports `SEED`): that over-inclusion is what a
// derived list costs and it is accepted rather than re-narrowed by hand (`checks.md`, by-path tier
// trigger lists). The tier reads no non-imported file at runtime, so there is no runtime-read input to
// list beside the cone.
//
// Advisory, not a trigger: nothing runs this tier automatically — a person reads this header and runs
// the command above when an edit touches one of these paths.

describe(`edit — corridor flatness over ${CORPUS_DRAGS} random clamped drags`, () => {
    // The spec's Validation "Corridor flatness" criterion, extended to edited documents: over the whole
    // corpus, `checkSurfaceFlatness` reads exactly 0 violations / 0.0000 m on both axes at SPACING and
    // SPACING/2. The chord's affine target makes the in-corridor reconstruction error vanish
    // identically (barycentric interpolation reproduces an affine field exactly at any cell size and
    // any road angle), so this reads zero on every clamped drag — not just the boot document.
    //
    // The corpus draws from the *unbounded* band and clamps via `clampToBound` +
    // `clampDragTarget` before `applyEdit`; the only filter is grade (`dragCorpus.ts`).
    //
    // The lattice is the banded builder, whose equivalence to the full builder for this oracle is the
    // null control in `flatness.test.ts` ("the banded lattice reads what the full lattice reads").
    const drags = dragCorpus(CORPUS_DRAGS);

    // A fixture-shape pin, not a witness for the corpus's grade filter: that filter rejects 0 of the
    // first 200 attempts, so this length can only move if the fixture's own draw or clamps change
    // (measurement beside the filter in `dragCorpus.ts`).
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
