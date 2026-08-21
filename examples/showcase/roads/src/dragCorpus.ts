// The frozen drag fixture the corridor-flatness scan runs over, shared by exactly two readers so the
// default suite and the by-path tier scan the *same* corpus rather than two hand-picked ones
// (`coding.md` Suite speed: a golden gate leaving the default suite leaves a cheap sentinel behind
// against the same frozen fixture):
//   - `editCorridor.tier.ts` — the full 200-drag corpus, run by path when `flatness.ts`, `editPure.ts`,
//     `overlay/network.ts`, `terrain/flatten.ts` or `terrain/profile.ts` is touched.
//   - `edit.test.ts` — the sentinel, the corpus's own first `SENTINEL_DRAGS` entries.
// The corpus is a prefix-stable sequence: `dragCorpus(n)` is the first `n` entries of `dragCorpus(m)`
// for any `m >= n`, so the sentinel is a slice of the tier's population and not a second fixture.

import { meshHeightAt } from "./capture";
import { applyEdit, clampDragTarget, clampToBound } from "./editPure";
import { buildBandedLatticeVertices, checkSurfaceFlatness, type FlatnessResult } from "./flatness";
import { generateNetwork } from "./overlay/network";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";
import { CELLS, SPACING, WORLD_HALF } from "./terrain/grid";
import { makePermutation } from "./terrain/noise";
import { heightAtCpu, MAX_GRADE } from "./terrain/profile";
import { SEED } from "./terrain/terrain";

/** a seeded RNG — deterministic so a failing drag reproduces. */
export function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** how many of the corpus's drags the default suite keeps as its sentinel. */
export const SENTINEL_DRAGS = 5;

/** the whole corpus the by-path tier scans. */
export const CORPUS_DRAGS = 200;

export interface CorpusDrag {
    readonly end: 0 | 1;
    readonly x: number;
    readonly z: number;
}

/**
 * the first `count` drags of the frozen corpus: random targets drawn from the *unbounded* band (wider
 * than the world, so past-bound and past-old-ceiling targets are included), passed through
 * `clampToBound` + `clampDragTarget` the way the live drag does, and filtered by grade alone — the
 * flatness oracle's longitudinal bound is `MAX_GRADE`, so a chord steeper than that produces
 * violations by design rather than by reconstruction error.
 *
 * @example
 * const drags = dragCorpus(5); // the sentinel slice
 */
export function dragCorpus(count: number): CorpusDrag[] {
    const rng = mulberry32(12345);
    const perm = makePermutation(SEED);
    const drags: CorpusDrag[] = [];
    let attempts = 0;
    while (drags.length < count && attempts < 50000) {
        attempts++;
        const end = (rng() < 0.5 ? 0 : 1) as 0 | 1;
        const x = (rng() * 2 - 1) * WORLD_HALF * 3;
        const z = (rng() * 2 - 1) * WORLD_HALF * 3;
        const [cx, cz] = clampToBound(x, z);
        const [fx, fz] = clampDragTarget(generateNetwork(), end, cx, cz);
        const edited = applyEdit(generateNetwork(), end, fx, fz);
        const [a, b] = edited.polylines[0].points;
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const hA = heightAtCpu(a[0], a[1], perm);
        const hB = heightAtCpu(b[0], b[1], perm);
        if (Math.abs(hB - hA) / len > MAX_GRADE) continue;
        drags.push({ end, x: fx, z: fz });
    }
    return drags;
}

/**
 * the corridor-flatness readings for one corpus drag, at `SPACING` and at `SPACING/2` — one derivation
 * read by both the tier and the sentinel, so the two cannot drift into asserting different things.
 *
 * The lattice is the **banded** builder: `checkSurfaceFlatness` only ever samples strictly inside the
 * footprint, which is exactly the reader `buildBandedLatticeVertices` is valid for, and the full
 * builder spent ~2.6 s per drag rebuilding 257² + 513² vertices of which ~1 % were ever read.
 */
export function scanDrag(drag: CorpusDrag): { coarse: FlatnessResult; fine: FlatnessResult } {
    const perm = makePermutation(SEED);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);
    const edited = applyEdit(generateNetwork(), drag.end, drag.x, drag.z);
    const { segments, cutDepth } = buildNetworkGeometry(edited, SEED);
    const falloff = computeFalloff(cutDepth);

    const coarseRaw = buildBandedLatticeVertices(SPACING, CELLS, segments, falloff, natural);
    const coarse = checkSurfaceFlatness(
        (sx, sz) => meshHeightAt(coarseRaw, sx, sz, SPACING, CELLS),
        edited,
    );

    const fineSpacing = SPACING / 2;
    const fineCells = CELLS * 2;
    const fineRaw = buildBandedLatticeVertices(fineSpacing, fineCells, segments, falloff, natural);
    const fine = checkSurfaceFlatness(
        (sx, sz) => meshHeightAt(fineRaw, sx, sz, fineSpacing, fineCells),
        edited,
    );

    return { coarse, fine };
}
