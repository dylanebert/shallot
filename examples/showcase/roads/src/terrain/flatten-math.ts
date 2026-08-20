// The pure-math half of `flatten.ts` — the flatten field's own constants and the falloff derivation,
// extracted so modules that only need the math (not the GPU kernel) can import without pulling in
// `@dylanebert/shallot`'s device-bound module graph. `test/roads.spec.ts` imports `captureProbePoints`
// from `overlay/network.ts`, which only needs `computeFalloff` and `FLAT_CORE_MARGIN`; importing these
// from `flatten.ts` would transitively load `@dylanebert/shallot`'s `loading/index.ts`, which imports
// `package.json` without the `with { type: "json" }` attribute Node v26+ requires — a pre-existing
// package-level issue that blocks Playwright's Node-side test loading. This module breaks that chain
// by depending only on `./grid`'s `SPACING`.

import { SPACING } from "./grid";

// FALLOFF is no longer `= SPACING` (stage 6's choice, derived only from a capture probe's own grid
// alignment, not a road-geometry argument). Once the profile is the straight chord the cut depth at a given point
// can grow past what a fixed 4 m band eases gracefully — a deep cut eased back over 4 m reads as a cliff
// wall, not a shoulder. {@link computeFalloff} re-derives it per network, from the network's own measured
// cut depth (the largest |natural - target| the flatten pipeline actually produces this reseed) and a
// cited side-slope limit, so the transition is only ever as wide as the deepest cut demands.
//
// AASHTO's Roadside Design Guide draws the line between a "traversable, non-recoverable" and a "critical,
// non-traversable" roadside slope at 1V:3H (~33%) — steeper reads as a cliff face to a driver leaving the
// shoulder. That's the target slope for the flatten transition's own *steepest* point, not its average:
// the cosine ease `0.5 - 0.5·cos(π·t)` has derivative `0.5·π·sin(π·t)`, peaking at `t = 0.5` (the
// transition's midpoint) at `π/2` — so a cut of depth `D` eased over a falloff distance `F` peaks at slope
// `D·(π/2)/F`. Solving for `F` at the side-slope limit gives the derivation below.
export const SIDE_SLOPE_LIMIT = 1 / 3; // AASHTO Roadside Design Guide, 1V:3H, rise/run

// Stage 15 (2026-08-19, the reconstruction-axis fix `flatness.ts` diagnosed): a triangle straddling the
// footprint edge can have a corner *outside* the old `halfWidth`-radius flat core, eased toward off-road
// terrain that varies by metres over one grid cell — the mesh reads that blend at a point whose analytic
// position is still inside the road. Widening the flat core (the `coreDist <= 0` region `flattenHeight`
// hands `targetHeight` outright) by a grid cell's own diagonal guarantees every vertex that can support
// a footprint-intersecting triangle sits inside it: a cell is `SPACING` square, so the farthest any of a
// straddling triangle's corners can sit from the footprint edge is that cell's diagonal, `√2·SPACING`.
// Cell geometry, not a fitted number.
export const FLAT_CORE_MARGIN = Math.SQRT2 * SPACING;

/** the falloff distance (metres) whose cosine-ease transition peaks at exactly {@link SIDE_SLOPE_LIMIT}
 *  for a cut of `cutDepth` metres — floored at {@link SPACING}, since a transition narrower than the mesh's
 *  own vertex spacing can't be resolved by the heightfield regardless of what the formula asks for.
 * @example computeFalloff(0) // SPACING — no cut, the floor wins
 * @example computeFalloff(4) // (Math.PI / 2) * 4 / SIDE_SLOPE_LIMIT, well past the floor */
export function computeFalloff(cutDepth: number): number {
    return Math.max(SPACING, ((Math.PI / 2) * cutDepth) / SIDE_SLOPE_LIMIT);
}
