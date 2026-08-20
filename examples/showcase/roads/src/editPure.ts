// The pure, device-free halves of the edit plugin — `applyEdit`, `clampToBound`, the admissibility
// predicate, and the chord-length / tile-count helpers. This module imports nothing from
// `@dylanebert/shallot`, so importing it under `bun test` (or under Playwright's Node side) does not
// pull in the engine's device-bound module graph — the Node ≥26 hazard Approach stage 4 names (the
// package's bare `package.json` import is rejected there). The device-bound plugin (`edit.ts`) imports
// these and re-exports them for its own consumers; `edit.test.ts` exercises the pure halves from here.

import type { StrokeDocument } from "./overlay/document";
import { documentDirtyTiles } from "./overlay/document";
import { ROAD_HALF_WIDTH, ROAD_MAX_LENGTH, ROAD_MIN_LENGTH } from "./overlay/network";
import { WORLD_HALF } from "./terrain/grid";

// --- pure, device-free helpers (no @dylanebert/shallot imports) ---

/** the world-bound margin: keeps the handle sphere (radius `halfWidth/2`) and the road's own half-width
 *  inside the grid so the road never extends past the terrain's edge. Derived from the road, like the
 *  handle radius. */
const BOUND_MARGIN = ROAD_HALF_WIDTH;

/** the handle's world-space radius — half the road's half-width, so the handle sits on the road without
 *  dwarfing it. The sphere mesh has radius 0.5, so the Transform scale is `HANDLE_RADIUS / 0.5`. */
export const HANDLE_RADIUS = ROAD_HALF_WIDTH / 2;

/** clamp (x, z) to the world bounds so the handle never leaves the grid: `|x|, |z| ≤ WORLD_HALF − margin`.
 *  Pure, device-free. */
export function clampToBound(x: number, z: number): [number, number] {
    const bound = WORLD_HALF - BOUND_MARGIN;
    return [Math.max(-bound, Math.min(bound, x)), Math.max(-bound, Math.min(bound, z))];
}

/** the chord length of a one-road document — the distance between its two endpoints. */
export function chordLength(doc: StrokeDocument): number {
    const [a, b] = doc.polylines[0].points;
    return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

/** pure, device-free: return a new document with endpoint `end` (0 or 1) moved to `(x, z)`. The input
 *  document is not mutated — a new `StrokeDocument` is returned with the moved endpoint replacing the
 *  original, every other point unchanged. */
export function applyEdit(doc: StrokeDocument, end: 0 | 1, x: number, z: number): StrokeDocument {
    const line = doc.polylines[0];
    const points = [...line.points] as [number, number][];
    points[end] = [x, z];
    return {
        polylines: [{ points, halfWidth: line.halfWidth }],
    };
}

/** whether moving endpoint `end` to `(x, z)` keeps the chord within the `ROAD_MIN_LENGTH`–`ROAD_MAX_LENGTH`
 *  band. The ceiling is atlas capacity, not taste: a chord past `ROAD_MAX_LENGTH` touches more than
 *  `ATLAS_LAYERS` tiles, and `allocate` throws `capacity exceeded (64 layers)` mid-drag. Pure,
 *  device-free. */
export function isAdmissibleDrag(doc: StrokeDocument, end: 0 | 1, x: number, z: number): boolean {
    const edited = applyEdit(doc, end, x, z);
    const len = chordLength(edited);
    return len >= ROAD_MIN_LENGTH && len <= ROAD_MAX_LENGTH;
}

/** the number of atlas tiles a document touches — `documentDirtyTiles`'s count, which is the exact
 *  per-primitive union the overlay's own oracle pins. Device-free (delegates to `document.ts`'s pure
 *  math). */
export function residentTileCount(doc: StrokeDocument): number {
    return documentDirtyTiles(doc).length;
}
