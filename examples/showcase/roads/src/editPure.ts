// The pure, device-free halves of the edit plugin — `applyEdit`, `clampToBound`, the drag-floor clamp,
// the ray-to-bound projection and `chordLength`. This module imports nothing from
// `@dylanebert/shallot`, so importing it under `bun test` (or under Playwright's Node side) does not
// pull in the engine's device-bound module graph — the Node ≥26 hazard Approach stage 4 names (the
// package's bare `package.json` import is rejected there). The device-bound plugin (`edit.ts`) imports
// these and re-exports them for its own consumers; `edit.test.ts` exercises the pure halves from here.

import type { StrokeDocument } from "./overlay/document";
import { ROAD_HALF_WIDTH, ROAD_MIN_LENGTH } from "./overlay/network";
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

/** clamp a drag target so the resulting chord holds the {@link ROAD_MIN_LENGTH} floor. If moving
 *  endpoint `end` to `(x, z)` would shorten the chord under `ROAD_MIN_LENGTH`, the target is projected
 *  along the drag direction (from the endpoint's current position toward `(x, z)`) to the nearest point
 *  that maintains the floor — the last valid position on the drag ray. A target already at or above the
 *  floor is returned unchanged. Pure, device-free.
 *
 *  The ceiling (`ROAD_MAX_LENGTH`) is deleted: a chord is any length the world contains, and capacity is
 *  sized by measurement — the worst-case single-document swath under the capsule dirty-set test, sized
 *  with headroom over the measured worst case — not by a refusal (`roads-interactive.md`'s Locked
 *  decision). See `overlay/tiles.ts`'s `ATLAS_LAYERS` for the derivation and the measured number. */
export function clampDragTarget(
    doc: StrokeDocument,
    end: 0 | 1,
    x: number,
    z: number,
): [number, number] {
    const line = doc.polylines[0];
    const other = (end === 0 ? line.points[1] : line.points[0]) as [number, number];
    const current = line.points[end] as [number, number];
    const [ox, oz] = other;
    const [px, pz] = current;

    // distance from target to the other endpoint
    const tdx = x - ox;
    const tdz = z - oz;
    const tdist = Math.hypot(tdx, tdz);
    if (tdist >= ROAD_MIN_LENGTH) return [x, z];

    // drag direction: from current position toward target
    const ddx = x - px;
    const ddz = z - pz;
    const dlen = Math.hypot(ddx, ddz);
    if (dlen < 1e-9) return [x, z]; // target equals current — no-op

    // solve |V + s*D|^2 = ROAD_MIN_LENGTH^2 for s, where V = P - O, D = (ddx, ddz)
    // the chord is valid at s=0 (current) and invalid at s=1 (target), so the crossing in (0, 1] is
    // the smaller root — the last valid point along the drag ray
    const vx = px - ox;
    const vz = pz - oz;
    const a = ddx * ddx + ddz * ddz;
    const b = 2 * (vx * ddx + vz * ddz);
    const c = vx * vx + vz * vz - ROAD_MIN_LENGTH * ROAD_MIN_LENGTH;
    const disc = b * b - 4 * a * c;
    if (disc < 0) {
        // no intersection — shouldn't happen if current is valid and target is invalid;
        // fallback: project target onto the floor circle
        return [ox + (tdx / tdist) * ROAD_MIN_LENGTH, oz + (tdz / tdist) * ROAD_MIN_LENGTH];
    }
    const s = (-b - Math.sqrt(disc)) / (2 * a);
    if (s < 0) {
        return [ox + (tdx / tdist) * ROAD_MIN_LENGTH, oz + (tdz / tdist) * ROAD_MIN_LENGTH];
    }
    return [px + ddx * s, pz + ddz * s];
}

/** Project a ray onto the world bound — find where the ray's (x, z) trajectory hits the world bound
 *  box, clamped to the bound. Used when the march misses (the ray doesn't cross the surface within
 *  `MARCH_MAX`), so the drag still has a target inside the world bound for every frame — the
 *  clamp-never-refuse law applied to the target's derivation, not just its constraints. Pure,
 *  device-free.
 *
 *  RED-FIRST WITNESS (stage 9, roads-interactive.md): the mutation that produces the red is removing
 *  `clampToBound` from every return path of this function, so a shallow ray (e.g. dir [0.999, -0.001,
 *  0.001] from origin [0, 200, 0]) yields a target far outside the world bound. The arm then fails
 *  its `|x| ≤ Bound` assertion. The failure text witnessed before the fix:
 *   "shallow past MARCH_MAX (x-axis): |x|=199800 past bound 508" / "Expected: <= 508" / "Received: 199800"
 *
 *  History (not this arm's output): against the shipped shape, the *old* `marchFlattenField` returned
 *  null on a miss and the caller held `lastValidTarget` — the handle froze under a moving cursor. That
 *  defect is what motivated replacing the null return with `projectRayToBound`, but this arm does not
 *  call `marchFlattenField`; it discriminates the clamp on `projectRayToBound`'s return paths.
 */
export function projectRayToBound(
    origin: readonly [number, number, number],
    dir: readonly [number, number, number],
): [number, number] {
    const [ox, oy, oz] = origin;
    const [dx, dy, dz] = dir;
    const bound = WORLD_HALF - BOUND_MARGIN;

    // If the ray goes downward, find where it crosses the ground plane (y = 0) and clamp to the bound.
    // This handles a march that misses because the crossing sits past MARCH_MAX — the ray does point at
    // the ground, just very far away.
    if (dy < -1e-9) {
        const t = -oy / dy;
        if (t > 0) {
            return clampToBound(ox + t * dx, oz + t * dz);
        }
    }

    // The ray goes upward or is horizontal — find where its (x, z) projection exits the world bound
    // box. Parameterize the ray in (x, z): (ox + t·dx, oz + t·dz). Find the smallest t > 0 where the
    // point is on the box boundary (|x| = bound or |z| = bound) and the other coordinate is within.
    let bestT = Infinity;
    if (Math.abs(dx) > 1e-9) {
        for (const t of [(bound - ox) / dx, (-bound - ox) / dx]) {
            if (t > 0) {
                const z = oz + t * dz;
                if (Math.abs(z) <= bound + 1e-6 && t < bestT) bestT = t;
            }
        }
    }
    if (Math.abs(dz) > 1e-9) {
        for (const t of [(bound - oz) / dz, (-bound - oz) / dz]) {
            if (t > 0) {
                const x = ox + t * dx;
                if (Math.abs(x) <= bound + 1e-6 && t < bestT) bestT = t;
            }
        }
    }

    if (bestT === Infinity) return clampToBound(ox, oz);
    return clampToBound(ox + bestT * dx, oz + bestT * dz);
}
