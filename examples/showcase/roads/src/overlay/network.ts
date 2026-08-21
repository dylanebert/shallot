import { gridX, gridZ, SPACING, worldX, worldZ } from "../terrain/grid";
import { documentDistance, type Polyline, type StrokeDocument } from "./document";

// The road network: one straight-segment road, at a fixed standard chord — pure XZ placement (no
// engine/GPU imports — `bun test` exercises it with no device).
//
// One road cannot contend with itself, so this generator carries no clearance/non-overlap machinery
// (`roads-interactive.md`'s locked decision: "one road keeps both trivially: no overlap predicate, no
// per-primitive falloff, no junction" — deleted at stage 1 along with the carpark and every arm that
// existed to witness non-overlap).
//
// Route selection is gone too (`roads-interactive.md`'s 2026-08-20 rescope): its job was picking a chord
// that needs less earthwork, and the drag hands that job to the person instead — optimizing a road on an
// axis the interaction abandons at the first grab is a cost optimizer maintained for a state the user
// drags away from. The boot road is simply a standard road: one fixed chord, no placement search and no
// seeded draw. `generateNetwork` takes no seed; `regenerate(seed)`'s seed drives the terrain's noise
// permutation alone (`terrain/terrain.ts`), never the road's position.

export const ROAD_COUNT = 1;
export const ROAD_HALF_WIDTH = 4; // metres — matches terrain/grid.ts's SPACING, `overlay/stroke.ts`'s own convention
export const ROAD_MIN_LENGTH = 80; // metres — the drag's clamp floor: a drag that would shorten past
// this clamps the endpoint to the closest point holding the floor (stage 4c). Its reason is the artifact
// (a degenerate chord, an unreadable dash phase), not a buffer size — so it survives as a clamp.

/** the boot road's fixed standard chord — centred on the origin, along +X, length 200 m (above
 *  {@link ROAD_MIN_LENGTH}). Not seeded, not searched: a person's drag is what picks a route that needs
 *  less earthwork now (`roads-interactive.md`'s locked decision). */
const STANDARD_CHORD: readonly [readonly [number, number], readonly [number, number]] = [
    [-100, 0],
    [100, 0],
];

/** the chord (endpoints + halfWidth) of the one road in `doc` — the analytic fs's marking geometry
 *  input. One road means one chord; the fs receives this as a uniform and computes marking distances
 *  from it directly, without decoding a texel. */
export function chordOf(doc: StrokeDocument): {
    a: readonly [number, number];
    b: readonly [number, number];
    halfWidth: number;
} {
    const line = doc.polylines[0];
    return {
        a: line.points[0],
        b: line.points[1],
        halfWidth: line.halfWidth,
    };
}

/** the road network: {@link ROAD_COUNT} (one) straight single-segment road at the fixed
 *  {@link STANDARD_CHORD}. One road cannot contend with itself, so this carries no clearance/non-overlap
 *  machinery — that non-overlap guarantee, and every arm that existed to witness it, belonged to the
 *  multi-road generator this stage retired (`roads-interactive.md` stage 1). No seed: the road's position
 *  is authored, not drawn — a person's drag is the only thing that ever moves it (stage 4). */
export function generateNetwork(): StrokeDocument {
    const polylines: Polyline[] = [
        { points: [STANDARD_CHORD[0], STANDARD_CHORD[1]], halfWidth: ROAD_HALF_WIDTH },
    ];
    return { polylines };
}

/**
 * deterministic capture-gate probe points for {@link generateNetwork}'s output — grid-aligned so the
 * capture's world→screen bridge can read the real generated terrain height at an exact grid vertex
 * (`terrain/grid.ts`'s `gridX`/`gridZ`), generalizing `overlay/stroke.ts`'s hand-authored on/off-road pair
 * (fixed, axis-aligned points valid only for that stroke's own straight-along-+X shape) to the standard
 * chord.
 *
 * Reads `doc.polylines[0]` — the one road {@link ROAD_COUNT} ever generates — rather than re-deriving a
 * "nearest to origin" search over a set that's always size 1: relabeled from the prior multi-road
 * generator's own nearest-midpoint pick, not re-anchored (`roads-interactive.md` stage 1's footprint —
 * "probe points read road 0"). The on-road point is that midpoint's nearest grid vertex; the off-road
 * point steps outward one grid cell at a time along the segment's own normal, trying both normal
 * directions so a natural terrain dip on one side doesn't stall the search — whichever direction clears
 * the road first is a clean single-boundary crossing. Both points are confirmed inside/outside by
 * {@link documentDistance} at derivation time rather than assumed from the geometry alone, so a network
 * change that breaks the pairing fails loud instead of silently reading the wrong pixels.
 */
export function captureProbePoints(): {
    onRoad: readonly [x: number, z: number];
    offRoad: readonly [x: number, z: number];
} {
    const doc = generateNetwork();
    const nearest = doc.polylines[0];

    const [[ax, az], [bx, bz]] = nearest.points;
    const mx = (ax + bx) / 2;
    const mz = (az + bz) / 2;
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz) || 1;
    const nx = -dz / len;
    const nz = dx / len;
    const halfWidth = nearest.halfWidth;

    const snap = (x: number, z: number): [number, number] => [worldX(gridX(x)), worldZ(gridZ(z))];

    const onRoad = snap(mx, mz);
    if (documentDistance(onRoad[0], onRoad[1], doc) >= 0) {
        throw new Error(
            `captureProbePoints: derived on-road point (${onRoad[0]}, ${onRoad[1]}) is not inside the network`,
        );
    }

    for (const sign of [1, -1]) {
        for (let step = 1; step <= 32; step++) {
            const reach = halfWidth + step * SPACING;
            const offRoad = snap(mx + sign * nx * reach, mz + sign * nz * reach);
            if (documentDistance(offRoad[0], offRoad[1], doc) > 0) return { onRoad, offRoad };
        }
    }
    throw new Error("captureProbePoints: no off-road grid point found within search radius");
}
