import { gridX, gridZ, SPACING, WORLD_HALF, worldX, worldZ } from "../terrain/grid";
import { mulberry32 } from "../terrain/noise";
import {
    documentDistance,
    type PolygonStamp,
    type Polyline,
    type StrokeDocument,
} from "./document";

// The seeded procedural network: "a handful" of straight-segment roads plus one carpark polygon, placed
// by a deterministic RNG (`terrain/noise.ts`'s `mulberry32`, the same seeded-RNG shape the permutation
// table uses — one source, not two independently authored copies). Pure XZ placement, no engine/GPU
// imports (the same device-free split `overlay/tiles.ts`/`overlay/document.ts` use) — `bun test`
// exercises the generator's determinism with no device.
//
// "Terrain-conformed" is the flatten step's job, not this one's (`terrain/flatten.ts`): a road's
// *longitudinal* profile follows the natural terrain height along its own centreline, so a straight-line
// XZ path is already terrain-conformed once flattened — this module only needs to place plausible XZ
// geometry, never reach for the terrain's height function itself.

export const ROAD_COUNT = 5; // "a handful" — enough to read as a network, not a maze
export const ROAD_HALF_WIDTH = 4; // metres — matches terrain/grid.ts's SPACING, `overlay/stroke.ts`'s own convention
const ROAD_MIN_LENGTH = 80; // metres
const ROAD_MAX_LENGTH = 220; // metres — keeps a single road well under the world's 1024 m span
const CARPARK_HALF = 20; // metres — half-extent of the one carpark stamp (a 40 m × 40 m lot)
const CARPARK_MARGIN = 4; // metres — clearance between the carpark's near edge and the anchor road's own edge

// keep every primitive off the world edge: a road endpoint or carpark corner within this margin of
// WORLD_HALF would push `overlay/tiles.ts`'s `dirtyTiles` clamp to the boundary column, silently
// truncating its own AABB rather than reading as a placement bug. Derived, not tuned — two terms, proven
// by construction, not by sampling:
//
//   1. A road's *far* endpoint is `x0 + cos(heading) * length`, reaching the FULL `ROAD_MAX_LENGTH` from
//      its start in either direction — not half of it (the bug an earlier version of this constant had:
//      `ROAD_MAX_LENGTH / 2` bounded only the start point, so a full-length road placed near the old
//      margin's edge still walked its endpoint past WORLD_HALF — demonstrated at seed 615,
//      `network.test.ts`'s own pinned regression witness).
//   2. The carpark's own reach *beyond whichever road primitive it's anchored to* — {@link CARPARK_REACH}
//      below — since its centre is offset from road 0's midpoint by up to `ROAD_HALF_WIDTH +
//      CARPARK_MARGIN + CARPARK_HALF`, and a corner reaches a further `CARPARK_HALF` past its own centre.
//
// With `bound = WORLD_HALF - WORLD_MARGIN` bounding a road's *start* point (x0, z0): a road endpoint
// satisfies `|x1| = |x0 + cosθ·length| <= bound + ROAD_MAX_LENGTH = WORLD_HALF - CARPARK_REACH` (and the
// same for z, sinθ); road 0's midpoint sits between its own two endpoints, so it inherits that same
// `WORLD_HALF - CARPARK_REACH` bound too; and a carpark corner, reaching at most CARPARK_REACH beyond that
// midpoint, therefore stays within exactly `WORLD_HALF` — never past it, on either axis, for any seed.
const CARPARK_REACH = ROAD_HALF_WIDTH + CARPARK_MARGIN + CARPARK_HALF * 2; // 48
const WORLD_MARGIN = ROAD_MAX_LENGTH + CARPARK_REACH;

/** the seeded procedural road network: {@link ROAD_COUNT} straight single-segment roads at random
 *  positions/headings/lengths within the world footprint (minus {@link WORLD_MARGIN}), plus one carpark
 *  polygon anchored perpendicular to the first road's midpoint. Deterministic in `seed` — the same seed
 *  always returns the identical document (`network.test.ts` pins this directly against
 *  `structuredClone`-style deep equality); a different seed almost certainly returns a different one (RNG
 *  collision across every coordinate is the only way it wouldn't). */
export function generateNetwork(seed: number): StrokeDocument {
    const rng = mulberry32(seed);
    const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
    const bound = WORLD_HALF - WORLD_MARGIN;

    const polylines: Polyline[] = [];
    for (let i = 0; i < ROAD_COUNT; i++) {
        const x0 = rand(-bound, bound);
        const z0 = rand(-bound, bound);
        const heading = rand(0, Math.PI * 2);
        const length = rand(ROAD_MIN_LENGTH, ROAD_MAX_LENGTH);
        const x1 = x0 + Math.cos(heading) * length;
        const z1 = z0 + Math.sin(heading) * length;
        polylines.push({
            points: [
                [x0, z0],
                [x1, z1],
            ],
            halfWidth: ROAD_HALF_WIDTH,
        });
    }

    // one carpark, offset laterally from the first road's midpoint by its own half-extent plus the
    // clearance margin — clear of the road's own footprint, not overlapping it.
    const [[ax, az], [bx, bz]] = polylines[0].points;
    const mx = (ax + bx) / 2;
    const mz = (az + bz) / 2;
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz) || 1;
    const nx = -dz / len; // unit perpendicular to the road direction
    const nz = dx / len;
    const offset = ROAD_HALF_WIDTH + CARPARK_MARGIN + CARPARK_HALF;
    const cx = mx + nx * offset;
    const cz = mz + nz * offset;
    const polygon: PolygonStamp = {
        points: [
            [cx - CARPARK_HALF, cz - CARPARK_HALF],
            [cx + CARPARK_HALF, cz - CARPARK_HALF],
            [cx + CARPARK_HALF, cz + CARPARK_HALF],
            [cx - CARPARK_HALF, cz + CARPARK_HALF],
        ],
    };

    return { polylines, polygons: [polygon] };
}

/**
 * deterministic capture-gate probe points for {@link generateNetwork}'s output at `seed` — grid-aligned
 * so the capture's world→screen bridge can read the real generated terrain height at an exact grid vertex
 * (`terrain/grid.ts`'s `gridX`/`gridZ`), generalizing `overlay/stroke.ts`'s hand-authored on/off-road pair
 * (fixed, axis-aligned points valid only for that stroke's own straight-along-+X shape) to a network whose
 * roads land at arbitrary headings and positions.
 *
 * Picks whichever road's midpoint sits nearest the world origin, not always `polylines[0]`: the scene's
 * fixed orbit camera (`public/scenes/roads.scene`) frames the whole 1024 m grid from its default target,
 * the world origin, so a road far out near the world's edge projects to only a few screen pixels — its
 * on/off-road pair would then sit closer than one screen pixel apart, sampling the same rounded pixel
 * twice (measured: a seed whose nearest road sat 260 m out read identical on/off-road luminance on the
 * real device). The on-road point is that midpoint's nearest grid vertex; the off-road point steps
 * outward one grid cell at a time along the segment's own normal, trying both normal directions since
 * only `polylines[0]` has a carpark anchored to one particular side (`generateNetwork`, above) — whichever
 * direction clears the road first is a clean single-boundary crossing. Both points are confirmed
 * inside/outside by {@link documentDistance} at derivation time rather than assumed from the geometry
 * alone, so a network change that breaks the pairing fails loud instead of silently reading the wrong
 * pixels.
 */
export function captureProbePoints(seed: number): {
    onRoad: readonly [x: number, z: number];
    offRoad: readonly [x: number, z: number];
} {
    const doc = generateNetwork(seed);

    let nearest = doc.polylines[0];
    let nearestDist = Number.POSITIVE_INFINITY;
    for (const line of doc.polylines) {
        const [[ax, az], [bx, bz]] = line.points;
        const d = Math.hypot((ax + bx) / 2, (az + bz) / 2);
        if (d < nearestDist) {
            nearestDist = d;
            nearest = line;
        }
    }

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
            `captureProbePoints: derived on-road point (${onRoad[0]}, ${onRoad[1]}) is not inside the network at seed ${seed}`,
        );
    }

    for (const sign of [1, -1]) {
        for (let step = 1; step <= 32; step++) {
            const reach = halfWidth + step * SPACING;
            const offRoad = snap(mx + sign * nx * reach, mz + sign * nz * reach);
            if (documentDistance(offRoad[0], offRoad[1], doc) > 0) return { onRoad, offRoad };
        }
    }
    throw new Error(
        `captureProbePoints: no off-road grid point found within search radius at seed ${seed}`,
    );
}
