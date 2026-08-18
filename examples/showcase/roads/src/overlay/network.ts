import { WORLD_HALF } from "../terrain/grid";
import { mulberry32 } from "../terrain/noise";
import type { PolygonStamp, Polyline, StrokeDocument } from "./document";

// The seeded procedural network: "a handful" of straight-segment roads plus one carpark polygon, placed
// by a deterministic RNG (`terrain/noise.ts`'s `mulberry32`, the same seeded-RNG shape the permutation
// table uses — one source, `coding.md`'s One-source-of-truth). Pure XZ placement, no engine/GPU imports
// (the same device-free split `overlay/tiles.ts`/`overlay/document.ts` use) — `bun test` exercises the
// generator's determinism with no device.
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
// truncating its own AABB rather than reading as a placement bug — the margin is one road's own maximum
// half-width-plus-length contribution, generous enough that no primitive this generator emits can reach
// the clamp.
const WORLD_MARGIN = ROAD_MAX_LENGTH / 2 + CARPARK_HALF * 2;

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
