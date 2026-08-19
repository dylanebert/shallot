import { computeFalloff, FLAT_CORE_MARGIN } from "../terrain/flatten";
import { gridX, gridZ, SPACING, WORLD_HALF, worldX, worldZ } from "../terrain/grid";
import { makePermutation, mulberry32 } from "../terrain/noise";
import { heightAtCpu, PROFILE_STEP } from "../terrain/profile";
import {
    documentDistance,
    type PolygonStamp,
    type Polyline,
    polygonDistance,
    type StrokeDocument,
    segmentDistance,
} from "./document";

// The seeded procedural network: "a handful" of straight-segment roads plus one carpark polygon, placed
// by a deterministic RNG (`terrain/noise.ts`'s `mulberry32`, the same seeded-RNG shape the permutation
// table uses — one source, not two independently authored copies). Pure XZ placement + height sampling
// for cut-depth derivation (no engine/GPU imports — `bun test` exercises the generator's determinism with
// no device).
//
// Stage 18 (non-overlap by construction): every road and the carpark are placed by rejection sampling so
// no two primitives' footprints come within `computeFalloff(cutDepth) + ROAD_HALF_WIDTH + FLAT_CORE_MARGIN +
// √2·SPACING` — the clearance derived from the network's own cut depth, never a hardcoded constant. The
// √2·SPACING lattice-diagonal term (see the derivation at the placement predicate) covers the mesh vertices
// the exactness oracle interpolates from, which can sit one cell diagonal outside the footprint edge; the
// shipped `F + h + FCM` clearance was short by √2·SPACING and left h − √2·SPACING = −1.66 m of slack. A
// deeper cut widens both the falloff and the required separation. Cut depth is measured *between* chord
// endpoints (the chord's target equals natural height at its endpoints by construction, so sampling at the
// profile's own points reads 0 and collapses `computeFalloff` to its `SPACING` floor — the same trap
// `buildNetworkGeometry` guards against, followed here). The carpark is placed independently (no longer
// anchored to road 0's midpoint), clear of every road by the same clearance plus a PROFILE_STEP sampling
// margin.

export const ROAD_COUNT = 5; // "a handful" — enough to read as a network, not a maze
export const ROAD_HALF_WIDTH = 4; // metres — matches terrain/grid.ts's SPACING, `overlay/stroke.ts`'s own convention
const ROAD_MIN_LENGTH = 80; // metres
const ROAD_MAX_LENGTH = 220; // metres — keeps a single road well under the world's 1024 m span
const CARPARK_HALF = 20; // metres — half-extent of the one carpark stamp (a 40 m × 40 m lot)
const MAX_ATTEMPTS = 2000; // rejection-sampling cap — average is ~2–4, not 2000

// Every primitive stays inside WORLD_HALF: road start points are within `bound = WORLD_HALF - WORLD_MARGIN`,
// so a full-length road's far endpoint reaches `bound + ROAD_MAX_LENGTH < WORLD_HALF` (the +1 margin makes
// the inequality strict, matching the test's `< WORLD_HALF` assertion). The carpark's centre is also within
// `bound`, so its corners reach `bound + CARPARK_HALF < WORLD_HALF` (CARPARK_HALF = 20 ≪ WORLD_MARGIN).
const WORLD_MARGIN = ROAD_MAX_LENGTH + 1;

/** the cut depth of a single road, measured *between* its chord endpoints — the largest
 *  `|natural - chord_target|` along the chord at {@link PROFILE_STEP} arc-length increments. The chord's
 *  target equals natural height at the endpoints by construction (`buildPolylineProfile`), so the profile's
 *  own points read exactly 0; the actual cut lives between them, where natural terrain deviates from the
 *  straight chord. Mirrors `buildNetworkGeometry`'s own cut-depth loop. */
function computeRoadCutDepth(
    points: ReadonlyArray<readonly [number, number]>,
    perm: Uint32Array,
): number {
    const first = points[0];
    const last = points[points.length - 1];
    const aH = heightAtCpu(first[0], first[1], perm);
    const bH = heightAtCpu(last[0], last[1], perm);
    const chordLen = Math.hypot(last[0] - first[0], last[1] - first[1]);
    let maxDev = 0;
    const n = Math.max(1, Math.ceil(chordLen / PROFILE_STEP));
    for (let s = 0; s <= n; s++) {
        const t = s / n;
        const x = first[0] + (last[0] - first[0]) * t;
        const z = first[1] + (last[1] - first[1]) * t;
        const chordHeight = aH + (bH - aH) * t;
        maxDev = Math.max(maxDev, Math.abs(heightAtCpu(x, z, perm) - chordHeight));
    }
    return maxDev;
}

/** minimum distance between two line segments' centrelines — 0 if they cross, otherwise the
 *  minimum of the four endpoint-to-segment-centreline distances. Exact (no sampling). */
function segmentCentrelineDistance(
    a1: readonly [number, number],
    a2: readonly [number, number],
    b1: readonly [number, number],
    b2: readonly [number, number],
): number {
    const cross = (ux: number, uz: number, vx: number, vz: number) => ux * vz - uz * vx;
    const d1 = cross(b2[0] - b1[0], b2[1] - b1[1], a1[0] - b1[0], a1[1] - b1[1]);
    const d2 = cross(b2[0] - b1[0], b2[1] - b1[1], a2[0] - b1[0], a2[1] - b1[1]);
    const d3 = cross(a2[0] - a1[0], a2[1] - a1[1], b1[0] - a1[0], b1[1] - a1[1]);
    const d4 = cross(a2[0] - a1[0], a2[1] - a1[1], b2[0] - a1[0], b2[1] - a1[1]);
    if (d1 > 0 !== d2 > 0 && d3 > 0 !== d4 > 0) return 0;

    const seg = (
        p: readonly [number, number],
        a: readonly [number, number],
        b: readonly [number, number],
    ) => segmentDistance(p[0], p[1], { ax: a[0], az: a[1], bx: b[0], bz: b[1], halfWidth: 0 });

    return Math.min(seg(b1, a1, a2), seg(b2, a1, a2), seg(a1, b1, b2), seg(a2, b1, b2));
}

/** minimum signed distance from a segment centreline to a polygon boundary — negative if the segment
 *  passes through the polygon. Samples along the segment at {@link PROFILE_STEP} intervals and also
 *  checks each polygon vertex to the segment centreline (exact). */
function segmentToPolygonDistance(
    a1: readonly [number, number],
    a2: readonly [number, number],
    poly: PolygonStamp,
): number {
    const len = Math.hypot(a2[0] - a1[0], a2[1] - a1[1]);
    const n = Math.max(1, Math.ceil(len / PROFILE_STEP));
    let minDist = Number.POSITIVE_INFINITY;
    for (let s = 0; s <= n; s++) {
        const t = s / n;
        const x = a1[0] + (a2[0] - a1[0]) * t;
        const z = a1[1] + (a2[1] - a1[1]) * t;
        minDist = Math.min(minDist, polygonDistance(x, z, poly));
    }
    for (const [px, pz] of poly.points) {
        minDist = Math.min(
            minDist,
            segmentDistance(px, pz, { ax: a1[0], az: a1[1], bx: a2[0], bz: a2[1], halfWidth: 0 }),
        );
    }
    return minDist;
}

/** the footprint-to-footprint distance between two road segments — the centreline distance
 *  minus both half-widths. Negative if the footprints overlap. */
export function roadFootprintDistance(
    a1: readonly [number, number],
    a2: readonly [number, number],
    b1: readonly [number, number],
    b2: readonly [number, number],
): number {
    return segmentCentrelineDistance(a1, a2, b1, b2) - 2 * ROAD_HALF_WIDTH;
}

/** the footprint-to-polygon-boundary distance between a road segment and a polygon stamp —
 *  the centreline-to-boundary distance minus the road's half-width. Negative if they overlap. */
export function roadPolygonFootprintDistance(
    a1: readonly [number, number],
    a2: readonly [number, number],
    poly: PolygonStamp,
): number {
    return segmentToPolygonDistance(a1, a2, poly) - ROAD_HALF_WIDTH;
}

/** the seeded procedural road network: {@link ROAD_COUNT} straight single-segment roads plus one carpark
 *  polygon, placed by rejection sampling so no two primitives' footprints come within
 *  `computeFalloff(cutDepth) + ROAD_HALF_WIDTH + FLAT_CORE_MARGIN + √2·SPACING` — the clearance derived from
 *  the network's own cut depth (measured between chord endpoints), never a hardcoded constant. The
 *  √2·SPACING term covers the lattice vertices the exactness oracle interpolates from (see the derivation
 *  at the placement predicate). The carpark is placed independently, clear of every road by the same
 *  clearance plus a PROFILE_STEP sampling margin. Deterministic in `seed` — the same seed always returns
 *  the identical document (`network.test.ts` pins this directly); a different seed almost certainly returns
 *  a different one. */
export function generateNetwork(seed: number): StrokeDocument {
    const rng = mulberry32(seed);
    const rand = (lo: number, hi: number) => lo + rng() * (hi - lo);
    const perm = makePermutation(seed);
    const bound = WORLD_HALF - WORLD_MARGIN;

    const polylines: Polyline[] = [];
    let maxCutDepth = 0;

    for (let i = 0; i < ROAD_COUNT; i++) {
        let placed = false;
        for (let attempt = 0; attempt < MAX_ATTEMPTS && !placed; attempt++) {
            const x0 = rand(-bound, bound);
            const z0 = rand(-bound, bound);
            const heading = rand(0, Math.PI * 2);
            const length = rand(ROAD_MIN_LENGTH, ROAD_MAX_LENGTH);
            const x1 = x0 + Math.cos(heading) * length;
            const z1 = z0 + Math.sin(heading) * length;

            if (Math.abs(x1) >= WORLD_HALF || Math.abs(z1) >= WORLD_HALF) continue;

            const candidate: [number, number][] = [
                [x0, z0],
                [x1, z1],
            ];

            // Clearance derived from the tentative network's own cut depth — a candidate that
            // deepens the max also widens the required separation, so all pairs (including
            // already-placed ones) are re-checked at the new clearance.
            const candidateCutDepth = computeRoadCutDepth(candidate, perm);
            const tentativeMaxCutDepth = Math.max(maxCutDepth, candidateCutDepth);
            const falloff = computeFalloff(tentativeMaxCutDepth);
            // R = F + h + FCM + √2·SPACING — the footprint-edge clearance the generator enforces.
            //
            // Derivation (stage-18 repair R1): a road contributes to the blend while its centreline
            // distance is < F + h + FCM (`networkCore`: coreDist = dist_to_centreline − h − FCM,
            // active while coreDist < F). `checkSurfaceFlatness` samples the centreline and both
            // edge-offset lines, and the piecewise-linear reconstruction at a sample point
            // interpolates the enclosing triangle's vertices, which can sit up to one cell diagonal
            // √2·SPACING outside the footprint edge. So a contributing vertex can sit at centreline
            // distance ≥ D − h − √2·SPACING (D = centreline separation). Non-contamination needs
            // D − h − √2·SPACING ≥ F + h + FCM, i.e. the footprint-edge clearance
            // R = D − 2h ≥ F + FCM + √2·SPACING. The generator enforces R = F + h + FCM + √2·SPACING,
            // which is the bound (F + FCM + √2·SPACING) plus h = 4 m of slack — restoring the 4 m
            // margin the shipped R = F + h + FCM lost (slack h − √2·SPACING = 4 − 5.657 = −1.66 m,
            // i.e. contamination was possible). The disjointness arm asserts the bound itself
            // (F + FCM + √2·SPACING), so the check is a claim about the mechanism rather than a
            // restatement of this constant.
            //
            // Why √2·SPACING and not √2·(SPACING/2): the exactness oracle runs at both SPACING and
            // SPACING/2, and the coarser lattice (SPACING) is the weaker case — its vertices sit
            // farther apart, so a triangle corner can wander √2·SPACING outside the footprint. The
            // finer lattice's diagonal (√2·SPACING/2) is smaller, so the coarse bound dominates and
            // covers both resolutions.
            const clearance = falloff + ROAD_HALF_WIDTH + FLAT_CORE_MARGIN + Math.SQRT2 * SPACING;

            const allLines = [...polylines, { points: candidate, halfWidth: ROAD_HALF_WIDTH }];
            let ok = true;
            for (let a = 0; a < allLines.length && ok; a++) {
                for (let b = a + 1; b < allLines.length && ok; b++) {
                    const [a1, a2] = allLines[a].points;
                    const [b1, b2] = allLines[b].points;
                    if (roadFootprintDistance(a1, a2, b1, b2) < clearance) ok = false;
                }
            }

            if (ok) {
                polylines.push({ points: candidate, halfWidth: ROAD_HALF_WIDTH });
                maxCutDepth = tentativeMaxCutDepth;
                placed = true;
            }
        }
        if (!placed) throw new Error(`generateNetwork: could not place road ${i} at seed ${seed}`);
    }

    // Carpark placed independently — clear of every road by the same R = F + h + FCM +
    // √2·SPACING clearance (the lattice-vertex argument applies to the polygon's footprint edge too),
    // plus a PROFILE_STEP sampling margin: `segmentToPolygonDistance` samples the segment at
    // PROFILE_STEP intervals, so it can overestimate the true centreline-to-boundary distance by up to
    // PROFILE_STEP/2 (polygonDistance is 1-Lipschitz, so PROFILE_STEP/2 is a bound, not a guess —
    // R6). The arm asserts the road–carpark pair against the bound + PROFILE_STEP/2 for the same
    // reason; the generator enforces the bound + h + PROFILE_STEP, which is tighter by h −
    // PROFILE_STEP/2 = 4 − 2 = 2 m.
    const falloff = computeFalloff(maxCutDepth);
    const clearance =
        falloff + ROAD_HALF_WIDTH + FLAT_CORE_MARGIN + Math.SQRT2 * SPACING + PROFILE_STEP;

    let carpark: PolygonStamp | null = null;
    for (let attempt = 0; attempt < MAX_ATTEMPTS && !carpark; attempt++) {
        const cx = rand(-bound, bound);
        const cz = rand(-bound, bound);
        const polygon: PolygonStamp = {
            points: [
                [cx - CARPARK_HALF, cz - CARPARK_HALF],
                [cx + CARPARK_HALF, cz - CARPARK_HALF],
                [cx + CARPARK_HALF, cz + CARPARK_HALF],
                [cx - CARPARK_HALF, cz + CARPARK_HALF],
            ],
        };

        let inWorld = true;
        for (const [x, z] of polygon.points) {
            if (Math.abs(x) >= WORLD_HALF || Math.abs(z) >= WORLD_HALF) {
                inWorld = false;
                break;
            }
        }
        if (!inWorld) continue;

        let ok = true;
        for (const line of polylines) {
            const [a1, a2] = line.points;
            if (roadPolygonFootprintDistance(a1, a2, polygon) < clearance) {
                ok = false;
                break;
            }
        }

        if (ok) carpark = polygon;
    }
    if (!carpark) throw new Error(`generateNetwork: could not place carpark at seed ${seed}`);

    return { polylines, polygons: [carpark] };
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
