import {
    DASH_DUTY,
    DASH_OFFSET,
    DASH_PERIOD,
    dirtyTiles,
    EDGE_INSET,
    LINE_HALF_WIDTH,
    type Rect,
    TEXEL_SIZE,
    TILE_SIZE,
    tileCoordOf,
    tileOrigin,
} from "./tiles";

// The stroke document: pure data (one road's polyline + width) plus the CPU analytic distance math the
// differential oracle checks the GPU rasterizer (`rasterize.ts`) against. No GPU/engine
// imports, so `bun test` exercises every formula here without a device — the same device-free split
// `tiles.ts` and `terrain/noise.ts` use.
//
// `segmentDistance` below is deliberately a *different* derivation from `rasterize.ts`'s
// `segmentDistanceGpu`: this one finds the signed perpendicular offset via a 2D cross product once the
// projection falls strictly between the endpoints, falling back to the nearer endpoint distance outside
// that range; `rasterize.ts`'s GPU form instead clamps the projection parameter `t` to [0, 1] and always
// measures straight-line distance to the clamped point. Same geometric quantity, two independently
// written code paths — an agreement check between two things written from one source tests
// self-consistency, not correctness.

/** one road-width centreline — consecutive points define its segments, `halfWidth` is constant along its
 *  whole length (a real network's per-segment width variation is out of this stage's scope). */
export interface Polyline {
    readonly points: ReadonlyArray<readonly [number, number]>;
    readonly halfWidth: number;
}

/** the rasterizer's whole input: every polyline stroke for one redraw — one road, no carpark. */
export interface StrokeDocument {
    readonly polylines: readonly Polyline[];
}

/** one polyline segment, flattened to its own endpoints + width — the unit both the CPU oracle and the
 *  GPU kernel's per-primitive distance function operate on. */
export interface Segment {
    readonly ax: number;
    readonly az: number;
    readonly bx: number;
    readonly bz: number;
    readonly halfWidth: number;
}

/** every polyline in `doc`, flattened to its consecutive-point segments — pure data marshaling (not
 *  distance math), shared by the CPU oracle and the GPU buffer packer alike; sharing this step doesn't
 *  weaken the differential, since the two sides only diverge in how they measure distance to a segment,
 *  not in which segments exist. */
export function flattenSegments(doc: StrokeDocument): Segment[] {
    const out: Segment[] = [];
    for (const line of doc.polylines) {
        for (let i = 0; i < line.points.length - 1; i++) {
            const [ax, az] = line.points[i];
            const [bx, bz] = line.points[i + 1];
            out.push({ ax, az, bx, bz, halfWidth: line.halfWidth });
        }
    }
    return out;
}

/** signed distance in world metres from (px, pz) to one segment's road edge — negative inside,
 *  zero at the boundary. Perpendicular-offset-via-cross-product once the projection lands strictly
 *  between the endpoints, nearer-endpoint distance otherwise — see the module header for why this is a
 *  different derivation from `rasterize.ts`'s clamped-projection form. */
export function segmentDistance(px: number, pz: number, seg: Segment): number {
    const { ax, az, bx, bz, halfWidth } = seg;
    const abx = bx - ax;
    const abz = bz - az;
    const len = Math.hypot(abx, abz);
    if (len === 0) return Math.hypot(px - ax, pz - az) - halfWidth;
    const ux = abx / len;
    const uz = abz / len;
    const along = (px - ax) * ux + (pz - az) * uz; // signed distance along the segment from A
    if (along <= 0) return Math.hypot(px - ax, pz - az) - halfWidth;
    if (along >= len) return Math.hypot(px - bx, pz - bz) - halfWidth;
    const cross = (px - ax) * uz - (pz - az) * ux; // signed perpendicular offset (2D cross product)
    return Math.abs(cross) - halfWidth;
}

/** the document's analytic distance at (px, pz) — the minimum (nearest, most-inside) signed distance
 *  over every segment. The CPU half of the differential oracle. */
export function documentDistance(px: number, pz: number, doc: StrokeDocument): number {
    let best = Number.POSITIVE_INFINITY;
    for (const seg of flattenSegments(doc)) {
        const d = segmentDistance(px, pz, seg);
        if (d < best) best = d;
    }
    return best;
}

/** the signed distance to the nearest road marking at (px, pz) for one segment — negative inside a
 *  marking, zero at its boundary, positive outside. Two solid edge lines inset from the road edge
 *  (centred at d = −EDGE_INSET on the existing edge distance) and a broken centreline (perpendicular
 *  distance to the centreline via d + halfWidth, station along the chord via fract(s / DASH_PERIOD) <
 *  DASH_DUTY). Cross-product derivation, like {@link segmentDistance} — the CPU twin the GPU's
 *  `markingDistanceFromChord` (terrain/terrain.ts, clamped-projection form) is checked against, independently.
 *
 *  Geometry per the Locked decision: one chord, so the dash phase has no joint to break at.
 *  Two-lane two-way: broken yellow centreline, solid white edges. */
export function markingDistanceForSegment(px: number, pz: number, seg: Segment): number {
    const { ax, az, bx, bz } = seg;
    const abx = bx - ax;
    const abz = bz - az;
    const len = Math.hypot(abx, abz);
    if (len === 0) return Number.POSITIVE_INFINITY;
    const ux = abx / len;
    const uz = abz / len;
    const along = (px - ax) * ux + (pz - az) * uz; // signed station along the chord from A
    const cross = (px - ax) * uz - (pz - az) * ux; // signed perpendicular offset (2D cross product)
    const edgeDist = segmentDistance(px, pz, seg); // the existing edge distance (handles endpoints)

    // edge line: solid, centred at d = −EDGE_INSET, width LINE_WIDTH
    const edgeLineDist = Math.abs(edgeDist + EDGE_INSET) - LINE_HALF_WIDTH;

    // centreline: broken, lateral distance from the perpendicular offset, longitudinal from the dash phase
    const lateral = Math.abs(cross) - LINE_HALF_WIDTH;
    let longitudinal: number;
    if (along < 0) {
        longitudinal = -along; // before A — centreline does not extend past the endpoint
    } else if (along > len) {
        longitudinal = along - len; // after B
    } else {
        const phase =
            (along + DASH_OFFSET) / DASH_PERIOD - Math.floor((along + DASH_OFFSET) / DASH_PERIOD); // fract((s + DASH_OFFSET) / DASH_PERIOD)
        if (phase < DASH_DUTY) {
            // inside a dash — distance to the nearer dash boundary (negative = inside)
            longitudinal = -Math.min(phase, DASH_DUTY - phase) * DASH_PERIOD;
        } else {
            // inside a gap — distance to the nearer dash (positive = outside)
            longitudinal = Math.min(phase - DASH_DUTY, 1 - phase) * DASH_PERIOD;
        }
    }
    const centreDist = Math.max(lateral, longitudinal);

    return Math.min(edgeLineDist, centreDist);
}

/** the document's analytic marking distance at (px, pz) — the minimum (nearest) signed distance to any
 *  road marking over every segment. The CPU half of the marking differential oracle. */
export function markingDistance(px: number, pz: number, doc: StrokeDocument): number {
    let best = Number.POSITIVE_INFINITY;
    for (const seg of flattenSegments(doc)) {
        const d = markingDistanceForSegment(px, pz, seg);
        if (d < best) best = d;
    }
    return best;
}

const MARGIN = TEXEL_SIZE; // the one-texel margin `stroke.ts`'s `strokeRect` used, so the fwidth-
// antialiased edge (terrain.ts's fs) never samples a texel this document didn't write.

/** one segment's world-space AABB, expanded by its half-width plus {@link MARGIN} — the cheap candidate
 *  prefilter for the capsule test: every tile the capsule touches is inside this rect, so it narrows the
 *  per-tile test to a small set without walking the whole grid. */
function segmentRect(seg: Segment): Rect {
    return {
        minX: Math.min(seg.ax, seg.bx) - seg.halfWidth - MARGIN,
        maxX: Math.max(seg.ax, seg.bx) + seg.halfWidth + MARGIN,
        minZ: Math.min(seg.az, seg.bz) - seg.halfWidth - MARGIN,
        maxZ: Math.max(seg.az, seg.bz) + seg.halfWidth + MARGIN,
    };
}

/** raw (no halfWidth subtraction) distance from (px, pz) to the segment's centreline — the nearest
 *  point on the infinite line clamped to the segment's endpoints. */
function pointToSegmentDistance(px: number, pz: number, seg: Segment): number {
    const { ax, az, bx, bz } = seg;
    const abx = bx - ax;
    const abz = bz - az;
    const len = Math.hypot(abx, abz);
    if (len === 0) return Math.hypot(px - ax, pz - az);
    const ux = abx / len;
    const uz = abz / len;
    const along = (px - ax) * ux + (pz - az) * uz;
    if (along <= 0) return Math.hypot(px - ax, pz - az);
    if (along >= len) return Math.hypot(px - bx, pz - bz);
    const cross = (px - ax) * uz - (pz - az) * ux;
    return Math.abs(cross);
}

/** distance from (px, pz) to an axis-aligned rect — 0 inside, the nearest edge distance outside. */
function pointToRectDistance(px: number, pz: number, rect: Rect): number {
    const dx = Math.max(rect.minX - px, 0, px - rect.maxX);
    const dz = Math.max(rect.minZ - pz, 0, pz - rect.maxZ);
    return Math.hypot(dx, dz);
}

/** whether a segment intersects an axis-aligned rect — the separating-axis theorem over the X axis,
 *  the Z axis, and the segment's own normal. */
function segmentIntersectsRect(seg: Segment, rect: Rect): boolean {
    const segMinX = Math.min(seg.ax, seg.bx);
    const segMaxX = Math.max(seg.ax, seg.bx);
    const segMinZ = Math.min(seg.az, seg.bz);
    const segMaxZ = Math.max(seg.az, seg.bz);
    if (segMaxX < rect.minX || segMinX > rect.maxX || segMaxZ < rect.minZ || segMinZ > rect.maxZ)
        return false;
    // separating axis along the segment's normal: (-dz, dx)
    const dx = seg.bx - seg.ax;
    const dz = seg.bz - seg.az;
    const na = -dz * seg.ax + dx * seg.az;
    const nb = -dz * seg.bx + dx * seg.bz;
    const segLo = Math.min(na, nb);
    const segHi = Math.max(na, nb);
    const r0 = -dz * rect.minX + dx * rect.minZ;
    const r1 = -dz * rect.maxX + dx * rect.minZ;
    const r2 = -dz * rect.minX + dx * rect.maxZ;
    const r3 = -dz * rect.maxX + dx * rect.maxZ;
    const rectLo = Math.min(r0, r1, r2, r3);
    const rectHi = Math.max(r0, r1, r2, r3);
    return segHi >= rectLo && segLo <= rectHi;
}

/** the minimum distance from a world-space rect to a segment's centreline — 0 if they intersect,
 *  otherwise the nearest point-to-point distance (min of rect-corner-to-segment and
 *  segment-endpoint-to-rect). Used by {@link documentDirtyTiles}'s capsule test to decide whether a
 *  tile is within `halfWidth + margin` of the segment. Module-local: the unbaked-hole arm deliberately
 *  derives its own distance by brute-force sweep rather than calling this, so exporting it would leave an
 *  export with no outside reader. */
function segmentRectDistance(seg: Segment, rect: Rect): number {
    if (segmentIntersectsRect(seg, rect)) return 0;
    let best = Math.min(
        pointToRectDistance(seg.ax, seg.az, rect),
        pointToRectDistance(seg.bx, seg.bz, rect),
    );
    best = Math.min(best, pointToSegmentDistance(rect.minX, rect.minZ, seg));
    best = Math.min(best, pointToSegmentDistance(rect.maxX, rect.minZ, seg));
    best = Math.min(best, pointToSegmentDistance(rect.minX, rect.maxZ, seg));
    best = Math.min(best, pointToSegmentDistance(rect.maxX, rect.maxZ, seg));
    return best;
}

/** the tile columns (tx, tz) → the tile's world-space rect. */
function tileWorldRect(tx: number, tz: number): Rect {
    const [ox, oz] = tileOrigin(tx, tz);
    return { minX: ox, minZ: oz, maxX: ox + TILE_SIZE, maxZ: oz + TILE_SIZE };
}

/** whether (px, pz) sits on a road — the same coverage region the overlay's fwidth-thresholded composite
 *  renders as pavement (`documentDistance <= 0`, `terrain.ts`'s fs).
 * @example drivable(0, 0, strokeDocument()) // true — the origin sits on the hand-authored stroke's centreline
 * @example drivable(1000, 1000, strokeDocument()) // false — off the world footprint entirely */
export function drivable(px: number, pz: number, doc: StrokeDocument): boolean {
    return documentDistance(px, pz, doc) <= 0;
}

/**
 * the document's exact dirty-tile set: the union, over every segment *individually*, of the tiles whose
 * rect is within `halfWidth + margin` of the segment — a capsule (offset-rectangle) test, not the
 * segment's axis-aligned bounding box. The AABB is kept as a cheap candidate prefilter
 * ({@link segmentRect} → {@link dirtyTiles}), but the emitted set is the capsule test's: a tile is dirty
 * iff the distance from that tile's rect to the segment is `<= halfWidth + margin`.
 *
 * Why the capsule and not the AABB: for an axis-aligned chord the AABB *is* the swath and the count is
 * exact (a full-width chord reads 32 tiles); for a diagonal chord the AABB is the whole enclosing rect,
 * so a corner-to-corner chord reads 256 — every tile in the world — against a true swath of at most
 * `2 × TILES_PER_SIDE − 1 = 31` tiles before width. The capsule narrows the dirty set to the chord's
 * actual footprint.
 *
 * Deliberately not one bounding rect over the whole document — two primitives far apart would otherwise
 * mark every tile *between* them too, which is exactly the over-approximation the spec's
 * "only-touched-tiles oracle" rules out. Sorted ascending by tile id, de-duplicated. Empty documents
 * throw — an empty edit (`markDirty` on nothing) is a caller bug, not a valid zero-tile mark.
 */
export function documentDirtyTiles(doc: StrokeDocument): number[] {
    const ids = new Set<number>();
    for (const seg of flattenSegments(doc))
        for (const id of dirtyTiles(segmentRect(seg))) {
            const [tx, tz] = tileCoordOf(id);
            if (segmentRectDistance(seg, tileWorldRect(tx, tz)) <= seg.halfWidth + MARGIN)
                ids.add(id);
        }
    if (ids.size === 0) {
        throw new Error("documentDirtyTiles: an empty document (no polylines) touches no tile");
    }
    return [...ids].sort((a, b) => a - b);
}
