import { dirtyTiles, type Rect, TEXEL_SIZE } from "./tiles";

// The stroke document: pure data (polylines + width, polygon stamps) plus the CPU analytic distance math
// stage 5's differential oracle checks the GPU rasterizer (`rasterize.ts`) against. No GPU/engine
// imports, so `bun test` exercises every formula here without a device (`testing.md`'s logic tier) — the
// same device-free split `tiles.ts` and `terrain/noise.ts` use.
//
// `segmentDistance` below is deliberately a *different* derivation from `rasterize.ts`'s
// `segmentDistanceGpu`: this one finds the signed perpendicular offset via a 2D cross product once the
// projection falls strictly between the endpoints, falling back to the nearer endpoint distance outside
// that range; `rasterize.ts`'s GPU form instead clamps the projection parameter `t` to [0, 1] and always
// measures straight-line distance to the clamped point. Same geometric quantity, two independently
// written code paths — `checks.md`'s rule that an agreement check between two things written from one
// source tests self-consistency, not correctness.

/** one road-width centreline — consecutive points define its segments, `halfWidth` is constant along its
 *  whole length (a real network's per-segment width variation is out of this stage's scope). */
export interface Polyline {
    readonly points: ReadonlyArray<readonly [number, number]>;
    readonly halfWidth: number;
}

/** a filled area stamp (a carpark, a plaza) — its boundary is the polygon edge, no width parameter. */
export interface PolygonStamp {
    readonly points: ReadonlyArray<readonly [number, number]>;
}

/** the rasterizer's whole input: every polyline stroke plus every polygon stamp for one redraw. */
export interface StrokeDocument {
    readonly polylines: readonly Polyline[];
    readonly polygons: readonly PolygonStamp[];
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

/** signed distance in world metres from (px, pz) to a polygon stamp's boundary — negative inside (a
 *  ray-cast winding test), magnitude the nearest edge distance (each edge a zero-width segment). */
export function polygonDistance(px: number, pz: number, poly: PolygonStamp): number {
    const pts = poly.points;
    let inside = false;
    let nearestEdge = Number.POSITIVE_INFINITY;
    for (let i = 0; i < pts.length; i++) {
        const [ax, az] = pts[i];
        const [bx, bz] = pts[(i + 1) % pts.length];
        // standard ray-cast: does the horizontal ray from (px, pz) cross edge (a, b)?
        if (az > pz !== bz > pz) {
            const xCross = ax + ((pz - az) / (bz - az)) * (bx - ax);
            if (px < xCross) inside = !inside;
        }
        const edge = segmentDistance(px, pz, { ax, az, bx, bz, halfWidth: 0 });
        if (edge < nearestEdge) nearestEdge = edge;
    }
    return inside ? -nearestEdge : nearestEdge;
}

/** the document's analytic distance at (px, pz) — the minimum (nearest, most-inside) signed distance
 *  over every segment and polygon. The CPU half of stage 5's differential oracle. */
export function documentDistance(px: number, pz: number, doc: StrokeDocument): number {
    let best = Number.POSITIVE_INFINITY;
    for (const seg of flattenSegments(doc)) {
        const d = segmentDistance(px, pz, seg);
        if (d < best) best = d;
    }
    for (const poly of doc.polygons) {
        const d = polygonDistance(px, pz, poly);
        if (d < best) best = d;
    }
    return best;
}

const MARGIN = TEXEL_SIZE; // the one-texel margin `stroke.ts`'s `strokeRect` used, so the fwidth-
// antialiased edge (terrain.ts's fs) never samples a texel this document didn't write.

/** one segment's world-space AABB, expanded by its half-width plus {@link MARGIN}. */
function segmentRect(seg: Segment): Rect {
    return {
        minX: Math.min(seg.ax, seg.bx) - seg.halfWidth - MARGIN,
        maxX: Math.max(seg.ax, seg.bx) + seg.halfWidth + MARGIN,
        minZ: Math.min(seg.az, seg.bz) - seg.halfWidth - MARGIN,
        maxZ: Math.max(seg.az, seg.bz) + seg.halfWidth + MARGIN,
    };
}

/** one polygon's point AABB, plus {@link MARGIN}. */
function polygonRect(poly: PolygonStamp): Rect {
    let minX = Number.POSITIVE_INFINITY;
    let minZ = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxZ = Number.NEGATIVE_INFINITY;
    for (const [x, z] of poly.points) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minZ = Math.min(minZ, z);
        maxZ = Math.max(maxZ, z);
    }
    return { minX: minX - MARGIN, maxX: maxX + MARGIN, minZ: minZ - MARGIN, maxZ: maxZ + MARGIN };
}

/** whether (px, pz) sits on a road or inside a carpark stamp — the same coverage region the overlay's
 *  fwidth-thresholded composite renders as pavement (`documentDistance <= 0`, `terrain.ts`'s fs). No
 *  separate "is it a road vs a carpark" distinction: both are drivable surface for this query's purpose
 *  (the spec names one `drivable(x, z)` query, not a per-primitive-kind one).
 * @example drivable(0, 0, strokeDocument()) // true — the origin sits on the hand-authored stroke's centreline
 * @example drivable(1000, 1000, strokeDocument()) // false — off the world footprint entirely */
export function drivable(px: number, pz: number, doc: StrokeDocument): boolean {
    return documentDistance(px, pz, doc) <= 0;
}

/**
 * the document's exact dirty-tile set: the union, over every segment and every polygon *individually*, of
 * `dirtyTiles` on that one primitive's own AABB (`tiles.ts`). Deliberately not one bounding rect over the
 * whole document — two primitives far apart would otherwise mark every tile *between* them too, which is
 * exactly the over-approximation the spec's "only-touched-tiles oracle" (assert the redrawn set equals
 * the analytic set, not the visual) rules out. Sorted ascending by tile id, de-duplicated. Empty
 * documents throw — an empty edit (`markDirty` on nothing) is a caller bug, not a valid zero-tile mark.
 */
export function documentDirtyTiles(doc: StrokeDocument): number[] {
    const ids = new Set<number>();
    for (const seg of flattenSegments(doc))
        for (const id of dirtyTiles(segmentRect(seg))) ids.add(id);
    for (const poly of doc.polygons) for (const id of dirtyTiles(polygonRect(poly))) ids.add(id);
    if (ids.size === 0) {
        throw new Error(
            "documentDirtyTiles: an empty document (no polylines or polygons) touches no tile",
        );
    }
    return [...ids].sort((a, b) => a - b);
}
