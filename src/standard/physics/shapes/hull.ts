import type { AABB, Mat3, Plane, Vec3 } from "../common/math";

export { computeHullAABB } from "./read-geometry";

import { S_GEO_REFERENCE, SHAPE_STRIDE } from "../kernel/shapecolumns";
import type { WorldState } from "../world/world";
import { drawHull } from "./readgeometry";

/** A hull vertex: index of one half-edge with this vertex as origin (b3HullVertex). */
export type HullVertex = { edge: number };
/** Half-edge indices in the native uint8 topology (b3HullHalfEdge). */
export type HullHalfEdge = { next: number; twin: number; origin: number; face: number };
/** A hull face, identified by one of its half-edges (b3HullFace). */
export type HullFace = { edge: number };
/** Independent hull geometry; immutable after construction. */
export type HullData = {
    aabb: AABB;
    surfaceArea: number;
    volume: number;
    innerRadius: number;
    center: Vec3;
    centralInertia: Mat3;
    vertexCount: number;
    edgeCount: number;
    faceCount: number;
    points: Vec3[];
    vertices: HullVertex[];
    edges: HullHalfEdge[];
    faces: HullFace[];
    planes: Plane[];
    hash: bigint;
};
export function shapeHullInnerRadius(world: WorldState, shape: number): number {
    const store = world.shapeStore;
    store.refreshViews();
    const ptr = store.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
    return store.materialF[(ptr >>> 2) + 12];
}
/** Independent hull geometry for a requested debug-draw callback. */
export function readShapeHull(world: WorldState, shape: number): HullData {
    const store = world.shapeStore;
    store.refreshViews();
    return readHullAt(world, store.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE]);
}
const hullViews = new WeakMap<WorldState, DataView>();
export function readHullAt(world: WorldState, ptr: number): HullData {
    world.shapeStore.refreshViews();
    const buffer = world.shapeStore.materialF.buffer;
    let view = hullViews.get(world);
    if (!view || view.buffer !== buffer) {
        view = new DataView(buffer);
        hullViews.set(world, view);
    }
    return drawHull(view, ptr);
}
/** Deep clone a hull (b3CloneHull). */
export function cloneHull(hull: HullData): HullData {
    return {
        aabb: { lowerBound: { ...hull.aabb.lowerBound }, upperBound: { ...hull.aabb.upperBound } },
        surfaceArea: hull.surfaceArea,
        volume: hull.volume,
        innerRadius: hull.innerRadius,
        center: { ...hull.center },
        centralInertia: {
            cx: { ...hull.centralInertia.cx },
            cy: { ...hull.centralInertia.cy },
            cz: { ...hull.centralInertia.cz },
        },
        vertexCount: hull.vertexCount,
        edgeCount: hull.edgeCount,
        faceCount: hull.faceCount,
        points: hull.points.map((p) => ({ ...p })),
        vertices: hull.vertices.map((v) => ({ ...v })),
        edges: hull.edges.map((e) => ({ ...e })),
        faces: hull.faces.map((f) => ({ ...f })),
        planes: hull.planes.map((pl) => ({ normal: { ...pl.normal }, offset: pl.offset })),
        hash: hull.hash,
    };
}
