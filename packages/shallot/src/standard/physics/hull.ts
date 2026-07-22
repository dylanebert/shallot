// Convex-hull authoring geometry — the backend-neutral registry of convex hull shapes a `Body` references
// by id. Backend-specific GPU packing (the flat buffer format `standard/avbd`'s collide pass reads) lives
// in `avbd/hull.ts`; this file owns only the geometry itself, shared by any backend. No GJK/EPA; no
// quickhull build here — authored hulls come from explicit geometry until a mesh→hull path lands.

import { Registry } from "../../engine";

type Vec3 = [number, number, number];

/** one polygonal hull face: outward unit normal + plane offset (`dot(normal, v) = offset` on the face) + CCW vertex indices. */
export interface HullFace {
    normal: Vec3;
    offset: number;
    verts: number[];
}

/** convex hull geometry: local vertices, polygonal faces, the unique edge directions for the SAT, and a registry name. */
export interface Hull {
    name: string;
    verts: Vec3[];
    faces: HullFace[];
    edges: Vec3[];
}

/** the registered convex hulls. A `Body` with `ShapeKind.Hull` references one by id; a backend packs them into its own GPU format. Register geometry with `Hulls.register({ name, verts, faces, edges })`; re-registering a name reuses the id. */
export const Hulls = new Registry<Hull>();

// the built-in unit cube (full-size 2, verts ±1) reserved at id 0 — a box collider is THIS hull scaled by
// its half-extents, so a hull SAT reads box and hull through ONE branch-free accessor path. Vertex/face/
// edge order matches the AVBD oracle `boxHull([2,2,2])`.
const UNIT_CUBE: Omit<Hull, "name"> = {
    verts: [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    faces: [
        { normal: [1, 0, 0], offset: 1, verts: [1, 2, 6, 5] },
        { normal: [-1, 0, 0], offset: 1, verts: [0, 4, 7, 3] },
        { normal: [0, 1, 0], offset: 1, verts: [2, 3, 7, 6] },
        { normal: [0, -1, 0], offset: 1, verts: [0, 1, 5, 4] },
        { normal: [0, 0, 1], offset: 1, verts: [4, 5, 6, 7] },
        { normal: [0, 0, -1], offset: 1, verts: [0, 3, 2, 1] },
    ],
    edges: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
};
/** the reserved hull id of the built-in unit cube: a box reads this hull scaled by its half-extents. */
export const UNIT_CUBE_ID = Hulls.register({ name: "__unit_cube__", ...UNIT_CUBE });
