// Convex-hull authoring geometry — the registry of convex hulls a `Shape` references by id. A solver
// builds its colliders from it (marshal-shape.ts). Authored hulls come from explicit geometry.

import { Registry, type Resource } from "../../engine";

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

/** World-owned convex hulls. Register geometry through `world.resource(Hulls)`; re-registering a name reuses its id. A `Shape` references an id in its own World's registry. */
export const Hulls: Resource<Registry<Hull>> = {
    create: () => {
        const hulls = new Registry<Hull>();
        hulls.register({ name: "__unit_cube__", ...structuredClone(UNIT_CUBE) });
        return hulls;
    },
};

// The built-in unit cube (full-size 2, verts ±1) reserved at id 0. A box collider is this hull scaled by
// Shape.scale, so the kernel reads boxes and other hulls through one path. Vertex/face/edge order is
// the canonical `boxHull([2,2,2])` layout.
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
/** Reserved hull id of the built-in unit cube; `Shape.scale` supplies its half-extents. */
export const UNIT_CUBE_ID = 0;
