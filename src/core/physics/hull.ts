// Convex-hull authoring geometry — the registry of convex hulls a `Shape` references by id. It holds
// Box3D's hull input (`b3CreateHull`'s points and vertex budget); a solver builds its own hull from it
// (marshal-shape.ts).

import { Registry, type Resource } from "../../engine";

/**
 * A convex hull's input: the hull is the convex hull of `points`. Interior points drop and coplanar
 * faces merge when a solver builds it. A solver refuses fewer than four points, or a `points` length
 * that is not a multiple of 3. Replace a hull by registering its name again rather than mutating a
 * registered one.
 */
export interface Hull {
    name: string;
    /** Flat xyz points in shape-local meters, before `Shape.scale`. */
    points: Float32Array;
    /** Most vertices the built hull keeps, clamped to [4, 128]; default 128 (`B3_MAX_HULL_VERTICES`). */
    maxVertexCount?: number;
}

/** World-owned convex hulls. Register geometry through `world.resource(Hulls)`; re-registering a name reuses its id. A `Shape` references an id in its own World's registry. */
export const Hulls: Resource<Registry<Hull>> = {
    create: () => {
        const hulls = new Registry<Hull>();
        hulls.register({ name: "__unit_cube__", points: UNIT_CUBE_POINTS.slice() });
        return hulls;
    },
};

// The built-in unit cube (full-size 2, points ±1) reserved at id 0. A box collider is this hull
// scaled by Shape.scale, so the kernel reads boxes and other hulls through one path.
const UNIT_CUBE_POINTS = new Float32Array([
    -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
]);
/** Reserved hull id of the built-in unit cube; `Shape.scale` supplies its half-extents. */
export const UNIT_CUBE_ID = 0;
