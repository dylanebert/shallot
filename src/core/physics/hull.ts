import { Registry, type Resource } from "../../engine";

/**
 * Input to Box3D's `b3CreateHull`. Interior points drop and coplanar
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

const UNIT_CUBE_POINTS = new Float32Array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
]);
/** Reserved hull id of the built-in 1 m unit cube; `Shape.scale` supplies its full lengths. */
export const UNIT_CUBE_ID = 0;
