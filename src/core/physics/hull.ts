import { Registry, type Resource } from "../../engine";

/**
 * Convex geometry in shape-local meters, before `Shape.scale`.
 * A box builds Box3D's `b3MakeBoxHull`; points build `b3CreateHull`, which drops interior points
 * and merges coplanar faces. Replace geometry by registering its name again, never by mutation.
 */
export type Hull = { name: string } & (
    | {
          /** Flat xyz points; at least four points, with a length divisible by three. */
          points: Float32Array;
          /** Most vertices the built hull keeps, clamped to [4, 128]; default 128. */
          maxVertexCount?: number;
          halfExtents?: never;
      }
    | {
          /** Positive box half-extents in meters. */
          halfExtents: readonly [number, number, number];
          points?: never;
          maxVertexCount?: never;
      }
);

/** World-owned convex hulls. Re-registering a name reuses its id; Shape.geometry names an id in its own World. */
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
