import type { Resource } from "../../engine";
import { PhysicsAssets } from "./assets";

/**
 * Convex geometry in shape-local meters, before `Shape.scale`.
 * A box builds Box3D's `b3MakeBoxHull`; points build `b3CreateHull`, which drops interior points
 * and merges coplanar faces. Add new geometry and assign its handle to replace geometry, never mutate an added definition.
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

/** World-owned convex hulls. `add` returns a handle for `Shape.geometry`; names are display labels. */
export const Hulls: Resource<PhysicsAssets<Hull>> = {
    create: () => {
        const hulls = new PhysicsAssets<Hull>();
        hulls.add({ name: "__unit_cube__", points: UNIT_CUBE_POINTS.slice() });
        return hulls;
    },
};

const UNIT_CUBE_POINTS = new Float32Array([
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
]);
/** Reserved hull handle of the built-in 1 m unit cube; `Shape.scale` supplies its full lengths. */
export const UNIT_CUBE_ID = 0;
