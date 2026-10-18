import type { Resource } from "../../engine";
import { PhysicsAssets } from "./assets";

/**
 * Convex geometry in shape-local meters, before `Shape.scale`.
 * At unit scale, boxes build Box3D's `b3MakeBoxHull` or `b3MakeTransformedBoxHull`, cylinders build
 * `b3CreateCylinder`, and points build `b3CreateHull`, which drops interior points and merges coplanar faces.
 * Scaling a transformed box or non-circular cylinder rebuilds from its scaled vertices.
 * Add new geometry and assign its handle to replace geometry, never mutate an added definition.
 */
export type Hull = { name: string } & (
    | {
          /** Flat xyz points; at least four points, with a length divisible by three. */
          points: Float32Array;
          /** Most vertices the built hull keeps, clamped to [4, 128]; default 128. */
          maxVertexCount?: number;
          halfExtents?: never;
          transform?: never;
          cylinder?: never;
      }
    | {
          /** Positive box half-extents in meters. */
          halfExtents: readonly [number, number, number];
          points?: never;
          maxVertexCount?: never;
          /** Box-local placement in body-local meters, before Shape.scale. Rotation is a normalized xyzw quaternion. */
          transform?: {
              position: readonly [number, number, number];
              rotation: readonly [number, number, number, number];
          };
          cylinder?: never;
      }
    | {
          /** Box3D cylinder along +Y, from yOffset to yOffset + height. Positive finite height and radius;
           * finite yOffset; integer sides in [3, 32]. Builds b3CreateCylinder, not a smooth analytic cylinder. */
          cylinder: { height: number; radius: number; yOffset: number; sides: number };
          points?: never;
          halfExtents?: never;
          maxVertexCount?: never;
          transform?: never;
      }
);

/** World-owned convex hulls. `add` returns a handle for `Shape.geometry`; names are display labels. */
export const Hulls: Resource<PhysicsAssets<Hull>> = {
    create: () => {
        const hulls = new PhysicsAssets<Hull>((hull) => {
            if (hull.cylinder) {
                const { height, radius, yOffset, sides } = hull.cylinder;
                if (
                    !(
                        Number.isFinite(height) &&
                        height > 0 &&
                        Number.isFinite(radius) &&
                        radius > 0 &&
                        Number.isFinite(yOffset) &&
                        Number.isInteger(sides) &&
                        sides >= 3 &&
                        sides <= 32
                    )
                )
                    throw new Error(
                        `physics: hull "${hull.name}" requires positive finite cylinder height and radius, finite yOffset and integer sides in [3, 32]`,
                    );
            }
            if (hull.transform) {
                const { position, rotation } = hull.transform;
                if (
                    !position.every(Number.isFinite) ||
                    !rotation.every(Number.isFinite) ||
                    Math.abs(Math.hypot(...rotation) - 1) > 1e-5
                )
                    throw new Error(
                        `physics: hull "${hull.name}" requires a finite box transform and normalized rotation`,
                    );
            }
        });
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
