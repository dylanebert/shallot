import type { Resource } from "../../engine";
import { PhysicsAssets } from "./assets";
import type { SurfaceMaterial } from "./shape";

type Vec3 = [number, number, number];

/** A child's placement in compound-local space: `position` in meters, `rotation` a unit quaternion `(x, y, z, w)`. */
type Placement = { position: Vec3; rotation: [number, number, number, number] };

/**
 * Box3D's `b3CompoundDef`: child shapes in compound-local meters, baked into one shape with one
 * broad-phase proxy. A solver shares a build per definition and body Transform.scale in each World,
 * keyed by the compound and its child geometry handles. Children carry their own materials, the default
 * surface material (friction 0.6) when omitted; the Shape's material columns and `scale` do not
 * apply. Body Transform.scale multiplies child positions and each child's geometry in its own frame,
 * retaining child rotations. Body-scale changes recreate the shape at the next fixed sync; sphere
 * and capsule children require uniform component magnitudes. A solver refuses a child naming a missing hull or mesh, a hull or mesh it refuses on its
 * own, and a mesh child whose `materials` count differs from its mesh's or exceeds 4. Replace a
 * compound by adding a new definition and assigning its handle, never by mutation.
 */
export interface Compound {
    name: string;
    capsules?: { center1: Vec3; center2: Vec3; radius: number; material?: SurfaceMaterial }[];
    /** `Hulls` handles, each built at its added size; `UNIT_CUBE_ID` is a 1 m cube. */
    hulls?: { hull: number; transform: Placement; material?: SurfaceMaterial }[];
    /**
     * `PhysicsMeshes` handles. `scale` is non-uniform in the child's space and may be negative.
     * `materials` holds one material per material index of the built mesh (its highest kept
     * `materialIndices` entry plus one, or one without them), at most 4
     * (`B3_MAX_COMPOUND_MESH_MATERIALS`); default one default material.
     */
    meshes?: {
        mesh: number;
        transform: Placement;
        scale: Vec3;
        materials?: SurfaceMaterial[];
    }[];
    spheres?: { center: Vec3; radius: number; material?: SurfaceMaterial }[];
}

/** World-owned compounds. `add` returns a handle for `Shape.geometry`; names are display labels.
 * Children reference handles in the same World's `Hulls` and `PhysicsMeshes`. */
export const Compounds: Resource<PhysicsAssets<Compound>> = {
    create: () => new PhysicsAssets(),
};
