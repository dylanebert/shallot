// Compound authoring geometry — the registry of compounds a `Shape` of kind Compound references by
// id. It holds Box3D's compound input (`b3CompoundDef`), naming its hulls and meshes by their own
// registry ids; a solver builds and shares its own compound from it.

import { Registry, type Resource } from "../../engine";
import type { SurfaceMaterial } from "./shape";

type Vec3 = [number, number, number];

/** A child's placement in compound-local space: `position` in meters, `rotation` a unit quaternion `(x, y, z, w)`. */
type Placement = { position: Vec3; rotation: [number, number, number, number] };

/**
 * A compound's input: child shapes in compound-local meters, baked into one shape with one
 * broad-phase proxy. A solver builds it once per World, and again for the next Shape once a child's
 * hull or mesh is registered again by name. Children carry their own materials, Box3D's default
 * surface material (friction 0.6) when omitted; the Shape's material columns and `scale` do not
 * apply. A solver refuses a child naming a missing hull or mesh, a hull or mesh it refuses on its
 * own, and a mesh child whose `materials` count differs from its mesh's or exceeds 4. Replace a
 * compound by registering its name again rather than mutating a registered one.
 */
export interface Compound {
    name: string;
    capsules?: { center1: Vec3; center2: Vec3; radius: number; material?: SurfaceMaterial }[];
    /** `Hulls` ids, each built at its registered size; `UNIT_CUBE_ID` is a 1 m cube. */
    hulls?: { hull: number; transform: Placement; material?: SurfaceMaterial }[];
    /**
     * `PhysicsMeshes` ids. `scale` is non-uniform in the child's space and may be negative.
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

/** World-owned compounds. Register geometry through `world.resource(Compounds)`; re-registering a name reuses its id. A `Shape` references an id in its own World's registry, and a compound's children reference ids in that World's `Hulls` and `PhysicsMeshes`. */
export const Compounds: Resource<Registry<Compound>> = {
    create: () => new Registry(),
};
