// Triangle-mesh authoring geometry — the registry of meshes a `Shape` of kind Mesh references by id.
// It holds Box3D's mesh input (`b3MeshDef`); a solver builds and shares its own mesh from it.

import { Registry, type Resource } from "../../engine";

/**
 * A triangle mesh's input. A solver builds it once per World and culls degenerate triangles. It
 * refuses an `indices` length that is not a multiple of 3, an index past the last vertex, a
 * `materialIndices` length other than the triangle count, a `vertices` length that is not whole
 * vertices at `stride`, and a mesh with no triangle left. Replace a mesh by registering its name
 * again rather than mutating a registered one.
 */
export interface PhysicsMesh {
    name: string;
    /** xyz vertex positions in shape-local meters, before `Shape.scale`; vertex `i` starts at `i * stride`. */
    vertices: Float32Array;
    /** Floats from one vertex to the next, at least 3; 0 (the default) means 3. Floats past a vertex's xyz are skipped. */
    stride?: number;
    /** Three vertex indices per triangle, counter-clockwise seen from the front unless `clockwiseWinding`. */
    indices: Uint32Array;
    /** One index per triangle into the Shape's `ShapeMaterials` set. */
    materialIndices?: Uint8Array;
    /** Distance in meters within which `weldVertices` merges vertices; default 0. */
    weldTolerance?: number;
    /** Whether to merge vertices within `weldTolerance`; default false. */
    weldVertices?: boolean;
    /** Whether the tree splits at the median instead of by surface area, faster to build for grid-like meshes; default false. */
    useMedianSplit?: boolean;
    /** Whether to find shared edges, so contacts on interior edges are smooth; default false. */
    identifyEdges?: boolean;
    /** Whether `indices` wind clockwise; default false. */
    clockwiseWinding?: boolean;
}

/** World-owned triangle meshes. Register geometry through `world.resource(PhysicsMeshes)`; re-registering a name reuses its id. A `Shape` references an id in its own World's registry. */
export const PhysicsMeshes: Resource<Registry<PhysicsMesh>> = {
    create: () => new Registry(),
};
