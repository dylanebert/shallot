import type { Resource } from "../../engine";
import { PhysicsAssets } from "./assets";

/**
 * Box3D's `b3MeshDef`. A solver builds it once per World and culls degenerate triangles. It
 * refuses an `indices` length that is not a multiple of 3, an index past the last vertex, a
 * `materialIndices` length other than the triangle count, and a mesh with no triangle left.
 * `vertices` must fill whole strides or end immediately after the last vertex's xyz.
 * Replace a mesh by adding a new definition and assigning its handle, never by mutation.
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

/** World-owned triangle meshes. `add` returns a handle for `Shape.geometry`; names are display labels. */
export const PhysicsMeshes: Resource<PhysicsAssets<PhysicsMesh>> = {
    create: () => new PhysicsAssets(),
};
