import type { AABB, Vec3 } from "../common/math";

export {
    createBoxMesh,
    createGridMesh,
    createHollowBoxMesh,
    createMesh,
    createTorusMesh,
    createWaveMesh,
} from "./build-nonconvex";
export { computeMeshAABB, getMeshTriangle, safeScale } from "./read-geometry";

/** Triangle mesh edge flags (b3MeshEdgeFlags), packed one byte per triangle. */
export const MeshEdgeFlags = {
    ConcaveEdge1: 0x01,
    ConcaveEdge2: 0x02,
    ConcaveEdge3: 0x04,
    InverseConcaveEdge1: 0x10,
    InverseConcaveEdge2: 0x20,
    InverseConcaveEdge3: 0x40,
    // A flat edge is both concave-from-front and concave-from-back (coplanar neighbours).
    FlatEdge1: 0x11,
    FlatEdge2: 0x22,
    FlatEdge3: 0x44,
    AllFlatEdges: 0x77,
} as const;

/** A BVH node (b3MeshNode). Internal nodes carry `axis` + `childOffset`; leaves carry the triangle run. */
export type MeshNode = {
    lowerBound: Vec3;
    upperBound: Vec3;
    leaf: boolean;
    axis: number;
    childOffset: number;
    triangleCount: number;
    triangleOffset: number;
};

/** A mesh triangle: three vertex indices (b3MeshTriangle). */
export type MeshTriangle = { index1: number; index2: number; index3: number };

/** Immutable mesh collision data (b3MeshData), modeled as a struct of arrays. */
export type MeshData = {
    bounds: AABB;
    surfaceArea: number;
    treeHeight: number;
    degenerateCount: number;
    nodes: MeshNode[];
    vertices: Vec3[];
    triangles: MeshTriangle[];
    materialIndices: number[];
    materialCount: number;
    flags: number[];
};

/** A mesh shape: immutable data reused across shapes, with a per-shape (possibly reflected) scale. */
export type Mesh = { data: MeshData; scale: Vec3 };

/** Mesh construction input (b3MeshDef). */
export type MeshDef = {
    vertices: Vec3[];
    indices: number[];
    materialIndices?: number[] | null;
    weldTolerance?: number;
    weldVertices?: boolean;
    useMedianSplit?: boolean;
    identifyEdges?: boolean;
};

/** A single scaled mesh triangle in mesh space (b3Triangle): world-scaled vertices, vertex indices,
 * and the edge flags, with winding + flags flipped when the shape scale reflects. */
export type Triangle = {
    vertices: [Vec3, Vec3, Vec3];
    i1: number;
    i2: number;
    i3: number;
    flags: number;
};
