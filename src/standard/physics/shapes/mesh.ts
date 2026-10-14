import {
    type AABB,
    aabb,
    f32,
    MIN_SCALE,
    maxf,
    minf,
    type Transform,
    type Vec3,
    vec3,
} from "../common/math";

export {
    createBoxMesh,
    createGridMesh,
    createHollowBoxMesh,
    createMesh,
    createTorusMesh,
    createWaveMesh,
} from "./build-nonconvex";

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

/** Clamp scale magnitude away from zero, preserving sign (b3SafeScale). */
export function safeScale(a: Vec3, out: Vec3 = vec3.zero()): Vec3 {
    out.x = Math.fround((a.x >= 0 ? 1 : -1) * Math.max(Math.fround(Math.abs(a.x)), MIN_SCALE));
    out.y = Math.fround((a.y >= 0 ? 1 : -1) * Math.max(Math.fround(Math.abs(a.y)), MIN_SCALE));
    out.z = Math.fround((a.z >= 0 ? 1 : -1) * Math.max(Math.fround(Math.abs(a.z)), MIN_SCALE));
    return out;
}

/** World-space AABB enclosing a scaled, transformed mesh (b3ComputeMeshAABB). */
const meshBounds = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };
export function computeMeshAABB(
    data: MeshData,
    transform: Transform,
    scale: Vec3,
    out: AABB = { lowerBound: vec3.zero(), upperBound: vec3.zero() },
): AABB {
    const bounds = meshBounds;
    const lx = f32(scale.x * data.bounds.lowerBound.x),
        ux = f32(scale.x * data.bounds.upperBound.x);
    const ly = f32(scale.y * data.bounds.lowerBound.y),
        uy = f32(scale.y * data.bounds.upperBound.y);
    const lz = f32(scale.z * data.bounds.lowerBound.z),
        uz = f32(scale.z * data.bounds.upperBound.z);
    bounds.lowerBound.x = minf(lx, ux);
    bounds.upperBound.x = maxf(lx, ux);
    bounds.lowerBound.y = minf(ly, uy);
    bounds.upperBound.y = maxf(ly, uy);
    bounds.lowerBound.z = minf(lz, uz);
    bounds.upperBound.z = maxf(lz, uz);
    return aabb.transformOut(transform, bounds, out);
}

/** A single scaled mesh triangle in mesh space (b3Triangle): world-scaled vertices, vertex indices,
 * and the edge flags, with winding + flags flipped when the shape scale reflects. */
export type Triangle = {
    vertices: [Vec3, Vec3, Vec3];
    i1: number;
    i2: number;
    i3: number;
    flags: number;
};

/**
 * Fetch a mesh triangle by index, scaled into mesh space (b3GetMeshTriangle). When the scale reflects
 * (negative determinant) the winding is reversed (indices 2/3 swapped) and concave edges become
 * convex, so the inverse-concave flags are remapped to concave.
 */
export function getMeshTriangle(mesh: Mesh, triangleIndex: number): Triangle {
    const data = mesh.data;
    const triangle = data.triangles[triangleIndex];
    const triangleFlags = data.flags[triangleIndex];
    const scale = mesh.scale;
    const vertices = data.vertices;

    const v1 = vec3.mul(scale, vertices[triangle.index1]);
    const reflected = f32(f32(scale.x * scale.y) * scale.z) < 0;

    if (reflected) {
        let flags = 0;
        if (triangleFlags & MeshEdgeFlags.InverseConcaveEdge1) flags |= MeshEdgeFlags.ConcaveEdge1;
        if (triangleFlags & MeshEdgeFlags.InverseConcaveEdge2) flags |= MeshEdgeFlags.ConcaveEdge2;
        if (triangleFlags & MeshEdgeFlags.InverseConcaveEdge3) flags |= MeshEdgeFlags.ConcaveEdge3;
        return {
            vertices: [
                v1,
                vec3.mul(scale, vertices[triangle.index3]),
                vec3.mul(scale, vertices[triangle.index2]),
            ],
            i1: triangle.index1,
            i2: triangle.index3,
            i3: triangle.index2,
            flags,
        };
    }

    return {
        vertices: [
            v1,
            vec3.mul(scale, vertices[triangle.index2]),
            vec3.mul(scale, vertices[triangle.index3]),
        ],
        i1: triangle.index1,
        i2: triangle.index2,
        i3: triangle.index3,
        flags: triangleFlags,
    };
}
