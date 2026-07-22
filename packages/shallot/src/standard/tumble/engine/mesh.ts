// Triangle mesh: welding, BVH build (binned SAH / median split), edge identification, and the
// BVH bounds-query traversal. Ported from Box3D's mesh.c (Erin Catto + Dirk Gregorius, MIT). Meshes
// are static-only collision geometry; a shape references immutable mesh data plus a per-shape scale.
//
// The C stores the mesh as one byte blob with offsets into node/vertex/triangle/material/flag arrays
// (SIMD-aligned bitfield nodes); the port models it as a plain struct of arrays. The raw bytes are
// never hashed by the sim, so the representation is free to change — only the BVH build order and the
// query's triangle-visit order are load-bearing for bit-exactness. fround discipline per the README.

import { HUGE, LINEAR_SLOP, OVERLAP_SLOP } from "./core";
import {
    type CastOutput,
    computeProxyAABB,
    type DistanceInput,
    emptyCache,
    emptyCastOutput,
    makeLocalProxy,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeCastPairInput,
    type ShapeProxy,
    shapeCast,
    shapeDistance,
} from "./distance";
import type { Capsule } from "./geometry";
import {
    type AABB,
    aabb,
    absf,
    computeCosSin,
    FLT_EPSILON,
    FLT_MAX,
    f32,
    intersectRayTriangle,
    MIN_SCALE,
    maxf,
    PI,
    type Plane,
    quat,
    type Transform,
    testBoundsRayOverlap,
    type Vec3,
    vec3,
    xf,
} from "./math";
import type { PlaneResult } from "./mover";

const BIN_COUNT = 8;
const DESIRED_TRIANGLES_PER_LEAF = 4;
const MAXIMUM_TRIANGLES_PER_LEAF = 8;
const NULL_INDEX = -1;

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

// --- small local geometry helpers (b3AABB_AddPoint, B3_BOUNDS3_EMPTY, b3IsSaneAABB, ...) ---------

const emptyBounds = (): AABB => ({
    lowerBound: { x: FLT_MAX, y: FLT_MAX, z: FLT_MAX },
    upperBound: { x: -FLT_MAX, y: -FLT_MAX, z: -FLT_MAX },
});

const addPoint = (a: AABB, p: Vec3): AABB => ({
    lowerBound: vec3.min(a.lowerBound, p),
    upperBound: vec3.max(a.upperBound, p),
});

const isSaneAABB = (a: AABB): boolean => {
    const lo = a.lowerBound;
    const hi = a.upperBound;
    if (!Number.isFinite(lo.x) || !Number.isFinite(lo.y) || !Number.isFinite(lo.z)) return false;
    if (!Number.isFinite(hi.x) || !Number.isFinite(hi.y) || !Number.isFinite(hi.z)) return false;
    if (lo.x > hi.x || lo.y > hi.y || lo.z > hi.z) return false;
    if (lo.x < -HUGE || lo.y < -HUGE || lo.z < -HUGE) return false;
    if (hi.x > HUGE || hi.y > HUGE || hi.z > HUGE) return false;
    return true;
};

/** Index of the largest component (b3MajorAxis). Ties resolve toward the lower axis, matching C. */
const majorAxis = (v: Vec3): number => (v.x < v.y ? (v.y < v.z ? 2 : 1) : v.x < v.z ? 2 : 0);

const component = (v: Vec3, axis: number): number => (axis === 0 ? v.x : axis === 1 ? v.y : v.z);

/** Clamp scale magnitude away from zero, preserving sign (b3SafeScale). */
export const safeScale = (a: Vec3): Vec3 => {
    const sign = vec3.sign(a);
    const abs = vec3.abs(a);
    const clamped: Vec3 = {
        x: maxf(abs.x, MIN_SCALE),
        y: maxf(abs.y, MIN_SCALE),
        z: maxf(abs.z, MIN_SCALE),
    };
    return vec3.mul(sign, clamped);
};

// --- vertex welding (spatial hash) --------------------------------------------------------------

type VertexNode = { vertexIndex: number; nextNodeIndex: number };

// uint64 hash-combine matching b3SpatialHash's cell key; int32 components sign-extend to uint64.
const cellKey = (x: number, y: number, z: number): bigint => {
    let key = 0n;
    const combine = (c: number): bigint => {
        const uc = BigInt.asUintN(64, BigInt(c));
        return BigInt.asUintN(
            64,
            key ^ BigInt.asUintN(64, uc + 0x9e3779b9n + (key << 6n) + (key >> 2n)),
        );
    };
    key = combine(x);
    key = combine(y);
    key = combine(z);
    return key;
};

// Weld nearby vertices into a unique set; returns the deduped vertices plus the remapped indices
// (b3WeldVertices). Bucketing keys the 3x3x3 neighborhood, then confirms each candidate within tolerance.
const weldVertices = (
    srcVertices: Vec3[],
    srcIndices: number[],
    tolerance: number,
): { vertices: Vec3[]; indices: number[] } => {
    const vertexCount = srcVertices.length;
    const cellSize = f32(2 * tolerance);
    const nodes: VertexNode[] = [];
    const map = new Map<bigint, number>();
    const vertexMapping = new Array<number>(vertexCount).fill(0);
    const dstVertices: Vec3[] = [];

    const findDuplicate = (currentIndex: number): number => {
        const vertex = srcVertices[currentIndex];
        const baseX = Math.floor(f32(vertex.x / cellSize));
        const baseY = Math.floor(f32(vertex.y / cellSize));
        const baseZ = Math.floor(f32(vertex.z / cellSize));

        for (let dx = -1; dx <= 1; ++dx) {
            for (let dy = -1; dy <= 1; ++dy) {
                for (let dz = -1; dz <= 1; ++dz) {
                    const key = cellKey(baseX + dx, baseY + dy, baseZ + dz);
                    const head = map.get(key);
                    if (head === undefined) continue;
                    let nodeIndex = head;
                    while (nodeIndex !== NULL_INDEX) {
                        const node = nodes[nodeIndex];
                        const other = srcVertices[node.vertexIndex];
                        if (
                            absf(f32(vertex.x - other.x)) <= tolerance &&
                            absf(f32(vertex.y - other.y)) <= tolerance &&
                            absf(f32(vertex.z - other.z)) <= tolerance
                        ) {
                            return node.vertexIndex;
                        }
                        nodeIndex = node.nextNodeIndex;
                    }
                }
            }
        }

        const currentKey = cellKey(baseX, baseY, baseZ);
        const head = map.get(currentKey);
        if (head !== undefined) {
            nodes.push({ vertexIndex: currentIndex, nextNodeIndex: head });
            map.set(currentKey, nodes.length - 1);
        } else {
            map.set(currentKey, nodes.length);
            nodes.push({ vertexIndex: currentIndex, nextNodeIndex: NULL_INDEX });
        }
        return NULL_INDEX;
    };

    let uniqueCount = 0;
    for (let i = 0; i < vertexCount; ++i) {
        const duplicateIndex = findDuplicate(i);
        if (duplicateIndex === NULL_INDEX) {
            vertexMapping[i] = uniqueCount;
            dstVertices.push({ x: srcVertices[i].x, y: srcVertices[i].y, z: srcVertices[i].z });
            uniqueCount += 1;
        } else {
            vertexMapping[i] = vertexMapping[duplicateIndex];
        }
    }

    const indices = srcIndices.map((srcIndex) => vertexMapping[srcIndex]);
    return { vertices: dstVertices, indices };
};

// --- BVH build ----------------------------------------------------------------------------------

type Primitive = { aabb: AABB; center: Vec3; triangleIndex: number };

type Split = { axis: number; index: number; leftBounds: AABB; rightBounds: AABB };

// Binned SAH split over primitives[start, start+count) (b3SplitBinnedSah). Reorders in place; returns
// axis = -1 when no valid split is found.
const splitBinnedSah = (prims: Primitive[], start: number, count: number): Split => {
    const split: Split = {
        axis: -1,
        index: -1,
        leftBounds: emptyBounds(),
        rightBounds: emptyBounds(),
    };

    let bounds: AABB = { lowerBound: prims[start].center, upperBound: prims[start].center };
    for (let i = 1; i < count; ++i) bounds = addPoint(bounds, prims[start + i].center);

    let bestBucket = -1;
    let bestCost = FLT_MAX;

    for (let axis = 0; axis < 3; ++axis) {
        const extent = aabb.extents(bounds);
        if (component(extent, axis) < LINEAR_SLOP) continue;

        const bucketCount: number[] = new Array(BIN_COUNT).fill(0);
        const bucketBounds: AABB[] = [];
        for (let i = 0; i < BIN_COUNT; ++i) bucketBounds.push(emptyBounds());

        const span = f32(component(bounds.upperBound, axis) - component(bounds.lowerBound, axis));
        const factor = f32(f32(BIN_COUNT * f32(1 - FLT_EPSILON)) / span);
        for (let i = 0; i < count; ++i) {
            const center = prims[start + i].center;
            const index = Math.trunc(
                f32(factor * f32(component(center, axis) - component(bounds.lowerBound, axis))),
            );
            bucketCount[index] += 1;
            bucketBounds[index] = aabb.union(bucketBounds[index], prims[start + i].aabb);
        }

        for (let i = 0; i < BIN_COUNT - 1; ++i) {
            let leftCount = 0;
            let leftBounds = emptyBounds();
            for (let k = 0; k <= i; ++k) {
                leftCount += bucketCount[k];
                leftBounds = aabb.union(leftBounds, bucketBounds[k]);
            }
            let rightCount = 0;
            let rightBounds = emptyBounds();
            for (let k = i + 1; k < BIN_COUNT; ++k) {
                rightCount += bucketCount[k];
                rightBounds = aabb.union(rightBounds, bucketBounds[k]);
            }
            if (leftCount > 0 && rightCount > 0) {
                const cost = f32(
                    f32(leftCount * aabb.area(leftBounds)) +
                        f32(rightCount * aabb.area(rightBounds)),
                );
                if (cost < bestCost) {
                    bestBucket = i;
                    bestCost = cost;
                    split.axis = axis;
                    split.index = leftCount;
                    split.leftBounds = leftBounds;
                    split.rightBounds = rightBounds;
                }
            }
        }
    }

    if (bestBucket >= 0) {
        const axis = split.axis;
        const span = f32(component(bounds.upperBound, axis) - component(bounds.lowerBound, axis));
        const factor = f32(f32(BIN_COUNT * f32(1 - FLT_EPSILON)) / span);
        let splitIndex = 0;
        for (let i = 0; i < count; ++i) {
            const center = prims[start + i].center;
            const index = Math.trunc(
                f32(factor * f32(component(center, axis) - component(bounds.lowerBound, axis))),
            );
            if (index <= bestBucket) {
                const temp = prims[start + i];
                prims[start + i] = prims[start + splitIndex];
                prims[start + splitIndex] = temp;
                splitIndex++;
            }
        }
    }

    return split;
};

// Split the middle by count (b3SplitHalf). Always produces a valid split.
const splitHalf = (prims: Primitive[], start: number, count: number): Split => {
    const splitIndex = (count / 2) | 0;
    let leftBounds = emptyBounds();
    for (let i = 0; i < splitIndex; ++i) leftBounds = aabb.union(leftBounds, prims[start + i].aabb);
    let rightBounds = emptyBounds();
    for (let i = splitIndex; i < count; ++i)
        rightBounds = aabb.union(rightBounds, prims[start + i].aabb);
    const bounds = aabb.union(leftBounds, rightBounds);
    const axis = majorAxis(aabb.extents(bounds));
    return { axis, index: splitIndex, leftBounds, rightBounds };
};

// Hoare-partition the longest centroid axis about its midpoint (b3SplitMedian). count > 2.
const splitMedian = (prims: Primitive[], start: number, count: number): Split => {
    let lowerBound = prims[start].center;
    let upperBound = prims[start].center;
    for (let i = 1; i < count; ++i) {
        lowerBound = vec3.min(lowerBound, prims[start + i].center);
        upperBound = vec3.max(upperBound, prims[start + i].center);
    }
    const d = vec3.sub(upperBound, lowerBound);
    const c = vec3.scale(0.5, vec3.add(lowerBound, upperBound));

    let axis: number;
    let pivot: number;
    if (d.x >= d.y && d.x >= d.z) {
        axis = 0;
        pivot = c.x;
    } else if (d.y >= d.z) {
        axis = 1;
        pivot = c.y;
    } else {
        axis = 2;
        pivot = c.z;
    }

    let i1 = 0;
    let i2 = count;
    while (i1 < i2) {
        while (i1 < i2 && component(prims[start + i1].center, axis) < pivot) i1 += 1;
        while (i1 < i2 && component(prims[start + i2 - 1].center, axis) >= pivot) i2 -= 1;
        if (i1 < i2) {
            const temp = prims[start + i1];
            prims[start + i1] = prims[start + i2 - 1];
            prims[start + i2 - 1] = temp;
            i1 += 1;
            i2 -= 1;
        }
    }

    if (i1 === 0 || i1 === count - 1) i1 = (count / 2) | 0;

    let leftBounds = emptyBounds();
    for (let i = 0; i < i1; ++i) leftBounds = aabb.union(leftBounds, prims[start + i].aabb);
    let rightBounds = emptyBounds();
    for (let i = i1; i < count; ++i) rightBounds = aabb.union(rightBounds, prims[start + i].aabb);

    return { axis, index: i1, leftBounds, rightBounds };
};

const storeLeaf = (bounds: AABB, triangleCount: number, triangleOffset: number): MeshNode => ({
    lowerBound: bounds.lowerBound,
    upperBound: bounds.upperBound,
    leaf: true,
    axis: 0,
    childOffset: 0,
    triangleCount,
    triangleOffset,
});

// Recursively build the BVH into `nodes` (DFS preorder append), reordering the primitive range in
// place (b3BuildRecursive). Returns the node index and subtree height.
const buildRecursive = (
    nodes: MeshNode[],
    prims: Primitive[],
    start: number,
    count: number,
    useMedianSplit: boolean,
): { index: number; height: number } => {
    if (count > DESIRED_TRIANGLES_PER_LEAF) {
        let split = useMedianSplit
            ? splitMedian(prims, start, count)
            : splitBinnedSah(prims, start, count);

        if (split.axis < 0) {
            if (count > MAXIMUM_TRIANGLES_PER_LEAF) {
                split = splitHalf(prims, start, count);
            } else {
                let bounds = emptyBounds();
                for (let i = 0; i < count; ++i) bounds = aabb.union(bounds, prims[start + i].aabb);
                const index = nodes.length;
                nodes.push(storeLeaf(bounds, count, start));
                return { index, height: 1 };
            }
        }

        const index = nodes.length;
        nodes.push(storeLeaf(emptyBounds(), 0, 0)); // placeholder, filled after recursion
        const left = buildRecursive(nodes, prims, start, split.index, useMedianSplit);
        const right = buildRecursive(
            nodes,
            prims,
            start + split.index,
            count - split.index,
            useMedianSplit,
        );

        const bounds = aabb.union(split.leftBounds, split.rightBounds);
        const node = nodes[index];
        node.leaf = false;
        node.axis = split.axis;
        node.childOffset = right.index - index;
        node.lowerBound = bounds.lowerBound;
        node.upperBound = bounds.upperBound;
        node.triangleOffset = 0;
        node.triangleCount = 0;

        return { index, height: (left.height > right.height ? left.height : right.height) + 1 };
    }

    let bounds = emptyBounds();
    for (let i = 0; i < count; ++i) bounds = aabb.union(bounds, prims[start + i].aabb);
    const index = nodes.length;
    nodes.push(storeLeaf(bounds, count, start));
    return { index, height: 1 };
};

// Reorder triangles + material indices into BVH depth-first-order and reassign each leaf's offset to
// the running counter (b3SortMeshTriangles). Casts and volume queries then return sorted runs.
const sortMeshTriangles = (mesh: MeshData): void => {
    const tempTriangles: MeshTriangle[] = [];
    const tempMaterials: number[] = [];
    const stack: number[] = [0];
    let offset = 0;

    while (stack.length > 0) {
        const nodeIndex = stack.pop() as number;
        const node = mesh.nodes[nodeIndex];
        if (!node.leaf) {
            // Left child follows its parent; right child is parent + childOffset. Push right, then
            // left, so left is processed first (matching the C DFS visit order).
            stack.push(nodeIndex + node.childOffset);
            stack.push(nodeIndex + 1);
        } else {
            for (let t = 0; t < node.triangleCount; ++t) {
                const index = node.triangleOffset + t;
                tempTriangles.push(mesh.triangles[index]);
                tempMaterials.push(mesh.materialIndices[index]);
            }
            node.triangleOffset = offset;
            offset += node.triangleCount;
        }
    }

    for (let i = 0; i < mesh.triangles.length; ++i) {
        mesh.triangles[i] = tempTriangles[i];
        mesh.materialIndices[i] = tempMaterials[i];
    }
};

// --- edge / concave-flag identification ---------------------------------------------------------

type MeshEdge = {
    vertex1: number;
    vertex2: number;
    triangle1: number;
    triangle2: number;
    triangleCount: number;
    triangleEdgeIndex1: number;
    triangleEdgeIndex2: number;
};

const signedVolume = (v1: Vec3, v2: Vec3, v3: Vec3, p: Vec3): number => {
    const e1 = vec3.sub(v2, v1);
    const e2 = vec3.sub(v3, v1);
    const n = vec3.cross(e1, e2);
    return vec3.dot(n, vec3.sub(p, v1));
};

// Mark shared edges as concave / inverse-concave from the dihedral angle + signed volume of the
// opposite vertex (b3IdentifyEdges). Feeds the triangle-vs-hull collision's edge culling.
const identifyEdges = (mesh: MeshData): void => {
    const triangles = mesh.triangles;
    const vertices = mesh.vertices;
    const flags = mesh.flags;
    const triangleCount = triangles.length;
    const edgeCount = 3 * triangleCount;

    const edges: MeshEdge[] = new Array(edgeCount);
    const normals: Vec3[] = new Array(triangleCount);

    for (let i = 0; i < triangleCount; ++i) {
        const tri = triangles[i];
        const i1 = tri.index1;
        const i2 = tri.index2;
        const i3 = tri.index3;
        const mk = (a: number, b: number, edgeIndex: number): MeshEdge => ({
            vertex1: a < b ? a : b,
            vertex2: a > b ? a : b,
            triangle1: i,
            triangle2: NULL_INDEX,
            triangleEdgeIndex1: edgeIndex,
            triangleEdgeIndex2: 0xff,
            triangleCount: 1,
        });
        edges[3 * i + 0] = mk(i1, i2, 0);
        edges[3 * i + 1] = mk(i2, i3, 1);
        edges[3 * i + 2] = mk(i3, i1, 2);

        const v1 = vertices[i1];
        const v2 = vertices[i2];
        const v3 = vertices[i3];
        const n = vec3.cross(vec3.sub(v2, v1), vec3.sub(v3, v1));
        normals[i] = vec3.normalize(n);
    }

    // Edges are keyed by their ordered vertex pair (C uses a packed uint64; a string is exact for any
    // vertex count and only identifies the pair — the map value picks the first-inserted shared edge).
    const map = new Map<string, number>();
    const edgeKey = (e: MeshEdge): string => `${e.vertex1},${e.vertex2}`;
    map.set(edgeKey(edges[0]), 0);
    for (let i = 1; i < edgeCount; ++i) {
        const edge = edges[i];
        const key = edgeKey(edge);
        const other = map.get(key);
        if (other === undefined) {
            map.set(key, i);
        } else {
            const base = edges[other];
            if (base.triangleCount === 1) {
                base.triangle2 = edge.triangle1;
                base.triangleEdgeIndex2 = edge.triangleEdgeIndex1;
            }
            base.triangleCount += 1;
        }
    }

    const cos5Deg = f32(0.9962);
    const concaveFlags = [
        MeshEdgeFlags.ConcaveEdge1,
        MeshEdgeFlags.ConcaveEdge2,
        MeshEdgeFlags.ConcaveEdge3,
    ];
    const inverseFlags = [
        MeshEdgeFlags.InverseConcaveEdge1,
        MeshEdgeFlags.InverseConcaveEdge2,
        MeshEdgeFlags.InverseConcaveEdge3,
    ];

    for (let i = 0; i < edgeCount; ++i) {
        const edge = edges[i];
        if (edge.triangleCount !== 2) continue;

        const triangle1 = triangles[edge.triangle1];
        const triangle2 = triangles[edge.triangle2];
        const j1 = triangle2.index1;
        const j2 = triangle2.index2;
        const j3 = triangle2.index3;

        let opposite = NULL_INDEX;
        if (edge.triangleEdgeIndex2 === 0) opposite = j3;
        else if (edge.triangleEdgeIndex2 === 1) opposite = j1;
        else opposite = j2;

        const v1 = vertices[triangle1.index1];
        const v2 = vertices[triangle1.index2];
        const v3 = vertices[triangle1.index3];
        const p = vertices[opposite];

        const volume = signedVolume(v1, v2, v3, p);
        const cosAngle = vec3.dot(normals[edge.triangle1], normals[edge.triangle2]);
        if (volume > 0 || cosAngle > cos5Deg) {
            flags[edge.triangle1] |= concaveFlags[edge.triangleEdgeIndex1];
            flags[edge.triangle2] |= concaveFlags[edge.triangleEdgeIndex2];
        }
        if (volume < 0 || cosAngle > cos5Deg) {
            flags[edge.triangle1] |= inverseFlags[edge.triangleEdgeIndex1];
            flags[edge.triangle2] |= inverseFlags[edge.triangleEdgeIndex2];
        }
    }
};

// --- mesh creation ------------------------------------------------------------------------------

/**
 * Build immutable mesh collision data from vertices + triangle indices (b3CreateMesh): optional
 * welding, degenerate-triangle culling, BVH build (SAH or median split), DFS triangle sort, and
 * optional edge identification. Returns null when the input is invalid or the bounds are insane.
 *
 * @example
 * const mesh = createMesh({ vertices, indices, useMedianSplit: false, identifyEdges: true });
 */
export function createMesh(def: MeshDef): MeshData | null {
    if (def.vertices.length < 3 || def.indices.length < 3) return null;
    const triangleCountIn = (def.indices.length / 3) | 0;
    if (triangleCountIn <= 0) return null;

    let vertices: Vec3[];
    let indices: number[];
    if (def.weldVertices && (def.weldTolerance ?? 0) > 0) {
        const welded = weldVertices(def.vertices, def.indices, def.weldTolerance as number);
        vertices = welded.vertices;
        indices = welded.indices;
    } else {
        vertices = def.vertices.map((v) => ({ x: v.x, y: v.y, z: v.z }));
        indices = def.indices.slice();
    }

    let meshBounds = emptyBounds();
    const primitives: Primitive[] = [];
    let degenerateCount = 0;
    const minArea = f32(f32(f32(f32(0.01) * LINEAR_SLOP) * LINEAR_SLOP));
    let surfaceArea = 0;
    let materialCount = 1;

    for (let index = 0; index < triangleCountIn; ++index) {
        const index1 = indices[3 * index + 0];
        const index2 = indices[3 * index + 1];
        const index3 = indices[3 * index + 2];
        const vertex1 = vertices[index1];
        const vertex2 = vertices[index2];
        const vertex3 = vertices[index3];

        const normal = vec3.cross(vec3.sub(vertex2, vertex1), vec3.sub(vertex3, vertex1));
        const area = f32(0.5 * vec3.length(normal));

        if (area < minArea) {
            if (index1 !== index2 && index1 !== index3 && index2 !== index3) degenerateCount += 1;
            continue;
        }

        surfaceArea = f32(surfaceArea + area);

        const box: AABB = {
            lowerBound: vec3.min(vertex1, vec3.min(vertex2, vertex3)),
            upperBound: vec3.max(vertex1, vec3.max(vertex2, vertex3)),
        };
        primitives.push({ aabb: box, center: aabb.center(box), triangleIndex: index });

        if (def.materialIndices != null) {
            const m = def.materialIndices[index] + 1;
            materialCount = materialCount > m ? materialCount : m;
        }

        meshBounds = aabb.union(meshBounds, box);
    }

    const triangleCount = primitives.length;
    if (!isSaneAABB(meshBounds)) return null;

    const nodes: MeshNode[] = [];
    const { height } = buildRecursive(
        nodes,
        primitives,
        0,
        triangleCount,
        def.useMedianSplit ?? false,
    );

    const triangles: MeshTriangle[] = new Array(triangleCount);
    const materialIndices: number[] = new Array(triangleCount).fill(0);
    const flags: number[] = new Array(triangleCount).fill(0);
    for (let index = 0; index < triangleCount; ++index) {
        const primitive = primitives[index];
        triangles[index] = {
            index1: indices[3 * primitive.triangleIndex + 0],
            index2: indices[3 * primitive.triangleIndex + 1],
            index3: indices[3 * primitive.triangleIndex + 2],
        };
        if (def.materialIndices != null) {
            materialIndices[index] = def.materialIndices[primitive.triangleIndex];
        }
    }

    const mesh: MeshData = {
        bounds: meshBounds,
        surfaceArea,
        treeHeight: height,
        degenerateCount,
        nodes,
        vertices,
        triangles,
        materialIndices,
        materialCount,
        flags,
    };

    sortMeshTriangles(mesh);
    if (def.identifyEdges) identifyEdges(mesh);

    return mesh;
}

/** Box mesh centered at `center` with half-extents `extent` (b3CreateBoxMesh). 8 verts, 12 triangles. */
export function createBoxMesh(center: Vec3, extent: Vec3, identifyEdges: boolean): MeshData {
    const x = extent.x;
    const y = extent.y;
    const z = extent.z;
    const corners: Vec3[] = [
        { x, y, z },
        { x: -x, y, z },
        { x: -x, y: -y, z },
        { x, y: -y, z },
        { x, y, z: -z },
        { x: -x, y, z: -z },
        { x: -x, y: -y, z: -z },
        { x, y: -y, z: -z },
    ];
    const vertices = corners.map((c) => vec3.add(c, center));
    // biome-ignore format: triangle index rows mirror the C face layout
    const indices = [
        0, 1, 3, 1, 2, 3, // front
        0, 4, 1, 1, 4, 5, // top
        0, 3, 7, 4, 0, 7, // right
        4, 7, 5, 6, 5, 7, // back
        1, 5, 2, 6, 2, 5, // left
        3, 2, 7, 6, 7, 2, // bottom
    ];
    return createMesh({ vertices, indices, useMedianSplit: false, identifyEdges }) as MeshData;
}

/** Hollow box mesh — inward-facing faces (b3CreateHollowBoxMesh). Always identifies edges. */
export function createHollowBoxMesh(center: Vec3, extent: Vec3): MeshData {
    const x = extent.x;
    const y = extent.y;
    const z = extent.z;
    const corners: Vec3[] = [
        { x, y, z },
        { x: -x, y, z },
        { x: -x, y: -y, z },
        { x, y: -y, z },
        { x, y, z: -z },
        { x: -x, y, z: -z },
        { x: -x, y: -y, z: -z },
        { x, y: -y, z: -z },
    ];
    const vertices = corners.map((c) => vec3.add(c, center));
    // biome-ignore format: inward face winding mirrors the C layout
    const indices = [
        3, 1, 0, 3, 2, 1, // front
        1, 4, 0, 5, 4, 1, // top
        7, 3, 0, 7, 0, 4, // right
        5, 7, 4, 7, 5, 6, // back
        2, 5, 1, 5, 2, 6, // left
        7, 2, 3, 2, 7, 6, // bottom
    ];
    return createMesh({
        vertices,
        indices,
        useMedianSplit: false,
        identifyEdges: true,
    }) as MeshData;
}

/** X/Z grid mesh, `xCount`×`zCount` cells of `cellWidth`, centered at origin (b3CreateGridMesh). */
export function createGridMesh(
    xCount: number,
    zCount: number,
    cellWidth: number,
    materialCount: number,
    identifyEdges: boolean,
): MeshData {
    const vertices: Vec3[] = [];
    const xWidth = f32(cellWidth * xCount);
    const zWidth = f32(cellWidth * zCount);

    let x = f32(-0.5 * xWidth);
    for (let ix = 0; ix <= xCount; ++ix) {
        let z = f32(-0.5 * zWidth);
        for (let iz = 0; iz <= zCount; ++iz) {
            vertices.push({ x, y: 0, z });
            z = f32(z + cellWidth);
        }
        x = f32(x + cellWidth);
    }

    const indices: number[] = [];
    const materialIndices: number[] = [];
    let materialIndex = 0;
    for (let ix = 0; ix < xCount; ++ix) {
        for (let iz = 0; iz < zCount; ++iz) {
            const index1 = iz + (zCount + 1) * ix;
            const index2 = index1 + 1;
            const index3 = index2 + (zCount + 1);
            const index4 = index3 - 1;
            indices.push(index1, index2, index3, index3, index4, index1);
            if (materialCount > 0) {
                materialIndices.push(materialIndex % materialCount, materialIndex % materialCount);
            }
            materialIndex += 1;
        }
    }

    return createMesh({
        vertices,
        indices,
        materialIndices: materialCount > 0 ? materialIndices : null,
        useMedianSplit: true,
        identifyEdges,
    }) as MeshData;
}

/**
 * Build a wavy grid mesh: a grid whose vertex heights follow a product of two sine waves
 * (b3CreateWaveMesh). Used by the "trees" benchmark ground.
 *
 * @example
 * const mesh = createWaveMesh(150, 200, 1, 0.4, 0.05, 0.1);
 */
export function createWaveMesh(
    xCount: number,
    zCount: number,
    cellWidth: number,
    amplitude: number,
    rowFrequency: number,
    columnFrequency: number,
): MeshData {
    const vertices: Vec3[] = [];
    const xWidth = f32(cellWidth * xCount);
    const zWidth = f32(cellWidth * zCount);

    const omegaZ = f32(f32(f32(2 * PI) * rowFrequency) * cellWidth);
    const omegaX = f32(f32(f32(2 * PI) * columnFrequency) * cellWidth);

    let x = f32(-0.5 * xWidth);
    for (let ix = 0; ix <= xCount; ++ix) {
        // Box3D's wave mesh is its one geometry helper that uses libm sinf rather than the
        // portable b3ComputeCosSin, so there is no cross-platform-deterministic sine to mirror.
        // Math.fround(Math.sin(x)) reproduces sinf bit-for-bit on the glibc reference build.
        const rowHeight = f32(Math.sin(f32(omegaX * ix)));
        let z = f32(-0.5 * zWidth);
        for (let iz = 0; iz <= zCount; ++iz) {
            const columnHeight = f32(Math.sin(f32(omegaZ * iz)));
            const y = f32(f32(amplitude * rowHeight) * columnHeight);
            vertices.push({ x, y, z });
            z = f32(z + cellWidth);
        }
        x = f32(x + cellWidth);
    }

    const indices: number[] = [];
    for (let ix = 0; ix < xCount; ++ix) {
        for (let iz = 0; iz < zCount; ++iz) {
            const index1 = iz + (zCount + 1) * ix;
            const index2 = index1 + 1;
            const index3 = index2 + (zCount + 1);
            const index4 = index3 - 1;
            indices.push(index1, index2, index3, index3, index4, index1);
        }
    }

    return createMesh({ vertices, indices, useMedianSplit: true, identifyEdges: true }) as MeshData;
}

/**
 * Build a torus (donut) mesh from a radial × tubular vertex grid (b3CreateTorusMesh). Uses the
 * portable trig, so it is bit-exact against the C reference. Used by the "rain" benchmark.
 *
 * @example
 * const mesh = createTorusMesh(16, 16, 3.75, 1);
 */
export function createTorusMesh(
    radialResolution: number,
    tubularResolution: number,
    radius: number,
    thickness: number,
): MeshData {
    // B3_TWO_PI (6.283185307f) is bit-equal to f32(2 * PI) — 0x40c90fdb.
    const twoPi = f32(2 * PI);
    const vertices: Vec3[] = [];
    for (let radialIndex = 0; radialIndex < radialResolution; ++radialIndex) {
        for (let tubularIndex = 0; tubularIndex < tubularResolution; ++tubularIndex) {
            const u = f32(f32(f32(tubularIndex) / tubularResolution) * twoPi);
            const v = f32(f32(f32(radialIndex) / radialResolution) * twoPi);
            const csU = computeCosSin(u);
            const csV = computeCosSin(v);
            const ring = f32(radius + f32(thickness * csV.cosine));
            const px = f32(ring * csU.cosine);
            const py = f32(ring * csU.sine);
            const pz = f32(thickness * csV.sine);
            vertices.push({ x: px, y: py, z: pz });
        }
    }

    const indices: number[] = [];
    for (let radialIndex1 = 0; radialIndex1 < radialResolution; ++radialIndex1) {
        const radialIndex2 = (radialIndex1 + 1) % radialResolution;
        for (let tubularIndex1 = 0; tubularIndex1 < tubularResolution; ++tubularIndex1) {
            const tubularIndex2 = (tubularIndex1 + 1) % tubularResolution;
            const index1 = radialIndex1 * tubularResolution + tubularIndex1;
            const index2 = radialIndex1 * tubularResolution + tubularIndex2;
            const index3 = radialIndex2 * tubularResolution + tubularIndex2;
            const index4 = radialIndex2 * tubularResolution + tubularIndex1;
            indices.push(index1, index2, index3, index3, index4, index1);
        }
    }

    return createMesh({
        vertices,
        indices,
        useMedianSplit: false,
        identifyEdges: true,
    }) as MeshData;
}

// --- queries ------------------------------------------------------------------------------------

/** World-space AABB enclosing a scaled, transformed mesh (b3ComputeMeshAABB). */
export function computeMeshAABB(data: MeshData, transform: Transform, scale: Vec3): AABB {
    const scaledLower = vec3.mul(scale, data.bounds.lowerBound);
    const scaledUpper = vec3.mul(scale, data.bounds.upperBound);
    const bounds: AABB = {
        lowerBound: vec3.min(scaledLower, scaledUpper),
        upperBound: vec3.max(scaledLower, scaledUpper),
    };
    return aabb.transform(transform, bounds);
}

// AABB-AABB overlap in unscaled mesh space (b3TestBoundsOverlap): all separation components <= 0.
const testBoundsOverlap = (
    nodeMin: Vec3,
    nodeMax: Vec3,
    boundsMin: Vec3,
    boundsMax: Vec3,
): boolean => {
    const s = vec3.max(vec3.sub(boundsMin, nodeMax), vec3.sub(nodeMin, boundsMax));
    return s.x <= 0 && s.y <= 0 && s.z <= 0;
};

const anyGreater3 = (v: Vec3): boolean => v.x > 0 || v.y > 0 || v.z > 0;

// 13-axis triangle-vs-AABB SAT overlap (b3TestBoundsTriangleOverlap), in unscaled mesh space.
export const testBoundsTriangleOverlap = (
    nodeCenter: Vec3,
    nodeExtent: Vec3,
    vertex1: Vec3,
    vertex2: Vec3,
    vertex3: Vec3,
): boolean => {
    const v1 = vec3.sub(vertex1, nodeCenter);
    const v2 = vec3.sub(vertex2, nodeCenter);
    const v3 = vec3.sub(vertex3, nodeCenter);

    const triMin = vec3.min(v1, vec3.min(v2, v3));
    const triMax = vec3.max(v1, vec3.max(v2, v3));
    const faceSep = vec3.max(vec3.sub(triMin, nodeExtent), vec3.neg(vec3.add(triMax, nodeExtent)));
    if (anyGreater3(faceSep)) return false;

    const edge1 = vec3.sub(v2, v1);
    const edge2 = vec3.sub(v3, v2);
    const edge3 = vec3.sub(v1, v3);
    const normal = vec3.cross(edge1, edge2);

    const triSep = f32(absf(vec3.dot(normal, v1)) - vec3.dot(vec3.abs(normal), nodeExtent));
    if (triSep > 0) return false;

    const edgeSep = (edge: Vec3, sum: Vec3, otherEdge: Vec3): Vec3 =>
        vec3.sub(
            vec3.sub(vec3.abs(vec3.cross(edge, sum)), vec3.abs(vec3.cross(edge, otherEdge))),
            vec3.scale(2, vec3.modifiedCross(vec3.abs(edge), nodeExtent)),
        );

    if (anyGreater3(edgeSep(edge1, vec3.add(v1, v3), edge3))) return false;
    if (anyGreater3(edgeSep(edge2, vec3.add(v1, v2), edge1))) return false;
    if (anyGreater3(edgeSep(edge3, vec3.add(v2, v3), edge2))) return false;

    return true;
};

/** Callback for {@link queryMesh}: scaled triangle vertices + triangle index; return false to stop. */
export type MeshQueryFcn = (a: Vec3, b: Vec3, c: Vec3, triangleIndex: number) => boolean;

/**
 * Visit every mesh triangle overlapping `bounds` (world-space AABB translated into local mesh space
 * by the caller), in BVH depth-first order (b3QueryMesh). Scaled triangle vertices are handed to
 * `fcn`, with winding flipped when the scale reflects.
 */
export function queryMesh(mesh: Mesh, bounds: AABB, fcn: MeshQueryFcn): void {
    const meshScale = mesh.scale;
    const clockwise = f32(f32(meshScale.x * meshScale.y) * meshScale.z) > 0;

    const invScale: Vec3 = {
        x: f32(1 / meshScale.x),
        y: f32(1 / meshScale.y),
        z: f32(1 / meshScale.z),
    };
    const temp1 = vec3.mul(invScale, bounds.lowerBound);
    const temp2 = vec3.mul(invScale, bounds.upperBound);
    const boundsMin = vec3.min(temp1, temp2);
    const boundsMax = vec3.max(temp1, temp2);
    const boundsCenter = vec3.scale(0.5, vec3.add(boundsMin, boundsMax));
    const boundsExtent = vec3.sub(boundsMax, boundsCenter);

    const data = mesh.data;
    const nodes = data.nodes;
    const triangles = data.triangles;
    const vertices = data.vertices;

    const stack: number[] = [];
    let nodeIndex = 0;

    while (true) {
        const node = nodes[nodeIndex];
        if (testBoundsOverlap(node.lowerBound, node.upperBound, boundsMin, boundsMax)) {
            if (node.leaf) {
                const triangleOffset = node.triangleOffset;
                for (let index = 0; index < node.triangleCount; ++index) {
                    const triangleIndex = triangleOffset + index;
                    const triangle = triangles[triangleIndex];
                    const vertex1 = vertices[triangle.index1];
                    const vertex2 = vertices[triangle.index2];
                    const vertex3 = vertices[triangle.index3];
                    if (
                        testBoundsTriangleOverlap(
                            boundsCenter,
                            boundsExtent,
                            vertex1,
                            vertex2,
                            vertex3,
                        )
                    ) {
                        const a = vec3.mul(meshScale, vertex1);
                        const b = clockwise
                            ? vec3.mul(meshScale, vertex2)
                            : vec3.mul(meshScale, vertex3);
                        const c = clockwise
                            ? vec3.mul(meshScale, vertex3)
                            : vec3.mul(meshScale, vertex2);
                        if (fcn(a, b, c, triangleIndex) === false) return;
                    }
                }
            } else {
                stack.push(nodeIndex + node.childOffset);
                nodeIndex = nodeIndex + 1;
                continue;
            }
        }
        if (stack.length === 0) break;
        nodeIndex = stack.pop() as number;
    }
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

// Front-to-back child order for a BVH ray/shape cast: descend the near child, stash the far one.
// The mesh node stores its right child at `nodeIndex + childOffset`, its left at `nodeIndex + 1`.
const inverseScaled = (meshScale: Vec3): Vec3 => ({
    x: f32(1 / meshScale.x),
    y: f32(1 / meshScale.y),
    z: f32(1 / meshScale.z),
});

/**
 * Ray vs mesh (b3RayCastMesh). Traverses the BVH front-to-back in inverse-scaled space, testing each
 * candidate triangle in scaled space; returns the nearest hit within `maxFraction`. The point/normal
 * are in the mesh's local frame (the caller lifts them to world).
 */
export function rayCastMesh(mesh: Mesh, input: RayCastInput): CastOutput {
    const data = mesh.data;
    const meshScale = mesh.scale;

    const output = emptyCastOutput();
    output.fraction = input.maxFraction;
    output.triangleIndex = NULL_INDEX;

    const rayStart = input.origin;
    const rayDelta = input.translation;

    const invScale = inverseScaled(meshScale);
    const clockwise = f32(f32(meshScale.x * meshScale.y) * meshScale.z) < 0;

    const invScaledRayStart = vec3.mul(invScale, rayStart);
    const invScaledRayDelta = vec3.mul(invScale, rayDelta);
    let invScaledRayEnd = vec3.add(
        invScaledRayStart,
        vec3.scale(output.fraction, invScaledRayDelta),
    );
    let invScaledRayMin = vec3.min(invScaledRayStart, invScaledRayEnd);
    let invScaledRayMax = vec3.max(invScaledRayStart, invScaledRayEnd);

    const nodes = data.nodes;
    const triangles = data.triangles;
    const vertices = data.vertices;
    const materialIndices = data.materialIndices;

    const stack: number[] = [];
    let nodeIndex = 0;
    while (true) {
        const node = nodes[nodeIndex];
        if (
            testBoundsOverlap(node.lowerBound, node.upperBound, invScaledRayMin, invScaledRayMax) &&
            testBoundsRayOverlap(
                node.lowerBound,
                node.upperBound,
                invScaledRayStart,
                invScaledRayDelta,
            )
        ) {
            if (node.leaf) {
                const triangleOffset = node.triangleOffset;
                for (let index = 0; index < node.triangleCount; ++index) {
                    const triangleIndex = triangleOffset + index;
                    const triangle = triangles[triangleIndex];
                    const vertex1 = vec3.mul(meshScale, vertices[triangle.index1]);
                    const vertex2 = clockwise
                        ? vec3.mul(meshScale, vertices[triangle.index3])
                        : vec3.mul(meshScale, vertices[triangle.index2]);
                    const vertex3 = clockwise
                        ? vec3.mul(meshScale, vertices[triangle.index2])
                        : vec3.mul(meshScale, vertices[triangle.index3]);

                    const alpha = intersectRayTriangle(
                        rayStart,
                        rayDelta,
                        vertex1,
                        vertex2,
                        vertex3,
                    );
                    if (alpha < output.fraction) {
                        const edge1 = vec3.sub(vertex2, vertex1);
                        const edge2 = vec3.sub(vertex3, vertex1);
                        output.normal = vec3.normalize(vec3.cross(edge1, edge2));
                        output.point = vec3.mulAdd(input.origin, alpha, input.translation);
                        output.fraction = alpha;
                        output.triangleIndex = triangleIndex;
                        output.materialIndex = materialIndices[triangleIndex];
                        output.hit = true;

                        invScaledRayEnd = vec3.add(
                            invScaledRayStart,
                            vec3.scale(alpha, invScaledRayDelta),
                        );
                        invScaledRayMin = vec3.min(invScaledRayStart, invScaledRayEnd);
                        invScaledRayMax = vec3.max(invScaledRayStart, invScaledRayEnd);
                    }
                }
            } else {
                const axis = node.axis;
                const delta =
                    axis === 0
                        ? invScaledRayDelta.x
                        : axis === 1
                          ? invScaledRayDelta.y
                          : invScaledRayDelta.z;
                if (delta > 0) {
                    stack.push(nodeIndex + node.childOffset);
                    nodeIndex = nodeIndex + 1;
                } else {
                    stack.push(nodeIndex + 1);
                    nodeIndex = nodeIndex + node.childOffset;
                }
                continue;
            }
        }
        if (stack.length === 0) break;
        nodeIndex = stack.pop() as number;
    }

    return output;
}

/**
 * Shape cast (a swept convex proxy) vs mesh (b3ShapeCastMesh). Sweeps the proxy's bounding-box center
 * through the BVH front-to-back, running the full GJK shape cast against each overlapping triangle;
 * returns the nearest hit. Point/normal are in the mesh's local frame.
 */
export function shapeCastMesh(mesh: Mesh, input: ShapeCastInput): CastOutput {
    const data = mesh.data;
    const meshScale = mesh.scale;

    let output = emptyCastOutput();
    output.fraction = input.maxFraction;
    output.triangleIndex = NULL_INDEX;

    const shapeBounds = aabb.make(input.proxy.points, input.proxy.count, input.proxy.radius);
    const shapeExtent = aabb.extents(shapeBounds);

    const rayStart = aabb.center(shapeBounds);
    const rayDelta = input.translation;
    let rayEnd = vec3.add(rayStart, vec3.scale(output.fraction, rayDelta));
    let rayMin = vec3.min(rayStart, rayEnd);
    let rayMax = vec3.max(rayStart, rayEnd);

    const invScale = inverseScaled(meshScale);
    const absInvScale = vec3.abs(invScale);
    const clockwise = f32(f32(meshScale.x * meshScale.y) * meshScale.z) < 0;

    const invScaledRayStart = vec3.mul(invScale, rayStart);
    const invScaledRayDelta = vec3.mul(invScale, rayDelta);
    let invScaledRayEnd = vec3.add(
        invScaledRayStart,
        vec3.scale(output.fraction, invScaledRayDelta),
    );
    let invScaledRayMin = vec3.min(invScaledRayStart, invScaledRayEnd);
    let invScaledRayMax = vec3.max(invScaledRayStart, invScaledRayEnd);
    const invScaledShapeExtent = vec3.mul(absInvScale, shapeExtent);

    const nodes = data.nodes;
    const triangles = data.triangles;
    const vertices = data.vertices;
    const materialIndices = data.materialIndices;

    const pairInput: ShapeCastPairInput = {
        proxyA: { points: [], count: 0, radius: 0 },
        proxyB: input.proxy,
        transform: xf.identity(),
        translationB: input.translation,
        maxFraction: output.fraction,
        canEncroach: input.canEncroach,
    };

    const stack: number[] = [];
    let nodeIndex = 0;
    while (true) {
        const node = nodes[nodeIndex];
        const nodeMin = vec3.sub(node.lowerBound, invScaledShapeExtent);
        const nodeMax = vec3.add(node.upperBound, invScaledShapeExtent);
        if (
            testBoundsOverlap(nodeMin, nodeMax, invScaledRayMin, invScaledRayMax) &&
            testBoundsRayOverlap(nodeMin, nodeMax, invScaledRayStart, invScaledRayDelta)
        ) {
            if (node.leaf) {
                const triangleOffset = node.triangleOffset;
                for (let index = 0; index < node.triangleCount; ++index) {
                    const triangleIndex = triangleOffset + index;
                    const triangle = triangles[triangleIndex];
                    const vertex1 = vec3.mul(meshScale, vertices[triangle.index1]);
                    const vertex2 = clockwise
                        ? vec3.mul(meshScale, vertices[triangle.index3])
                        : vec3.mul(meshScale, vertices[triangle.index2]);
                    const vertex3 = clockwise
                        ? vec3.mul(meshScale, vertices[triangle.index2])
                        : vec3.mul(meshScale, vertices[triangle.index3]);

                    const triangleMin = vec3.sub(
                        vec3.min(vertex1, vec3.min(vertex2, vertex3)),
                        shapeExtent,
                    );
                    const triangleMax = vec3.add(
                        vec3.max(vertex1, vec3.max(vertex2, vertex3)),
                        shapeExtent,
                    );
                    if (testBoundsOverlap(triangleMin, triangleMax, rayMin, rayMax)) {
                        const origin = vertex1;
                        pairInput.proxyA = {
                            points: [
                                vec3.zero(),
                                vec3.sub(vertex2, origin),
                                vec3.sub(vertex3, origin),
                            ],
                            count: 3,
                            radius: 0,
                        };
                        pairInput.transform = { p: vec3.neg(origin), q: quat.identity() };
                        pairInput.maxFraction = output.fraction;

                        const pairOutput = shapeCast(pairInput);
                        if (pairOutput.hit) {
                            pairOutput.point = vec3.add(pairOutput.point, origin);
                            output = pairOutput;
                            output.triangleIndex = triangleIndex;
                            output.materialIndex = materialIndices[triangleIndex];

                            rayEnd = vec3.add(rayStart, vec3.scale(pairOutput.fraction, rayDelta));
                            rayMin = vec3.min(rayStart, rayEnd);
                            rayMax = vec3.max(rayStart, rayEnd);

                            invScaledRayEnd = vec3.add(
                                invScaledRayStart,
                                vec3.scale(pairOutput.fraction, invScaledRayDelta),
                            );
                            invScaledRayMin = vec3.min(invScaledRayStart, invScaledRayEnd);
                            invScaledRayMax = vec3.max(invScaledRayStart, invScaledRayEnd);
                        }
                    }
                }
            } else {
                const axis = node.axis;
                const delta =
                    axis === 0
                        ? invScaledRayDelta.x
                        : axis === 1
                          ? invScaledRayDelta.y
                          : invScaledRayDelta.z;
                if (delta > 0) {
                    stack.push(nodeIndex + node.childOffset);
                    nodeIndex = nodeIndex + 1;
                } else {
                    stack.push(nodeIndex + 1);
                    nodeIndex = nodeIndex + node.childOffset;
                }
                continue;
            }
        }
        if (stack.length === 0) break;
        nodeIndex = stack.pop() as number;
    }

    return output;
}

/**
 * True if `proxy` (in world space, with the mesh at `transform`) overlaps the mesh (b3OverlapMesh).
 * Pulls the proxy into mesh-local space, walks the BVH in inverse-scaled space, and runs GJK against
 * each candidate triangle in scaled space; reports a hit within the overlap slop. Winding-agnostic.
 */
export function overlapMesh(mesh: Mesh, transform: Transform, proxy: ShapeProxy): boolean {
    const cache = emptyCache();
    const localProxy = makeLocalProxy(proxy, transform);
    const box = computeProxyAABB(localProxy);

    const meshScale = mesh.scale;
    const invScale = inverseScaled(meshScale);
    // Scale may reflect, so the unscaled bounds min/max can swap; recompute them.
    const temp1 = vec3.mul(invScale, box.lowerBound);
    const temp2 = vec3.mul(invScale, box.upperBound);
    const invScaledBoundsMin = vec3.min(temp1, temp2);
    const invScaledBoundsMax = vec3.max(temp1, temp2);
    const invScaledBoundsCenter = vec3.scale(0.5, vec3.add(invScaledBoundsMin, invScaledBoundsMax));
    const invScaledBoundsExtent = vec3.sub(invScaledBoundsMax, invScaledBoundsCenter);

    const input: DistanceInput = {
        proxyA: { points: [], count: 0, radius: 0 },
        proxyB: localProxy,
        transform: xf.identity(),
        useRadii: true,
    };

    const nodes = mesh.data.nodes;
    const triangles = mesh.data.triangles;
    const vertices = mesh.data.vertices;

    const stack: number[] = [];
    let nodeIndex = 0;
    while (true) {
        const node = nodes[nodeIndex];
        if (
            testBoundsOverlap(
                node.lowerBound,
                node.upperBound,
                invScaledBoundsMin,
                invScaledBoundsMax,
            )
        ) {
            if (node.leaf) {
                const triangleOffset = node.triangleOffset;
                for (let index = 0; index < node.triangleCount; ++index) {
                    const triangleIndex = triangleOffset + index;
                    const triangle = triangles[triangleIndex];
                    const vertex1 = vertices[triangle.index1];
                    const vertex2 = vertices[triangle.index2];
                    const vertex3 = vertices[triangle.index3];
                    if (
                        testBoundsTriangleOverlap(
                            invScaledBoundsCenter,
                            invScaledBoundsExtent,
                            vertex1,
                            vertex2,
                            vertex3,
                        )
                    ) {
                        input.proxyA = {
                            points: [
                                vec3.mul(meshScale, vertex1),
                                vec3.mul(meshScale, vertex2),
                                vec3.mul(meshScale, vertex3),
                            ],
                            count: 3,
                            radius: 0,
                        };
                        cache.count = 0;
                        const distanceOutput = shapeDistance(input, cache);
                        if (distanceOutput.distance < OVERLAP_SLOP) return true;
                    }
                }
            } else {
                stack.push(nodeIndex + node.childOffset);
                nodeIndex = nodeIndex + 1;
                continue;
            }
        }
        if (stack.length === 0) break;
        nodeIndex = stack.pop() as number;
    }

    return false;
}

/**
 * Collision planes between a capsule mover and a triangle mesh (b3CollideMoverAndMesh), in the mesh's
 * frame. Walks the BVH in unscaled space, tests each candidate triangle against the mover's core
 * segment (in scaled space, winding-agnostic), and emits one plane per triangle within reach. Deep
 * overlap is dropped (no SAT for movers). Stops at `capacity` planes.
 */
export function collideMoverAndMesh(mesh: Mesh, capacity: number, mover: Capsule): PlaneResult[] {
    const planes: PlaneResult[] = [];
    if (capacity === 0) {
        return planes;
    }

    const input: DistanceInput = {
        proxyA: { points: [], count: 0, radius: 0 },
        proxyB: { points: [mover.center1, mover.center2], count: 2, radius: 0 },
        transform: xf.identity(),
        useRadii: false,
    };
    const cache = emptyCache();

    const r = { x: mover.radius, y: mover.radius, z: mover.radius };
    const boundsMin = vec3.sub(vec3.min(mover.center1, mover.center2), r);
    const boundsMax = vec3.add(vec3.max(mover.center1, mover.center2), r);

    // Scale may reflect, so the unscaled bounds min/max can swap; recompute them.
    const meshScale = mesh.scale;
    const invScale = inverseScaled(meshScale);
    const temp1 = vec3.mul(invScale, boundsMin);
    const temp2 = vec3.mul(invScale, boundsMax);
    const invScaledBoundsMin = vec3.min(temp1, temp2);
    const invScaledBoundsMax = vec3.max(temp1, temp2);
    const invScaledBoundsCenter = vec3.scale(0.5, vec3.add(invScaledBoundsMin, invScaledBoundsMax));
    const invScaledBoundsExtent = vec3.sub(invScaledBoundsMax, invScaledBoundsCenter);

    const nodes = mesh.data.nodes;
    const triangles = mesh.data.triangles;
    const vertices = mesh.data.vertices;

    const stack: number[] = [];
    let nodeIndex = 0;
    while (true) {
        const node = nodes[nodeIndex];
        if (
            testBoundsOverlap(
                node.lowerBound,
                node.upperBound,
                invScaledBoundsMin,
                invScaledBoundsMax,
            )
        ) {
            if (node.leaf) {
                const triangleOffset = node.triangleOffset;
                for (let index = 0; index < node.triangleCount; ++index) {
                    const triangleIndex = triangleOffset + index;
                    const triangle = triangles[triangleIndex];
                    const vertex1 = vertices[triangle.index1];
                    const vertex2 = vertices[triangle.index2];
                    const vertex3 = vertices[triangle.index3];
                    if (
                        testBoundsTriangleOverlap(
                            invScaledBoundsCenter,
                            invScaledBoundsExtent,
                            vertex1,
                            vertex2,
                            vertex3,
                        )
                    ) {
                        // Distance in scaled space. Winding order doesn't matter.
                        input.proxyA = {
                            points: [
                                vec3.mul(meshScale, vertex1),
                                vec3.mul(meshScale, vertex2),
                                vec3.mul(meshScale, vertex3),
                            ],
                            count: 3,
                            radius: 0,
                        };
                        cache.count = 0;
                        const output = shapeDistance(input, cache);

                        // distance 0 is deep overlap, dropped (no SAT for movers).
                        if (output.distance !== 0 && output.distance <= mover.radius) {
                            const plane: Plane = {
                                normal: output.normal,
                                offset: f32(mover.radius - output.distance),
                            };
                            planes.push({ plane, point: output.pointA });
                            if (planes.length === capacity) {
                                return planes;
                            }
                        }
                    }
                }
            } else {
                stack.push(nodeIndex + node.childOffset);
                nodeIndex = nodeIndex + 1;
                continue;
            }
        }
        if (stack.length === 0) break;
        nodeIndex = stack.pop() as number;
    }

    return planes;
}
