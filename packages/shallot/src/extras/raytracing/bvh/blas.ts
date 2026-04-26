import type { Vec3, BVHNode, MortonPair, AABB } from "./structs";
import {
    LEAF_FLAG,
    isLeaf,
    leafIndex,
    BLAS_TRIANGLE_SIZE,
    TREE_NODE_STRIDE,
    SHAPE_AABB_STRIDE,
    AABB_SENTINEL,
    MORTON_QUANTIZATION,
    MAX_PROPAGATION_ITERS,
    OCT_ENCODING_SCALE,
} from "./structs";
import { meshRegistry } from "../../../standard/render";
import { MAX_SHAPES, type MeshData } from "../../../standard/render/core";
import { capacity } from "../../../engine";

export interface BLASTriangle {
    v0: Vec3;
    e1: Vec3;
    e2: Vec3;
    n0: Vec3;
    n1: Vec3;
    n2: Vec3;
}

export interface BLASData {
    nodes: BVHNode[];
    sortedTriIds: number[];
    sortedPairs: MortonPair[];
    parents: number[];
    aabbMin: Vec3;
    aabbMax: Vec3;
    triCount: number;
}

export interface BLASAtlas {
    blasData: Map<number, BLASData>;
    triIdsBuffer: GPUBuffer;
    trianglesBuffer: GPUBuffer;
    triangles: Map<number, BLASTriangle[]>;
    nodesBuffer: GPUBuffer;
    parentIndicesBuffer: GPUBuffer;
    baseTrianglesBuffer: GPUBuffer;
    boundsFlagsBuffer: GPUBuffer;
    metas: BLASMeta[];
    entityBlasMetaBuffer: GPUBuffer;
    perEntityShapeAABBs: GPUBuffer;
    shapeDataBuffer: GPUBuffer;
    dynamicInstances: Map<number, number[]>;
}

function vec3(x: number, y: number, z: number): Vec3 {
    return { x, y, z };
}

function vec3Min(a: Vec3, b: Vec3): Vec3 {
    return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), z: Math.min(a.z, b.z) };
}

function vec3Max(a: Vec3, b: Vec3): Vec3 {
    return { x: Math.max(a.x, b.x), y: Math.max(a.y, b.y), z: Math.max(a.z, b.z) };
}

function vec3Add(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function vec3Sub(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function vec3Scale(v: Vec3, s: number): Vec3 {
    return { x: v.x * s, y: v.y * s, z: v.z * s };
}

function octEncode(n: Vec3): number {
    const absSum = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z);
    let vx = n.x / absSum;
    let vy = n.y / absSum;
    const vz = n.z / absSum;
    if (vz < 0) {
        const signX = vx >= 0 ? 1 : -1;
        const signY = vy >= 0 ? 1 : -1;
        const newVx = (1 - Math.abs(vy)) * signX;
        const newVy = (1 - Math.abs(vx)) * signY;
        vx = newVx;
        vy = newVy;
    }
    const x = Math.floor(
        Math.max(0, Math.min(OCT_ENCODING_SCALE, (vx * 0.5 + 0.5) * OCT_ENCODING_SCALE)),
    );
    const y = Math.floor(
        Math.max(0, Math.min(OCT_ENCODING_SCALE, (vy * 0.5 + 0.5) * OCT_ENCODING_SCALE)),
    );
    return ((y << 16) | x) >>> 0;
}

export function extractShapeTriangles(mesh: MeshData): BLASTriangle[] {
    const triangles: BLASTriangle[] = [];
    const { vertices, indices, indexCount } = mesh;
    const stride = 8;

    for (let i = 0; i < indexCount; i += 3) {
        const i0 = indices[i];
        const i1 = indices[i + 1];
        const i2 = indices[i + 2];

        const v0 = vec3(
            vertices[i0 * stride],
            vertices[i0 * stride + 1],
            vertices[i0 * stride + 2],
        );
        const v1 = vec3(
            vertices[i1 * stride],
            vertices[i1 * stride + 1],
            vertices[i1 * stride + 2],
        );
        const v2 = vec3(
            vertices[i2 * stride],
            vertices[i2 * stride + 1],
            vertices[i2 * stride + 2],
        );

        const n0 = vec3(
            vertices[i0 * stride + 3],
            vertices[i0 * stride + 4],
            vertices[i0 * stride + 5],
        );
        const n1 = vec3(
            vertices[i1 * stride + 3],
            vertices[i1 * stride + 4],
            vertices[i1 * stride + 5],
        );
        const n2 = vec3(
            vertices[i2 * stride + 3],
            vertices[i2 * stride + 4],
            vertices[i2 * stride + 5],
        );

        triangles.push({
            v0,
            e1: vec3Sub(v1, v0),
            e2: vec3Sub(v2, v0),
            n0,
            n1,
            n2,
        });
    }

    return triangles;
}

export function packTriangles(tris: BLASTriangle[]): Uint32Array {
    const stride = BLAS_TRIANGLE_SIZE / 4;
    const data = new Uint32Array(tris.length * stride);
    const floats = new Float32Array(data.buffer);
    for (let i = 0; i < tris.length; i++) {
        const t = tris[i];
        const b = i * stride;
        floats[b] = t.v0.x;
        floats[b + 1] = t.v0.y;
        floats[b + 2] = t.v0.z;
        data[b + 3] = 0;
        floats[b + 4] = t.e1.x;
        floats[b + 5] = t.e1.y;
        floats[b + 6] = t.e1.z;
        data[b + 7] = 0;
        floats[b + 8] = t.e2.x;
        floats[b + 9] = t.e2.y;
        floats[b + 10] = t.e2.z;
        data[b + 11] = 0;
        data[b + 12] = octEncode(t.n0);
        data[b + 13] = octEncode(t.n1);
        data[b + 14] = octEncode(t.n2);
        data[b + 15] = 0;
    }
    return data;
}

function computeBounds(triangles: BLASTriangle[]): AABB {
    if (triangles.length === 0) {
        return { min: vec3(0, 0, 0), max: vec3(0, 0, 0) };
    }

    let min = vec3(Infinity, Infinity, Infinity);
    let max = vec3(-Infinity, -Infinity, -Infinity);

    for (const tri of triangles) {
        const v0 = tri.v0;
        const v1 = vec3Add(v0, tri.e1);
        const v2 = vec3Add(v0, tri.e2);
        min = vec3Min(min, v0);
        min = vec3Min(min, v1);
        min = vec3Min(min, v2);
        max = vec3Max(max, v0);
        max = vec3Max(max, v1);
        max = vec3Max(max, v2);
    }

    return { min, max };
}

function expandBits(v: number): number {
    let x = v & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    return x >>> 0;
}

function mortonCode3D(x: number, y: number, z: number): number {
    return ((expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z)) >>> 0;
}

function computeMortonCode(centroid: Vec3, bounds: AABB): number {
    const size = {
        x: bounds.max.x - bounds.min.x,
        y: bounds.max.y - bounds.min.y,
        z: bounds.max.z - bounds.min.z,
    };

    const safeSize = {
        x: Math.max(size.x, 1e-6),
        y: Math.max(size.y, 1e-6),
        z: Math.max(size.z, 1e-6),
    };

    const normalized = {
        x: (centroid.x - bounds.min.x) / safeSize.x,
        y: (centroid.y - bounds.min.y) / safeSize.y,
        z: (centroid.z - bounds.min.z) / safeSize.z,
    };

    const clamped = {
        x: Math.max(0, Math.min(1, normalized.x)),
        y: Math.max(0, Math.min(1, normalized.y)),
        z: Math.max(0, Math.min(1, normalized.z)),
    };

    const quantized = {
        x: Math.floor(clamped.x * MORTON_QUANTIZATION),
        y: Math.floor(clamped.y * MORTON_QUANTIZATION),
        z: Math.floor(clamped.z * MORTON_QUANTIZATION),
    };

    return mortonCode3D(quantized.x, quantized.y, quantized.z);
}

function buildMortonPairs(triangles: BLASTriangle[], bounds: AABB): MortonPair[] {
    return triangles.map((tri, i) => {
        const centroid = vec3Add(tri.v0, vec3Scale(vec3Add(tri.e1, tri.e2), 1 / 3));
        return {
            code: computeMortonCode(centroid, bounds),
            triangleId: i,
        };
    });
}

function radixSort(pairs: MortonPair[]): MortonPair[] {
    const n = pairs.length;
    if (n === 0) return [];

    let input = [...pairs];
    let output = new Array<MortonPair>(n);

    for (let pass = 0; pass < 4; pass++) {
        const bitOffset = pass * 8;
        const histogram = new Array<number>(256).fill(0);

        for (const pair of input) {
            const digit = (pair.code >>> bitOffset) & 0xff;
            histogram[digit]++;
        }

        let sum = 0;
        for (let i = 0; i < 256; i++) {
            const count = histogram[i];
            histogram[i] = sum;
            sum += count;
        }

        for (const pair of input) {
            const digit = (pair.code >>> bitOffset) & 0xff;
            output[histogram[digit]] = pair;
            histogram[digit]++;
        }

        [input, output] = [output, input];
    }

    return input;
}

function clz32(x: number): number {
    if (x === 0) return 32;
    return Math.clz32(x >>> 0);
}

function delta(sortedPairs: MortonPair[], i: number, j: number): number {
    const n = sortedPairs.length;
    if (j < 0 || j >= n) {
        return -1;
    }
    const codeI = sortedPairs[i].code >>> 0;
    const codeJ = sortedPairs[j].code >>> 0;
    if (codeI === codeJ) {
        return clz32((i ^ j) >>> 0) + 32;
    }
    return clz32((codeI ^ codeJ) >>> 0);
}

function determineRange(sortedPairs: MortonPair[], i: number): [number, number] {
    const n = sortedPairs.length;

    if (i === 0) {
        return [0, n - 1];
    }

    const deltaLeft = delta(sortedPairs, i, i - 1);
    const deltaRight = delta(sortedPairs, i, i + 1);
    const d = deltaRight > deltaLeft ? 1 : -1;

    const deltaMin = Math.min(deltaLeft, deltaRight);

    let lmax = 2;
    while (delta(sortedPairs, i, i + lmax * d) > deltaMin) {
        lmax *= 2;
    }

    let l = 0;
    let t = Math.floor(lmax / 2);
    while (t >= 1) {
        if (delta(sortedPairs, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
        t = Math.floor(t / 2);
    }

    const j = i + l * d;
    const first = Math.min(i, j);
    const last = Math.max(i, j);

    return [first, last];
}

function findSplit(sortedPairs: MortonPair[], first: number, last: number): number {
    const deltaNode = delta(sortedPairs, first, last);

    let split = first;
    let stride = last - first;

    do {
        stride = Math.floor((stride + 1) / 2);
        const middle = split + stride;

        if (middle < last) {
            const splitDelta = delta(sortedPairs, first, middle);

            if (splitDelta > deltaNode) {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

function buildKarrasTree(sortedPairs: MortonPair[]): {
    nodes: BVHNode[];
    parents: number[];
} {
    const n = sortedPairs.length;

    if (n === 0) {
        return { nodes: [], parents: [] };
    }

    if (n === 1) {
        return { nodes: [], parents: [-1] };
    }

    const numInternal = n - 1;
    const nodes: BVHNode[] = new Array(numInternal);
    const parents: number[] = new Array(2 * n).fill(-1);

    for (let i = 0; i < numInternal; i++) {
        const [first, last] = determineRange(sortedPairs, i);
        const gamma = findSplit(sortedPairs, first, last);

        const leftIsLeaf = Math.min(first, last) === gamma;
        const rightIsLeaf = Math.max(first, last) === gamma + 1;

        let leftChild: number;
        let rightChild: number;

        if (leftIsLeaf) {
            leftChild = (gamma | LEAF_FLAG) >>> 0;
            parents[gamma] = i;
        } else {
            leftChild = gamma;
            parents[n + gamma] = i;
        }

        if (rightIsLeaf) {
            rightChild = ((gamma + 1) | LEAF_FLAG) >>> 0;
            parents[gamma + 1] = i;
        } else {
            rightChild = gamma + 1;
            parents[n + (gamma + 1)] = i;
        }

        nodes[i] = {
            min: vec3(AABB_SENTINEL, AABB_SENTINEL, AABB_SENTINEL),
            max: vec3(-AABB_SENTINEL, -AABB_SENTINEL, -AABB_SENTINEL),
            leftChild,
            rightChild,
        };
    }

    return { nodes, parents };
}

function getTriangleBounds(tri: BLASTriangle): AABB {
    const v0 = tri.v0;
    const v1 = vec3Add(v0, tri.e1);
    const v2 = vec3Add(v0, tri.e2);
    const min = vec3Min(vec3Min(v0, v1), v2);
    const max = vec3Max(vec3Max(v0, v1), v2);
    return { min, max };
}

function propagateBounds(
    nodes: BVHNode[],
    triangles: BLASTriangle[],
    pairs: MortonPair[],
    parents: number[],
): void {
    const n = triangles.length;
    if (n <= 1) return;

    const boundsFlags = new Array(n - 1).fill(0);

    for (let leafIdx = 0; leafIdx < n; leafIdx++) {
        let current = leafIdx;
        let isLeafNode = true;

        for (let iter = 0; iter < MAX_PROPAGATION_ITERS; iter++) {
            const parent = isLeafNode ? parents[current] : parents[n + current];

            if (parent === -1 || parent === undefined) {
                break;
            }

            const oldFlag = boundsFlags[parent];
            boundsFlags[parent]++;

            if (oldFlag === 0) {
                break;
            }

            const node = nodes[parent];
            const left = node.leftChild;
            const right = node.rightChild;

            let leftBounds: AABB;
            let rightBounds: AABB;

            if (isLeaf(left)) {
                const leftTri = triangles[pairs[leafIndex(left)].triangleId];
                leftBounds = getTriangleBounds(leftTri);
            } else {
                leftBounds = { min: nodes[left].min, max: nodes[left].max };
            }

            if (isLeaf(right)) {
                const rightTri = triangles[pairs[leafIndex(right)].triangleId];
                rightBounds = getTriangleBounds(rightTri);
            } else {
                rightBounds = { min: nodes[right].min, max: nodes[right].max };
            }

            nodes[parent].min = vec3Min(leftBounds.min, rightBounds.min);
            nodes[parent].max = vec3Max(leftBounds.max, rightBounds.max);

            current = parent;
            isLeafNode = false;

            if (parent === 0) {
                break;
            }
        }
    }
}

export function buildShapeBLAS(triangles: BLASTriangle[]): BLASData {
    const triCount = triangles.length;

    if (triCount === 0) {
        return {
            nodes: [],
            sortedTriIds: [],
            sortedPairs: [],
            parents: [],
            aabbMin: vec3(0, 0, 0),
            aabbMax: vec3(0, 0, 0),
            triCount: 0,
        };
    }

    const bounds = computeBounds(triangles);

    if (triCount === 1) {
        return {
            nodes: [],
            sortedTriIds: [0],
            sortedPairs: [{ code: 0, triangleId: 0 }],
            parents: [-1],
            aabbMin: bounds.min,
            aabbMax: bounds.max,
            triCount: 1,
        };
    }

    const pairs = buildMortonPairs(triangles, bounds);
    const sortedPairs = radixSort(pairs);
    const { nodes, parents } = buildKarrasTree(sortedPairs);

    propagateBounds(nodes, triangles, sortedPairs, parents);

    const sortedTriIds = sortedPairs.map((p) => p.triangleId);

    const rootBounds = nodes.length > 0 ? { min: nodes[0].min, max: nodes[0].max } : bounds;

    return {
        nodes,
        sortedTriIds,
        sortedPairs,
        parents,
        aabbMin: rootBounds.min,
        aabbMax: rootBounds.max,
        triCount,
    };
}

function createRefitBuffers(
    device: GPUDevice,
    blasData: Map<number, BLASData>,
    trianglesMap: Map<number, BLASTriangle[]>,
    metas: BLASMeta[],
    totalTreeNodes: number,
    totalParentIndices: number,
    dynInstances: Map<number, number[]>,
): { treeNodesBuffer: GPUBuffer; parentIndicesBuffer: GPUBuffer; boundsFlagsBuffer: GPUBuffer } {
    const treeNodesData = new Uint32Array(
        Math.max(totalTreeNodes * TREE_NODE_STRIDE, TREE_NODE_STRIDE),
    );
    const treeNodesFloat = new Float32Array(treeNodesData.buffer);
    const parentIndicesData = new Uint32Array(Math.max(totalParentIndices, 1));

    let treeNodeWriteOffset = 0;
    let parentWriteOffset = 0;

    for (let shapeId = 0; shapeId < metas.length; shapeId++) {
        const meta = metas[shapeId];
        if (meta.triCount === 0) continue;
        const blas = blasData.get(shapeId);
        const triangles = trianglesMap.get(shapeId);
        if (!blas || !triangles) continue;

        const n = blas.triCount;
        const dynEntities = dynInstances.get(shapeId);
        const instanceCount = dynEntities ? dynEntities.length + 1 : 1;

        if (n <= 1) {
            for (let inst = 0; inst < instanceCount; inst++) {
                treeNodesData.fill(
                    0,
                    treeNodeWriteOffset * TREE_NODE_STRIDE,
                    treeNodeWriteOffset * TREE_NODE_STRIDE + TREE_NODE_STRIDE,
                );
                treeNodeWriteOffset++;
            }
            parentIndicesData[parentWriteOffset++] = 0;
        } else {
            const nodeCount = 2 * n - 1;

            for (let inst = 0; inst < instanceCount; inst++) {
                for (let i = 0; i < n - 1; i++) {
                    const bvhNode = blas.nodes[i];
                    const base = (treeNodeWriteOffset + i) * TREE_NODE_STRIDE;
                    treeNodesFloat[base + 0] = bvhNode.min.x;
                    treeNodesFloat[base + 1] = bvhNode.min.y;
                    treeNodesFloat[base + 2] = bvhNode.min.z;
                    treeNodesData[base + 3] = bvhNode.leftChild >>> 0;
                    treeNodesFloat[base + 4] = bvhNode.max.x;
                    treeNodesFloat[base + 5] = bvhNode.max.y;
                    treeNodesFloat[base + 6] = bvhNode.max.z;
                    treeNodesData[base + 7] = bvhNode.rightChild >>> 0;
                }

                for (let leafIdx = 0; leafIdx < n; leafIdx++) {
                    const leafNodeIdx = n - 1 + leafIdx;
                    const base = (treeNodeWriteOffset + leafNodeIdx) * TREE_NODE_STRIDE;
                    const triId = blas.sortedPairs[leafIdx].triangleId;
                    const tri = triangles[triId];
                    const v0 = tri.v0;
                    const v1 = vec3Add(v0, tri.e1);
                    const v2 = vec3Add(v0, tri.e2);
                    const tMin = vec3Min(vec3Min(v0, v1), v2);
                    const tMax = vec3Max(vec3Max(v0, v1), v2);
                    treeNodesFloat[base + 0] = tMin.x;
                    treeNodesFloat[base + 1] = tMin.y;
                    treeNodesFloat[base + 2] = tMin.z;
                    treeNodesData[base + 3] = 0;
                    treeNodesFloat[base + 4] = tMax.x;
                    treeNodesFloat[base + 5] = tMax.y;
                    treeNodesFloat[base + 6] = tMax.z;
                    treeNodesData[base + 7] = 0;
                }

                treeNodeWriteOffset += nodeCount;
            }

            for (let i = 0; i < 2 * n; i++) {
                parentIndicesData[parentWriteOffset + i] = blas.parents[i] >>> 0;
            }

            parentWriteOffset += 2 * n;
        }
    }

    const treeNodesBuffer = device.createBuffer({
        label: "blas-tree-nodes",
        size: Math.max(treeNodesData.byteLength, 32),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(treeNodesBuffer, 0, treeNodesData);

    const parentIndicesBuffer = device.createBuffer({
        label: "blas-parent-indices",
        size: Math.max(parentIndicesData.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(parentIndicesBuffer, 0, parentIndicesData);

    const boundsFlagsBuffer = device.createBuffer({
        label: "blas-bounds-flags",
        size: Math.max(totalTreeNodes * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return { treeNodesBuffer, parentIndicesBuffer, boundsFlagsBuffer };
}

export interface BLASMeta {
    triIdOffset: number;
    triOffset: number;
    baseTriOffset: number;
    triCount: number;
    treeNodeOffset: number;
    parentOffset: number;
}

export function createBLASAtlas(
    device: GPUDevice,
    getMesh: (id: number) => MeshData | undefined,
    dynamicInstances?: Map<number, number[]>,
): BLASAtlas {
    const dynInstances = dynamicInstances ?? new Map<number, number[]>();
    const shapeCount = meshRegistry.count();
    const blasData = new Map<number, BLASData>();
    const trianglesMap = new Map<number, BLASTriangle[]>();
    const metas: BLASMeta[] = [];

    let totalTriIds = 0;
    let totalTris = 0;
    let totalBaseTris = 0;
    let totalTreeNodes = 0;
    let totalParentIndices = 0;

    const analyticalShapeAABBs: Record<number, { min: Vec3; max: Vec3 }> = {
        0: { min: vec3(-0.5, -0.5, -0.5), max: vec3(0.5, 0.5, 0.5) },
        1: { min: vec3(-0.5, -0.5, -0.5), max: vec3(0.5, 0.5, 0.5) },
        2: { min: vec3(-0.5, -1.0, -0.5), max: vec3(0.5, 1.0, 0.5) },
        3: { min: vec3(-0.5, 0.0, -0.5), max: vec3(0.5, 0.0, 0.5) },
    };

    for (let shapeId = 0; shapeId < shapeCount; shapeId++) {
        const isAnalytical = shapeId < 4 && !dynInstances.has(shapeId);
        const mesh = getMesh(shapeId);
        if (!mesh || mesh.indexCount === 0 || isAnalytical) {
            metas.push({
                triIdOffset: 0,
                triOffset: 0,
                baseTriOffset: 0,
                triCount: 0,
                treeNodeOffset: 0,
                parentOffset: 0,
            });
            continue;
        }

        const triangles = extractShapeTriangles(mesh);
        const blas = buildShapeBLAS(triangles);
        blasData.set(shapeId, blas);
        trianglesMap.set(shapeId, triangles);

        const n = blas.triCount;
        const treeNodeCount = n <= 1 ? 1 : 2 * n - 1;
        const parentCount = n <= 1 ? 1 : 2 * n;
        const dynEntities = dynInstances.get(shapeId);
        const instanceCount = dynEntities ? dynEntities.length + 1 : 1;

        metas.push({
            triIdOffset: totalTriIds,
            triOffset: totalTris,
            baseTriOffset: totalBaseTris,
            triCount: blas.triCount,
            treeNodeOffset: totalTreeNodes,
            parentOffset: totalParentIndices,
        });

        totalTriIds += blas.sortedTriIds.length;
        totalBaseTris += triangles.length;
        totalTris += triangles.length * instanceCount;
        totalTreeNodes += treeNodeCount * instanceCount;
        totalParentIndices += parentCount;
    }

    const triIdsData = new Uint32Array(Math.max(totalTriIds, 1));
    const trianglesData = new Uint32Array(
        Math.max(totalTris * (BLAS_TRIANGLE_SIZE / 4), BLAS_TRIANGLE_SIZE / 4),
    );
    const baseTrianglesData = new Uint32Array(
        Math.max(totalBaseTris * (BLAS_TRIANGLE_SIZE / 4), BLAS_TRIANGLE_SIZE / 4),
    );

    let triIdOffset = 0;
    let baseTriOffset = 0;

    for (let shapeId = 0; shapeId < shapeCount; shapeId++) {
        const blas = blasData.get(shapeId);
        const triangles = trianglesMap.get(shapeId);
        if (!blas || !triangles) continue;

        for (const triId of blas.sortedTriIds) {
            triIdsData[triIdOffset++] = triId;
        }

        const meta = metas[shapeId];
        for (let t = 0; t < triangles.length; t++) {
            const tri = triangles[t];
            const stride = BLAS_TRIANGLE_SIZE / 4;

            const slotBase = meta.triOffset * stride + t * stride;
            const slotFloat = new Float32Array(trianglesData.buffer, slotBase * 4, stride);
            slotFloat[0] = tri.v0.x;
            slotFloat[1] = tri.v0.y;
            slotFloat[2] = tri.v0.z;
            trianglesData[slotBase + 3] = 0;
            slotFloat[4] = tri.e1.x;
            slotFloat[5] = tri.e1.y;
            slotFloat[6] = tri.e1.z;
            trianglesData[slotBase + 7] = 0;
            slotFloat[8] = tri.e2.x;
            slotFloat[9] = tri.e2.y;
            slotFloat[10] = tri.e2.z;
            trianglesData[slotBase + 11] = 0;
            trianglesData[slotBase + 12] = octEncode(tri.n0);
            trianglesData[slotBase + 13] = octEncode(tri.n1);
            trianglesData[slotBase + 14] = octEncode(tri.n2);
            trianglesData[slotBase + 15] = 0;

            const baseBase = baseTriOffset * stride;
            const baseFloat = new Float32Array(baseTrianglesData.buffer, baseBase * 4, stride);
            baseFloat[0] = tri.v0.x;
            baseFloat[1] = tri.v0.y;
            baseFloat[2] = tri.v0.z;
            baseTrianglesData[baseBase + 3] = 0;
            baseFloat[4] = tri.e1.x;
            baseFloat[5] = tri.e1.y;
            baseFloat[6] = tri.e1.z;
            baseTrianglesData[baseBase + 7] = 0;
            baseFloat[8] = tri.e2.x;
            baseFloat[9] = tri.e2.y;
            baseFloat[10] = tri.e2.z;
            baseTrianglesData[baseBase + 11] = 0;
            baseTrianglesData[baseBase + 12] = octEncode(tri.n0);
            baseTrianglesData[baseBase + 13] = octEncode(tri.n1);
            baseTrianglesData[baseBase + 14] = octEncode(tri.n2);
            baseTrianglesData[baseBase + 15] = 0;

            baseTriOffset++;
        }
    }

    const triIdsBuffer = device.createBuffer({
        label: "blas-triIds",
        size: Math.max(triIdsData.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(triIdsBuffer, 0, triIdsData);

    const trianglesBuffer = device.createBuffer({
        label: "blas-triangles",
        size: Math.max(totalTris * BLAS_TRIANGLE_SIZE, BLAS_TRIANGLE_SIZE),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(trianglesBuffer, 0, trianglesData);

    const baseTrianglesBuffer = device.createBuffer({
        label: "blas-base-triangles",
        size: Math.max(totalBaseTris * BLAS_TRIANGLE_SIZE, BLAS_TRIANGLE_SIZE),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(baseTrianglesBuffer, 0, baseTrianglesData);

    const { treeNodesBuffer, parentIndicesBuffer, boundsFlagsBuffer } = createRefitBuffers(
        device,
        blasData,
        trianglesMap,
        metas,
        totalTreeNodes,
        totalParentIndices,
        dynInstances,
    );

    const entityBlasMetaBuffer = device.createBuffer({
        label: "entity-blas-meta",
        size: capacity() * 4 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const perEntityShapeAABBs = device.createBuffer({
        label: "per-entity-shape-aabbs",
        size: capacity() * SHAPE_AABB_STRIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const ShapeDataStride = 16;
    const shapeDataArray = new Float32Array(MAX_SHAPES * ShapeDataStride);
    const shapeDataU32 = new Uint32Array(shapeDataArray.buffer);
    for (let shapeId = 0; shapeId < metas.length; shapeId++) {
        const meta = metas[shapeId];
        const base = shapeId * ShapeDataStride;
        const analytical = analyticalShapeAABBs[shapeId];
        if (meta.triCount === 0) {
            if (analytical) {
                shapeDataArray[base + 0] = analytical.min.x;
                shapeDataArray[base + 1] = analytical.min.y;
                shapeDataArray[base + 2] = analytical.min.z;
                shapeDataArray[base + 4] = analytical.max.x;
                shapeDataArray[base + 5] = analytical.max.y;
                shapeDataArray[base + 6] = analytical.max.z;
            }
            continue;
        }
        const blas = blasData.get(shapeId);
        if (!blas) continue;
        shapeDataArray[base + 0] = blas.aabbMin.x;
        shapeDataArray[base + 1] = blas.aabbMin.y;
        shapeDataArray[base + 2] = blas.aabbMin.z;
        shapeDataArray[base + 4] = blas.aabbMax.x;
        shapeDataArray[base + 5] = blas.aabbMax.y;
        shapeDataArray[base + 6] = blas.aabbMax.z;
        const n = meta.triCount;
        const nodeCount = n <= 1 ? 1 : 2 * n - 1;
        shapeDataU32[base + 8] = meta.treeNodeOffset;
        shapeDataU32[base + 9] = meta.triIdOffset;
        shapeDataU32[base + 10] = meta.triOffset;
        shapeDataU32[base + 11] = n;
        shapeDataU32[base + 12] = nodeCount;
    }
    const shapeDataBuffer = device.createBuffer({
        label: "shape-data",
        size: shapeDataArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(shapeDataBuffer, 0, shapeDataArray);

    return {
        blasData,
        triIdsBuffer,
        trianglesBuffer,
        triangles: trianglesMap,
        nodesBuffer: treeNodesBuffer,
        parentIndicesBuffer,
        baseTrianglesBuffer,
        boundsFlagsBuffer,
        metas,
        entityBlasMetaBuffer,
        perEntityShapeAABBs,
        shapeDataBuffer,
        dynamicInstances: dynInstances,
    };
}
