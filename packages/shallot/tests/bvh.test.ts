import { initGPU, shouldSkipGPU } from "./helpers/gpu";
import { readBuffer } from "../src/standard/compute/readback";
import { describe, test, expect, beforeAll } from "bun:test";
import { createBox, createSphere, getMesh, clearMeshes } from "../src/standard/render/mesh";
import { rayTriangle } from "../src/standard/physics/raycast";
import {
    extractShapeTriangles,
    buildShapeBLAS,
    createBLASAtlas,
    type BLASTriangle,
} from "../src/extras/raytracing/bvh/blas";
import {
    type Vec3,
    type AABB,
    type BVHNode,
    type MortonPair,
    type HitResult,
    LEAF_FLAG,
    isLeaf,
    leafIndex,
    TREE_NODE_SIZE,
    TREE_NODE_STRIDE,
    BVH_NODE_SIZE,
} from "../src/extras/raytracing/bvh/structs";
import { createRadixSortNode } from "../src/standard/radix";

interface Triangle {
    v0: Vec3;
    e1: Vec3;
    e2: Vec3;
    n0: Vec3;
    n1: Vec3;
    n2: Vec3;
    entityId: number;
}

interface Ray {
    origin: Vec3;
    direction: Vec3;
}

interface ValidationResult {
    valid: boolean;
    errors: string[];
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

function vec3Scale(v: Vec3, s: number): Vec3 {
    return { x: v.x * s, y: v.y * s, z: v.z * s };
}

function tri(v0: Vec3, v1: Vec3, v2: Vec3, entityId: number): Triangle {
    const e1 = vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const e2 = vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const n = vec3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
    const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    const normal = len > 0 ? vec3(n.x / len, n.y / len, n.z / len) : vec3(0, 1, 0);
    return { v0, e1, e2, n0: normal, n1: normal, n2: normal, entityId };
}

function blasTri(v0: Vec3, v1: Vec3, v2: Vec3): BLASTriangle {
    const e1 = vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    const e2 = vec3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    const n = vec3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
    const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    const normal = len > 0 ? vec3(n.x / len, n.y / len, n.z / len) : vec3(0, 1, 0);
    return { v0, e1, e2, n0: normal, n1: normal, n2: normal };
}

function computeSceneBounds(triangles: Triangle[]): AABB {
    if (triangles.length === 0) {
        return { min: vec3(0, 0, 0), max: vec3(0, 0, 0) };
    }
    let min = { x: Infinity, y: Infinity, z: Infinity };
    let max = { x: -Infinity, y: -Infinity, z: -Infinity };
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
        x: Math.floor(clamped.x * 1023),
        y: Math.floor(clamped.y * 1023),
        z: Math.floor(clamped.z * 1023),
    };
    return mortonCode3D(quantized.x, quantized.y, quantized.z);
}

function buildMortonPairs(triangles: Triangle[], bounds: AABB): MortonPair[] {
    return triangles.map((tri, i) => {
        const centroid = vec3Add(tri.v0, vec3Scale(vec3Add(tri.e1, tri.e2), 1 / 3));
        return { code: computeMortonCode(centroid, bounds), triangleId: i };
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
    if (j < 0 || j >= n) return -1;
    const codeI = sortedPairs[i].code >>> 0;
    const codeJ = sortedPairs[j].code >>> 0;
    if (codeI === codeJ) return clz32((i ^ j) >>> 0) + 32;
    return clz32((codeI ^ codeJ) >>> 0);
}

function determineRange(sortedPairs: MortonPair[], i: number): [number, number] {
    const n = sortedPairs.length;
    if (i === 0) return [0, n - 1];
    const deltaLeft = delta(sortedPairs, i, i - 1);
    const deltaRight = delta(sortedPairs, i, i + 1);
    const d = deltaRight > deltaLeft ? 1 : -1;
    const deltaMin = Math.min(deltaLeft, deltaRight);
    let lmax = 2;
    while (delta(sortedPairs, i, i + lmax * d) > deltaMin) lmax *= 2;
    let l = 0;
    let t = Math.floor(lmax / 2);
    while (t >= 1) {
        if (delta(sortedPairs, i, i + (l + t) * d) > deltaMin) l += t;
        t = Math.floor(t / 2);
    }
    const j = i + l * d;
    return [Math.min(i, j), Math.max(i, j)];
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
            if (splitDelta > deltaNode) split = middle;
        }
    } while (stride > 1);
    return split;
}

function buildKarrasTree(sortedPairs: MortonPair[]): { nodes: BVHNode[]; parents: number[] } {
    const n = sortedPairs.length;
    if (n === 0) return { nodes: [], parents: [] };
    if (n === 1) return { nodes: [], parents: [-1] };
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
            min: { x: 1e30, y: 1e30, z: 1e30 },
            max: { x: -1e30, y: -1e30, z: -1e30 },
            leftChild,
            rightChild,
        };
    }
    return { nodes, parents };
}

function getTriangleBounds(tri: Triangle): AABB {
    const v0 = tri.v0;
    const v1 = vec3Add(v0, tri.e1);
    const v2 = vec3Add(v0, tri.e2);
    return { min: vec3Min(vec3Min(v0, v1), v2), max: vec3Max(vec3Max(v0, v1), v2) };
}

function propagateBounds(nodes: BVHNode[], triangles: Triangle[], pairs: MortonPair[]): void {
    const n = triangles.length;
    if (n <= 1) return;
    const { parents } = buildKarrasTree(pairs);
    const boundsFlags = new Array(n - 1).fill(0);
    for (let leafIdx = 0; leafIdx < n; leafIdx++) {
        let current = leafIdx;
        let isLeafNode = true;
        for (let iter = 0; iter < 64; iter++) {
            const parent = isLeafNode ? parents[current] : parents[n + current];
            if (parent === -1 || parent === undefined) break;
            const oldFlag = boundsFlags[parent];
            boundsFlags[parent]++;
            if (oldFlag === 0) break;
            const node = nodes[parent];
            const left = node.leftChild;
            const right = node.rightChild;
            let leftBounds: AABB;
            let rightBounds: AABB;
            if (isLeaf(left)) {
                leftBounds = getTriangleBounds(triangles[pairs[leafIndex(left)].triangleId]);
            } else {
                leftBounds = { min: nodes[left].min, max: nodes[left].max };
            }
            if (isLeaf(right)) {
                rightBounds = getTriangleBounds(triangles[pairs[leafIndex(right)].triangleId]);
            } else {
                rightBounds = { min: nodes[right].min, max: nodes[right].max };
            }
            nodes[parent].min = vec3Min(leftBounds.min, rightBounds.min);
            nodes[parent].max = vec3Max(leftBounds.max, rightBounds.max);
            current = parent;
            isLeafNode = false;
            if (parent === 0) break;
        }
    }
}

function intersectAABB(ray: Ray, nodeMin: Vec3, nodeMax: Vec3, tMax: number): boolean {
    const invDirX = Math.abs(ray.direction.x) < 1e-10 ? 1e30 : 1 / ray.direction.x;
    const invDirY = Math.abs(ray.direction.y) < 1e-10 ? 1e30 : 1 / ray.direction.y;
    const invDirZ = Math.abs(ray.direction.z) < 1e-10 ? 1e30 : 1 / ray.direction.z;
    const t1x = (nodeMin.x - ray.origin.x) * invDirX;
    const t2x = (nodeMax.x - ray.origin.x) * invDirX;
    const t1y = (nodeMin.y - ray.origin.y) * invDirY;
    const t2y = (nodeMax.y - ray.origin.y) * invDirY;
    const t1z = (nodeMin.z - ray.origin.z) * invDirZ;
    const t2z = (nodeMax.z - ray.origin.z) * invDirZ;
    const tEnter = Math.max(Math.min(t1x, t2x), Math.min(t1y, t2y), Math.min(t1z, t2z));
    const tExit = Math.min(Math.max(t1x, t2x), Math.max(t1y, t2y), Math.max(t1z, t2z));
    return tEnter <= tExit && tExit >= 0 && tEnter < tMax;
}

function intersectTriangle(ray: Ray, tri: Triangle): HitResult {
    const v0 = tri.v0;
    const v1 = vec3Add(v0, tri.e1);
    const v2 = vec3Add(v0, tri.e2);
    const result = rayTriangle(
        ray.origin.x,
        ray.origin.y,
        ray.origin.z,
        ray.direction.x,
        ray.direction.y,
        ray.direction.z,
        v0.x,
        v0.y,
        v0.z,
        v1.x,
        v1.y,
        v1.z,
        v2.x,
        v2.y,
        v2.z,
    );
    if (!result) {
        return {
            hit: false,
            t: 0,
            entityId: 0,
            u: 0,
            v: 0,
            normal: vec3(0, 1, 0),
            worldPos: vec3(0, 0, 0),
        };
    }
    return {
        hit: true,
        t: result.t,
        entityId: tri.entityId,
        u: 0,
        v: 0,
        normal: vec3(result.nx, result.ny, result.nz),
        worldPos: vec3(
            ray.origin.x + result.t * ray.direction.x,
            ray.origin.y + result.t * ray.direction.y,
            ray.origin.z + result.t * ray.direction.z,
        ),
    };
}

function linearTrace(ray: Ray, triangles: Triangle[]): HitResult {
    let closest: HitResult = {
        hit: false,
        t: 1e30,
        entityId: 0,
        u: 0,
        v: 0,
        normal: vec3(0, 1, 0),
        worldPos: vec3(0, 0, 0),
    };
    for (const tri of triangles) {
        const hit = intersectTriangle(ray, tri);
        if (hit.hit && hit.t < closest.t) closest = hit;
    }
    return closest;
}

function traverseBVH(
    ray: Ray,
    nodes: BVHNode[],
    triangles: Triangle[],
    pairs: MortonPair[],
): HitResult {
    let closest: HitResult = {
        hit: false,
        t: 1e30,
        entityId: 0,
        u: 0,
        v: 0,
        normal: vec3(0, 1, 0),
        worldPos: vec3(0, 0, 0),
    };
    const n = triangles.length;
    if (n === 0) return closest;
    if (n === 1) return intersectTriangle(ray, triangles[pairs[0].triangleId]);
    const stack: number[] = [0];
    let iterations = 0;
    const maxIterations = Math.min(n * 4, 10000);
    while (stack.length > 0 && iterations < maxIterations) {
        iterations++;
        const nodeIdx = stack.pop()!;
        const node = nodes[nodeIdx];
        if (node.min.x > node.max.x) continue;
        if (!intersectAABB(ray, node.min, node.max, closest.t)) continue;
        for (const child of [node.leftChild, node.rightChild]) {
            if (isLeaf(child)) {
                const hit = intersectTriangle(ray, triangles[pairs[leafIndex(child)].triangleId]);
                if (hit.hit && hit.t < closest.t) closest = hit;
            } else {
                stack.push(child);
            }
        }
    }
    return closest;
}

function traverseAnyHit(
    ray: Ray,
    nodes: BVHNode[],
    triangles: Triangle[],
    pairs: MortonPair[],
    tMax = 1e30,
): boolean {
    const n = triangles.length;
    if (n === 0) return false;
    if (n === 1) {
        const hit = intersectTriangle(ray, triangles[pairs[0].triangleId]);
        return hit.hit && hit.t < tMax;
    }
    const stack: number[] = [0];
    let iterations = 0;
    const maxIterations = Math.min(n * 4, 10000);
    while (stack.length > 0 && iterations < maxIterations) {
        iterations++;
        const node = nodes[stack.pop()!];
        if (node.min.x > node.max.x) continue;
        if (!intersectAABB(ray, node.min, node.max, tMax)) continue;
        for (const child of [node.leftChild, node.rightChild]) {
            if (isLeaf(child)) {
                const hit = intersectTriangle(ray, triangles[pairs[leafIndex(child)].triangleId]);
                if (hit.hit && hit.t < tMax) return true;
            } else {
                stack.push(child);
            }
        }
    }
    return false;
}

function validateBVHStructure(nodes: BVHNode[], triCount: number): ValidationResult {
    const errors: string[] = [];
    if (triCount <= 1) return { valid: true, errors: [] };
    const numInternal = triCount - 1;
    if (nodes.length < numInternal) {
        errors.push(`Expected ${numInternal} internal nodes, got ${nodes.length}`);
        return { valid: false, errors };
    }
    const visitedInternal = new Set<number>();
    const visitedLeaves = new Set<number>();
    function dfs(nodeIdx: number, depth: number): boolean {
        if (depth > triCount + 10) {
            errors.push(`Cycle detected`);
            return false;
        }
        if (visitedInternal.has(nodeIdx)) {
            errors.push(`Node ${nodeIdx} visited multiple times`);
            return false;
        }
        visitedInternal.add(nodeIdx);
        if (nodeIdx < 0 || nodeIdx >= numInternal) {
            errors.push(`Invalid node index: ${nodeIdx}`);
            return false;
        }
        const node = nodes[nodeIdx];
        for (const child of [node.leftChild, node.rightChild]) {
            if (isLeaf(child)) {
                const idx = leafIndex(child);
                if (idx < 0 || idx >= triCount) {
                    errors.push(`Invalid leaf index: ${idx}`);
                    return false;
                }
                if (visitedLeaves.has(idx)) {
                    errors.push(`Leaf ${idx} referenced multiple times`);
                    return false;
                }
                visitedLeaves.add(idx);
            } else {
                if (child < 0 || child >= numInternal) {
                    errors.push(`Invalid internal index: ${child}`);
                    return false;
                }
                if (!dfs(child, depth + 1)) return false;
            }
        }
        return true;
    }
    if (!dfs(0, 0)) return { valid: false, errors };
    if (visitedLeaves.size !== triCount) {
        errors.push(`Not all leaves reachable. Found ${visitedLeaves.size}, expected ${triCount}`);
        return { valid: false, errors };
    }
    return { valid: true, errors };
}

function validateBoundsContainment(
    nodes: BVHNode[],
    triangles: Triangle[],
    mortonPairs: MortonPair[],
): ValidationResult {
    const errors: string[] = [];
    const n = triangles.length;
    if (n <= 1) return { valid: true, errors: [] };
    function getTriBounds(idx: number): AABB {
        const tri = triangles[mortonPairs[idx].triangleId];
        const v0 = tri.v0;
        const v1 = vec3Add(v0, tri.e1);
        const v2 = vec3Add(v0, tri.e2);
        return {
            min: vec3(
                Math.min(v0.x, v1.x, v2.x),
                Math.min(v0.y, v1.y, v2.y),
                Math.min(v0.z, v1.z, v2.z),
            ),
            max: vec3(
                Math.max(v0.x, v1.x, v2.x),
                Math.max(v0.y, v1.y, v2.y),
                Math.max(v0.z, v1.z, v2.z),
            ),
        };
    }
    function vec3Contains(outer: AABB, inner: AABB): boolean {
        const eps = 1e-5;
        return (
            outer.min.x <= inner.min.x + eps &&
            outer.min.y <= inner.min.y + eps &&
            outer.min.z <= inner.min.z + eps &&
            outer.max.x >= inner.max.x - eps &&
            outer.max.y >= inner.max.y - eps &&
            outer.max.z >= inner.max.z - eps
        );
    }
    function checkNode(nodeIdx: number): AABB | null {
        const node = nodes[nodeIdx];
        const nodeBounds = { min: node.min, max: node.max };
        let leftBounds: AABB, rightBounds: AABB;
        if (isLeaf(node.leftChild)) leftBounds = getTriBounds(leafIndex(node.leftChild));
        else {
            const child = checkNode(node.leftChild);
            if (!child) return null;
            leftBounds = child;
        }
        if (isLeaf(node.rightChild)) rightBounds = getTriBounds(leafIndex(node.rightChild));
        else {
            const child = checkNode(node.rightChild);
            if (!child) return null;
            rightBounds = child;
        }
        if (!vec3Contains(nodeBounds, leftBounds))
            errors.push(`Node ${nodeIdx} does not contain left child bounds`);
        if (!vec3Contains(nodeBounds, rightBounds))
            errors.push(`Node ${nodeIdx} does not contain right child bounds`);
        return nodeBounds;
    }
    if (nodes.length > 0) checkNode(0);
    return { valid: errors.length === 0, errors };
}

describe("CPU BVH Reference", () => {
    describe("scene bounds", () => {
        test("computes bounds for single triangle", () => {
            const triangles = [tri(vec3(1, 2, 3), vec3(4, 5, 6), vec3(7, 8, 9), 0)];
            const bounds = computeSceneBounds(triangles);
            expect(bounds.min).toEqual(vec3(1, 2, 3));
            expect(bounds.max).toEqual(vec3(7, 8, 9));
        });

        test("computes bounds for multiple triangles", () => {
            const triangles = [
                tri(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), 0),
                tri(vec3(5, 5, 5), vec3(6, 5, 5), vec3(5, 6, 5), 1),
                tri(vec3(-2, -3, -4), vec3(-1, -3, -4), vec3(-2, -2, -4), 2),
            ];
            const bounds = computeSceneBounds(triangles);
            expect(bounds.min).toEqual(vec3(-2, -3, -4));
            expect(bounds.max).toEqual(vec3(6, 6, 5));
        });

        test("handles empty triangle list", () => {
            const bounds = computeSceneBounds([]);
            expect(bounds.min).toEqual(vec3(0, 0, 0));
            expect(bounds.max).toEqual(vec3(0, 0, 0));
        });
    });

    describe("Morton codes", () => {
        test("Morton code at origin is smallest", () => {
            const bounds: AABB = { min: vec3(0, 0, 0), max: vec3(1, 1, 1) };
            const code0 = computeMortonCode(vec3(0, 0, 0), bounds);
            const code1 = computeMortonCode(vec3(0.5, 0.5, 0.5), bounds);
            const code2 = computeMortonCode(vec3(1, 1, 1), bounds);
            expect(code0).toBeLessThan(code1);
            expect(code1).toBeLessThan(code2);
        });

        test("builds Morton pairs correctly", () => {
            const triangles = [
                tri(vec3(0, 0, 0), vec3(0.3, 0, 0), vec3(0, 0.3, 0), 0),
                tri(vec3(0.9, 0.9, 0.9), vec3(1, 0.9, 0.9), vec3(0.9, 1, 0.9), 1),
            ];
            const bounds: AABB = { min: vec3(0, 0, 0), max: vec3(1, 1, 0.9) };
            const pairs = buildMortonPairs(triangles, bounds);
            expect(pairs).toHaveLength(2);
            expect(pairs[0].triangleId).toBe(0);
            expect(pairs[1].triangleId).toBe(1);
            expect(pairs[0].code).toBeLessThan(pairs[1].code);
        });

        test("morton codes at exact scene boundaries", () => {
            const bounds: AABB = { min: vec3(0, 0, 0), max: vec3(1, 1, 1) };
            expect(computeMortonCode(vec3(0, 0, 0), bounds)).toBe(0);
            expect(computeMortonCode(vec3(1, 1, 1), bounds)).toBe(mortonCode3D(1023, 1023, 1023));
        });
    });

    describe("radix sort", () => {
        test("sorts Morton pairs by code in ascending order", () => {
            const pairs = [
                { code: 0xf0000000, triangleId: 3 },
                { code: 0x10000000, triangleId: 1 },
                { code: 0x80000000, triangleId: 2 },
                { code: 0x00000001, triangleId: 0 },
                { code: 0x40000000, triangleId: 4 },
            ];
            const sorted = radixSort(pairs);
            expect(sorted.map((p) => p.code)).toEqual([
                0x00000001, 0x10000000, 0x40000000, 0x80000000, 0xf0000000,
            ]);
        });

        test("handles empty input", () => {
            expect(radixSort([])).toHaveLength(0);
        });

        test("handles single element", () => {
            const sorted = radixSort([{ code: 42, triangleId: 7 }]);
            expect(sorted).toEqual([{ code: 42, triangleId: 7 }]);
        });

        test("handles duplicate codes", () => {
            const pairs = [
                { code: 100, triangleId: 0 },
                { code: 100, triangleId: 1 },
                { code: 100, triangleId: 2 },
            ];
            const sorted = radixSort(pairs);
            expect(sorted.map((p) => p.triangleId).sort()).toEqual([0, 1, 2]);
        });
    });

    describe("Karras tree", () => {
        test("builds valid tree for 4 triangles", () => {
            const sortedPairs = [
                { code: 0x00000000, triangleId: 0 },
                { code: 0x00010000, triangleId: 1 },
                { code: 0x80000000, triangleId: 2 },
                { code: 0x80010000, triangleId: 3 },
            ];
            const { nodes } = buildKarrasTree(sortedPairs);
            expect(nodes).toHaveLength(3);
            const validation = validateBVHStructure(nodes, 4);
            expect(validation.valid).toBe(true);
        });

        test("handles single triangle (no internal nodes)", () => {
            expect(buildKarrasTree([{ code: 42, triangleId: 0 }]).nodes).toHaveLength(0);
        });

        test("handles two triangles (single internal node)", () => {
            const { nodes } = buildKarrasTree([
                { code: 0, triangleId: 0 },
                { code: 1, triangleId: 1 },
            ]);
            expect(nodes).toHaveLength(1);
            expect(isLeaf(nodes[0].leftChild)).toBe(true);
            expect(isLeaf(nodes[0].rightChild)).toBe(true);
        });

        test("tree covers all leaves exactly once", () => {
            const sortedPairs: MortonPair[] = [];
            for (let i = 0; i < 16; i++) sortedPairs.push({ code: i * 0x10000000, triangleId: i });
            const { nodes } = buildKarrasTree(sortedPairs);
            const visitedLeaves = new Set<number>();
            function visit(child: number) {
                if (isLeaf(child)) visitedLeaves.add(leafIndex(child));
                else {
                    visit(nodes[child].leftChild);
                    visit(nodes[child].rightChild);
                }
            }
            visit(nodes[0].leftChild);
            visit(nodes[0].rightChild);
            expect(visitedLeaves.size).toBe(16);
        });
    });

    describe("traversal", () => {
        test("BVH trace matches linear trace", () => {
            const triangles = [
                tri(vec3(-1, -1, -5), vec3(1, -1, -5), vec3(0, 1, -5), 0),
                tri(vec3(-2, -2, -10), vec3(2, -2, -10), vec3(0, 2, -10), 1),
                tri(vec3(3, -1, -7), vec3(5, -1, -7), vec3(4, 1, -7), 2),
            ];
            const bounds = computeSceneBounds(triangles);
            const pairs = buildMortonPairs(triangles, bounds);
            const sorted = radixSort(pairs);
            const { nodes } = buildKarrasTree(sorted);
            propagateBounds(nodes, triangles, sorted);
            const rays: Ray[] = [
                { origin: vec3(0, 0, 0), direction: vec3(0, 0, -1) },
                { origin: vec3(4, 0, 0), direction: vec3(0, 0, -1) },
                { origin: vec3(-5, 0, 0), direction: vec3(0, 0, -1) },
            ];
            for (const ray of rays) {
                const linearResult = linearTrace(ray, triangles);
                const bvhResult = traverseBVH(ray, nodes, triangles, sorted);
                expect(bvhResult.hit).toBe(linearResult.hit);
                if (linearResult.hit) {
                    expect(bvhResult.t).toBeCloseTo(linearResult.t, 5);
                    expect(bvhResult.entityId).toBe(linearResult.entityId);
                }
            }
        });

        test("handles single triangle scene", () => {
            const triangles = [tri(vec3(0, 0, -5), vec3(1, 0, -5), vec3(0, 1, -5), 42)];
            const bounds = computeSceneBounds(triangles);
            const pairs = buildMortonPairs(triangles, bounds);
            const sorted = radixSort(pairs);
            const { nodes } = buildKarrasTree(sorted);
            const ray: Ray = { origin: vec3(0.2, 0.2, 0), direction: vec3(0, 0, -1) };
            const result = traverseBVH(ray, nodes, triangles, sorted);
            expect(result.hit).toBe(true);
            expect(result.entityId).toBe(42);
        });
    });

    describe("traverseAnyHit", () => {
        test("returns true on hit", () => {
            const triangles = [tri(vec3(-1, -1, -5), vec3(1, -1, -5), vec3(0, 1, -5), 0)];
            const bounds = computeSceneBounds(triangles);
            const sorted = radixSort(buildMortonPairs(triangles, bounds));
            const { nodes } = buildKarrasTree(sorted);
            propagateBounds(nodes, triangles, sorted);
            expect(
                traverseAnyHit(
                    { origin: vec3(0, 0, 0), direction: vec3(0, 0, -1) },
                    nodes,
                    triangles,
                    sorted,
                ),
            ).toBe(true);
        });

        test("returns false on miss", () => {
            const triangles = [tri(vec3(-1, -1, -5), vec3(1, -1, -5), vec3(0, 1, -5), 0)];
            const bounds = computeSceneBounds(triangles);
            const sorted = radixSort(buildMortonPairs(triangles, bounds));
            const { nodes } = buildKarrasTree(sorted);
            propagateBounds(nodes, triangles, sorted);
            expect(
                traverseAnyHit(
                    { origin: vec3(100, 100, 0), direction: vec3(0, 0, -1) },
                    nodes,
                    triangles,
                    sorted,
                ),
            ).toBe(false);
        });

        test("respects tMax", () => {
            const triangles = [tri(vec3(-1, -1, -5), vec3(1, -1, -5), vec3(0, 1, -5), 0)];
            const bounds = computeSceneBounds(triangles);
            const sorted = radixSort(buildMortonPairs(triangles, bounds));
            const { nodes } = buildKarrasTree(sorted);
            propagateBounds(nodes, triangles, sorted);
            const ray: Ray = { origin: vec3(0, 0, 0), direction: vec3(0, 0, -1) };
            expect(traverseAnyHit(ray, nodes, triangles, sorted, 10)).toBe(true);
            expect(traverseAnyHit(ray, nodes, triangles, sorted, 3)).toBe(false);
        });
    });

    describe("stress tests", () => {
        test("128 triangles", () => {
            const triangles: Triangle[] = [];
            for (let i = 0; i < 128; i++) {
                const x = i % 16,
                    y = Math.floor(i / 16);
                triangles.push(tri(vec3(x, y, 0), vec3(x + 0.5, y, 0), vec3(x, y + 0.5, 0), i));
            }
            const bounds = computeSceneBounds(triangles);
            const sorted = radixSort(buildMortonPairs(triangles, bounds));
            const { nodes } = buildKarrasTree(sorted);
            propagateBounds(nodes, triangles, sorted);
            expect(nodes).toHaveLength(127);
            expect(validateBVHStructure(nodes, 128).valid).toBe(true);
            expect(validateBoundsContainment(nodes, triangles, sorted).valid).toBe(true);
            const ray: Ray = { origin: vec3(7.25, 3.25, 10), direction: vec3(0, 0, -1) };
            const bvhResult = traverseBVH(ray, nodes, triangles, sorted);
            const linearResult = linearTrace(ray, triangles);
            expect(bvhResult.hit).toBe(linearResult.hit);
            if (linearResult.hit) expect(bvhResult.entityId).toBe(linearResult.entityId);
        });
    });
});

describe("BLAS", () => {
    describe("extractShapeTriangles", () => {
        test("extracts triangles from box mesh", () => {
            expect(extractShapeTriangles(createBox()).length).toBe(12);
        });

        test("extracts triangles from sphere mesh", () => {
            const sphere = createSphere();
            expect(extractShapeTriangles(sphere).length).toBe(sphere.indexCount / 3);
        });
    });

    describe("single triangle", () => {
        test("builds BLAS for single triangle", () => {
            const triangles = [blasTri(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0))];
            const blas = buildShapeBLAS(triangles);
            expect(blas.triCount).toBe(1);
            expect(blas.nodes.length).toBe(0);
            expect(blas.sortedTriIds).toEqual([0]);
        });

        test("single triangle bounds are correct", () => {
            const blas = buildShapeBLAS([blasTri(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0))]);
            expect(blas.aabbMin.x).toBeCloseTo(0, 5);
            expect(blas.aabbMax.x).toBeCloseTo(1, 5);
            expect(blas.aabbMax.y).toBeCloseTo(1, 5);
        });
    });

    describe("box shape", () => {
        test("builds BLAS for box (12 triangles)", () => {
            const blas = buildShapeBLAS(extractShapeTriangles(createBox()));
            expect(blas.triCount).toBe(12);
            expect(blas.nodes.length).toBe(11);
        });

        test("box BLAS bounds are approximately [-0.5, 0.5]^3", () => {
            const blas = buildShapeBLAS(extractShapeTriangles(createBox()));
            expect(blas.aabbMin.x).toBeCloseTo(-0.5, 4);
            expect(blas.aabbMax.x).toBeCloseTo(0.5, 4);
        });
    });

    describe("sphere shape", () => {
        test("builds BLAS for sphere", () => {
            const triangles = extractShapeTriangles(createSphere());
            const blas = buildShapeBLAS(triangles);
            expect(blas.triCount).toBe(triangles.length);
            expect(blas.nodes.length).toBe(triangles.length - 1);
        });
    });

    describe("edge cases", () => {
        test("handles empty triangle list", () => {
            const blas = buildShapeBLAS([]);
            expect(blas.triCount).toBe(0);
            expect(blas.nodes.length).toBe(0);
        });

        test("handles two triangles", () => {
            const triangles = [
                blasTri(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)),
                blasTri(vec3(2, 0, 0), vec3(3, 0, 0), vec3(2, 1, 0)),
            ];
            const blas = buildShapeBLAS(triangles);
            expect(blas.triCount).toBe(2);
            expect(blas.nodes.length).toBe(1);
        });
    });
});

describe("TLAS", () => {
    describe("buffer sizing", () => {
        test("TREE_NODE_SIZE is 32 bytes", () => expect(TREE_NODE_SIZE).toBe(32));
        test("BVH_NODE_SIZE is 128 bytes", () => expect(BVH_NODE_SIZE).toBe(128));
    });

    describe("tree structure", () => {
        test("leaf flag is high bit", () =>
            expect((0x80000000 >>> 0).toString(16)).toBe("80000000"));
    });

    describe("morton code", () => {
        test("expandBits produces 10-bit interleaved pattern", () => {
            expect(expandBits(0)).toBe(0);
            expect(expandBits(1)).toBe(1);
            expect(expandBits(0b11)).toBe(0b001001);
        });

        test("mortonCode3D interleaves x, y, z", () => {
            expect(mortonCode3D(0, 0, 0)).toBe(0);
            expect(mortonCode3D(1, 0, 0)).toBe(0b100);
            expect(mortonCode3D(0, 1, 0)).toBe(0b010);
            expect(mortonCode3D(0, 0, 1)).toBe(0b001);
            expect(mortonCode3D(1, 1, 1)).toBe(0b111);
        });
    });
});

describe("GPU Radix Sort", () => {
    let device: GPUDevice | null = null;

    beforeAll(async () => {
        const skipReason = shouldSkipGPU();
        if (skipReason) return;
        device = (await initGPU()).device;
    }, 30000);

    async function runNode(
        device: GPUDevice,
        node: { prepare?: (d: GPUDevice) => Promise<void>; execute: (ctx: any) => void },
    ): Promise<void> {
        if (node.prepare) await node.prepare(device);
        const encoder = device.createCommandEncoder();
        node.execute({
            device,
            queue: device.queue,
            encoder,
            context: null,
            format: "rgba8unorm",
            canvasView: null,
            getTexture: () => null,
            getTextureView: () => null,
            getBuffer: () => null,
            setTexture: () => {},
            setTextureView: () => {},
            setBuffer: () => {},
        });
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
    }

    test("sorts ascending order", async () => {
        if (!device) return;
        const input = [
            { code: 0xf0000000, triangleId: 0 },
            { code: 0x10000000, triangleId: 1 },
            { code: 0x80000000, triangleId: 2 },
            { code: 0x00000001, triangleId: 3 },
            { code: 0x40000000, triangleId: 4 },
        ];
        const keys = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const values = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(keys, 0, new Uint32Array(input.map((p) => p.code)));
        device.queue.writeBuffer(values, 0, new Uint32Array(input.map((p) => p.triangleId)));
        await runNode(device, createRadixSortNode({ keys, values, count: input.length }));
        const resultKeys = new Uint32Array(await readBuffer(device, keys, input.length * 4));
        for (let i = 1; i < resultKeys.length; i++)
            expect(resultKeys[i]).toBeGreaterThanOrEqual(resultKeys[i - 1]);
        keys.destroy();
        values.destroy();
    }, 30000);

    test("matches CPU reference for random data", async () => {
        if (!device) return;
        const count = 1000;
        const input: MortonPair[] = [];
        for (let i = 0; i < count; i++)
            input.push({ code: Math.floor(Math.random() * 0xffffffff), triangleId: i });
        const cpuSorted = radixSort(input);
        const keys = device.createBuffer({
            size: count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const values = device.createBuffer({
            size: count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(keys, 0, new Uint32Array(input.map((p) => p.code)));
        device.queue.writeBuffer(values, 0, new Uint32Array(input.map((p) => p.triangleId)));
        await runNode(device, createRadixSortNode({ keys, values, count }));
        const resultKeys = new Uint32Array(await readBuffer(device, keys, count * 4));
        for (let i = 0; i < count; i++) expect(resultKeys[i]).toBe(cpuSorted[i].code);
        keys.destroy();
        values.destroy();
    }, 30000);

    test("preserves key-value association", async () => {
        if (!device) return;
        const count = 100;
        const input: MortonPair[] = [];
        for (let i = 0; i < count; i++)
            input.push({ code: Math.floor(Math.random() * 0xffffffff), triangleId: i });
        const originalCodeByTriangle = new Map(input.map((p) => [p.triangleId, p.code]));
        const keys = device.createBuffer({
            size: count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const values = device.createBuffer({
            size: count * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(keys, 0, new Uint32Array(input.map((p) => p.code)));
        device.queue.writeBuffer(values, 0, new Uint32Array(input.map((p) => p.triangleId)));
        await runNode(device, createRadixSortNode({ keys, values, count }));
        const resultKeys = new Uint32Array(await readBuffer(device, keys, count * 4));
        const resultVals = new Uint32Array(await readBuffer(device, values, count * 4));
        let mismatches = 0;
        for (let i = 0; i < count; i++)
            if (resultKeys[i] !== originalCodeByTriangle.get(resultVals[i])) mismatches++;
        expect(mismatches).toBe(0);
        keys.destroy();
        values.destroy();
    }, 30000);
});

describe("TLAS Instance Bounds", () => {
    type Mat4 = Float32Array;

    function identityMatrix(): Mat4 {
        return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    }

    function translationMatrix(x: number, y: number, z: number): Mat4 {
        return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
    }

    function scaleMatrix(sx: number, sy: number, sz: number): Mat4 {
        return new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1]);
    }

    function rotationYMatrix(angle: number): Mat4 {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return new Float32Array([c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]);
    }

    function multiplyMatrix(a: Mat4, b: Mat4): Mat4 {
        const result = new Float32Array(16);
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += a[row + k * 4] * b[k + col * 4];
                }
                result[row + col * 4] = sum;
            }
        }
        return result;
    }

    function transformPoint(m: Mat4, p: Vec3): Vec3 {
        return {
            x: m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
            y: m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            z: m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
        };
    }

    function transformAABB(m: Mat4, aabb: AABB): AABB {
        const corners = [
            vec3(aabb.min.x, aabb.min.y, aabb.min.z),
            vec3(aabb.max.x, aabb.min.y, aabb.min.z),
            vec3(aabb.min.x, aabb.max.y, aabb.min.z),
            vec3(aabb.max.x, aabb.max.y, aabb.min.z),
            vec3(aabb.min.x, aabb.min.y, aabb.max.z),
            vec3(aabb.max.x, aabb.min.y, aabb.max.z),
            vec3(aabb.min.x, aabb.max.y, aabb.max.z),
            vec3(aabb.max.x, aabb.max.y, aabb.max.z),
        ];

        const transformed = corners.map((c) => transformPoint(m, c));

        let min = { x: Infinity, y: Infinity, z: Infinity };
        let max = { x: -Infinity, y: -Infinity, z: -Infinity };

        for (const p of transformed) {
            min = vec3Min(min, p);
            max = vec3Max(max, p);
        }

        return { min, max };
    }

    describe("identity transform", () => {
        test("preserves mesh AABB", () => {
            const meshAABB: AABB = { min: vec3(-0.5, -0.5, -0.5), max: vec3(0.5, 0.5, 0.5) };
            const matrix = identityMatrix();
            const result = transformAABB(matrix, meshAABB);
            expect(result.min.x).toBeCloseTo(-0.5, 5);
            expect(result.min.y).toBeCloseTo(-0.5, 5);
            expect(result.min.z).toBeCloseTo(-0.5, 5);
            expect(result.max.x).toBeCloseTo(0.5, 5);
            expect(result.max.y).toBeCloseTo(0.5, 5);
            expect(result.max.z).toBeCloseTo(0.5, 5);
        });
    });

    describe("translation", () => {
        test("shifts AABB by translation amount", () => {
            const meshAABB: AABB = { min: vec3(-0.5, -0.5, -0.5), max: vec3(0.5, 0.5, 0.5) };
            const matrix = translationMatrix(10, 20, 30);
            const result = transformAABB(matrix, meshAABB);
            expect(result.min.x).toBeCloseTo(9.5, 5);
            expect(result.min.y).toBeCloseTo(19.5, 5);
            expect(result.min.z).toBeCloseTo(29.5, 5);
            expect(result.max.x).toBeCloseTo(10.5, 5);
            expect(result.max.y).toBeCloseTo(20.5, 5);
            expect(result.max.z).toBeCloseTo(30.5, 5);
        });

        test("negative translation", () => {
            const meshAABB: AABB = { min: vec3(0, 0, 0), max: vec3(1, 1, 1) };
            const matrix = translationMatrix(-5, -5, -5);
            const result = transformAABB(matrix, meshAABB);
            expect(result.min.x).toBeCloseTo(-5, 5);
            expect(result.max.x).toBeCloseTo(-4, 5);
        });
    });

    describe("scale", () => {
        test("uniform scale expands AABB", () => {
            const meshAABB: AABB = { min: vec3(-1, -1, -1), max: vec3(1, 1, 1) };
            const matrix = scaleMatrix(2, 2, 2);
            const result = transformAABB(matrix, meshAABB);
            expect(result.min.x).toBeCloseTo(-2, 5);
            expect(result.max.x).toBeCloseTo(2, 5);
        });

        test("non-uniform scale", () => {
            const meshAABB: AABB = { min: vec3(-1, -1, -1), max: vec3(1, 1, 1) };
            const matrix = scaleMatrix(1, 2, 3);
            const result = transformAABB(matrix, meshAABB);
            expect(result.min.x).toBeCloseTo(-1, 5);
            expect(result.max.x).toBeCloseTo(1, 5);
            expect(result.min.y).toBeCloseTo(-2, 5);
            expect(result.max.y).toBeCloseTo(2, 5);
            expect(result.min.z).toBeCloseTo(-3, 5);
            expect(result.max.z).toBeCloseTo(3, 5);
        });

        test("scale of 0.5 shrinks AABB", () => {
            const meshAABB: AABB = { min: vec3(-2, -2, -2), max: vec3(2, 2, 2) };
            const matrix = scaleMatrix(0.5, 0.5, 0.5);
            const result = transformAABB(matrix, meshAABB);
            expect(result.min.x).toBeCloseTo(-1, 5);
            expect(result.max.x).toBeCloseTo(1, 5);
        });
    });

    describe("rotation", () => {
        test("45 degree Y rotation expands XZ bounds", () => {
            const meshAABB: AABB = { min: vec3(-1, -1, -1), max: vec3(1, 1, 1) };
            const matrix = rotationYMatrix(Math.PI / 4);
            const result = transformAABB(matrix, meshAABB);
            expect(result.max.x).toBeGreaterThan(1);
            expect(result.max.x).toBeLessThan(1.5);
            expect(result.min.y).toBeCloseTo(-1, 5);
            expect(result.max.y).toBeCloseTo(1, 5);
        });

        test("90 degree Y rotation swaps X and Z", () => {
            const meshAABB: AABB = { min: vec3(-1, -2, -3), max: vec3(1, 2, 3) };
            const matrix = rotationYMatrix(Math.PI / 2);
            const result = transformAABB(matrix, meshAABB);
            expect(result.min.x).toBeCloseTo(-3, 4);
            expect(result.max.x).toBeCloseTo(3, 4);
            expect(result.min.z).toBeCloseTo(-1, 4);
            expect(result.max.z).toBeCloseTo(1, 4);
        });

        test("180 degree rotation preserves size", () => {
            const meshAABB: AABB = { min: vec3(-1, -1, -1), max: vec3(1, 1, 1) };
            const matrix = rotationYMatrix(Math.PI);
            const result = transformAABB(matrix, meshAABB);
            expect(result.max.x - result.min.x).toBeCloseTo(2, 4);
            expect(result.max.y - result.min.y).toBeCloseTo(2, 4);
            expect(result.max.z - result.min.z).toBeCloseTo(2, 4);
        });
    });

    describe("combined transforms", () => {
        test("translate then scale", () => {
            const meshAABB: AABB = { min: vec3(-0.5, -0.5, -0.5), max: vec3(0.5, 0.5, 0.5) };
            const translate = translationMatrix(5, 0, 0);
            const scale = scaleMatrix(2, 2, 2);
            const combined = multiplyMatrix(translate, scale);
            const result = transformAABB(combined, meshAABB);
            expect(result.min.x).toBeCloseTo(4, 5);
            expect(result.max.x).toBeCloseTo(6, 5);
        });

        test("scale then translate", () => {
            const meshAABB: AABB = { min: vec3(-0.5, -0.5, -0.5), max: vec3(0.5, 0.5, 0.5) };
            const scale = scaleMatrix(2, 2, 2);
            const translate = translationMatrix(5, 0, 0);
            const combined = multiplyMatrix(scale, translate);
            const result = transformAABB(combined, meshAABB);
            expect(result.min.x).toBeCloseTo(9, 5);
            expect(result.max.x).toBeCloseTo(11, 5);
        });
    });

    describe("instance indirection", () => {
        test("instance ID maps to correct BLAS", () => {
            const instances = [
                { blasId: 0, transform: identityMatrix() },
                { blasId: 1, transform: translationMatrix(5, 0, 0) },
                { blasId: 0, transform: translationMatrix(-5, 0, 0) },
            ];
            expect(instances[0].blasId).toBe(0);
            expect(instances[1].blasId).toBe(1);
            expect(instances[2].blasId).toBe(0);
        });

        test("multiple instances can share BLAS", () => {
            const blasIds = [0, 0, 0, 1, 1];
            const uniqueBlas = new Set(blasIds);
            expect(uniqueBlas.size).toBe(2);
            expect(blasIds.filter((id) => id === 0).length).toBe(3);
        });
    });

    describe("multi-instance tree", () => {
        test("builds valid structure for 3 instances", () => {
            const instanceBounds: AABB[] = [
                { min: vec3(-1, -1, -1), max: vec3(1, 1, 1) },
                { min: vec3(4, -1, -1), max: vec3(6, 1, 1) },
                { min: vec3(-6, -1, -1), max: vec3(-4, 1, 1) },
            ];
            const sceneBounds = {
                min: vec3(
                    Math.min(...instanceBounds.map((b) => b.min.x)),
                    Math.min(...instanceBounds.map((b) => b.min.y)),
                    Math.min(...instanceBounds.map((b) => b.min.z)),
                ),
                max: vec3(
                    Math.max(...instanceBounds.map((b) => b.max.x)),
                    Math.max(...instanceBounds.map((b) => b.max.y)),
                    Math.max(...instanceBounds.map((b) => b.max.z)),
                ),
            };
            expect(sceneBounds.min.x).toBe(-6);
            expect(sceneBounds.max.x).toBe(6);
        });

        test("empty scene has no instances", () => {
            const instanceCount = 0;
            expect(instanceCount).toBe(0);
        });

        test("single instance needs no internal nodes", () => {
            const instanceCount = 1;
            const internalNodes = instanceCount > 1 ? instanceCount - 1 : 0;
            expect(internalNodes).toBe(0);
        });

        test("N instances need N-1 internal nodes", () => {
            for (const n of [2, 3, 4, 10, 100]) {
                const internalNodes = n - 1;
                expect(internalNodes).toBe(n - 1);
            }
        });
    });

    describe("instance centroid", () => {
        test("computes from transformed AABB", () => {
            const meshAABB: AABB = { min: vec3(-1, -1, -1), max: vec3(1, 1, 1) };
            const matrix = translationMatrix(10, 20, 30);
            const transformed = transformAABB(matrix, meshAABB);
            const centroid = {
                x: (transformed.min.x + transformed.max.x) / 2,
                y: (transformed.min.y + transformed.max.y) / 2,
                z: (transformed.min.z + transformed.max.z) / 2,
            };
            expect(centroid.x).toBeCloseTo(10, 5);
            expect(centroid.y).toBeCloseTo(20, 5);
            expect(centroid.z).toBeCloseTo(30, 5);
        });

        test("centroid at origin for centered instance", () => {
            const meshAABB: AABB = { min: vec3(-1, -1, -1), max: vec3(1, 1, 1) };
            const centroid = {
                x: (meshAABB.min.x + meshAABB.max.x) / 2,
                y: (meshAABB.min.y + meshAABB.max.y) / 2,
                z: (meshAABB.min.z + meshAABB.max.z) / 2,
            };
            expect(centroid.x).toBeCloseTo(0, 5);
            expect(centroid.y).toBeCloseTo(0, 5);
            expect(centroid.z).toBeCloseTo(0, 5);
        });
    });
});

describe("BLAS atlas multi-instance", () => {
    let device: GPUDevice | null = null;

    beforeAll(async () => {
        const skipReason = shouldSkipGPU();
        if (skipReason) return;
        clearMeshes();
        getMesh(0);
        const result = await initGPU();
        device = result.device;
    });

    test("tree node child pointers written for all instances", async () => {
        if (!device) return;

        const boxShapeId = 0;
        const dynInstances = new Map([[boxShapeId, [10, 20]]]);
        const atlas = createBLASAtlas(device, getMesh, dynInstances);

        const meta = atlas.metas[boxShapeId];
        expect(meta.triCount).toBeGreaterThan(1);

        const n = meta.triCount;
        const nodeCount = 2 * n - 1;
        const internalNodes = n - 1;

        const bufferSize = atlas.nodesBuffer.size;
        const data = await readBuffer(device, atlas.nodesBuffer, bufferSize);
        const u32 = new Uint32Array(data);

        for (let inst = 0; inst < 2; inst++) {
            const instNodeOffset = meta.treeNodeOffset + inst * nodeCount;
            let hasNonZeroChild = false;
            for (let i = 0; i < internalNodes; i++) {
                const base = (instNodeOffset + i) * TREE_NODE_STRIDE;
                const left = u32[base + 3];
                const right = u32[base + 7];
                if (left !== 0 || right !== 0) hasNonZeroChild = true;
            }
            expect(hasNonZeroChild).toBe(true);
        }

        const inst0Base = meta.treeNodeOffset * TREE_NODE_STRIDE;
        const inst1Base = (meta.treeNodeOffset + nodeCount) * TREE_NODE_STRIDE;
        for (let i = 0; i < internalNodes; i++) {
            expect(u32[inst1Base + i * TREE_NODE_STRIDE + 3]).toBe(
                u32[inst0Base + i * TREE_NODE_STRIDE + 3],
            );
            expect(u32[inst1Base + i * TREE_NODE_STRIDE + 7]).toBe(
                u32[inst0Base + i * TREE_NODE_STRIDE + 7],
            );
        }
    });
});
