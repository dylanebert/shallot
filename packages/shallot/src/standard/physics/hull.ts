import { getMesh, type MeshData } from "../render";
import { registry, type Registry } from "../../engine";
import { quickhull } from "./quickhull";

export interface ConvexHullFace {
    plane: Float32Array;
    vertexIndices: Uint32Array;
}

export interface ConvexHull {
    vertices: Float32Array;
    numVertices: number;
    faces: ConvexHullFace[];
    numFaces: number;
    uniqueEdges: Float32Array;
    numUniqueEdges: number;
    localCenter: Float32Array;
    extents: Float32Array;
}

const MAX_HULLS = 256;

export const hullRegistry: Registry<ConvexHull> = registry(MAX_HULLS);

export function hull(meshId: number): number {
    const existing = hullRegistry.getByName(String(meshId));
    if (existing !== undefined) return existing;
    const meshData = getMesh(meshId);
    if (!meshData) throw new Error(`mesh ${meshId} not found`);
    const h = computeHull(meshData);
    return hullRegistry.add(h, String(meshId));
}

const VERTEX_STRIDE = 8;
const NORMAL_TOL = 1e-6;

function extractUniquePositions(meshData: MeshData): Float64Array {
    const seen = new Map<string, number>();
    const coords: number[] = [];
    for (let i = 0; i < meshData.vertexCount; i++) {
        const off = i * VERTEX_STRIDE;
        const x = meshData.vertices[off];
        const y = meshData.vertices[off + 1];
        const z = meshData.vertices[off + 2];
        const key = `${Math.round(x * 1e5)},${Math.round(y * 1e5)},${Math.round(z * 1e5)}`;
        if (!seen.has(key)) {
            seen.set(key, coords.length / 3);
            coords.push(x, y, z);
        }
    }
    return new Float64Array(coords);
}

function mergeCoplanarFaces(
    tris: number[][],
    normals: Float64Array,
    pts: Float64Array,
): ConvexHullFace[] {
    const groups: number[][] = [];
    const groupNormals: number[] = [];
    const groupOffsets: number[] = [];

    for (let i = 0; i < tris.length; i++) {
        const nx = normals[i * 3],
            ny = normals[i * 3 + 1],
            nz = normals[i * 3 + 2];
        const v0 = tris[i][0];
        const offset = nx * pts[v0 * 3] + ny * pts[v0 * 3 + 1] + nz * pts[v0 * 3 + 2];
        let found = -1;
        for (let g = 0; g < groups.length; g++) {
            const go = g * 3;
            const dot =
                nx * groupNormals[go] + ny * groupNormals[go + 1] + nz * groupNormals[go + 2];
            if (dot > 1 - NORMAL_TOL && Math.abs(offset - groupOffsets[g]) < NORMAL_TOL) {
                found = g;
                break;
            }
        }
        if (found >= 0) {
            groups[found].push(i);
        } else {
            groups.push([i]);
            groupNormals.push(nx, ny, nz);
            groupOffsets.push(offset);
        }
    }

    const result: ConvexHullFace[] = [];
    for (let g = 0; g < groups.length; g++) {
        const gTris = groups[g];
        const nx = groupNormals[g * 3],
            ny = groupNormals[g * 3 + 1],
            nz = groupNormals[g * 3 + 2];
        const d = -groupOffsets[g];
        const plane = new Float32Array([nx, ny, nz, d]);
        if (gTris.length === 1) {
            result.push({ plane, vertexIndices: new Uint32Array(tris[gTris[0]]) });
        } else {
            result.push({ plane, vertexIndices: new Uint32Array(extractBoundary(tris, gTris)) });
        }
    }
    return result;
}

function extractBoundary(tris: number[][], group: number[]): number[] {
    const edgeCounts = new Map<string, number>();
    for (const ti of group) {
        const tri = tris[ti];
        for (let i = 0; i < 3; i++) {
            const a = tri[i],
                b = tri[(i + 1) % 3];
            const key = a < b ? `${a},${b}` : `${b},${a}`;
            edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
        }
    }
    const next = new Map<number, number>();
    for (const ti of group) {
        const tri = tris[ti];
        for (let i = 0; i < 3; i++) {
            const a = tri[i],
                b = tri[(i + 1) % 3];
            const key = a < b ? `${a},${b}` : `${b},${a}`;
            if (edgeCounts.get(key) === 1) next.set(a, b);
        }
    }
    if (next.size === 0) return tris[group[0]];
    const start = next.keys().next().value!;
    const loop: number[] = [start];
    let cur = next.get(start)!;
    while (cur !== start) {
        loop.push(cur);
        const n = next.get(cur);
        if (n === undefined) break;
        cur = n;
    }
    return loop;
}

function collectUniqueEdges(faces: ConvexHullFace[], vertices: Float32Array): Float32Array {
    const dirs: number[] = [];
    let count = 0;
    for (const face of faces) {
        const idx = face.vertexIndices;
        for (let i = 0; i < idx.length; i++) {
            const a = idx[i],
                b = idx[(i + 1) % idx.length];
            let dx = vertices[b * 3] - vertices[a * 3];
            let dy = vertices[b * 3 + 1] - vertices[a * 3 + 1];
            let dz = vertices[b * 3 + 2] - vertices[a * 3 + 2];
            const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (len < 1e-12) continue;
            dx /= len;
            dy /= len;
            dz /= len;
            if (
                dx < -1e-8 ||
                (Math.abs(dx) < 1e-8 && dy < -1e-8) ||
                (Math.abs(dx) < 1e-8 && Math.abs(dy) < 1e-8 && dz < 0)
            ) {
                dx = -dx;
                dy = -dy;
                dz = -dz;
            }
            let dup = false;
            for (let j = 0; j < count; j++) {
                if (
                    dx * dirs[j * 3] + dy * dirs[j * 3 + 1] + dz * dirs[j * 3 + 2] >
                    1 - NORMAL_TOL
                ) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                dirs.push(dx, dy, dz);
                count++;
            }
        }
    }
    return new Float32Array(dirs);
}

function computeHull(meshData: MeshData): ConvexHull {
    const pts = extractUniquePositions(meshData);
    const n = pts.length / 3;
    const triFaces = quickhull(pts, n);

    const triNormals = new Float64Array(triFaces.length * 3);
    for (let i = 0; i < triFaces.length; i++) {
        const [a, b, c] = triFaces[i];
        const p = pts;
        const ax = p[a * 3],
            ay = p[a * 3 + 1],
            az = p[a * 3 + 2];
        const bx = p[b * 3] - ax,
            by = p[b * 3 + 1] - ay,
            bz = p[b * 3 + 2] - az;
        const cx = p[c * 3] - ax,
            cy = p[c * 3 + 1] - ay,
            cz = p[c * 3 + 2] - az;
        let nx = by * cz - bz * cy,
            ny = bz * cx - bx * cz,
            nz = bx * cy - by * cx;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-15) {
            nx /= len;
            ny /= len;
            nz /= len;
        }
        triNormals[i * 3] = nx;
        triNormals[i * 3 + 1] = ny;
        triNormals[i * 3 + 2] = nz;
    }

    const faces = mergeCoplanarFaces(triFaces, triNormals, pts);
    const vertices = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) vertices[i] = pts[i];
    const uniqueEdges = collectUniqueEdges(faces, vertices);

    let cx = 0,
        cy = 0,
        cz = 0;
    for (let i = 0; i < n; i++) {
        cx += pts[i * 3];
        cy += pts[i * 3 + 1];
        cz += pts[i * 3 + 2];
    }
    cx /= n;
    cy /= n;
    cz /= n;

    let minX = Infinity,
        minY = Infinity,
        minZ = Infinity;
    let maxX = -Infinity,
        maxY = -Infinity,
        maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
        const x = pts[i * 3],
            y = pts[i * 3 + 1],
            z = pts[i * 3 + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }

    return {
        vertices,
        numVertices: n,
        faces,
        numFaces: faces.length,
        uniqueEdges,
        numUniqueEdges: uniqueEdges.length / 3,
        localCenter: new Float32Array([cx, cy, cz]),
        extents: new Float32Array([(maxX - minX) / 2, (maxY - minY) / 2, (maxZ - minZ) / 2]),
    };
}

export interface PackedHullGPU {
    data: Uint32Array;
    metaCount: number;
}

const HULL_META_STRIDE = 12;
const VERTEX_STRIDE_GPU = 4;
const FACE_STRIDE_GPU = 8;
const EDGE_STRIDE_GPU = 4;

export function packHullsForGPU(): PackedHullGPU {
    const hulls = hullRegistry.all();
    const metaCount = hulls.length;
    if (metaCount === 0) {
        return { data: new Uint32Array(HULL_META_STRIDE), metaCount: 0 };
    }

    let totalVerts = 0;
    let totalFaces = 0;
    let totalFaceIndices = 0;
    let totalEdges = 0;
    for (const h of hulls) {
        totalVerts += h.numVertices;
        totalFaces += h.numFaces;
        for (const f of h.faces) totalFaceIndices += f.vertexIndices.length;
        totalEdges += h.numUniqueEdges;
    }

    const metaSize = metaCount * HULL_META_STRIDE;
    const vertSize = totalVerts * VERTEX_STRIDE_GPU;
    const faceSize = totalFaces * FACE_STRIDE_GPU;
    const faceIdxSize = totalFaceIndices;
    const edgeSize = totalEdges * EDGE_STRIDE_GPU;
    const totalSize = metaSize + vertSize + faceSize + faceIdxSize + edgeSize;

    const buf = new ArrayBuffer(totalSize * 4);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    const vertBase = metaSize;
    const faceBase = vertBase + vertSize;
    const faceIdxBase = faceBase + faceSize;
    const edgeBase = faceIdxBase + faceIdxSize;

    let vertOff = 0;
    let faceOff = 0;
    let faceIdxOff = 0;
    let edgeOff = 0;

    for (let hi = 0; hi < hulls.length; hi++) {
        const h = hulls[hi];
        const metaOff = hi * HULL_META_STRIDE;
        u32[metaOff + 0] = vertBase + vertOff * VERTEX_STRIDE_GPU;
        u32[metaOff + 1] = h.numVertices;
        u32[metaOff + 2] = faceBase + faceOff * FACE_STRIDE_GPU;
        u32[metaOff + 3] = h.numFaces;
        u32[metaOff + 4] = edgeBase + edgeOff * EDGE_STRIDE_GPU;
        u32[metaOff + 5] = h.numUniqueEdges;
        f32[metaOff + 6] = h.extents[0] > 1e-12 ? 1.0 / h.extents[0] : 0;
        f32[metaOff + 7] = h.extents[1] > 1e-12 ? 1.0 / h.extents[1] : 0;
        f32[metaOff + 8] = h.extents[2] > 1e-12 ? 1.0 / h.extents[2] : 0;
        u32[metaOff + 9] = 0;
        u32[metaOff + 10] = 0;
        u32[metaOff + 11] = 0;

        for (let vi = 0; vi < h.numVertices; vi++) {
            const dst = vertBase + (vertOff + vi) * VERTEX_STRIDE_GPU;
            f32[dst + 0] = h.vertices[vi * 3 + 0];
            f32[dst + 1] = h.vertices[vi * 3 + 1];
            f32[dst + 2] = h.vertices[vi * 3 + 2];
            f32[dst + 3] = 0;
        }

        let localFaceIdxOff = 0;
        for (let fi = 0; fi < h.numFaces; fi++) {
            const face = h.faces[fi];
            const dst = faceBase + (faceOff + fi) * FACE_STRIDE_GPU;
            f32[dst + 0] = face.plane[0];
            f32[dst + 1] = face.plane[1];
            f32[dst + 2] = face.plane[2];
            f32[dst + 3] = face.plane[3];
            u32[dst + 4] = faceIdxBase + faceIdxOff + localFaceIdxOff;
            u32[dst + 5] = face.vertexIndices.length;
            u32[dst + 6] = 0;
            u32[dst + 7] = 0;

            for (let ii = 0; ii < face.vertexIndices.length; ii++) {
                u32[faceIdxBase + faceIdxOff + localFaceIdxOff + ii] = face.vertexIndices[ii];
            }
            localFaceIdxOff += face.vertexIndices.length;
        }

        for (let ei = 0; ei < h.numUniqueEdges; ei++) {
            const dst = edgeBase + (edgeOff + ei) * EDGE_STRIDE_GPU;
            f32[dst + 0] = h.uniqueEdges[ei * 3 + 0];
            f32[dst + 1] = h.uniqueEdges[ei * 3 + 1];
            f32[dst + 2] = h.uniqueEdges[ei * 3 + 2];
            f32[dst + 3] = 0;
        }

        vertOff += h.numVertices;
        faceOff += h.numFaces;
        faceIdxOff += localFaceIdxOff;
        edgeOff += h.numUniqueEdges;
    }

    return { data: u32, metaCount };
}
