import { describe, test, expect, beforeEach } from "bun:test";
import { hull, hullRegistry, type ConvexHull } from "../src/standard/physics/hull";
import {
    mesh,
    clearMeshes,
    createBox,
    createCone,
    createSphere,
    type MeshData,
} from "../src/standard/render/mesh";

const VERTEX_STRIDE = 8;

function createTetrahedron(): MeshData {
    const s = 1 / Math.sqrt(2);
    const verts = [
        [1, 0, -s],
        [-1, 0, -s],
        [0, 1, s],
        [0, -1, s],
    ];
    const faceIndices = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ];

    const vertices: number[] = [];
    const indices: number[] = [];
    let vi = 0;
    for (const face of faceIndices) {
        const [i0, i1, i2] = face;
        const p0 = verts[i0],
            p1 = verts[i1],
            p2 = verts[i2];
        const ex = p1[0] - p0[0],
            ey = p1[1] - p0[1],
            ez = p1[2] - p0[2];
        const fx = p2[0] - p0[0],
            fy = p2[1] - p0[1],
            fz = p2[2] - p0[2];
        let nx = ey * fz - ez * fy,
            ny = ez * fx - ex * fz,
            nz = ex * fy - ey * fx;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        nx /= len;
        ny /= len;
        nz /= len;
        for (const idx of face) {
            const p = verts[idx];
            vertices.push(p[0], p[1], p[2], nx, ny, nz, 0, 0);
            indices.push(vi++);
        }
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint16Array(indices),
        vertexCount: vi,
        indexCount: indices.length,
    };
}

function assertContainment(meshData: MeshData, h: ConvexHull): void {
    for (let i = 0; i < meshData.vertexCount; i++) {
        const vx = meshData.vertices[i * VERTEX_STRIDE];
        const vy = meshData.vertices[i * VERTEX_STRIDE + 1];
        const vz = meshData.vertices[i * VERTEX_STRIDE + 2];
        for (const face of h.faces) {
            const dist =
                face.plane[0] * vx + face.plane[1] * vy + face.plane[2] * vz + face.plane[3];
            expect(dist).toBeLessThanOrEqual(1e-5);
        }
    }
}

beforeEach(() => {
    hullRegistry.clear();
    clearMeshes();
});

describe("convex hull", () => {
    describe("cube", () => {
        let h: ConvexHull;
        let meshId: number;

        beforeEach(() => {
            meshId = mesh(createBox(), "test-box");
            h = hullRegistry.get(hull(meshId))!;
        });

        test("topology: 8 verts, 6 faces", () => {
            expect(h.numVertices).toBe(8);
            expect(h.numFaces).toBe(6);
            const eulerEdges = h.numVertices + h.numFaces - 2;
            expect(eulerEdges).toBe(12);
        });

        test("plane equations are axis-aligned with offset 0.5", () => {
            const normals = h.faces.map((f) => [f.plane[0], f.plane[1], f.plane[2]]);
            const offsets = h.faces.map((f) => f.plane[3]);

            const axes = [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ];
            for (const axis of axes) {
                const match = normals.findIndex(
                    (n) =>
                        Math.abs(n[0] - axis[0]) < 1e-6 &&
                        Math.abs(n[1] - axis[1]) < 1e-6 &&
                        Math.abs(n[2] - axis[2]) < 1e-6,
                );
                expect(match).not.toBe(-1);
                expect(Math.abs(offsets[match]) - 0.5).toBeCloseTo(0, 4);
            }
        });

        test("3 unique edge directions", () => {
            expect(h.numUniqueEdges).toBe(3);
        });

        test("all mesh vertices inside hull", () => {
            assertContainment(createBox(), h);
        });
    });

    describe("tetrahedron", () => {
        let h: ConvexHull;

        beforeEach(() => {
            const meshId = mesh(createTetrahedron(), "test-tet");
            h = hullRegistry.get(hull(meshId))!;
        });

        test("topology: 4 verts, 4 faces", () => {
            expect(h.numVertices).toBe(4);
            expect(h.numFaces).toBe(4);
            const eulerEdges = h.numVertices + h.numFaces - 2;
            expect(eulerEdges).toBe(6);
        });

        test("6 unique edge directions", () => {
            expect(h.numUniqueEdges).toBe(6);
        });

        test("face normals point outward", () => {
            for (const face of h.faces) {
                const vi = face.vertexIndices[0];
                const vx = h.vertices[vi * 3],
                    vy = h.vertices[vi * 3 + 1],
                    vz = h.vertices[vi * 3 + 2];
                const toCenter =
                    face.plane[0] * (h.localCenter[0] - vx) +
                    face.plane[1] * (h.localCenter[1] - vy) +
                    face.plane[2] * (h.localCenter[2] - vz);
                expect(toCenter).toBeLessThan(0);
            }
        });
    });

    describe("edge directions", () => {
        test("no duplicates", () => {
            const meshId = mesh(createBox(), "edge-test");
            const h = hullRegistry.get(hull(meshId))!;
            for (let i = 0; i < h.numUniqueEdges; i++) {
                for (let j = i + 1; j < h.numUniqueEdges; j++) {
                    const dot =
                        h.uniqueEdges[i * 3] * h.uniqueEdges[j * 3] +
                        h.uniqueEdges[i * 3 + 1] * h.uniqueEdges[j * 3 + 1] +
                        h.uniqueEdges[i * 3 + 2] * h.uniqueEdges[j * 3 + 2];
                    expect(Math.abs(dot)).toBeLessThan(1 - 1e-6);
                }
            }
        });

        test("all unit length", () => {
            const meshId = mesh(createTetrahedron(), "len-test");
            const h = hullRegistry.get(hull(meshId))!;
            for (let i = 0; i < h.numUniqueEdges; i++) {
                const x = h.uniqueEdges[i * 3];
                const y = h.uniqueEdges[i * 3 + 1];
                const z = h.uniqueEdges[i * 3 + 2];
                expect(Math.sqrt(x * x + y * y + z * z)).toBeCloseTo(1, 6);
            }
        });
    });

    test("hull returns cached result", () => {
        const meshId = mesh(createBox(), "cache-test");
        const id1 = hull(meshId);
        const id2 = hull(meshId);
        expect(id1).toBe(id2);
    });

    test("clear removes cached hulls", () => {
        const meshId = mesh(createBox(), "clear-test");
        hull(meshId);
        hullRegistry.clear();
        clearMeshes();
        const newMeshId = mesh(createBox(), "clear-test");
        expect(() => hull(newMeshId)).not.toThrow();
    });

    test("throws for unknown mesh", () => {
        expect(() => hull(9999)).toThrow();
    });

    describe("cone16 parity", () => {
        let h: ConvexHull;

        beforeEach(() => {
            const meshId = mesh(createCone(16), "parity-cone");
            h = hullRegistry.get(hull(meshId))!;
        });

        test("topology: 18 verts, 17 faces", () => {
            expect(h.numVertices).toBe(18);
            expect(h.numFaces).toBe(17);
        });

        test("has a bottom face with 16 vertices", () => {
            const bottomFace = h.faces.find((f) => f.vertexIndices.length === 16);
            expect(bottomFace).toBeDefined();
            expect(bottomFace!.plane[1]).toBeCloseTo(-1, 4);
        });

        test("all mesh vertices inside hull", () => {
            const meshData = createCone(16);
            assertContainment(meshData, h);
        });
    });

    describe("sphere", () => {
        let h: ConvexHull;

        beforeEach(() => {
            const meshId = mesh(createSphere(8, 4), "parity-sphere");
            h = hullRegistry.get(hull(meshId))!;
        });

        test("uses all 26 unique vertices", () => {
            expect(h.numVertices).toBe(26);
        });

        test("euler formula: V - E + F = 2", () => {
            let totalFaceVerts = 0;
            for (const face of h.faces) totalFaceVerts += face.vertexIndices.length;
            const edges = totalFaceVerts / 2;
            expect(h.numVertices - edges + h.numFaces).toBe(2);
        });

        test("all face normals point outward", () => {
            for (const face of h.faces) {
                const vi = face.vertexIndices[0];
                const vx = h.vertices[vi * 3],
                    vy = h.vertices[vi * 3 + 1],
                    vz = h.vertices[vi * 3 + 2];
                const toCenter =
                    face.plane[0] * (h.localCenter[0] - vx) +
                    face.plane[1] * (h.localCenter[1] - vy) +
                    face.plane[2] * (h.localCenter[2] - vz);
                expect(toCenter).toBeLessThan(0);
            }
        });

        test("all mesh vertices inside hull", () => {
            const meshData = createSphere(8, 4);
            assertContainment(meshData, h);
        });
    });
});
