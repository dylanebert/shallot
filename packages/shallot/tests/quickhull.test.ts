import { describe, test, expect } from "bun:test";
import { quickhull } from "../src/standard/physics/quickhull";

function pts(...coords: number[][]): Float64Array {
    const out = new Float64Array(coords.length * 3);
    for (let i = 0; i < coords.length; i++) {
        out[i * 3] = coords[i][0];
        out[i * 3 + 1] = coords[i][1];
        out[i * 3 + 2] = coords[i][2];
    }
    return out;
}

function checkEuler(faces: number[][], nVerts: number): boolean {
    const totalHalfEdges = faces.length * 3;
    const edges = totalHalfEdges / 2;
    return nVerts - edges + faces.length === 2;
}

function checkContainment(p: Float64Array, faces: number[][]): number {
    const n = p.length / 3;
    let maxViolation = 0;
    for (const [a, b, c] of faces) {
        const ax = p[a * 3],
            ay = p[a * 3 + 1],
            az = p[a * 3 + 2];
        const ux = p[b * 3] - ax,
            uy = p[b * 3 + 1] - ay,
            uz = p[b * 3 + 2] - az;
        const vx = p[c * 3] - ax,
            vy = p[c * 3 + 1] - ay,
            vz = p[c * 3 + 2] - az;
        let nx = uy * vz - uz * vy,
            ny = uz * vx - ux * vz,
            nz = ux * vy - uy * vx;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-15) continue;
        nx /= len;
        ny /= len;
        nz /= len;
        const off = nx * ax + ny * ay + nz * az;
        for (let i = 0; i < n; i++) {
            const d = nx * p[i * 3] + ny * p[i * 3 + 1] + nz * p[i * 3 + 2] - off;
            if (d > maxViolation) maxViolation = d;
        }
    }
    return maxViolation;
}

const CUBE_PTS = pts(
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
);
const TET_PTS = pts([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]);

function spherePts(segments: number, rings: number): Float64Array {
    const coords: number[] = [];
    coords.push(0, 0.5, 0);
    for (let r = 1; r < rings; r++) {
        const phi = (r / rings) * Math.PI;
        for (let s = 0; s < segments; s++) {
            const theta = (s / segments) * Math.PI * 2;
            coords.push(
                0.5 * Math.sin(phi) * Math.cos(theta),
                0.5 * Math.cos(phi),
                0.5 * Math.sin(phi) * Math.sin(theta),
            );
        }
    }
    coords.push(0, -0.5, 0);
    return new Float64Array(coords);
}

function randomPts(n: number, seed: number): Float64Array {
    const coords: number[] = [];
    let s = seed;
    for (let i = 0; i < n * 3; i++) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        coords.push((s / 0x7fffffff) * 2 - 1);
    }
    return new Float64Array(coords);
}

// Gold face counts verified against quickhull3d reference (mauriciopoppe/quickhull3d)
describe("quickhull", () => {
    test("cube: 12 triangles", () => {
        expect(quickhull(CUBE_PTS, 8).length).toBe(12);
    });

    test("cube: containment", () => {
        expect(checkContainment(CUBE_PTS, quickhull(CUBE_PTS, 8))).toBeLessThan(1e-10);
    });

    test("tetrahedron: 4 faces", () => {
        expect(quickhull(TET_PTS, 4).length).toBe(4);
    });

    test("tetrahedron: containment", () => {
        expect(checkContainment(TET_PTS, quickhull(TET_PTS, 4))).toBeLessThan(1e-10);
    });

    test("sphere(8,4): 48 triangles", () => {
        const p = spherePts(8, 4);
        expect(quickhull(p, p.length / 3).length).toBe(48);
    });

    test("sphere(8,4): euler", () => {
        const p = spherePts(8, 4);
        const n = p.length / 3;
        expect(checkEuler(quickhull(p, n), n)).toBe(true);
    });

    test("sphere(8,4): containment", () => {
        const p = spherePts(8, 4);
        expect(checkContainment(p, quickhull(p, p.length / 3))).toBeLessThan(1e-10);
    });

    test("sphere(16,8): 224 triangles", () => {
        const p = spherePts(16, 8);
        expect(quickhull(p, p.length / 3).length).toBe(224);
    });

    test("random 50: 38 faces", () => {
        const p = randomPts(50, 12345);
        expect(quickhull(p, 50).length).toBe(38);
    });

    test("random 50: containment", () => {
        const p = randomPts(50, 12345);
        expect(checkContainment(p, quickhull(p, 50))).toBeLessThan(1e-10);
    });
});
