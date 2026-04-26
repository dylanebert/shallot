import type { MeshData } from "../../standard/render/core";
import type { Vec3, Triangle } from "./bvh/structs";

export type { Triangle };

export interface FlatTriangle {
    v0x: number;
    v0y: number;
    v0z: number;
    v1x: number;
    v1y: number;
    v1z: number;
    v2x: number;
    v2y: number;
    v2z: number;
    n0x: number;
    n0y: number;
    n0z: number;
    n1x: number;
    n1y: number;
    n1z: number;
    n2x: number;
    n2y: number;
    n2z: number;
    entityId: number;
}

function transformPoint(x: number, y: number, z: number, m: Float32Array): Vec3 {
    return {
        x: m[0] * x + m[4] * y + m[8] * z + m[12],
        y: m[1] * x + m[5] * y + m[9] * z + m[13],
        z: m[2] * x + m[6] * y + m[10] * z + m[14],
    };
}

function transformNormal(nx: number, ny: number, nz: number, m: Float32Array): Vec3 {
    const x = m[0] * nx + m[4] * ny + m[8] * nz;
    const y = m[1] * nx + m[5] * ny + m[9] * nz;
    const z = m[2] * nx + m[6] * ny + m[10] * nz;
    const len = Math.sqrt(x * x + y * y + z * z);
    if (len === 0) return { x: 0, y: 0, z: 0 };
    return { x: x / len, y: y / len, z: z / len };
}

export function extractTriangles(
    mesh: MeshData,
    entityId: number,
    transform?: Float32Array,
): Triangle[] {
    const triangles: Triangle[] = [];
    const { vertices, indices } = mesh;
    const stride = 8;

    for (let i = 0; i < indices.length; i += 3) {
        const i0 = indices[i] * stride;
        const i1 = indices[i + 1] * stride;
        const i2 = indices[i + 2] * stride;

        let v0: Vec3 = { x: vertices[i0], y: vertices[i0 + 1], z: vertices[i0 + 2] };
        let n0: Vec3 = { x: vertices[i0 + 3], y: vertices[i0 + 4], z: vertices[i0 + 5] };

        let v1: Vec3 = { x: vertices[i1], y: vertices[i1 + 1], z: vertices[i1 + 2] };
        let n1: Vec3 = { x: vertices[i1 + 3], y: vertices[i1 + 4], z: vertices[i1 + 5] };

        let v2: Vec3 = { x: vertices[i2], y: vertices[i2 + 1], z: vertices[i2 + 2] };
        let n2: Vec3 = { x: vertices[i2 + 3], y: vertices[i2 + 4], z: vertices[i2 + 5] };

        if (transform) {
            v0 = transformPoint(v0.x, v0.y, v0.z, transform);
            v1 = transformPoint(v1.x, v1.y, v1.z, transform);
            v2 = transformPoint(v2.x, v2.y, v2.z, transform);
            n0 = transformNormal(n0.x, n0.y, n0.z, transform);
            n1 = transformNormal(n1.x, n1.y, n1.z, transform);
            n2 = transformNormal(n2.x, n2.y, n2.z, transform);
        }

        const e1: Vec3 = { x: v1.x - v0.x, y: v1.y - v0.y, z: v1.z - v0.z };
        const e2: Vec3 = { x: v2.x - v0.x, y: v2.y - v0.y, z: v2.z - v0.z };

        triangles.push({ v0, e1, e2, n0, n1, n2, entityId });
    }

    return triangles;
}
