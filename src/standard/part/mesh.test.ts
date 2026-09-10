import { describe, expect, test } from "bun:test";
import { meshBounds } from "../render/mesh";
import { capsule, cube, sphere } from "./mesh";

// the shared interleaved layout: (posX posY posZ uvX)(normalX normalY normalZ uvY)
const pos = (v: Float32Array, i: number) => [v[i * 8], v[i * 8 + 1], v[i * 8 + 2]] as const;
const nrm = (v: Float32Array, i: number) => [v[i * 8 + 4], v[i * 8 + 5], v[i * 8 + 6]] as const;
const hypot = (x: number, y: number, z: number) => Math.sqrt(x * x + y * y + z * z);

// triangle winding agrees with the authored normals: the face normal
// (cross of two edges) points the same way as the averaged vertex normal.
// Returns the count of mis-wound non-degenerate triangles (UV-sphere poles
// produce zero-area triangles, skipped below the area floor)
function misWound(vertices: Float32Array, indices: Uint32Array): { bad: number; checked: number } {
    let bad = 0;
    let checked = 0;
    for (let t = 0; t < indices.length; t += 3) {
        const [ax, ay, az] = pos(vertices, indices[t]);
        const [bx, by, bz] = pos(vertices, indices[t + 1]);
        const [cx, cy, cz] = pos(vertices, indices[t + 2]);
        const e1x = bx - ax;
        const e1y = by - ay;
        const e1z = bz - az;
        const e2x = cx - ax;
        const e2y = cy - ay;
        const e2z = cz - az;
        const fx = e1y * e2z - e1z * e2y;
        const fy = e1z * e2x - e1x * e2z;
        const fz = e1x * e2y - e1y * e2x;
        if (hypot(fx, fy, fz) < 1e-6) continue; // degenerate pole triangle
        checked++;
        const n0 = nrm(vertices, indices[t]);
        const n1 = nrm(vertices, indices[t + 1]);
        const n2 = nrm(vertices, indices[t + 2]);
        const ax2 = n0[0] + n1[0] + n2[0];
        const ay2 = n0[1] + n1[1] + n2[1];
        const az2 = n0[2] + n1[2] + n2[2];
        if (fx * ax2 + fy * ay2 + fz * az2 <= 0) bad++;
    }
    return { bad, checked };
}

describe("cube", () => {
    const { vertices, indices } = cube();

    test("interleaved layout — vertex 0 is the +Z face BL corner", () => {
        // pos (-0.5, -0.5, 0.5), uv (0, 0), normal (0, 0, 1)
        expect(Array.from(vertices.slice(0, 8))).toEqual([-0.5, -0.5, 0.5, 0, 0, 0, 1, 0]);
    });

    test("24 vertices (4 per face), 36 indices (2 tris per face)", () => {
        expect(vertices.length).toBe(24 * 8);
        expect(indices.length).toBe(36);
    });

    test("every corner is a half-diagonal from center with an axis-unit normal", () => {
        for (let i = 0; i < vertices.length / 8; i++) {
            expect(hypot(...pos(vertices, i))).toBeCloseTo(Math.sqrt(0.75), 6);
            expect(hypot(...nrm(vertices, i))).toBeCloseTo(1, 6);
        }
    });

    test("bounding sphere is centered with the half-diagonal radius", () => {
        const [cx, cy, cz, r] = meshBounds(vertices);
        expect(cx).toBeCloseTo(0, 6);
        expect(cy).toBeCloseTo(0, 6);
        expect(cz).toBeCloseTo(0, 6);
        expect(r).toBeCloseTo(Math.sqrt(0.75), 6);
    });

    test("every triangle winds CCW outward", () => {
        const { bad, checked } = misWound(vertices, indices);
        expect(bad).toBe(0);
        expect(checked).toBe(12); // 6 faces × 2, none degenerate
    });
});

describe("sphere", () => {
    const { vertices, indices } = sphere();

    test("interleaved layout — vertex 0 is the top pole", () => {
        // y=0, theta=0: position (0, 0.5, 0), normal (0, 1, 0), uv (0, 0)
        expect(Array.from(vertices.slice(0, 8))).toEqual([0, 0.5, 0, 0, 0, 1, 0, 0]);
    });

    test("vertex + index counts follow the grid formula", () => {
        // (rings+1)(segments+1) vertices, rings*segments*2 triangles
        expect(vertices.length).toBe(17 * 33 * 8);
        expect(indices.length).toBe(16 * 32 * 6);
    });

    test("every vertex lies on the radius-0.5 surface with a unit normal", () => {
        for (let i = 0; i < vertices.length / 8; i++) {
            expect(hypot(...pos(vertices, i))).toBeCloseTo(0.5, 6);
            expect(hypot(...nrm(vertices, i))).toBeCloseTo(1, 6);
        }
    });

    test("bounding sphere is the unit primitive: centered, radius 0.5", () => {
        const [cx, cy, cz, r] = meshBounds(vertices);
        expect(cx).toBeCloseTo(0, 6);
        expect(cy).toBeCloseTo(0, 6);
        expect(cz).toBeCloseTo(0, 6);
        expect(r).toBeCloseTo(0.5, 6);
    });

    test("every triangle winds CCW outward", () => {
        const { bad, checked } = misWound(vertices, indices);
        expect(bad).toBe(0);
        expect(checked).toBeGreaterThan(0);
    });
});

describe("capsule", () => {
    const { vertices, indices } = capsule();

    // distance from the central segment (0,-0.5,0)→(0,0.5,0); equals the radius
    // for every surface point of a radius-0.5 capsule
    const distToAxis = (px: number, py: number, pz: number) =>
        hypot(px, Math.max(-0.5, Math.min(0.5, py)) - py, pz);

    test("vertex + index counts cover both caps, the seam rings, and the band", () => {
        // 2 caps of (halfRings+1)(segments+1) + 2 seam rings of (segments+1)
        expect(vertices.length).toBe((2 * 9 * 33 + 2 * 33) * 8);
        // 2 caps of halfRings*segments*2 tris + a cylinder band of segments*2
        expect(indices.length).toBe((8 * 32 * 2 + 8 * 32 * 2 + 32 * 2) * 3);
    });

    test("every vertex sits radius 0.5 off the central segment with a unit normal", () => {
        for (let i = 0; i < vertices.length / 8; i++) {
            expect(distToAxis(...pos(vertices, i))).toBeCloseTo(0.5, 6);
            expect(hypot(...nrm(vertices, i))).toBeCloseTo(1, 6);
        }
    });

    test("bounding sphere spans the poles: centered, radius 1.0", () => {
        // caps reach y = ±(halfHeight + radius) = ±1, the farthest points
        const [cx, cy, cz, r] = meshBounds(vertices);
        expect(cx).toBeCloseTo(0, 6);
        expect(cy).toBeCloseTo(0, 6);
        expect(cz).toBeCloseTo(0, 6);
        expect(r).toBeCloseTo(1, 6);
    });

    test("every triangle winds CCW outward — including the reversed bottom cap", () => {
        const { bad, checked } = misWound(vertices, indices);
        expect(bad).toBe(0);
        expect(checked).toBeGreaterThan(0);
    });
});
