import { describe, expect, test } from "bun:test";
import { mesh, packMeshes, quantizeMeshes } from "./mesh";

// These are the live CPU→GPU mesh boundary checks: record width, absolute index packing, and the
// unorm16 lattice consumed by the renderer's vertex decoder. They live beside mesh.ts rather than a
// renderer-specific surface suite because no shading contract participates in the encoding.
describe("mesh vertex contract", () => {
    test("rejects a vertices length that isn't a whole number of Vertex records", () => {
        expect(() =>
            mesh({ name: "bad", vertices: new Float32Array(7), indices: new Uint32Array([0]) }),
        ).toThrow(/not a multiple of 8/);
    });

    test("packMeshes concatenates and shifts indices by vertex base", () => {
        const packed = packMeshes([
            {
                name: "a",
                vertices: new Float32Array(4 * 8),
                indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
            },
            { name: "b", vertices: new Float32Array(3 * 8), indices: new Uint32Array([0, 1, 2]) },
        ]);
        expect([...packed.indices]).toEqual([0, 1, 2, 0, 2, 3, 4, 5, 6]);
        expect(packed.vertices.length).toBe((4 + 3) * 8);
        expect(packed.slices).toEqual([
            { name: "a", indexBase: 0, indexCount: 6, vertexBase: 0, vertexCount: 4 },
            { name: "b", indexBase: 6, indexCount: 3, vertexBase: 4, vertexCount: 3 },
        ]);
    });

    test("quantizeMeshes round-trips pos+uv within extent/65535, packs meshId, mirrors the position stream", () => {
        const packed = packMeshes([
            {
                name: "a",
                vertices: new Float32Array([
                    -2, 0.5, -10, 0, 0, 0, 1, 0, 3, 1.5, 7, 4, 0, 0, 1, 8, 0, 1, -1, 2, 0, 0, 1, 4,
                ]),
                indices: new Uint32Array([0, 1, 2]),
            },
            {
                name: "b",
                vertices: new Float32Array([
                    1, 1, 1, 0.25, 1, 0, 0, 0.5, 2, 2, 2, 0.75, 1, 0, 0, 1.5,
                ]),
                indices: new Uint32Array([0, 1]),
            },
        ]);
        const { main, position, quant } = quantizeMeshes(packed.vertices, packed.slices);

        for (const [meshId, s] of packed.slices.entries()) {
            const o = meshId * 12;
            const [pminX, pminY, pminZ, uminX] = quant.slice(o, o + 4);
            const [pextX, pextY, pextZ, uminY] = quant.slice(o + 4, o + 8);
            const [uextX, uextY] = quant.slice(o + 8, o + 10);
            for (let v = 0; v < s.vertexCount; v++) {
                const vi = s.vertexBase + v;
                const src = vi * 8;
                const w0 = main[vi * 4];
                const w1 = main[vi * 4 + 1];
                const w3 = main[vi * 4 + 3];
                expect(w1 >>> 16).toBe(meshId);
                expect(position[vi * 2]).toBe(w0);
                expect(position[vi * 2 + 1]).toBe(w1);
                const px = pminX + ((w0 & 0xffff) / 65535) * pextX;
                const py = pminY + ((w0 >>> 16) / 65535) * pextY;
                const pz = pminZ + ((w1 & 0xffff) / 65535) * pextZ;
                expect(Math.abs(px - packed.vertices[src])).toBeLessThanOrEqual(pextX / 65535);
                expect(Math.abs(py - packed.vertices[src + 1])).toBeLessThanOrEqual(pextY / 65535);
                expect(Math.abs(pz - packed.vertices[src + 2])).toBeLessThanOrEqual(pextZ / 65535);
                const u = uminX + ((w3 & 0xffff) / 65535) * uextX;
                const vv = uminY + ((w3 >>> 16) / 65535) * uextY;
                expect(Math.abs(u - packed.vertices[src + 3])).toBeLessThanOrEqual(
                    uextX / 65535 + 1e-9,
                );
                expect(Math.abs(vv - packed.vertices[src + 7])).toBeLessThanOrEqual(
                    uextY / 65535 + 1e-9,
                );
            }
        }
    });

    test("quantizeMeshes puts x, y and z on the same unorm16 lattice", () => {
        const straddle = 1.4384299516677856;
        const packed = packMeshes([
            {
                name: "a",
                vertices: new Float32Array([
                    -2,
                    -2,
                    -2,
                    0,
                    0,
                    0,
                    1,
                    0,
                    straddle,
                    straddle,
                    straddle,
                    0,
                    0,
                    0,
                    1,
                    0,
                    3,
                    3,
                    3,
                    0,
                    0,
                    0,
                    1,
                    0,
                ]),
                indices: new Uint32Array([0, 1, 2]),
            },
        ]);
        const { main } = quantizeMeshes(packed.vertices, packed.slices);
        const w0 = main[4];
        const w1 = main[5];
        expect(w0 >>> 16).toBe(w0 & 0xffff);
        expect(w1 & 0xffff).toBe(w0 & 0xffff);
    });
});
