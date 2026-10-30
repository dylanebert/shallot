import { expect, test } from "bun:test";
import { type GltfJson, parse } from "./gltf";

function expectVertex(v: Float32Array, base: number, exp: number[]) {
    for (let i = 0; i < 8; i++) {
        if (i === 3 || i === 7) expect(v[base + i]).toBeCloseTo(exp[i], 5);
        else expect(v[base + i]).toBe(exp[i]);
    }
}

test("decodes the same geometry when attributes are tightly packed (no byteStride)", () => {
    // POSITION/NORMAL/TEXCOORD in their own tight bufferViews — the stride defaults to the element size
    const buf = new ArrayBuffer((3 + 3 + 2) * 3 * 4);
    const dv = new DataView(buf);
    const pos = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const nrm = [0, 0, 1, 0, 1, 0, 1, 0, 0];
    const uv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    pos.forEach((x, i) => {
        dv.setFloat32(i * 4, x, true);
    });
    nrm.forEach((x, i) => {
        dv.setFloat32(36 + i * 4, x, true);
    });
    uv.forEach((x, i) => {
        dv.setFloat32(72 + i * 4, x, true);
    });
    const gltf: GltfJson = {
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: 36 },
            { buffer: 0, byteOffset: 36, byteLength: 36 },
            { buffer: 0, byteOffset: 72, byteLength: 24 },
        ],
        accessors: [
            { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
            { bufferView: 1, componentType: 5126, count: 3, type: "VEC3" },
            { bufferView: 2, componentType: 5126, count: 3, type: "VEC2" },
        ],
        meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 } }] }],
        nodes: [{ mesh: 0 }],
        scenes: [{ nodes: [0] }],
    };
    const { meshes } = parse(gltf, [buf]);
    expectVertex(meshes[0].vertices, 0, [1, 2, 3, 0.1, 0, 0, 1, 0.2]);
    expectVertex(meshes[0].vertices, 16, [7, 8, 9, 0.5, 1, 0, 0, 0.6]);
});

test("widens ubyte and uint indices to u32", () => {
    // one triangle, POSITION tight in bv0, indices in bv1 — non-sequential values catch a byte-width
    // misread (a ubyte stream read as ushort would shift every index)
    const cases: {
        ct: number;
        bytes: number;
        write: (dv: DataView, o: number, v: number) => void;
    }[] = [
        { ct: 5121, bytes: 1, write: (dv, o, v) => dv.setUint8(o, v) },
        { ct: 5125, bytes: 4, write: (dv, o, v) => dv.setUint32(o, v, true) },
    ];
    for (const { ct, bytes, write } of cases) {
        const idx = [2, 0, 1];
        const posLen = 9 * 4;
        const buf = new ArrayBuffer(posLen + idx.length * bytes);
        const dv = new DataView(buf);
        [1, 2, 3, 4, 5, 6, 7, 8, 9].forEach((x, i) => {
            dv.setFloat32(i * 4, x, true);
        });
        idx.forEach((v, i) => {
            write(dv, posLen + i * bytes, v);
        });
        const gltf: GltfJson = {
            bufferViews: [
                { buffer: 0, byteOffset: 0, byteLength: posLen },
                { buffer: 0, byteOffset: posLen, byteLength: idx.length * bytes },
            ],
            accessors: [
                { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
                { bufferView: 1, componentType: ct, count: 3, type: "SCALAR" },
            ],
            meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 0 }, indices: 1 }] }],
            nodes: [{ mesh: 0 }],
            scenes: [{ nodes: [0] }],
        };
        const { meshes } = parse(gltf, [buf]);
        expect(meshes[0].indices).toBeInstanceOf(Uint32Array);
        expect([...meshes[0].indices]).toEqual([2, 0, 1]);
    }
});
