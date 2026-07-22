import { describe, expect, test } from "bun:test";
import { HULL_FACE_STRIDE, HULL_HEADER, packHulls } from "../../src/standard/avbd/hull";
import { type Hull, Hulls, UNIT_CUBE_ID } from "../../src/standard/physics/hull";
import { boxHull, tetHull } from "./hull";

// The flat `hullData` layout the GPU collide pass (collide.ts HULL_WGSL) reads — a serialization boundary,
// so a round-trip test pins it before the WGSL accessors index it on the real GPU. Decode the packed
// buffer the same way the WGSL does (header stride 8, face stride 6) and assert it reproduces the source
// hull's verts / faces / edges. Catches an offset/stride bug deterministically on CPU.

interface Decoded {
    verts: number[][];
    faces: { normal: number[]; offset: number; verts: number[] }[];
    edges: number[][];
}

function decode(buf: Uint32Array, id: number): Decoded {
    const f32 = new Float32Array(buf.buffer);
    const ho = id * HULL_HEADER;
    const vertBase = buf[ho + 0];
    const vertCount = buf[ho + 1];
    const faceBase = buf[ho + 2];
    const faceCount = buf[ho + 3];
    const edgeBase = buf[ho + 4];
    const edgeCount = buf[ho + 5];
    const faceIdxBase = buf[ho + 6];

    const verts: number[][] = [];
    for (let i = 0; i < vertCount; i++) {
        const o = vertBase + i * 3;
        verts.push([f32[o], f32[o + 1], f32[o + 2]]);
    }
    const faces: Decoded["faces"] = [];
    for (let f = 0; f < faceCount; f++) {
        const o = faceBase + f * HULL_FACE_STRIDE;
        const localOff = buf[o + 4];
        const fvCount = buf[o + 5];
        const fv: number[] = [];
        for (let j = 0; j < fvCount; j++) fv.push(buf[faceIdxBase + localOff + j]);
        faces.push({ normal: [f32[o], f32[o + 1], f32[o + 2]], offset: f32[o + 3], verts: fv });
    }
    const edges: number[][] = [];
    for (let e = 0; e < edgeCount; e++) {
        const o = edgeBase + e * 3;
        edges.push([f32[o], f32[o + 1], f32[o + 2]]);
    }
    return { verts, faces, edges };
}

function expectHull(got: Decoded, want: Omit<Hull, "name">): void {
    expect(got.verts.length).toBe(want.verts.length);
    for (let i = 0; i < want.verts.length; i++)
        for (let k = 0; k < 3; k++) expect(got.verts[i][k]).toBeCloseTo(want.verts[i][k], 6);
    expect(got.faces.length).toBe(want.faces.length);
    for (let f = 0; f < want.faces.length; f++) {
        for (let k = 0; k < 3; k++)
            expect(got.faces[f].normal[k]).toBeCloseTo(want.faces[f].normal[k], 6);
        expect(got.faces[f].offset).toBeCloseTo(want.faces[f].offset, 6);
        expect(got.faces[f].verts).toEqual(want.faces[f].verts);
    }
    expect(got.edges.length).toBe(want.edges.length);
    for (let e = 0; e < want.edges.length; e++)
        for (let k = 0; k < 3; k++) expect(got.edges[e][k]).toBeCloseTo(want.edges[e][k], 6);
}

describe("hull packing round-trip", () => {
    test("a box-hull + a tet-hull pack and decode back to their source geometry", () => {
        const box = boxHull([2, 1, 3]);
        const tet = tetHull(0.5);
        const boxId = Hulls.register({ name: "pack-box", ...box });
        const tetId = Hulls.register({ name: "pack-tet", ...tet });
        const buf = packHulls();

        expectHull(decode(buf, boxId), box);
        expectHull(decode(buf, tetId), tet);
    });

    test("the built-in unit cube packs at UNIT_CUBE_ID (a box collides as this cube × half-extents)", () => {
        // the scale-unified box path reads this hull for EVERY box × hull collision, so its geometry is
        // load-bearing: 8 verts at ±1, 6 axis-aligned faces (offset 1), 3 unique edge directions.
        const cube = decode(packHulls(), UNIT_CUBE_ID);
        expect(cube.verts.length).toBe(8);
        for (const v of cube.verts) for (const c of v) expect(Math.abs(c)).toBeCloseTo(1, 6);
        expect(cube.faces.length).toBe(6);
        for (const f of cube.faces) {
            expect(f.offset).toBeCloseTo(1, 6);
            expect(
                Math.abs(f.normal[0]) + Math.abs(f.normal[1]) + Math.abs(f.normal[2]),
            ).toBeCloseTo(1, 6);
        }
        expect(cube.edges).toEqual([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]);
    });
});
