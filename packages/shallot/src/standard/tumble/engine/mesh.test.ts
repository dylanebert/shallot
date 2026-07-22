import { describe, expect, test } from "bun:test";
import gold from "./geometry.gold.json";
import type { Vec3 } from "./math";
import {
    createBoxMesh,
    createGridMesh,
    createHollowBoxMesh,
    createTorusMesh,
    createWaveMesh,
    type MeshData,
    type MeshNode,
} from "./mesh";

const dv = new DataView(new ArrayBuffer(4));
function fromBits(hex: string): number {
    dv.setUint32(0, Number.parseInt(hex, 16));
    return dv.getFloat32(0);
}
function bits(f: number): string {
    dv.setFloat32(0, f);
    return dv.getUint32(0).toString(16).padStart(8, "0");
}
function bitEqual(got: number, want: string, label: string) {
    const w = fromBits(want);
    if (!Object.is(got, w)) {
        throw new Error(`${label}: got 0x${bits(got)} (${got}), want ${want} (${w})`);
    }
}
function vecEqual(got: Vec3, want: string[], label: string) {
    bitEqual(got.x, want[0], `${label}.x`);
    bitEqual(got.y, want[1], `${label}.y`);
    bitEqual(got.z, want[2], `${label}.z`);
}

type NodeGold = {
    leaf: boolean;
    axis: number;
    childOffset: number;
    triangleCount: number;
    triangleOffset: number;
    lowerBound: string[];
    upperBound: string[];
};
type MeshGold = {
    name: string;
    nodeCount: number;
    vertexCount: number;
    triangleCount: number;
    materialCount: number;
    degenerateCount: number;
    treeHeight: number;
    surfaceArea: string;
    boundsLower: string[];
    boundsUpper: string[];
    nodes: NodeGold[];
    vertices: string[][];
    triangles: number[][];
    materialIndices: number[];
    flags: number[];
};

const meshGold = (name: string) => gold.meshes.find((m) => m.name === name) as unknown as MeshGold;

function assertMesh(mesh: MeshData, g: MeshGold) {
    expect(mesh.nodes.length).toBe(g.nodeCount);
    expect(mesh.vertices.length).toBe(g.vertexCount);
    expect(mesh.triangles.length).toBe(g.triangleCount);
    expect(mesh.materialCount).toBe(g.materialCount);
    expect(mesh.degenerateCount).toBe(g.degenerateCount);
    expect(mesh.treeHeight).toBe(g.treeHeight);
    bitEqual(mesh.surfaceArea, g.surfaceArea, `${g.name} surfaceArea`);
    vecEqual(mesh.bounds.lowerBound, g.boundsLower, `${g.name} boundsLower`);
    vecEqual(mesh.bounds.upperBound, g.boundsUpper, `${g.name} boundsUpper`);

    for (let i = 0; i < g.nodeCount; ++i) {
        const n: MeshNode = mesh.nodes[i];
        const e = g.nodes[i];
        expect(n.leaf).toBe(e.leaf);
        expect(n.axis).toBe(e.axis);
        expect(n.childOffset).toBe(e.childOffset);
        expect(n.triangleCount).toBe(e.triangleCount);
        expect(n.triangleOffset).toBe(e.triangleOffset);
        vecEqual(n.lowerBound, e.lowerBound, `${g.name} node[${i}].lower`);
        vecEqual(n.upperBound, e.upperBound, `${g.name} node[${i}].upper`);
    }

    for (let i = 0; i < g.vertexCount; ++i) {
        vecEqual(mesh.vertices[i], g.vertices[i], `${g.name} vertex[${i}]`);
    }

    for (let i = 0; i < g.triangleCount; ++i) {
        const t = mesh.triangles[i];
        expect([t.index1, t.index2, t.index3]).toEqual(g.triangles[i]);
    }

    expect(mesh.materialIndices).toEqual(g.materialIndices);
    expect(mesh.flags).toEqual(g.flags);
}

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

describe("mesh creation bit-exact vs C reference", () => {
    test("box mesh (SAH split, edge identification)", () => {
        const box = createBoxMesh(v(0, 0, 0), v(1, 1, 1), true);
        assertMesh(box, meshGold("box"));
    });

    test("grid mesh (median split, per-triangle materials)", () => {
        const grid = createGridMesh(4, 4, 1, 3, true);
        assertMesh(grid, meshGold("grid"));
    });

    test("hollow box mesh (inward faces, SAH)", () => {
        const hollow = createHollowBoxMesh(v(0.5, -0.25, 1), v(2, 1, 0.5));
        assertMesh(hollow, meshGold("hollow"));
    });

    test("torus mesh (portable trig, SAH)", () => {
        const torus = createTorusMesh(8, 6, 3, 1);
        assertMesh(torus, meshGold("torus"));
    });
});

// The wave mesh is Box3D's one geometry helper using libm sinf rather than the portable trig, so
// its heights have no cross-platform-deterministic reference — assert structure, not bit-exactness.
describe("wave mesh (libm sine heights, structural)", () => {
    test("grid topology + flat boundary rows + amplitude-bounded interior", () => {
        const xCount = 6;
        const zCount = 5;
        const cellWidth = 1;
        const amplitude = 0.4;
        const wave = createWaveMesh(xCount, zCount, cellWidth, amplitude, 0.05, 0.1);

        expect(wave.vertices.length).toBe((xCount + 1) * (zCount + 1));
        expect(wave.triangles.length).toBe(2 * xCount * zCount);

        // sin(0) === 0, so the ix=0 row and the iz=0 column are flat (±0, since a zero row height
        // times a negative column sine yields -0 — the exact f32 result the C reference produces).
        for (let iz = 0; iz <= zCount; ++iz) expect(Math.abs(wave.vertices[iz].y)).toBe(0);
        for (let ix = 0; ix <= xCount; ++ix) {
            expect(Math.abs(wave.vertices[(zCount + 1) * ix].y)).toBe(0);
        }

        // Interior heights ride the sine product, so |y| is nonzero yet bounded by amplitude.
        let maxAbs = 0;
        for (const vert of wave.vertices) maxAbs = Math.max(maxAbs, Math.abs(vert.y));
        expect(maxAbs).toBeGreaterThan(0);
        expect(maxAbs).toBeLessThanOrEqual(Math.fround(amplitude));
    });
});
