import { expect } from "bun:test";
import { check } from "../../../harness/check";
import type { Vec3 } from "../common/math";
import gold from "./geometry.gold.json";
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

function assertMesh(mesh: MeshData, g: MeshGold, vector: string) {
    const at = (what: string) => `${vector}: ${g.name} ${what}`;
    expect(mesh.nodes.length, at("nodeCount")).toBe(g.nodeCount);
    expect(mesh.vertices.length, at("vertexCount")).toBe(g.vertexCount);
    expect(mesh.triangles.length, at("triangleCount")).toBe(g.triangleCount);
    expect(mesh.materialCount, at("materialCount")).toBe(g.materialCount);
    expect(mesh.degenerateCount, at("degenerateCount")).toBe(g.degenerateCount);
    expect(mesh.treeHeight, at("treeHeight")).toBe(g.treeHeight);
    bitEqual(mesh.surfaceArea, g.surfaceArea, at("surfaceArea"));
    vecEqual(mesh.bounds.lowerBound, g.boundsLower, at("boundsLower"));
    vecEqual(mesh.bounds.upperBound, g.boundsUpper, at("boundsUpper"));

    for (let i = 0; i < g.nodeCount; ++i) {
        const n: MeshNode = mesh.nodes[i];
        const e = g.nodes[i];
        expect(n.leaf, at(`node[${i}].leaf`)).toBe(e.leaf);
        expect(n.axis, at(`node[${i}].axis`)).toBe(e.axis);
        expect(n.childOffset, at(`node[${i}].childOffset`)).toBe(e.childOffset);
        expect(n.triangleCount, at(`node[${i}].triangleCount`)).toBe(e.triangleCount);
        expect(n.triangleOffset, at(`node[${i}].triangleOffset`)).toBe(e.triangleOffset);
        vecEqual(n.lowerBound, e.lowerBound, at(`node[${i}].lower`));
        vecEqual(n.upperBound, e.upperBound, at(`node[${i}].upper`));
    }

    for (let i = 0; i < g.vertexCount; ++i) {
        vecEqual(mesh.vertices[i], g.vertices[i], at(`vertex[${i}]`));
    }

    for (let i = 0; i < g.triangleCount; ++i) {
        const t = mesh.triangles[i];
        expect([t.index1, t.index2, t.index3], at(`triangle[${i}]`)).toEqual(g.triangles[i]);
    }

    expect(mesh.materialIndices, at("materialIndices")).toEqual(g.materialIndices);
    expect(mesh.flags, at("flags")).toEqual(g.flags);
}

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

check(
    "triangle mesh builders match the C reference bit for bit",
    {
        claim: "a triangle mesh builder drifts from the Box3D C reference in its BVH nodes, vertices, winding, edge flags or surface area, and the mesh gold no longer describes what the TypeScript port builds",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const cases: [string, () => MeshData, string][] = [
            [
                "box (SAH split, edge identification)",
                () => createBoxMesh(v(0, 0, 0), v(1, 1, 1), true),
                "box",
            ],
            [
                "grid (median split, per-triangle materials)",
                () => createGridMesh(4, 4, 1, 3, true),
                "grid",
            ],
            [
                "hollow box (inward faces, SAH)",
                () => createHollowBoxMesh(v(0.5, -0.25, 1), v(2, 1, 0.5)),
                "hollow",
            ],
            ["torus (portable trig, SAH)", () => createTorusMesh(8, 6, 3, 1), "torus"],
        ];
        for (const [vector, build, name] of cases) {
            assertMesh(build(), meshGold(name), vector);
        }
    },
);

// The wave mesh is Box3D's one geometry helper using libm sinf rather than the portable trig, so
// its heights have no cross-platform-deterministic reference — assert structure, not bit-exactness.
check(
    "the triangle wave mesh grids, flattens its seed rows and stays inside its amplitude",
    {
        claim: "createWaveMesh emits the wrong triangle count for its cell grid, lifts the zero-sine boundary row or column off the plane, or rides its sine product past the requested amplitude",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const xCount = 6;
        const zCount = 5;
        const cellWidth = 1;
        const amplitude = 0.4;
        const wave = createWaveMesh(xCount, zCount, cellWidth, amplitude, 0.05, 0.1);

        expect(wave.vertices.length, "wave: vertexCount").toBe((xCount + 1) * (zCount + 1));
        expect(wave.triangles.length, "wave: triangleCount").toBe(2 * xCount * zCount);

        // sin(0) === 0, so the ix=0 row and the iz=0 column are flat (±0, since a zero row height
        // times a negative column sine yields -0 — the exact f32 result the C reference produces).
        for (let iz = 0; iz <= zCount; ++iz) {
            expect(Math.abs(wave.vertices[iz].y), `wave: ix=0 row vertex[${iz}].y`).toBe(0);
        }
        for (let ix = 0; ix <= xCount; ++ix) {
            expect(
                Math.abs(wave.vertices[(zCount + 1) * ix].y),
                `wave: iz=0 column vertex[${(zCount + 1) * ix}].y`,
            ).toBe(0);
        }

        // Interior heights ride the sine product, so |y| is nonzero yet bounded by amplitude.
        let maxAbs = 0;
        for (const vert of wave.vertices) maxAbs = Math.max(maxAbs, Math.abs(vert.y));
        expect(maxAbs, "wave: interior heights are all flat").toBeGreaterThan(0);
        expect(maxAbs, "wave: interior height exceeds amplitude").toBeLessThanOrEqual(
            Math.fround(amplitude),
        );
    },
);
