import { expect, test } from "bun:test";
import {
    createBoxMesh,
    createGridMesh,
    createHollowBoxMesh,
    createMesh,
    createTorusMesh,
    createWaveMesh,
    type MeshData,
} from "./mesh";

function orderedTraversal(mesh: MeshData | null): void {
    expect(mesh).not.toBeNull();
    if (!mesh) throw new Error("expected a mesh with nondegenerate triangles");
    const indices: number[] = [];
    const stack = [0];
    while (stack.length) {
        const index = stack.pop()!;
        const node = mesh.nodes[index];
        if (node.leaf) {
            for (let i = 0; i < node.triangleCount; ++i) {
                indices.push(node.triangleOffset + i);
            }
        } else {
            stack.push(index + node.childOffset, index + 1);
        }
    }
    expect(indices).toEqual(Array.from({ length: mesh.triangles.length }, (_, i) => i));
}

let state = 0x537a1107;
function random(): number {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
}

test("createMesh emits increasing triangle indices in left-first traversal, including degenerate and many-triangle inputs", () => {
    for (const count of [1, 8, 257, 2048]) {
        const vertices = [];
        const indices: number[] = [];
        for (let i = 0; i < count; ++i) {
            const base = vertices.length;
            const x = 20 * random() - 10;
            const y = 20 * random() - 10;
            const z = 20 * random() - 10;
            vertices.push({ x, y, z }, { x: x + 0.5, y, z }, { x, y: y + 0.5, z });
            indices.push(base, base + 1, base + 2);
            if (i % 3 === 0) indices.push(base, base, base + 2);
            if (i % 5 === 0) {
                const degenerate = vertices.length;
                vertices.push({ x, y, z }, { x, y, z }, { x, y, z });
                indices.push(degenerate, degenerate + 1, degenerate + 2);
            }
        }
        for (const useMedianSplit of [false, true]) {
            for (const weldVertices of [false, true]) {
                orderedTraversal(
                    createMesh({
                        vertices,
                        indices,
                        useMedianSplit,
                        weldVertices,
                        weldTolerance: 0.001,
                    }),
                );
            }
        }
    }
});

test("box, hollow box, grid, wave and torus builders emit increasing triangle indices in left-first traversal over seeded inputs", () => {
    for (let i = 0; i < 8; ++i) {
        const center = { x: random(), y: random(), z: random() };
        const extent = { x: 0.1 + random(), y: 0.1 + random(), z: 0.1 + random() };
        orderedTraversal(createBoxMesh(center, extent, i % 2 === 0));
        orderedTraversal(createHollowBoxMesh(center, extent));
        const x = 3 + Math.floor(30 * random());
        const z = 3 + Math.floor(30 * random());
        const width = 0.1 + random();
        orderedTraversal(createGridMesh(x, z, width, 3, i % 2 === 0));
        orderedTraversal(createWaveMesh(x, z, width, random(), random(), random()));
        orderedTraversal(createTorusMesh(x, z, 3 + random(), 0.2 + random()));
    }
    orderedTraversal(createBoxMesh({ x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 1 }, false));
    orderedTraversal(createHollowBoxMesh({ x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 1 }));
    expect(
        createMesh({
            vertices: [
                { x: 0, y: 0, z: 0 },
                { x: 1, y: 0, z: 0 },
                { x: 2, y: 0, z: 0 },
            ],
            indices: [0, 1, 2],
        }),
    ).toBeNull();
});
