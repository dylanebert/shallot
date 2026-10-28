import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import type { PhysicsMesh } from "@dylanebert/shallot/physics";
import {
    createPhysicsMesh,
    destroyPhysicsMesh,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import gold from "../../../crates/physics/tests/nonconvex_build.gold.json";
import { drawMeshData } from "./shapes/readgeometry";

setDefaultTimeout(TIMEOUT.node);

const rawMeshes: [string, PhysicsMesh][] = ["raw", "weld", "median", "clockwise"].map((name) => {
    const native = gold.find((entry) => entry.name === name)!;
    const args = new Float32Array(new Uint32Array(native.args).buffer);
    return [
        name,
        {
            name,
            vertices: new Float32Array(new Uint32Array(native.vertices!).buffer),
            indices: new Uint32Array(native.indices!),
            materialIndices: native.materials?.length
                ? new Uint8Array(native.materials)
                : undefined,
            weldTolerance: args[0],
            weldVertices: !!args[1],
            useMedianSplit: !!args[2],
            identifyEdges: !!args[3],
            clockwiseWinding: !!args[4],
        },
    ];
});

for (const [name, definition] of [
    ...rawMeshes,
    [
        "grid",
        {
            name: "grid",
            grid: { xCount: 4, zCount: 5, cellWidth: 0.3, materialCount: 3, identifyEdges: true },
        },
    ],
    [
        "torus",
        {
            name: "torus",
            torus: { radialResolution: 8, tubularResolution: 6, radius: 3, thickness: 1 },
        },
    ],
] satisfies [string, PhysicsMesh][]) {
    test(`public ${name} build reports every native field and every tree level; rebuilding is independent`, async () => {
        const native = gold.find((entry) => entry.name === name)!;
        const mesh = drawMeshData(new DataView(new Uint32Array(native.words!).buffer), 0);
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            const first = createPhysicsMesh(app.world, definition);
            const second = createPhysicsMesh(app.world, definition);
            expect(second).not.toBe(first);
            expect(second).toEqual(first);
            expect(first.triangleCount).toBe(mesh.triangles.length);
            expect(first.vertexCount).toBe(mesh.vertices.length);
            expect(first.degenerateCount).toBe(mesh.degenerateCount);
            expect(first.surfaceArea).toBe(mesh.surfaceArea);
            expect(first.treeHeight).toBe(mesh.treeHeight);
            expect(first.nodes).toHaveLength(mesh.nodes.length);
            // Independent breadth-first walk of the native image, retaining native sibling order.
            let queue = [0];
            let level = 0;
            while (queue.length) {
                for (const [position, index] of queue.entries()) {
                    const node = mesh.nodes[index]!;
                    expect(first.nodes[index]).toEqual({
                        bounds: { lowerBound: node.lowerBound, upperBound: node.upperBound },
                        level,
                        leaf: node.leaf,
                    });
                    expect(first.nodes[index]!.bounds).toBe(first.nodeBounds[level]![position]);
                }
                expect(first.nodeBounds[level]).toEqual(
                    queue.map((index) => ({
                        lowerBound: mesh.nodes[index]!.lowerBound,
                        upperBound: mesh.nodes[index]!.upperBound,
                    })),
                );
                queue = queue.flatMap((index) =>
                    mesh.nodes[index]!.leaf
                        ? []
                        : [index + 1, index + mesh.nodes[index]!.childOffset],
                );
                level++;
            }
            expect(first.nodeBounds).toHaveLength(level);
            destroyPhysicsMesh(app.world, first);
            expect(second.triangleCount).toBe(mesh.triangles.length);
            destroyPhysicsMesh(app.world, second);
        } finally {
            await app.dispose();
        }
    });
}
