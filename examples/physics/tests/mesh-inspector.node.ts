import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, createApp, PhysicsMeshes } from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { INK } from "../src/ink-presentation";
import { JOINT_LINE } from "../src/joint-presentation";
import { MeshSamplePlugin } from "../src/mesh";
import { MeshCreationState, MeshInspectorState } from "../src/mesh-inspector";
import { drawMeshViewerView, meshViewerBounds } from "../src/mesh-inspector-presentation";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
test("Viewer draws selected node bounds in ink and restores stepping after Creation Benchmark", async () => {
    const app = await createApp({ defaults: false, plugins: [MeshSamplePlugin] });
    try {
        const select = sceneSwitcher(app.world);
        select("meshViewer");
        const values = controlValues("meshViewer");
        expect(meshViewerBounds(app.world)).toEqual([]);
        for (const level of [0, 1, 3]) {
            changeControl(app.world, "meshViewer", [], values, "draw level", level);
            const bounds = meshViewerBounds(app.world);
            const edges: number[][] = [];
            drawMeshViewerView(app.world, (a, b, color, options) => {
                expect(color).toBe(INK);
                expect(options).toBe(JOINT_LINE);
                edges.push([...a, ...b]);
            });
            expect(edges).toHaveLength(12 * bounds.length);
            for (const [index, box] of bounds.entries()) {
                const nodeEdges = edges.slice(12 * index, 12 * (index + 1));
                expect(new Set(nodeEdges.map((edge) => JSON.stringify(edge))).size).toBe(12);
                for (const edge of nodeEdges) {
                    for (let axis = 0; axis < 3; axis++) {
                        expect([box[axis], box[axis + 3]]).toContain(edge[axis]);
                        expect([box[axis], box[axis + 3]]).toContain(edge[axis + 3]);
                    }
                    expect([0, 1, 2].filter((axis) => edge[axis] !== edge[axis + 3])).toHaveLength(
                        1,
                    );
                }
            }
        }
        for (const index of [1, 2, 3, 0]) {
            changeControl(app.world, "meshViewer", [], values, "index", index);
            const s = app.world.resource(MeshInspectorState);
            expect(scenes.meshViewer.controls["draw level"]!.bounds!(values, app.world)).toEqual([
                -1,
                s.height,
            ]);
            expect(app.world.resource(PhysicsMeshes).size).toBe(1);
        }
        select("meshCreation");
        expect(meshViewerBounds(app.world)).toEqual([]);
        expect([...app.world.query([Body])]).toEqual([]);
        expect(app.world.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(false);
        select("meshViewer");
        expect(app.world.resource(MeshCreationState).active).toBe(false);
        expect(app.world.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(true);
    } finally {
        await app.dispose();
    }
});
