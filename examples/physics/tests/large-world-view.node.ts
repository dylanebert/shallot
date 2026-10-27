import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, MeshInstance, MeshMaterial, qRotate, Shape, Transform } from "@dylanebert/shallot";
import { Gizmo, Orbit } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin, Viewer } from "../src/app";
import { drawingBounds, drawSet } from "../src/draw-set";
import { sceneSwitcher } from "../src/sample";

setDefaultTimeout(TIMEOUT.node);

test("Large World's picker attaches meshes, materials and edges only to the world's draw query set", async () => {
    const app = await openGpuApp({ plugins: [AppPlugin] });
    try {
        const world = app.world;
        const select = sceneSwitcher(world, world.resource(Viewer));
        select("largeWorld");
        world.tick();
        world.step(0);
        await world.frameFence;
        const camera = [...world.query([Camera])][0]!;
        function assertDrawSet() {
            const pose = world.get(camera, Transform)!;
            const expected = drawSet(
                world,
                drawingBounds(
                    pose.translation,
                    qRotate(...pose.rotation, 0, 0, -1),
                    1,
                    world.storage(Camera).fov.get(camera),
                ),
            );
            for (const component of [MeshInstance, MeshMaterial, Gizmo]) {
                expect(new Set(world.query([Shape, component]))).toEqual(expected);
            }
            expect(expected.size).toBeLessThan(41_000);
            expect(expected.size).toBeGreaterThan(40_000);
            return expected;
        }
        const opening = assertDrawSet();
        world.storage(Orbit).pan.set(camera, 3000, 0, 0);
        world.step(1 / 60);
        await world.frameFence;
        const moved = assertDrawSet();
        expect([...opening].some((eid) => !moved.has(eid))).toBe(true);
        select("boxStack");
        expect([...world.query([Shape, MeshInstance])]).toHaveLength(41);
    } finally {
        app.dispose();
    }
});
