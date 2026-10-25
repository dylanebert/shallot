import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    Meshes,
    type MeshHandle,
    MeshInstance,
    ScreenSpaceAmbientOcclusion,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Scene, ScenePlugin, setOcclusion } from "./scene";

setDefaultTimeout(TIMEOUT.node);
test("three unit cubes meet around a moving 0.4 sphere and High AO camera", async () => {
    const app = await openGpuApp({ plugins: [ScenePlugin] });
    const world = app.world;
    try {
        const { camera, cubes, sphere } = world.resource(Scene);
        console.info("SSAO adapter", world.gpu.adapter);
        const meshes = world.resource(Meshes);
        const instances = world.storage(MeshInstance);
        for (const eid of cubes)
            expect(meshes.get(instances.mesh.get(eid) as MeshHandle)!.bounds![3]).toBeCloseTo(
                Math.sqrt(3) / 2,
            );
        expect(meshes.get(instances.mesh.get(sphere) as MeshHandle)!.bounds![3]).toBeCloseTo(0.4);
        expect([...world.query([Camera])]).toEqual([camera]);
        expect(world.storage(ScreenSpaceAmbientOcclusion).qualityLevel.get(camera)).toBe(2);
        const position = world.storage(Transform).translation;
        expect(
            cubes.map((eid) => [position.x.get(eid), position.y.get(eid), position.z.get(eid)]),
        ).toEqual([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ]);
        world.tick();
        ScenePlugin.systems![0].update!(world);
        expect(position.y.get(sphere)).toBeCloseTo(Math.sin(world.time.elapsed / 1.7) * 0.7);
        world.step(0);
        setOcclusion(world, false);
        world.step(0);
        setOcclusion(world, true);
        world.step(0);
        expect(world.has(camera, ScreenSpaceAmbientOcclusion)).toBe(true);
    } finally {
        app.dispose();
    }
});
