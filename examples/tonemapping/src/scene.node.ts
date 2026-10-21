import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    ColorGrading,
    DirectionalLight,
    MeshInstance,
    Transform,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Scene, ScenePlugin } from "./scene";
import { setScene } from "./settings";

setDefaultTimeout(TIMEOUT.node);
test("Bevy camera and shadowed sun illuminate the plane, five-part cube and six spheres", async () => {
    const app = await openGpuApp({ plugins: [ScenePlugin] });
    try {
        const world = app.world;
        const { camera, subjects } = world.resource(Scene);
        expect([...world.query([Camera])]).toEqual([camera]);
        expect(world.has(camera, ColorGrading)).toBe(true);
        expect(world.has(camera, Orbit)).toBe(true);
        world.step(0);
        expect(subjects).toHaveLength(12);
        setScene(world, true);
        expect([...world.query([MeshInstance])]).toEqual([world.resource(Scene).pattern]);
        setScene(world, false);
        expect([...world.query([MeshInstance])]).toEqual(subjects);
        expect(world.storage(Transform).translation.x.get(camera)).toBeCloseTo(0.7);
        const sun = [...world.query([DirectionalLight])][0];
        expect(world.storage(DirectionalLight).illuminance.get(sun)).toBe(15000);
        expect(world.storage(DirectionalLight).shadowMapsEnabled.get(sun)).toBe(1);
    } finally {
        app.dispose();
    }
});
