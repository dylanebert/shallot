import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, Tonemapping, TonemappingMethod, Transform } from "@dylanebert/shallot";
import { Bloom } from "@dylanebert/shallot/bloom";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp as createApp } from "@dylanebert/shallot/testing/gpu";
import { Scene, ScenePlugin, setBloom } from "./scene";

setDefaultTimeout(TIMEOUT.node);
test("one black-clear TonyMcMapface camera blooms a grid of 100 bouncing spheres", async () => {
    const app = await createApp({ plugins: [ScenePlugin] });
    const world = app.world;
    try {
        const { camera, spheres } = world.resource(Scene);
        expect([...world.query([Camera])]).toEqual([camera]);
        expect(world.storage(Camera).clearColor.get(camera)).toBe(0);
        expect(world.storage(Tonemapping).method.get(camera)).toBe(TonemappingMethod.TonyMcMapface);
        expect(world.has(camera, Bloom)).toBe(true);
        expect(spheres.length).toBe(100);
        world.tick();
        ScenePlugin.systems![0].update!(world);
        const position = world.storage(Transform).translation;
        for (const eid of spheres)
            expect(position.y.get(eid)).toBeCloseTo(
                Math.sin(position.x.get(eid) + position.z.get(eid) + world.time.elapsed),
            );
        setBloom(world, false);
        expect(world.has(camera, Bloom)).toBe(false);
        setBloom(world, true);
        expect(world.has(camera, Bloom)).toBe(true);
    } finally {
        app.dispose();
    }
});
