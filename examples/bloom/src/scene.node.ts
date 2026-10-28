import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, Tonemapping, TonemappingMethod, Transform } from "@dylanebert/shallot";
import { Bloom } from "@dylanebert/shallot/bloom";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp as createApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin } from "./app";
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

test("the example's scene visibly blooms: bloom on and off frames differ", async () => {
    const app = await createApp({
        plugins: [
            {
                name: "CaptureBloom",
                dependencies: [AppPlugin],
                initialize(world) {
                    attachTexture(world, world.resource(Scene).camera, { width: 640, height: 360 });
                },
            },
        ],
    });
    try {
        const { world } = app;
        const camera = world.resource(Scene).camera;
        const frame = async () => {
            world.step(0);
            return (await captureTexture(world, camera)).rgba;
        };
        const on = await frame();
        setBloom(world, false);
        const off = await frame();
        let sum = 0;
        for (let i = 0; i < on.length; i++) if (i % 4 !== 3) sum += Math.abs(on[i] - off[i]);
        // The mean per-channel difference, in 8-bit steps, the browser capture test bounded.
        expect(sum / (on.length * 0.75)).toBeGreaterThan(0.1);
    } finally {
        app.dispose();
    }
});
