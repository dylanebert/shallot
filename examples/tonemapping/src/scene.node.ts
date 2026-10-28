import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    ColorGrading,
    DirectionalLight,
    MeshInstance,
    Transform,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin } from "./app";
import { Scene, ScenePlugin } from "./scene";
import { methods, setMethod, setScene } from "./settings";

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

test("the example's scene shows each method distinctly on both scenes", async () => {
    const app = await openGpuApp({
        plugins: [
            {
                name: "CaptureTonemapping",
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
        // The mean per-channel difference, in 8-bit steps.
        const difference = (a: Uint8ClampedArray, b: Uint8ClampedArray) => {
            let sum = 0;
            for (let i = 0; i < a.length; i++) if (i % 4 !== 3) sum += Math.abs(a[i] - b[i]);
            return sum / (a.length * 0.75);
        };
        const basic = await frame();
        for (const pattern of [false, true]) {
            setScene(world, pattern);
            const frames: Uint8ClampedArray[] = [];
            for (const [name, method] of methods) {
                setMethod(world, method);
                const image = await frame();
                for (const previous of frames)
                    expect(
                        difference(image, previous),
                        `${pattern ? "pattern" : "basic"}: ${name}`,
                    ).toBeGreaterThan(0.001);
                frames.push(image);
            }
            if (pattern) expect(difference(basic, frames[0])).toBeGreaterThan(1);
        }
        setMethod(world, methods[0][1]);
        setScene(world, false);
        expect(difference(basic, await frame())).toBeLessThan(0.01);
    } finally {
        app.dispose();
    }
});
