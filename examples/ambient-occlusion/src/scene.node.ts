import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    Meshes,
    type MeshHandle,
    MeshInstance,
    ScreenSpaceAmbientOcclusion,
    Transform,
} from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin } from "./app";
import { Scene, ScenePlugin, setOcclusion } from "./scene";
import { setQuality, setSetting, settings } from "./settings";

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

test("the example's scene visibly occludes at the crease and sphere, and radius, thickness and quality change it", async () => {
    const app = await openGpuApp({
        plugins: [
            {
                name: "CaptureOcclusion",
                dependencies: [AppPlugin],
                initialize(world) {
                    attachTexture(world, world.resource(Scene).camera, { width: 640, height: 360 });
                },
            },
        ],
    });
    try {
        const { world } = app;
        const frame = async () => {
            world.step(0);
            return (await captureTexture(world, world.resource(Scene).camera)).rgba;
        };
        // The mean per-channel difference in 8-bit steps, and one pixel's channel sum.
        const difference = (a: Uint8ClampedArray, b: Uint8ClampedArray) => {
            let sum = 0;
            for (let i = 0; i < a.length; i++) if (i % 4 !== 3) sum += Math.abs(a[i] - b[i]);
            return sum / (a.length * 0.75);
        };
        const sample = (rgba: Uint8ClampedArray, x: number, y: number) => {
            const i = (y * 640 + x) * 4;
            return rgba[i] + rgba[i + 1] + rgba[i + 2];
        };
        const on = await frame();
        setOcclusion(world, false);
        const off = await frame();
        expect(difference(on, off)).toBeGreaterThan(0.1);
        expect(sample(off, 320, 130) - sample(on, 320, 130), "crease").toBeGreaterThan(60);
        expect(sample(off, 320, 220) - sample(on, 320, 220), "sphere").toBeGreaterThan(20);
        expect(
            Math.abs(sample(on, 260, 110) - sample(off, 260, 110)) / 3,
            "far",
        ).toBeLessThanOrEqual(2);
        setOcclusion(world, true);
        expect(difference(await frame(), on)).toBeLessThan(0.1);
        for (const [index, value] of [8, 4].entries()) {
            setSetting(world, index, value);
            expect(difference(await frame(), on), settings[index].label).toBeGreaterThan(0.01);
            setSetting(world, index, settings[index].value);
        }
        for (const quality of [0, 1, 3]) {
            setQuality(world, quality);
            expect(difference(await frame(), on), `quality ${quality}`).toBeGreaterThan(0.001);
        }
    } finally {
        app.dispose();
    }
});
