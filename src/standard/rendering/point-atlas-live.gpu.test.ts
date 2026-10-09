import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile, gpuRequirements } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DepthPrepass,
    PointLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { createApp, type World } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { PointShadows } from "./shadows";

setDefaultTimeout(CEILING.gpu);

const subjects = compileGpuFile(import.meta.path, async () => {
    const owner = await createApp({
        defaults: false,
        plugins: [
            {
                name: "PointAtlasTestDevice",
                gpu: gpuRequirements([StandardRenderingPlugin, MeshRenderPlugin]),
            },
        ],
    });
    const device = rawDevice(owner.world.gpu.device);
    return { owner, device };
});

// an app whose world sets PointShadows.atlas in setup, on the shared device
function build(atlas: number) {
    return createApp({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
        device: subjects().device,
        setup: (world) => {
            world.resource(PointShadows).atlas = atlas;
        },
    });
}

// a cube between a shadowed point light and a wall, seen by a camera with or without a depth prepass
function scene(world: World, prepass: boolean): number {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 6, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    if (prepass) world.add(camera, DepthPrepass);
    attachTexture(world, camera, { width: 32, height: 32 });
    const material = world.resource(Materials).add(StandardMaterial());
    const wall = world.create();
    world.add(wall, Transform, { translation: [0, 0, -1, 0], scale: [20, 20, 0.2, 0] });
    world.add(wall, MeshInstance);
    world.add(wall, MeshMaterial, { material });
    const cube = world.create();
    world.add(cube, Transform, { translation: [0, 0, 1, 0] });
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, { material });
    world.add(world.create(), AmbientLight, { intensity: 0 });
    const light = world.create();
    world.add(light, Transform, { translation: [0.6, 0.6, 3, 0] });
    world.add(light, PointLight, { intensity: 20, range: 20 });
    world.storage(PointLight).shadowMapsEnabled.set(light, 1);
    return camera;
}

for (const [atlas0, atlas1] of [
    [2048, 1024],
    [1024, 2048],
    [2048, 256],
]) {
    for (const prepass of [false, true]) {
        test(`changing PointShadows.atlas ${atlas0} to ${atlas1} after a point light first casts renders as an app built with ${atlas1}${prepass ? ", under a depth prepass" : ""}`, async () => {
            const app = await build(atlas0);
            const fresh = await build(atlas1);
            try {
                const camera = scene(app.world, prepass);
                app.world.step(0);
                app.world.step(0);
                app.world.resource(PointShadows).atlas = atlas1;
                app.world.gpu.device.pushErrorScope("validation");
                app.world.step(0);
                app.world.step(0);
                expect((await app.world.gpu.device.popErrorScope())?.message ?? null).toBeNull();
                const reference = scene(fresh.world, prepass);
                fresh.world.step(0);
                fresh.world.step(0);
                expect((await captureTexture(app.world, camera)).rgba).toEqual(
                    (await captureTexture(fresh.world, reference)).rgba,
                );
            } finally {
                app.dispose();
                fresh.dispose();
            }
        });
    }
}

afterAll(() => {
    subjects().owner.dispose();
});
