import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile, disposeGpuApps, gpuRequirements } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    PointLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { createApp, lookAtRotation, type World } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { StandardRenderer } from "./forward";
import { StandardRenderingPlugin } from "./index";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { MeshRenderPlugin } from "./mesh-render";
import { DirectionalLightShadowMap, PointShadows } from "./shadows";

setDefaultTimeout(CEILING.gpu);

// app B's shadow settings, written the way their JSDoc instructs
function configure(world: World): void {
    world.resource(PointShadows).atlas = 1024;
    world.resource(DirectionalLightShadowMap).size = 512;
}

// a cube between a shadowed point light, a shadowed sun and a wall: the wall holds both shadows
function scene(world: World): number {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 6, 0] });
    world.add(camera, Camera);
    world.add(camera, AmbientLight, { brightness: 0 });
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 32, height: 32 });
    const material = world.resource(Materials).add(StandardMaterial());
    const wall = world.create();
    world.add(wall, Transform, { translation: [0, 0, -1, 0], scale: [20, 20, 0.2, 0] });
    world.add(wall, MeshInstance);
    world.add(wall, MeshMaterial, material);
    const cube = world.create();
    world.add(cube, Transform, { translation: [0, 0, 1, 0] });
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, material);
    const light = world.create();
    world.add(light, Transform, { translation: [0.6, 0.6, 3, 0] });
    world.add(light, PointLight, { intensity: 788064.8, range: 20 });
    world.storage(PointLight).shadowMapsEnabled.set(light, 1);
    const sun = world.create();
    const sunRotation = lookAtRotation(0, 0, 0, -0.3, -0.3, -1);
    world.add(sun, Transform, {
        rotation: [sunRotation.x, sunRotation.y, sunRotation.z, sunRotation.w],
    });
    world.add(sun, DirectionalLight, { illuminance: 9406.831 });
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    return camera;
}

// the shadows first appear on the second frame
async function shot(world: World, camera: number) {
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const error = (await world.gpu.device.popErrorScope())?.message ?? null;
    const { rgba } = await captureTexture(world, camera);
    return { rgba, error };
}

const subjects = compileGpuFile(import.meta.path, async () => {
    const owner = await createApp({
        defaults: false,
        plugins: [
            {
                name: "ShadowSettingsTestDevice",
                gpu: gpuRequirements([StandardRenderingPlugin, MeshRenderPlugin]),
            },
        ],
    });
    const device = rawDevice(owner.world.gpu.device);
    const plugins = [StandardRenderingPlugin, MeshRenderPlugin];
    const a = await createApp({ defaults: false, plugins, device });
    const cameraA = scene(a.world);
    // A's reference frame is taken before B exists
    const reference = await shot(a.world, cameraA);
    const b = await createApp({ defaults: false, plugins, device, setup: configure });
    const cameraB = scene(b.world);
    return { owner, a, cameraA, reference, b, cameraB };
});

test("a second app's shadow settings leave the first app's frame unchanged", async () => {
    const { a, cameraA, reference, b, cameraB } = subjects();
    expect(reference.error).toBeNull();
    expect((await shot(b.world, cameraB)).error).toBeNull();
    const { rgba, error } = await shot(a.world, cameraA);
    expect(error).toBeNull();
    expect(rgba).toEqual(reference.rgba);
});

afterAll(() => {
    const { owner, a, b } = subjects();
    return disposeGpuApps([owner, a, b]);
});
