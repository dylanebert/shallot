import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import {
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    SUN_DISK_EARTH_ANGULAR_SIZE,
    SunDisk,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import { Backgrounds, CameraBackground, StandardRenderer } from "../../standard/rendering";
import { Sky, SkyPlugin } from "./index";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [SkyPlugin] }]);

async function center(world: World, camera: number): Promise<number[]> {
    world.step(0);
    world.step(0);
    const { rgba } = await captureTexture(world, camera);
    return Array.from(rgba.subarray((8 * 17 + 8) * 4, (8 * 17 + 8) * 4 + 4));
}

function rotationToward(x: number, y: number, z: number): [number, number, number, number] {
    const q = lookAtRotation(0, 0, 0, x, y, z);
    return [q.x, q.y, q.z, q.w];
}

function addDisk(world: World, color: number, travel: readonly [number, number, number]): number {
    const light = world.create();
    world.add(light, Transform, { rotation: rotationToward(...travel) });
    world.add(light, DirectionalLight, { color, illuminance: 0 });
    world.add(light, SunDisk);
    return light;
}

test("the sky draws each SunDisk in its light color along its transformed direction", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 17, height: 17 });
    const sky = world.create();
    world.add(sky, Sky, {
        zenith: 0,
        horizon: 0,
        cloudCoverage: 0,
        cloudDensity: 0,
        hazeDensity: 0,
    });
    world.add(camera, CameraBackground, { name: world.resource(Backgrounds).id("sky") });

    const noDisk = await center(world, camera);
    const red = addDisk(world, 0xff0000, [0, 0, 1]);
    expect(world.storage(SunDisk).angularSize.get(red)).toBeCloseTo(SUN_DISK_EARTH_ANGULAR_SIZE, 7);
    const redSun = await center(world, camera);
    expect(redSun[0]).toBeGreaterThan(noDisk[0]);
    expect(redSun[1]).toBe(noDisk[1]);

    world.storage(Transform).rotation.set(camera, 0, Math.SQRT1_2, 0, Math.SQRT1_2);
    const noSecondDisk = await center(world, camera);
    addDisk(world, 0x00ff00, [1, 0, 0]);
    const greenSun = await center(world, camera);
    expect(greenSun[1]).toBeGreaterThan(noSecondDisk[1]);
});
