import { afterEach, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DirectionalLight,
    PointLight,
    SpotLight,
    VolumetricLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Fog, FogPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);
const SIZE = 48;
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);
const entities: number[] = [];
function entity() {
    const eid = subjects()[0]!.world.create();
    entities.push(eid);
    return eid;
}
afterEach(() => {
    for (const eid of entities) subjects()[0]!.world.destroy(eid);
    entities.length = 0;
});
function scene(scatteringAsymmetry = 0) {
    const { world } = subjects()[0]!;
    const camera = entity();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    // Orthographic clear depth gives every ray a finite, equal march; black isolates in-scatter.
    world.add(camera, Camera, {
        mode: CameraMode.Orthographic,
        size: 10,
        near: 0.1,
        far: 10,
        clearColor: 0,
    });
    world.add(camera, StandardRenderer);
    world.add(camera, AmbientLight, { brightness: 0 });
    attachTexture(world, camera, { width: SIZE, height: SIZE });
    world.add(entity(), Fog, {
        color: 0,
        density: 0.15,
        absorption: 0,
        scattering: 1,
        lightIntensity: 10 / Math.PI,
        scatteringAsymmetry,
        stepCount: 64,
        jitter: 0,
    });
    return camera;
}
async function frame(camera: number) {
    const { world } = subjects()[0]!;
    world.tick();
    world.step(1 / 60);
    world.step(1 / 60);
    return (await captureTexture(world, camera)).rgba;
}
function patch(rgba: Uint8ClampedArray, x = 24, y = 24) {
    let sum = 0;
    for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) sum += rgba[((y + dy) * SIZE + x + dx) * 4]!;
    }
    return sum / 9;
}

test("a directional light's shadow makes haze darker than adjacent lit haze", async () => {
    const { world } = subjects()[0]!;
    const camera = scene();
    const sun = entity();
    world.add(sun, Transform, { rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2] });
    world.add(sun, DirectionalLight, {
        illuminance: 1000,
        shadowMapsEnabled: 1,
        maximumDistance: 20,
    });
    world.add(sun, VolumetricLight);
    const roof = entity();
    world.add(roof, Transform, { translation: [-2, 2, 0, 0], scale: [3, 0.2, 24, 0] });
    world.add(roof, MeshInstance);
    const shadowed = await frame(camera);
    const shade = patch(shadowed, 19);
    const lit = patch(shadowed, 34);
    expect(lit).toBeGreaterThan(shade + 15);
    world.remove(roof, MeshInstance);
    const unobstructed = await frame(camera);
    expect(patch(unobstructed, 19)).toBeGreaterThan(shade + 15);
    expect(Math.abs(patch(unobstructed, 19) - patch(unobstructed, 34))).toBeLessThanOrEqual(2);
});

test("forward anisotropy brightens haze toward a directional light rather than away", async () => {
    const { world } = subjects()[0]!;
    const camera = scene(0.6);
    const sun = entity();
    world.add(sun, Transform, { rotation: [0, 1, 0, 0] });
    world.add(sun, DirectionalLight, { illuminance: 1000 });
    world.add(sun, VolumetricLight);
    const toward = patch(await frame(camera));
    world.storage(Transform).rotation.set(sun, 0, 0, 0, 1);
    const away = patch(await frame(camera));
    expect(toward).toBeGreaterThan(away + 30);
});

for (const kind of ["point", "spot"] as const) {
    test(`${kind} VolumetricLight lights haze only within its range${kind === "spot" ? " and cone" : ""}`, async () => {
        const { world } = subjects()[0]!;
        const camera = scene();
        const light = entity();
        world.add(light, Transform, { translation: [0, 0, 2, 0] });
        if (kind === "point") world.add(light, PointLight, { intensity: 10000, range: 2 });
        else
            world.add(light, SpotLight, {
                intensity: 100000,
                range: 4,
                innerAngle: 0.2,
                outerAngle: 0.4,
            });
        const unmarked = await frame(camera);
        expect(patch(unmarked)).toBe(0);
        world.add(light, VolumetricLight);
        const marked = await frame(camera);
        expect(patch(marked)).toBeGreaterThan(patch(unmarked) + 20);
        expect(patch(marked, 45)).toBe(0);
        if (kind === "spot") expect(patch(marked, 29)).toBe(0);
        world.storage(Transform).translation.set(light, 0, 0, 10, 0);
        expect(patch(await frame(camera))).toBe(0);
        world.remove(light, VolumetricLight);
        world.storage(Transform).translation.set(light, 0, 0, 2, 0);
        expect(patch(await frame(camera))).toBe(0);
    });
}

test("ambient in-scatter increases with the camera's AmbientLight brightness", async () => {
    const { world } = subjects()[0]!;
    const camera = scene();
    const dark = patch(await frame(camera));
    world.storage(AmbientLight).brightness.set(camera, 40);
    const dim = patch(await frame(camera));
    world.storage(AmbientLight).brightness.set(camera, 160);
    const bright = patch(await frame(camera));
    expect(dark).toBe(0);
    expect(dim).toBeGreaterThan(dark + 10);
    expect(bright).toBeGreaterThan(dim + 10);
});
