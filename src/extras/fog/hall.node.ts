import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DebandDither,
    DirectionalLight,
    Exposure,
    GlobalAmbientLight,
    Tonemapping,
    TonemappingMethod,
    VolumetricLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Fog, FogPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);
const WIDTH = 320;
const HEIGHT = 180;
const EV100 = 6;
const linear = (value: number) => {
    const x = value / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};

// volumetric-fog's first hall: 0.4 m walls around an 8.6 × 5 × 23.6 m interior, a 4 × 3 m window in the far
// wall and a high sun behind it, seen from the hall's other end. The ceiling and side walls meet the far wall
// at a grazing angle, so a column there sees its own wall metres short of the far wall its neighbours see, and
// the cells between lie in the sunlit air outside the hall.
test("haze along the far wall's seams stays dark while the sun lights air outside them", async () => {
    const { world } = subjects()[0]!;
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 12;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 1.7, 11, 0] });
    world.add(camera, Camera, { fov: 70, clearColor: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Exposure, { ev100: EV100 });
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    world.add(camera, DebandDither, { enabled: 0 });
    attachTexture(world, camera, { width: WIDTH, height: HEIGHT });
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.15, 0.14, 0.125, 1] }));
    for (const [translation, scale] of [
        [
            [0, -0.25, 0],
            [9, 0.5, 24],
        ],
        [
            [0, 5.25, 0],
            [9, 0.5, 24],
        ],
        [
            [-4.5, 2.5, 0],
            [0.4, 5, 24],
        ],
        [
            [4.5, 2.5, 0],
            [0.4, 5, 24],
        ],
        [
            [0, 2.5, 12],
            [9, 5, 0.4],
        ],
        [
            [-3.25, 2.5, -12],
            [2.5, 5, 0.4],
        ],
        [
            [3.25, 2.5, -12],
            [2.5, 5, 0.4],
        ],
        [
            [0, 0.75, -12],
            [4, 1.5, 0.4],
        ],
        [
            [0, 4.75, -12],
            [4, 0.5, 0.4],
        ],
    ] as const) {
        const block = world.create();
        world.add(block, Transform, { translation, scale });
        world.add(block, MeshInstance);
        world.add(block, MeshMaterial, material);
    }
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 8, -12, 0, 0, 0) });
    world.add(sun, DirectionalLight, {
        color: 0xffe8cc,
        illuminance: 70000,
        shadowMapsEnabled: 1,
        maximumDistance: 45,
    });
    world.add(sun, VolumetricLight);
    world.add(world.create(), Fog, {
        brightness: 0,
        density: 0.025,
        absorption: 0,
        scattering: 1,
        lightIntensity: 1 / Math.PI,
        scatteringAsymmetry: 0.4,
        heightFalloff: 0,
    });
    const frame = async () => {
        world.tick();
        // Every Halton phase twice, so history holds each column's jittered samples.
        for (let i = 0; i < 48; i++) world.step(0);
        const { rgba } = await captureTexture(world, camera);
        return (x: number, y: number) => {
            const at = (y * WIDTH + x) * 4;
            const value =
                0.2126 * linear(rgba[at]!) +
                0.7152 * linear(rgba[at + 1]!) +
                0.0722 * linear(rgba[at + 2]!);
            return value * 1.2 * 2 ** EV100;
        };
    };
    const lit = await frame();
    world.remove(sun, VolumetricLight);
    const unlit = await frame();
    const sunlight = (x: number, y: number) => lit(x, y) - unlit(x, y);
    // Strips up both side walls where they meet the far wall, clear of the window's shaft and its blur.
    let seam = 0;
    for (const [left, right] of [
        [120, 137],
        [188, 201],
    ] as const)
        for (let x = left; x < right; x++)
            for (let y = 62; y < 110; y++) seam = Math.max(seam, sunlight(x, y));
    const wall = unlit(130, 80);
    const shaft = sunlight(160, 90);
    console.log("hall seams", { seam, wall, shaft });
    expect(shaft).toBeGreaterThan(20 * wall);
    expect(seam).toBeLessThan(0.1 * wall);
});
