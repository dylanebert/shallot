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
const EV100 = 5;
const linear = (value: number) => {
    const x = value / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};

// `particles`' room: carbon walls 0.4 m thick around a 6.6 × 3.2 × 6.6 m interior, a 2 × 1.5 m window in
// the back wall, and a low sun behind it. The ambient is the room's sun-only interreflection balance,
// ρΦ / (πA(1 − ρ)) = 9.2 cd/m², dark enough that light leaking through a wall would show.
test("a dolly across a floor slice boundary keeps the shaft-edge step near the unmasked grid's", async () => {
    const { world } = subjects()[0]!;
    const ambient = world.resource(GlobalAmbientLight);
    ambient.color = 0xd8e4ff;
    ambient.brightness = 9.2;
    const camera = world.create();
    world.add(camera, Transform, {
        translation: [0, 1.7, 2, 0],
        rotation: lookAtRotation(0, 1.7, 2, 0, 0, -1),
    });
    world.add(camera, Camera, { fov: 70, clearColor: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Exposure, { ev100: EV100 });
    // Default presentation tonemapping, with EV5 and dither disabled.
    world.add(camera, DebandDither, { enabled: 0 });
    attachTexture(world, camera, { width: WIDTH, height: HEIGHT });
    const carbon = world
        .resource(Materials)
        .add(StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1] }));
    for (const [translation, scale] of [
        [
            [0, -0.25, 0],
            [7, 0.5, 7],
        ],
        [
            [0, 3.45, 0],
            [7, 0.5, 7],
        ],
        [
            [0, 1.6, 3.5],
            [7, 3.2, 0.4],
        ],
        [
            [-3.5, 1.6, 0],
            [0.4, 3.2, 7],
        ],
        [
            [3.5, 1.6, 0],
            [0.4, 3.2, 7],
        ],
        [
            [-2.25, 1.6, -3.5],
            [2.5, 3.2, 0.4],
        ],
        [
            [2.25, 1.6, -3.5],
            [2.5, 3.2, 0.4],
        ],
        [
            [0, 0.65, -3.5],
            [2, 1.3, 0.4],
        ],
        [
            [0, 3, -3.5],
            [2, 0.4, 0.4],
        ],
    ] as const) {
        const block = world.create();
        world.add(block, Transform, { translation, scale });
        world.add(block, MeshInstance);
        world.add(block, MeshMaterial, carbon);
    }
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 5, -8, 0, 1, 0) });
    world.add(sun, DirectionalLight, {
        color: 0xffe8cc,
        illuminance: 24000,
        shadowMapsEnabled: 1,
        maximumDistance: 20,
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
    let largest = 0,
        largestAt: unknown = null;
    let previous: Float64Array | null = null;
    for (let step = 20; step <= 26; step++) {
        const eyeY = 1.7 - (step * 0.01 * 1.7) / Math.hypot(3, 1.7),
            eyeZ = 2 - (step * 0.01 * 3) / Math.hypot(3, 1.7);
        world.storage(Transform).translation.set(camera, 0, eyeY, eyeZ, 0);
        world.tick();
        for (let i = 0; i < 48; i++) world.step(0);
        const { rgba } = await captureTexture(world, camera);
        const image = new Float64Array(WIDTH * HEIGHT);
        for (let y = 0; y < HEIGHT; y++)
            for (let x = 0; x < WIDTH; x++) {
                const at = (y * WIDTH + x) * 4;
                const value =
                    0.2126 * linear(rgba[at]!) +
                    0.7152 * linear(rgba[at + 1]!) +
                    0.0722 * linear(rgba[at + 2]!);
                image[y * WIDTH + x] = value;
                const f = 90 / Math.tan((35 * Math.PI) / 180),
                    n = Math.hypot(3, 1.7),
                    c = 3 / n,
                    s = 1.7 / n;
                const ry = -s + ((90 - y - 0.5) / f) * c,
                    rz = -c - ((90 - y - 0.5) / f) * s,
                    t = -eyeY / ry;
                const wx = ((x + 0.5 - 160) / f) * t,
                    wz = eyeZ + rz * t;
                if (
                    previous &&
                    t > 0 &&
                    Math.abs(wx) > 1.08 &&
                    Math.abs(wx) < 1.25 &&
                    wz > -0.5 &&
                    wz < 1.5
                ) {
                    const delta = Math.abs(value - previous[y * WIDTH + x]!);
                    if (delta > largest) {
                        largest = delta;
                        largestAt = {
                            step,
                            x,
                            y,
                            wx,
                            wz,
                            from: previous[y * WIDTH + x],
                            to: value,
                        };
                    }
                }
            }
        previous = image;
    }
    console.log("floor shaft-edge dolly", { largest, largestAt });
    // The unmasked grid's 1 cm dolly peaks at 0.005662 over the full metre; allow display quantization.
    expect(largest).toBeLessThan(0.007);
});
