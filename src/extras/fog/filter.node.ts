import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    captureTexture,
    DebandDither,
    Exposure,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { FogPlugin } from ".";
import { room } from "./room.fixture";

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

test("a dolly across a floor slice boundary keeps the shaft-edge step near the unmasked grid's", async () => {
    const { world } = subjects()[0]!;
    room(world);
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
    let largest = 0,
        largestAt: unknown = null;
    let previous: Float64Array | null = null;
    for (let step = 20; step <= 26; step++) {
        const eyeY = 1.7 - (step * 0.01 * 1.7) / Math.hypot(3, 1.7),
            eyeZ = 2 - (step * 0.01 * 3) / Math.hypot(3, 1.7);
        world.storage(Transform).translation.set(camera, 0, eyeY, eyeZ, 0);
        world.tick();
        for (let i = 0; i < 32; i++) world.step(0);
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
    // The unmasked grid's 1 cm dolly peaks at 0.005662 over the full meter; allow display quantization.
    expect(largest).toBeLessThan(0.007);
});
