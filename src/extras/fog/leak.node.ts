import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    captureTexture,
    DebandDither,
    Exposure,
    Tonemapping,
    TonemappingMethod,
    VolumetricLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
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
const EV100 = 4;
const linear = (value: number) => {
    const x = value / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};

test("haze just inside the room's walls stays dark while sunlit air lies just outside them", async () => {
    const { world } = subjects()[0]!;
    const sun = room(world);
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 1.7, 2, 0] });
    world.add(camera, Camera, { fov: 70, clearColor: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Exposure, { ev100: EV100 });
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    world.add(camera, DebandDither, { enabled: 0 });
    attachTexture(world, camera, { width: WIDTH, height: HEIGHT });
    const frame = async () => {
        world.tick();
        // Every Halton phase twice, so the history holds the jittered samples a leak comes from.
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
    // Strips across the back wall's top edge and both corners, clear of the window and its shaft.
    let edge = 0;
    for (let x = 70; x < 250; x++)
        for (let y = 46; y < 62; y++) edge = Math.max(edge, sunlight(x, y));
    for (const left of [68, 232])
        for (let x = left; x < left + 20; x++)
            for (let y = 62; y < 130; y++) edge = Math.max(edge, sunlight(x, y));
    const wall = unlit(100, 90);
    const shaft = sunlight(160, 120);
    console.log("room leak", { edge, wall, shaft });
    expect(shaft).toBeGreaterThan(20 * wall);
    expect(edge).toBeLessThan(0.1 * wall);
});
