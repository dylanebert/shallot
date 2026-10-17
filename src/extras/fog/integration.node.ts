import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Fog, FogPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);

function encode(linear: number): number {
    return Math.round(
        255 * (linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055),
    );
}

test("uniform extinction matches Beer-Lambert at the surface and clamps beyond the grid range", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { mode: CameraMode.Orthographic, near: 0.1, far: 20, clearColor: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 16, height: 16 });
    const fog = world.create();
    world.add(fog, Fog, {
        color: 0xffffff,
        density: 0.02,
        absorption: 1,
        scattering: 0,
        viewDistance: 8,
    });
    const pixel = async () => {
        world.step(0);
        world.step(0);
        return (await captureTexture(world, camera)).rgba[4 * (8 * 16 + 8)]!;
    };
    const atRange = await pixel();
    expect(Math.abs(atRange - encode(1 - Math.exp(-0.02 * (8 - 0.1))))).toBeLessThanOrEqual(1);
    world.storage(Camera).far.set(camera, 40);
    expect(await pixel()).toBe(atRange);
    world.storage(Fog).viewDistance.set(fog, 64);
    const atSurface = await pixel();
    expect(Math.abs(atSurface - encode(1 - Math.exp(-0.02 * (40 - 0.1))))).toBeLessThanOrEqual(1);
    expect(atSurface).toBeGreaterThan(atRange + 50);
    world.storage(Fog).density.set(fog, 0);
    expect(await pixel()).toBe(0);
});
