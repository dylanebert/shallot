import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
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
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);

function encode(linear: number): number {
    return Math.round(
        255 * (linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055),
    );
}

test("a sub-160×90 view caps its fog grid to its pixels and matches Beer-Lambert inside and beyond the range", async () => {
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
    const grids: GPUTexture[] = [];
    const device = world.gpu.device;
    const createTexture = device.createTexture.bind(device);
    device.createTexture = (descriptor) => {
        const texture = createTexture(descriptor);
        if (descriptor.label === `fog-grid/${camera}`) grids.push(texture);
        return texture;
    };
    let atRange: number;
    try {
        atRange = await pixel();
        expect(grids.at(-1)!.width).toBe(16);
        expect(grids.at(-1)!.height).toBe(16);
    } finally {
        device.createTexture = createTexture;
    }
    expect(Math.abs(atRange - encode(1 - Math.exp(-0.02 * (20 - 0.1))))).toBeLessThanOrEqual(1);
    world.storage(Camera).far.set(camera, 40);
    const beyond = await pixel();
    expect(Math.abs(beyond - encode(1 - Math.exp(-0.02 * (40 - 0.1))))).toBeLessThanOrEqual(1);
    world.storage(Fog).viewDistance.set(fog, 64);
    const atSurface = await pixel();
    expect(Math.abs(atSurface - encode(1 - Math.exp(-0.02 * (40 - 0.1))))).toBeLessThanOrEqual(1);
    expect(atSurface).toBe(beyond);
    expect(atSurface).toBeGreaterThan(atRange + 30);
    world.storage(Fog).density.set(fog, 0);
    expect(await pixel()).toBe(0);
    world.destroy(camera);
    world.destroy(fog);
});

test("clear sky ends at maxDistance and farther surfaces take the same ray-length haze", async () => {
    const { world } = subjects()[1]!;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { near: 0.001, far: 100, clearColor: 0, antialias: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 32, height: 32 });
    const surface = world.create();
    world.add(surface, Transform, { translation: [0, 0, -60, 0], scale: [20, 20, 1, 0] });
    world.add(surface, MeshInstance);
    world.add(
        surface,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: [0, 0, 0, 1] })),
    );
    const fog = world.create();
    world.add(fog, Fog, {
        color: 0xffffff,
        density: 0.02,
        absorption: 1,
        scattering: 0,
        viewDistance: 8,
    });
    world.storage(Fog).maxDistance.set(fog, 30);
    for (const distance of [30, 8, 4, 0, -1]) {
        if (distance !== 30) world.storage(Fog).maxDistance.set(fog, distance);
        world.step(0);
        world.step(0);
        const pixels = (await captureTexture(world, camera)).rgba;
        const expected = encode(1 - Math.exp(-0.02 * Math.max(0, distance)));
        const background = pixels[0]!;
        const beyond = pixels[(16 * 32 + 16) * 4]!;
        expect(Math.abs(background - expected)).toBeLessThanOrEqual(1);
        expect(Math.abs(beyond - expected)).toBeLessThanOrEqual(1);
        expect(Math.abs(beyond - background)).toBeLessThanOrEqual(1);
    }
    world.destroy(camera);
    world.destroy(surface);
    world.destroy(fog);
});

test("height fog crosses viewDistance on a long ground plane without a luminance or in-scatter seam", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 2, 0, 0] });
    world.add(camera, Camera, { near: 0.1, far: 150, fov: 60, clearColor: 0 });
    world.add(camera, AmbientLight, { color: 0x0000ff, brightness: 160 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 128, height: 128 });
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -0.5, -75, 0], scale: [300, 1, 300, 0] });
    world.add(ground, MeshInstance);
    world.add(
        ground,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ baseColor: [0, 0, 0, 1] })),
    );
    const fog = world.create();
    world.add(fog, Fog, {
        color: 0xff0000,
        density: 0.06,
        heightFalloff: 0.3,
        absorption: 0.4,
        scattering: 0.6,
        lightIntensity: 20,
        viewDistance: 128,
    });
    const frame = async () => {
        world.step(0);
        world.step(0);
        return (await captureTexture(world, camera)).rgba;
    };
    // A fully numerical column is the independent reference for the analytic continuation.
    // Red isolates transmittance; blue isolates nonzero ambient in-scatter.
    const numerical = await frame();
    world.storage(Fog).viewDistance.set(fog, 8);
    const continued = await frame();
    for (let y = 72; y < 118; y++) {
        const at = (y * 128 + 64) * 4;
        for (const channel of [0, 2]) {
            expect(numerical[at + channel]!).toBeGreaterThan(30);
            expect(
                Math.abs(continued[at + channel]! - numerical[at + channel]!),
            ).toBeLessThanOrEqual(2);
        }
        if (y > 72) {
            const luminance = (pixels: Uint8ClampedArray, row: number) => {
                const i = (row * 128 + 64) * 4;
                return 0.2126 * pixels[i]! + 0.0722 * pixels[i + 2]!;
            };
            const slope = (pixels: Uint8ClampedArray) =>
                luminance(pixels, y) - luminance(pixels, y - 1);
            expect(Math.abs(slope(continued) - slope(numerical))).toBeLessThan(0.5);
        }
    }
    world.destroy(camera);
    world.destroy(ground);
    world.destroy(fog);
});
