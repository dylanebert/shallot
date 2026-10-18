import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
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
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Fog, FogPlugin, NoFog } from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);
const encode = (x: number) =>
    Math.round(255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055));

test("Blend, Premultiplied, Add and Multiply apply fog at their own depth and NoFog leaves transparents unchanged", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { mode: CameraMode.Orthographic, near: 0.1, far: 30, antialias: 0 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 16, height: 16 });
    const materials = world.resource(Materials);
    const wall = world.create();
    world.add(wall, Transform, { translation: [0, 0, -20.5, 0], scale: [40, 40, 1, 0] });
    world.add(wall, MeshInstance);
    const wallRow = materials.add(StandardMaterial({ baseColor: [0, 0, 0, 1], unlit: true }));
    world.add(wall, MeshMaterial, wallRow);
    const surface = world.create();
    world.add(surface, Transform, { translation: [0, 0, -4.5, 0], scale: [20, 20, 1, 0] });
    world.add(surface, MeshInstance);
    const row = materials.add(StandardMaterial({ baseColor: [0, 0, 0, 0.5], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    world.add(surface, MeshMaterial, row);
    const fog = world.create();
    world.add(fog, Fog, {
        color: 0xffffff,
        density: 0.05,
        absorption: 1,
        scattering: 0,
        viewDistance: 32,
    });
    const pixel = async () => {
        world.step(0);
        world.step(0);
        return (await captureTexture(world, camera)).rgba[(8 * 16 + 8) * 4]!;
    };
    const nearT = Math.exp(-0.05 * (4 - 0.1));
    const wallT = Math.exp(-0.05 * (20 - 0.1));
    const actual = await pixel();
    const expected = encode(0.5 * (1 - nearT) + 0.5 * (1 - wallT));
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2);
    expect(Math.abs(actual - encode(1 - wallT))).toBeGreaterThan(20);
    world.add(camera, NoFog);
    expect(await pixel()).toBe(0);
    world.remove(camera, NoFog);
    materials.update(row, { baseColor: [0.4, 0, 0, 0.5] }, { alphaMode: AlphaMode.Add });
    expect(Math.abs((await pixel()) - encode(1 - wallT + 0.2 * nearT))).toBeLessThanOrEqual(2);
    world.add(camera, NoFog);
    expect(Math.abs((await pixel()) - encode(0.2))).toBeLessThanOrEqual(1);
    world.remove(camera, NoFog);
    materials.update(row, { baseColor: [0.2, 0, 0, 0.5] }, { alphaMode: AlphaMode.Premultiplied });
    const premultiplied = await pixel();
    const premultipliedExpected = 0.2 * nearT + 0.5 * (1 - nearT) + 0.5 * (1 - wallT);
    expect(Math.abs(premultiplied - encode(premultipliedExpected))).toBeLessThanOrEqual(2);
    expect(
        Math.abs(premultiplied - encode(premultipliedExpected + 0.5 * (1 - nearT))),
    ).toBeGreaterThan(8);

    world.storage(Fog).color.set(fog, 0x0000ff);
    materials.update(wallRow, { baseColor: [1, 1, 1, 1] });
    materials.update(row, {}, { alphaMode: AlphaMode.Multiply });
    const multiplied = await pixel();
    const multipliedExpected = wallT * (0.5 * (1 - nearT + 0.2 * nearT) + 0.5);
    expect(Math.abs(multiplied - encode(multipliedExpected))).toBeLessThanOrEqual(2);
    expect(Math.abs(multiplied - encode(wallT * (0.5 * 0.2 * nearT + 0.5)))).toBeGreaterThan(5);
});
