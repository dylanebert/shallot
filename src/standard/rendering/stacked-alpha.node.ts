import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { DEFAULT_PLUGINS } from "..";
import { AlphaMode, Materials, MeshMaterial, StandardMaterial, StandardRenderer } from ".";

setDefaultTimeout(TIMEOUT.node);

const SIZE = 32;
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [...DEFAULT_PLUGINS] }]);

function center(rgba: Uint8ClampedArray): number[] {
    const i = ((SIZE / 2) * SIZE + SIZE / 2) * 4;
    return Array.from(rgba.subarray(i, i + 4));
}

async function frame(world: World, eid: number): Promise<Uint8ClampedArray> {
    world.step(0);
    world.step(0);
    return (await captureTexture(world, eid)).rgba;
}

test("a camera stacked over another draws opaque and masked surfaces opaque whatever their material alpha", async () => {
    const { world } = subjects()[0];
    for (const [alphaMode, alpha] of [
        [AlphaMode.Opaque, 0.2],
        [AlphaMode.Mask(0.5), 0.6],
    ] as const) {
        for (const clearColor of [0x0000ff, 0x00ff00]) {
            const lower = world.create();
            world.add(lower, Transform, { translation: [0, 0, 5, 0] });
            world.add(lower, Camera, { clearColor, antialias: 0 });
            world.add(lower, Tonemapping, { method: TonemappingMethod.None });
            const target = attachTexture(world, lower, { width: SIZE, height: SIZE });
            const upper = world.create();
            world.add(upper, Transform, { translation: [0, 0, 5, 0] });
            world.add(upper, Camera, { order: 1, antialias: 0 });
            world.add(upper, StandardRenderer);
            world.add(upper, Tonemapping, { method: TonemappingMethod.None });
            attachTexture(world, upper, target);
            const cube = world.create();
            world.add(cube, Transform);
            world.add(cube, MeshInstance);
            const material = StandardMaterial({ baseColor: [1, 0, 0, alpha], unlit: true });
            world.add(cube, MeshMaterial, world.resource(Materials).add(material, { alphaMode }));

            expect(center(await frame(world, lower))).toEqual([255, 0, 0, 255]);
            for (const eid of [cube, upper, lower]) world.destroy(eid);
            world.step(0);
        }
    }
});

test("an AlphaToCoverage surface on a camera stacked over another covers it by coverage alone", async () => {
    const { world } = subjects()[0];
    const lower = world.create();
    world.add(lower, Transform, { translation: [0, 0, 5, 0] });
    world.add(lower, Camera, { clearColor: 0x0000ff });
    world.add(lower, Tonemapping, { method: TonemappingMethod.None });
    const target = attachTexture(world, lower, { width: SIZE, height: SIZE });
    const upper = world.create();
    world.add(upper, Transform, { translation: [0, 0, 5, 0] });
    world.add(upper, Camera, { order: 1 });
    world.add(upper, StandardRenderer);
    world.add(upper, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, upper, target);
    const cube = world.create();
    world.add(cube, Transform);
    world.add(cube, MeshInstance);
    const material = StandardMaterial({ baseColor: [1, 0, 0, 0.5], unlit: true });
    world.add(
        cube,
        MeshMaterial,
        world.resource(Materials).add(material, { alphaMode: AlphaMode.AlphaToCoverage }),
    );

    // Half the samples carry the red cube at full alpha, half the lower camera's blue.
    const [r, g, b, a] = center(await frame(world, lower));
    expect([g, a]).toEqual([0, 255]);
    expect(Math.abs(r! - 128)).toBeLessThanOrEqual(2);
    expect(Math.abs(r! + b! - 255)).toBeLessThanOrEqual(2);
    for (const eid of [cube, upper, lower]) world.destroy(eid);
    world.step(0);
});
