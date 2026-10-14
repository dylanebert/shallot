import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Outline, OutlinePlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

const SIZE = 48;
const BAND = [1, 0, 1, 1] as const;

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
]);

function scene(world: World) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { clearColor: 0x204060 });
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: SIZE, height: SIZE });
    const outlined = world.create();
    world.add(outlined, Transform);
    world.add(outlined, MeshInstance);
    return { camera, outlined };
}

async function frame(world: World, camera: number): Promise<Uint8ClampedArray> {
    world.step(0);
    world.step(0);
    return (await captureTexture(world, camera)).rgba;
}

const pixel = (rgba: Uint8ClampedArray, x: number, y: number) => {
    const i = (y * SIZE + x) * 4;
    return [rgba[i]!, rgba[i + 1]!, rgba[i + 2]!];
};

const same = (a: number[], b: number[]) => a.every((value, i) => Math.abs(value - b[i]!) <= 1);

/** Pixels that show the clear colour without the outline and the band colour with it. */
function bandPixels(without: Uint8ClampedArray, withOutline: Uint8ClampedArray): number {
    const clear = pixel(without, 0, 0);
    let count = 0;
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            if (!same(pixel(without, x, y), clear)) continue;
            const [r, g, b] = pixel(withOutline, x, y);
            if (r! - g! > 100 && b! - g! > 100) count++;
        }
    }
    return count;
}

function expectSceneKept(without: Uint8ClampedArray, withOutline: Uint8ClampedArray) {
    const centre = SIZE / 2;
    expect(same(pixel(withOutline, centre, centre), pixel(without, centre, centre))).toBe(true);
    expect(same(pixel(withOutline, 0, 0), pixel(without, 0, 0))).toBe(true);
    expect(same(pixel(withOutline, SIZE - 1, SIZE - 1), pixel(without, SIZE - 1, SIZE - 1))).toBe(
        true,
    );
}

test("an outlined mesh shows its band colour just outside its silhouette and keeps the scene elsewhere", async () => {
    const { world } = subjects()[0]!;
    const { camera, outlined } = scene(world);
    const without = await frame(world, camera);
    expect(same(pixel(without, SIZE / 2, SIZE / 2), pixel(without, 0, 0))).toBe(false);

    world.add(outlined, Outline, { color: BAND, width: 4 });
    const withOutline = await frame(world, camera);

    expect(bandPixels(without, withOutline)).toBeGreaterThan(0);
    expectSceneKept(without, withOutline);
});

test("an occlusion-aware outline shows its band while visible and none behind an occluder", async () => {
    const { world } = subjects()[1]!;
    const { camera, outlined } = scene(world);
    const without = await frame(world, camera);

    world.add(outlined, Outline, { color: BAND, width: 4, occlude: 1 });
    const visible = await frame(world, camera);
    expect(bandPixels(without, visible)).toBeGreaterThan(0);
    expectSceneKept(without, visible);

    const occluder = world.create();
    world.add(occluder, Transform, { translation: [0, 0, 2, 0], scale: [4, 4, 1, 1] });
    world.add(occluder, MeshInstance);
    world.remove(outlined, Outline);
    const hidden = await frame(world, camera);

    world.add(outlined, Outline, { color: BAND, width: 4, occlude: 1 });
    const occluded = await frame(world, camera);
    let difference = 0;
    for (let i = 0; i < hidden.length; i++)
        difference = Math.max(difference, Math.abs(hidden[i]! - occluded[i]!));
    expect(difference).toBeLessThanOrEqual(1);

    world.storage(Outline).occlude.set(outlined, 0);
    const onTop = await frame(world, camera);
    const centre = SIZE / 2;
    const [r, g, b] = pixel(onTop, centre, centre);
    expect(same([r!, g!, b!], pixel(hidden, centre, centre))).toBe(true);
    let band = 0;
    for (let i = 0; i < hidden.length; i += 4) {
        const [hr, hg, hb] = [onTop[i]!, onTop[i + 1]!, onTop[i + 2]!];
        if (hr - hg > 100 && hb - hg > 100 && !(hidden[i]! - hidden[i + 1]! > 100)) band++;
    }
    expect(band).toBeGreaterThan(0);
});
