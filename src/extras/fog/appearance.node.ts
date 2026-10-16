import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { AmbientLight, attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Fog, FogPlugin, NoFog } from ".";

setDefaultTimeout(TIMEOUT.node);

const SIZE = 48;
const HAZE = [1, 0, 0] as const;
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);

function camera(world: World, noFog = false): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, 5, 0] });
    world.add(eid, Camera, { clearColor: 0x204060 });
    world.add(eid, AmbientLight, { brightness: 199.61915 });
    world.add(eid, StandardRenderer);
    if (noFog) world.add(eid, NoFog);
    attachTexture(world, eid, { width: SIZE, height: SIZE });
    return eid;
}

function mesh(
    world: World,
    translation: [number, number, number, number],
    scale: [number, number, number, number],
    color: [number, number, number, number],
): void {
    const eid = world.create();
    world.add(eid, Transform, { translation, scale });
    world.add(eid, MeshInstance);
    world.add(
        eid,
        MeshMaterial,
        world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: color, perceptualRoughness: 1 })),
    );
}

async function frame(world: World, eid: number) {
    return (await captureTexture(world, eid)).rgba;
}

function pixel(rgba: Uint8ClampedArray, x: number, y: number): number[] {
    const at = (y * SIZE + x) * 4;
    return [rgba[at]!, rgba[at + 1]!, rgba[at + 2]!];
}

function maxDifference(a: number[], b: number[]): number {
    return Math.max(...a.map((value, i) => Math.abs(value - b[i]!)));
}

function hazeDistance(color: number[]): number {
    return (
        (color[0]! - HAZE[0] * 255) ** 2 +
        (color[1]! - HAZE[1] * 255) ** 2 +
        (color[2]! - HAZE[2] * 255) ** 2
    );
}

test("Fog tints a distant surface, preserves a near surface, and leaves NoFog unchanged", async () => {
    const { world } = subjects()[0]!;
    const fogCamera = camera(world);
    const noFogCamera = camera(world, true);
    mesh(world, [0, 0, -15, 0], [30, 30, 0.2, 0], [0.1, 0.8, 0.2, 1]);
    mesh(world, [0, 0, 4, 0], [0.3, 0.3, 0.3, 0], [0.1, 0.2, 0.8, 1]);

    world.step(0);
    world.step(0);
    const nearBefore = await frame(world, fogCamera);
    const noFogBefore = await frame(world, noFogCamera);

    const fog = world.create();
    world.add(fog, Fog, {
        density: 0.01,
        color: 0xff0000,
        absorption: 1,
        scattering: 0,
        stepCount: 32,
        jitter: 0,
    });
    world.step(0);
    world.step(0);
    const fogged = await frame(world, fogCamera);
    const noFogAfter = await frame(world, noFogCamera);

    const near = pixel(fogged, SIZE / 2, SIZE / 2);
    const nearReference = pixel(nearBefore, SIZE / 2, SIZE / 2);
    const far = pixel(fogged, 4, 4);
    const farReference = pixel(nearBefore, 4, 4);
    expect(maxDifference(near, nearReference)).toBeLessThanOrEqual(16);

    expect(far[0]!).toBeGreaterThan(farReference[0]! + 40);
    expect(hazeDistance(far)).toBeLessThan(hazeDistance(farReference));
    expect(maxDifference(pixel(noFogAfter, 4, 4), pixel(noFogBefore, 4, 4))).toBe(0);
    expect(
        maxDifference(
            pixel(noFogAfter, SIZE / 2, SIZE / 2),
            pixel(noFogBefore, SIZE / 2, SIZE / 2),
        ),
    ).toBe(0);
    expect(far[0]!).toBeGreaterThan(far[1]!);
});
