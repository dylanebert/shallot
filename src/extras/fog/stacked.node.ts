import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera } from "../../core/rendering";
import { offscreenTexture } from "../../core/rendering/view";
import { Transform } from "../../core/transform";
import { probeTexture } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { corners, LOWER, pixel, SIZE, stackedFrame, stackedScene } from "../stacked-camera.fixture";
import { Fog, FogPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

// 1.0 as an IEEE half, the scene target's rgba16float alpha.
const HALF_ONE = 0x3c00;

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);

test("fog on a camera stacked over another reaches the lower image only through that camera's coverage", async () => {
    const { world } = subjects()[0];
    const { lower, upper, cube } = stackedScene(world, 1);
    const clear = pixel(await stackedFrame(world, lower), SIZE / 2, SIZE / 2);
    const fog = world.create();
    world.add(fog, Fog, { density: 0.1, color: 0xffffff });
    const rgba = await stackedFrame(world, lower);
    expect(new Set(corners(rgba).map(String))).toEqual(new Set([String(LOWER)]));
    // A partly covered edge pixel is the fogged cube blended over the lower clear by one coverage, so
    // every channel gives the same blend weight, within dither.
    const surface = pixel(rgba, SIZE / 2, SIZE / 2);
    expect(surface[2]!).toBeGreaterThan(clear[2]! + 20);
    const near = (a: number[], b: number[]) =>
        a.every((value, c) => c > 2 || Math.abs(value - b[c]!) <= 2);
    let edges = 0;
    const outside: string[] = [];
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const value = pixel(rgba, x, y);
            if (near(value, LOWER) || near(value, surface)) continue;
            edges++;
            const weights = [0, 1, 2].map(
                (c) => (value[c]! - LOWER[c]!) / (surface[c]! - LOWER[c]!),
            );
            if (Math.max(...weights) - Math.min(...weights) > 0.1 || Math.min(...weights) < 0)
                outside.push(`${x},${y}: ${value} weights ${weights.map((w) => w.toFixed(2))}`);
        }
    }
    expect(edges).toBeGreaterThan(8);
    expect(outside).toEqual([]);
    for (const eid of [fog, cube, upper, lower]) world.destroy(eid);
    world.step(0);
});

test("a bottom camera's scene alpha stays 1 under AlphaToCoverage, so fog's coverage factor leaves its in-scatter whole", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { clearColor: 0x204060 });
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: SIZE, height: SIZE });
    const cube = world.create();
    world.add(cube, Transform);
    world.add(cube, MeshInstance);
    world.add(
        cube,
        MeshMaterial,
        world
            .resource(Materials)
            .add(StandardMaterial({ baseColor: [1, 0, 0, 0.5], unlit: true }), {
                alphaMode: AlphaMode.AlphaToCoverage,
            }),
    );
    const fog = world.create();
    world.add(fog, Fog, { density: 0.1, color: 0xffffff });
    await stackedFrame(world, camera);
    const scene = await probeTexture(world, offscreenTexture(world, camera)!);
    const halves = new Uint16Array(scene.bytes as ArrayBuffer);
    const alphas = new Set<number>();
    for (let i = 3; i < halves.length; i += 4) alphas.add(halves[i]!);
    expect([...alphas]).toEqual([HALF_ONE]);
    for (const eid of [cube, fog, camera]) world.destroy(eid);
    world.step(0);
});
