import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
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
import { Fog, FogPlugin, NoFog } from ".";

setDefaultTimeout(TIMEOUT.node);

const config = { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] };
const stacked = gpuScene(import.meta.path, config, (world) => stackedScene(world, 1));

test("fog on a camera stacked over another reaches the lower image only through that camera's coverage", async () => {
    const {
        world,
        scene: { lower, upper, cube },
    } = stacked();
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

const coverageScene = gpuScene(import.meta.path, config, (world) => {
    // Four lone cameras see identical fog in one frame, each over its own cube far from the others:
    // white and black opaque cubes measure in-scatter F and transmittance T at the cube's depth, and an
    // AlphaToCoverage cube, seen with and without fog, must read F + D·T from its unfogged color D.
    const cameras: number[] = [];
    const entities: number[] = [];
    const materials = world.resource(Materials);
    const looks = [
        { color: [1, 1, 1, 1], mode: AlphaMode.Opaque, fogged: true },
        { color: [0, 0, 0, 1], mode: AlphaMode.Opaque, fogged: true },
        { color: [1, 0, 0, 0.5], mode: AlphaMode.AlphaToCoverage, fogged: true },
        { color: [1, 0, 0, 0.5], mode: AlphaMode.AlphaToCoverage, fogged: false },
    ] as const;
    for (const [k, look] of looks.entries()) {
        const x = k * 100;
        const camera = world.create();
        world.add(camera, Transform, { translation: [x, 0, 5, 0] });
        world.add(camera, Camera, { clearColor: 0x204060 });
        world.add(camera, StandardRenderer);
        if (!look.fogged) world.add(camera, NoFog);
        attachTexture(world, camera, { width: SIZE, height: SIZE });
        cameras.push(camera);
        const cube = world.create();
        world.add(cube, Transform, { translation: [x, 0, 0, 0] });
        world.add(cube, MeshInstance);
        world.add(
            cube,
            MeshMaterial,
            materials.add(StandardMaterial({ baseColor: [...look.color], unlit: true }), {
                alphaMode: look.mode,
            }),
        );
        entities.push(cube);
    }
    const fog = world.create();
    world.add(fog, Fog, { density: 0.1, color: 0xffffff });
    entities.push(fog);
    return { cameras, entities };
});

test("a bottom camera's in-scatter ignores its scene alpha, as before stacking, under hardware AlphaToCoverage", async () => {
    const {
        world,
        scene: { cameras, entities },
    } = coverageScene();
    world.gpu.device.pushErrorScope("validation");
    for (let i = 0; i < 4; i++) world.step(0);
    expect((await world.gpu.device.popErrorScope())?.message ?? null).toBeNull();
    const [white, black, coverage, unfogged] = await Promise.all(
        cameras.map(async (camera) => {
            const scene = await probeTexture(world, offscreenTexture(world, camera)!);
            return new Float32Array(Array.from(new Uint16Array(scene.bytes as ArrayBuffer), half));
        }),
    );
    const off: string[] = [];
    let partial = 0;
    for (let y = SIZE / 2 - 2; y < SIZE / 2 + 2; y++) {
        for (let x = SIZE / 2 - 2; x < SIZE / 2 + 2; x++) {
            const at = (y * SIZE + x) * 4;
            if (unfogged![at + 3]! < 1) partial++;
            for (let c = 0; c < 3; c++) {
                const inScatter = black![at + c]!;
                const transmittance = white![at + c]! - inScatter;
                const expected = inScatter + unfogged![at + c]! * transmittance;
                if (Math.abs(coverage![at + c]! - expected) > 0.01 + 0.01 * expected)
                    off.push(`${x},${y}[${c}]: ${coverage![at + c]} != ${expected}`);
            }
        }
    }
    // Hardware alphaToCoverage stores material alpha in the samples it covers.
    expect(partial).toBeGreaterThan(0);
    expect(black![((SIZE / 2) * SIZE + SIZE / 2) * 4]!).toBeGreaterThan(0.05);
    expect(off).toEqual([]);
    for (const eid of [...entities, ...cameras]) world.destroy(eid);
    world.step(0);
});

function half(bits: number): number {
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x3ff;
    const sign = bits & 0x8000 ? -1 : 1;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
