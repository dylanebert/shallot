import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    EffectPasses,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { ImageBindings } from "../../core/rendering/image";
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
import { ChromaticAberration, EffectStackPlugin, LensDistortion, Vignette } from ".";

setDefaultTimeout(TIMEOUT.node);

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, EffectStackPlugin] },
    { defaults: false, plugins: [...DEFAULT_PLUGINS, EffectStackPlugin] },
]);

test("a vignette on a camera stacked over another shades only what that camera drew", async () => {
    const { world } = subjects()[0];
    const { lower, upper, cube } = stackedScene(world, 1);
    const plain = pixel(await stackedFrame(world, lower), SIZE / 2, SIZE / 2 + 6);
    world.add(upper, Vignette, { radius: 0.2 });
    const rgba = await stackedFrame(world, lower);
    expect(new Set(corners(rgba).map(String))).toEqual(new Set([String(LOWER)]));
    const shaded = pixel(rgba, SIZE / 2, SIZE / 2 + 6);
    expect(shaded[0]!).toBeLessThan(plain[0]! - 20);
    for (const eid of [cube, upper, lower]) world.destroy(eid);
    world.step(0);
});

for (const effect of [LensDistortion, ChromaticAberration, Vignette]) {
    test(`${effect === LensDistortion ? "lens distortion" : effect === ChromaticAberration ? "chromatic aberration" : "vignette"} defaults change the frame, zero is identity, removal unregisters the single pass`, async () => {
        const { world } = subjects()[0];
        const { lower, upper, cube } = stackedScene(world, 1, 512);
        const plain = await stackedFrame(world, lower);
        expect(world.resource(EffectPasses).get(upper)?.before.length ?? 0).toBe(0);
        world.add(upper, effect);
        const changed = await stackedFrame(world, lower);
        expect(changed).not.toEqual(plain);
        expect(world.resource(EffectPasses).get(upper)?.before.length).toBe(1);
        // The clear lower camera is visible wherever the upper drew no geometry.
        for (const [x, y] of [
            [0, 0],
            [511, 0],
            [0, 511],
            [511, 511],
        ]) {
            const at = (y! * 512 + x!) * 4;
            expect(Array.from(changed.subarray(at, at + 4))).toEqual(LOWER);
        }
        world.storage(effect).intensity.set(upper, 0);
        expect(await stackedFrame(world, lower)).toEqual(plain);
        expect(world.resource(EffectPasses).get(upper)?.before.length ?? 0).toBe(0);
        world.remove(upper, effect);
        expect(await stackedFrame(world, lower)).toEqual(plain);
        expect(world.resource(EffectPasses).get(upper)?.before.length ?? 0).toBe(0);
        for (const eid of [cube, upper, lower]) world.destroy(eid);
        world.step(0);
    });
}

test("all three effects share one pass and detach it when the last component leaves", async () => {
    const { world } = subjects()[0];
    const { lower, upper, cube } = stackedScene(world, 1);
    for (const effect of [LensDistortion, ChromaticAberration, Vignette]) world.add(upper, effect);
    await stackedFrame(world, lower);
    expect(world.resource(EffectPasses).get(upper)?.before.length).toBe(1);
    for (const effect of [LensDistortion, ChromaticAberration, Vignette])
        world.remove(upper, effect);
    await stackedFrame(world, lower);
    expect(world.resource(EffectPasses).get(upper)?.before.length ?? 0).toBe(0);
    for (const eid of [cube, upper, lower]) world.destroy(eid);
    world.step(0);
});

test("a custom chromatic LUT changes the fringes and unsetting it restores the default lookup", async () => {
    const { world } = subjects()[0];
    const { lower, upper, cube } = stackedScene(world, 1, 512);
    world.add(upper, ChromaticAberration);
    const baseline = await stackedFrame(world, lower);
    const bindings = world.resource(ImageBindings);
    const original = bindings.images;
    const lut = world.gpu.device.createTexture({
        size: [3, 1, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    world.own(lut);
    world.gpu.device.queue.writeTexture(
        { texture: lut },
        new Uint8Array([0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255]),
        { bytesPerRow: 12 },
        [3, 1],
    );
    bindings.images = lut;
    try {
        world.storage(ChromaticAberration).colorLut.set(upper, 0);
        expect(await stackedFrame(world, lower)).not.toEqual(baseline);
        world.storage(ChromaticAberration).colorLut.set(upper, -1);
        expect(await stackedFrame(world, lower)).toEqual(baseline);
    } finally {
        bindings.images = original;
        for (const eid of [cube, upper, lower]) world.destroy(eid);
        world.step(0);
    }
});

const RADIUS = 0.6;
const SMOOTHNESS = 2;

/** Bevy's vignette on a square view with default roundness, centre and edge compensation. */
function vignetted(scene: number, x: number, y: number): number {
    const u = (x + 0.5) / SIZE - 0.5;
    const v = (y + 0.5) / SIZE - 0.5;
    const distance = Math.hypot(u, v) / RADIUS;
    const factor = Math.max(0, 1 - distance * distance) ** SMOOTHNESS;
    return scene + (1 - scene) * (1 - factor);
}

function srgb(linear: number): number {
    const c = Math.max(linear, 0);
    const encoded = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
    return Math.round(Math.min(encoded, 1) * 255);
}

function half(bits: number): number {
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x3ff;
    if (exponent === 0) return 2 ** -14 * (fraction / 1024);
    return 2 ** (exponent - 15) * (1 + fraction / 1024);
}

test("a lone or bottom camera's white vignette mixes its scene by the unscaled formula where AlphaToCoverage leaves alpha below 1", async () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera, { clearColor: 0x204060 });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    world.add(camera, Vignette, { radius: RADIUS, smoothness: SMOOTHNESS, color: 0xffffff });
    const target = attachTexture(world, camera, { width: SIZE, height: SIZE });
    // A cube filling the view at alpha 0.5: hardware alpha-to-coverage leaves scene alpha below 1.
    const cube = world.create();
    world.add(cube, Transform, { translation: [0, 0, 2, 0], scale: [8, 8, 1, 0] });
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

    async function check(): Promise<void> {
        const rgba = await stackedFrame(world, camera);
        const scene = new Uint16Array(
            (await probeTexture(world, offscreenTexture(world, camera)!)).bytes as ArrayBuffer,
        );
        let partial = 0;
        const off: string[] = [];
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE; x++) {
                const at = (y * SIZE + x) * 4;
                if (half(scene[at + 3]!) < 0.99) partial++;
                for (let c = 0; c < 3; c++) {
                    const expected = srgb(vignetted(half(scene[at + c]!), x, y));
                    if (Math.abs(rgba[at + c]! - expected) > 1)
                        off.push(`${x},${y}[${c}]: ${rgba[at + c]} != ${expected}`);
                }
            }
        }
        expect(partial).toBeGreaterThan(SIZE * SIZE * 0.5);
        expect(off.slice(0, 8)).toEqual([]);
    }

    await check();
    // The bottom of a stack whose upper camera faces away renders as the lone camera does.
    const away = world.create();
    world.add(away, Transform, { translation: [0, 0, 5, 0], rotation: [0, 1, 0, 0] });
    world.add(away, Camera, { order: 1 });
    world.add(away, StandardRenderer);
    attachTexture(world, away, target);
    await check();
    for (const eid of [cube, away, camera]) world.destroy(eid);
    world.step(0);
});
