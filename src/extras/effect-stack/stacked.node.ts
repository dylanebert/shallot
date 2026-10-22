import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Vignette as LegacyVignette, VignettePlugin } from "@dylanebert/shallot/vignette";
import { gpuApps, gpuScene } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    EffectPasses,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "../../core/rendering";
import { ImageBindings } from "../../core/rendering/image";
import { tonemappingStateKey } from "../../core/rendering/tonemapping-state";
import { offscreenTexture } from "../../core/rendering/view";
import { Transform } from "../../core/transform";
import { precompile, type World } from "../../engine";
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

import { bevyPipeline, renderBevy } from "./bevy.fixture";
import { cases, compare, inputTexture, parityCapture, texture } from "./parity.fixture";

setDefaultTimeout(TIMEOUT.node);

const parity = parityCapture();
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            ...DEFAULT_PLUGINS,
            VignettePlugin,
            parity.plugin,
            {
                name: "EffectStackReference",
                dependencies: [EffectStackPlugin],
                initialize: initializeReference,
            },
        ],
    },
]);
const alphaScenes = [false, true].map((stacked) =>
    gpuScene(
        import.meta.path,
        { defaults: false, plugins: [...DEFAULT_PLUGINS, EffectStackPlugin] },
        (world) => initializeAlphaScene(world, stacked),
    ),
);

let referencePipeline: GPURenderPipeline;
let referenceInput: GPUTexture;
let referenceOutput: GPUTexture;
async function initializeReference(world: World) {
    referencePipeline = await bevyPipeline(world);
    precompile(world, "bevy-effect-stack", () => [referencePipeline]);
    referenceInput = inputTexture(world);
    referenceOutput = texture(world, "rgba16float");
}

// Full default-renderer composition shares this Node file's prepared world. Compare on
// one adapter: bilinear weights differ between adapters even for Bevy's own shader.
for (const c of cases) {
    test(`${c.name} matches live Bevy within one half-float step`, async () => {
        const world = subjects()[0].world;
        const reference = await renderBevy(
            world,
            referencePipeline,
            c,
            referenceInput,
            referenceOutput,
        );
        compare(await parity.render(world, c), reference, c.name);
    });
}

test("the deprecated vignette imports still shade only what their stacked camera drew", async () => {
    expect(VignettePlugin).toBe(EffectStackPlugin);
    const { world } = subjects()[0];
    const { lower, upper, cube } = stackedScene(world, 1);
    const plain = pixel(await stackedFrame(world, lower), SIZE / 2, SIZE / 2 + 6);
    world.add(upper, LegacyVignette, { radius: 0.2 });
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

for (const inactive of [Vignette, LensDistortion, ChromaticAberration]) {
    test(`inactive ${inactive === Vignette ? "vignette" : inactive === LensDistortion ? "distortion" : "aberration"} stays absent beside another active effect`, async () => {
        const { world } = subjects()[0];
        const { lower, upper, cube } = stackedScene(world, 4, 128);
        const companion = inactive === Vignette ? LensDistortion : Vignette;
        world.add(upper, companion);
        const baseline = await stackedFrame(world, lower);
        world.add(upper, inactive);
        for (const intensity of [
            0,
            1e-4,
            0.00005,
            ...(inactive === LensDistortion ? [-1e-4] : [-0.5]),
        ]) {
            world.storage(inactive).intensity.set(upper, intensity);
            expect(await stackedFrame(world, lower)).toEqual(baseline);
        }
        for (const eid of [cube, upper, lower]) world.destroy(eid);
        world.step(0);
    });
}

for (const effect of [LensDistortion, ChromaticAberration]) {
    test(`${effect === LensDistortion ? "distortion" : "aberration"} carries displaced edge coverage and bounds premultiplied RGB by alpha`, async () => {
        const { world } = subjects()[0];
        const size = 128;
        const { lower, upper, cube } = stackedScene(world, 4, size);
        world.remove(cube, MeshMaterial);
        world.add(
            cube,
            MeshMaterial,
            world
                .resource(Materials)
                .add(StandardMaterial({ baseColor: [1, 1, 1, 1], unlit: true })),
        );
        const intensity = effect === LensDistortion ? 4 : 0.2;
        world.add(upper, effect, { intensity });
        await stackedFrame(world, lower);
        const targets = world
            .resource(tonemappingStateKey)
            .targets.get(world.resource(Views).get(upper)!)!;
        // Read the actual HDR pass output, before tonemapping and lower-camera compositing.
        targets.hdr[0].texture.destroy();
        const output = world.gpu.device.createTexture({
            size: [size, size],
            format: "rgba16float",
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });
        targets.hdr[0] = { texture: output, view: output.createView() };
        await stackedFrame(world, lower);
        const source = new Uint16Array(
            (await probeTexture(world, offscreenTexture(world, upper)!)).bytes as ArrayBuffer,
        );
        const result = new Uint16Array((await probeTexture(world, output)).bytes as ArrayBuffer);
        const alphaAt = (x: number, y: number) =>
            half(
                source[
                    (Math.max(0, Math.min(size - 1, y)) * size +
                        Math.max(0, Math.min(size - 1, x))) *
                        4 +
                        3
                ]!,
            );
        const sample = (u: number, v: number) => {
            const x = u * size - 0.5;
            const y = v * size - 0.5;
            const ix = Math.floor(x);
            const iy = Math.floor(y);
            const fx = x - ix;
            const fy = y - iy;
            return (
                (alphaAt(ix, iy) * (1 - fx) + alphaAt(ix + 1, iy) * fx) * (1 - fy) +
                (alphaAt(ix, iy + 1) * (1 - fx) + alphaAt(ix + 1, iy + 1) * fx) * fy
            );
        };
        let displacedEdges = 0;
        for (let y = 0; y < size; y++)
            for (let x = 0; x < size; x++) {
                const u = (x + 0.5) / size;
                const v = (y + 0.5) / size;
                let expected: number;
                if (effect === LensDistortion) {
                    const dx = u - 0.5;
                    const dy = v - 0.5;
                    const radius = Math.max(Math.hypot(dx, dy), 1e-6);
                    const k1 = (intensity * (Math.abs(dx) + Math.abs(dy))) / radius;
                    const factor = 1 + k1 * radius * radius;
                    expected = sample(
                        Math.max(0, Math.min(1, dx * factor + 0.5)),
                        Math.max(0, Math.min(1, dy * factor + 0.5)),
                    );
                } else {
                    const count = Math.min(
                        Math.ceil(Math.hypot(0.5 - u, 0.5 - v) * intensity * size),
                        8,
                    );
                    if (count <= 1) continue;
                    const sums = [0, 0, 0];
                    const weights = [0, 0, 0];
                    for (let i = 0; i < count; i++) {
                        const t = (i + 0.5) / count;
                        const alpha = sample(
                            u + (0.5 - u) * intensity * t,
                            v + (0.5 - v) * intensity * t,
                        );
                        // Bilinear sampling across Bevy's three-texel RGB lookup, half a texel inset.
                        const lut = [
                            Math.max(0, 1 - 2 * t),
                            1 - Math.abs(2 * t - 1),
                            Math.max(0, 2 * t - 1),
                        ];
                        for (let c = 0; c < 3; c++) {
                            sums[c]! += alpha * lut[c]!;
                            weights[c]! += lut[c]!;
                        }
                    }
                    expected = Math.max(...sums.map((sum, c) => sum / weights[c]!));
                }
                const at = (y * size + x) * 4;
                const alpha = half(result[at + 3]!);
                const rgb = [0, 1, 2].map((c) => half(result[at + c]!));
                expect(Math.max(...rgb)).toBeLessThanOrEqual(alpha + 0.001);
                if (
                    expected > 0.05 &&
                    expected < 0.95 &&
                    Math.abs(expected - alphaAt(x, y)) > 0.15
                ) {
                    displacedEdges++;
                    expect(alpha).toBeCloseTo(expected, 2);
                    expect(Math.abs(alpha - alphaAt(x, y))).toBeGreaterThan(0.14);
                    expect(Math.max(...rgb)).toBeGreaterThan(0.04);
                }
            }
        expect(displacedEdges).toBeGreaterThan(0);
        for (const eid of [cube, upper, lower]) world.destroy(eid);
        world.step(0);
    });
}

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

function initializeAlphaScene(world: World, stacked: boolean) {
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

    const entities = [cube, camera];
    if (stacked) {
        // The upper camera faces away, leaving the bottom camera's image unchanged.
        const away = world.create();
        world.add(away, Transform, { translation: [0, 0, 5, 0], rotation: [0, 1, 0, 0] });
        world.add(away, Camera, { order: 1 });
        world.add(away, StandardRenderer);
        attachTexture(world, away, target);
        entities.push(away);
    }
    return { camera, entities };
}
test("a lone or bottom camera's white vignette mixes its scene by the unscaled formula where AlphaToCoverage leaves alpha below 1", async () => {
    for (const subject of alphaScenes) {
        const {
            world,
            scene: { camera, entities },
        } = subject();
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
        for (const eid of entities) world.destroy(eid);
        world.step(0);
    }
});
