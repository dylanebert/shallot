import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    captureTexture,
    detachView,
    EffectPasses,
    TonemappingSystem,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { type Plugin, precompile, type World } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { EffectStackPlugin, Vignette } from "../effect-stack";
import { compare } from "../effect-stack/parity.fixture";
import { Bloom, BloomCompositeMode, BloomPlugin } from ".";
import { bevyPipelines, renderBevy } from "./bevy.fixture";
import { blendFactor, stateKey } from "./state";

setDefaultTimeout(TIMEOUT.gpu);
const width = 64;
const height = 32;
let eid = -1;
let input: GPUTexture;
let output: GPUTexture;
let inputView: GPUTextureView;
let outputView: GPUTextureView;
const capture: Plugin = {
    name: "BloomCapture",
    dependencies: [BloomPlugin, EffectStackPlugin],
    initialize,
    systems: [
        {
            name: "bloom-capture",
            group: "draw",
            after: [BloomPlugin.systems![0]!, EffectStackPlugin.systems![0]!],
            before: [TonemappingSystem],
            update(world) {
                const pass = world.resource(EffectPasses).get(eid)?.before[0];
                if (pass) pass(world, eid, world.resource(Views).get(eid)!, inputView, outputView);
            },
        },
    ],
};
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [EffectStackPlugin, BloomPlugin, capture] },
]);
let reference: Awaited<ReturnType<typeof bevyPipelines>>;
let referenceOutput: GPUTexture;
function texture(world: World) {
    const texture = world.gpu.device.createTexture({
        size: [width, height],
        format: "rgba16float",
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.COPY_DST,
    });
    world.own(texture);
    return texture;
}
const bits = async (world: World, t: GPUTexture) =>
    new Uint16Array((await probeTexture(world, t)).bytes as ArrayBuffer);
async function initialize(world: World) {
    reference = await bevyPipelines(world);
    precompile(world, "bevy-bloom", () => [reference.first, reference.down, reference.up]);
    input = texture(world);
    output = texture(world);
    referenceOutput = texture(world);
    inputView = input.createView();
    outputView = output.createView();
    const data = new Uint16Array(width * height * 4);
    for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
            const at = (y * width + x) * 4;
            const bright = Math.abs(x - 32) < 4 && Math.abs(y - 16) < 4;
            data.set(
                bright ? [0x5000, 0x4800, 0x4400, 0x3c00] : [0x3000, 0x2c00, 0x2800, 0x3c00],
                at,
            );
        }
    world.gpu.device.queue.writeTexture({ texture: input }, data, { bytesPerRow: width * 8 }, [
        width,
        height,
    ]);
    eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera, { clearColor: 0x808080 });
    attachTexture(world, eid, { width, height });
    world.add(eid, Bloom);
}

test("NATURAL chain matches vendored Bevy driven by bloom/mod.rs within one half-float step", async () => {
    const world = subjects()[0].world;
    await renderBevy(world, reference, input, referenceOutput);
    world.step(0);
    compare(await bits(world, output), await bits(world, referenceOutput), "bloom NATURAL");
});

test("blend factors equal compute_blend_factor at NATURAL and SCREEN_BLUR", () => {
    for (const blur of [false, true]) {
        const settings = {
            intensity: blur ? 1 : 0.15,
            lowFrequencyBoost: blur ? 0 : 0.7,
            lowFrequencyBoostCurvature: blur ? 0 : 0.95,
            highPassFrequency: blur ? 1 / 3 : 1,
            compositeMode: BloomCompositeMode.EnergyConserving,
        };
        for (let mip = 0; mip < 8; mip++) {
            const angle = mip / 7;
            const boost =
                (1 - (1 - angle) ** (blur ? 1 : 20)) * (blur ? 0 : 0.7) * (blur ? 0 : 0.85);
            const cutoff =
                1 -
                Math.max(
                    0,
                    Math.min(1, (angle - settings.highPassFrequency) / settings.highPassFrequency),
                );
            expect(blendFactor(settings, mip, 7)).toBeCloseTo(
                (settings.intensity + boost) * cutoff,
                14,
            );
        }
    }
});

test("zero intensity leaves the frame unchanged and removal unregisters the pass", async () => {
    const world = subjects()[0].world;
    world.remove(eid, Bloom);
    world.step(0);
    const plain = (await captureTexture(world, eid)).rgba;
    world.add(eid, Bloom, { intensity: 0 });
    world.step(0);
    expect(world.resource(EffectPasses).get(eid)?.before.length ?? 0).toBe(0);
    expect((await captureTexture(world, eid)).rgba).toEqual(plain);
    world.storage(Bloom).intensity.set(eid, 0.15);
    world.step(0);
    expect(world.resource(EffectPasses).get(eid)?.before.length).toBe(1);
    world.remove(eid, Bloom);
    world.step(0);
    expect(world.resource(EffectPasses).get(eid)?.before.length ?? 0).toBe(0);
    world.add(eid, Bloom);
});

test("both composite modes, a threshold and non-uniform scale change the HDR frame", async () => {
    const world = subjects()[0].world;
    world.step(0);
    const natural = await bits(world, output);
    expect(natural).not.toEqual(await bits(world, input));
    world.storage(Bloom).compositeMode.set(eid, BloomCompositeMode.Additive);
    world.step(0);
    const additive = await bits(world, output);
    expect(additive).not.toEqual(natural);
    expect(additive).not.toEqual(await bits(world, input));
    world.storage(Bloom).threshold.set(eid, 4);
    world.storage(Bloom).thresholdSoftness.set(eid, 0.5);
    world.step(0);
    expect(await bits(world, output)).not.toEqual(additive);
    world.remove(eid, Bloom);
    world.add(eid, Bloom, { scale: [4, 1] });
    world.step(0);
    expect(await bits(world, output)).not.toEqual(natural);
    world.remove(eid, Bloom);
    world.add(eid, Bloom);
});

test("bloom stays first when EffectStackPlugin was installed first, including after reactivation", () => {
    const world = subjects()[0].world;
    world.add(eid, Vignette);
    world.step(0);
    const before = world.resource(EffectPasses).get(eid)!.before;
    expect(before.length).toBe(2);
    const bloom = before[0];
    const stack = before[1];
    world.storage(Bloom).intensity.set(eid, 0);
    world.step(0);
    expect(before).toEqual([stack]);
    world.storage(Bloom).intensity.set(eid, 0.15);
    world.step(0);
    expect(before[1]).toBe(stack);
    expect(before[0]).not.toBe(stack);
    expect(bloom).not.toBe(stack);
    world.remove(eid, Vignette);
});

test("a stacked camera keeps input alpha and never scatters into uncovered pixels", async () => {
    const world = subjects()[0].world;
    // Exercise the EffectPass contract directly: premultiplied partial and zero coverage.
    const data = new Uint16Array(width * height * 4);
    for (let i = 0; i < width * height; i++)
        data.set(i % width < width / 2 ? [0x4800, 0x4400, 0x4000, 0x3800] : [0, 0, 0, 0], i * 4);
    world.gpu.device.queue.writeTexture({ texture: input }, data, { bytesPerRow: width * 8 }, [
        width,
        height,
    ]);
    const lower = world.create();
    world.add(lower, Transform);
    world.add(lower, Camera, { order: -1 });
    attachTexture(world, lower, world.resource(Views).get(eid)!.texture!);
    world.step(0);
    expect(world.resource(Views).get(eid)!.sortedIndexForTarget).toBe(1);
    const result = await bits(world, output);
    for (let i = 0; i < data.length; i += 4) {
        expect(result[i + 3]).toBe(data[i + 3]);
        if (data[i + 3] === 0) expect(Array.from(result.subarray(i, i + 4))).toEqual([0, 0, 0, 0]);
    }
    world.destroy(lower);
});

test("steady draws reuse the chain and bind groups; an aspect resize reallocates Bevy's sizes", () => {
    const world = subjects()[0].world;
    world.step(0);
    world.step(0);
    const camera = world.resource(stateKey).cameras.get(eid)!;
    const chain = camera.chain!;
    const groups = camera.groups;
    const firstGroup = groups.get(inputView);
    expect(chain.texture.width).toBe(1024);
    expect(chain.texture.height).toBe(512);
    expect(chain.texture.mipLevelCount).toBe(8);
    world.step(0);
    world.step(0);
    expect(camera.chain).toBe(chain);
    expect(camera.groups).toBe(groups);
    expect(camera.groups.get(inputView)).toBe(firstGroup);
    detachView(world, eid);
    attachTexture(world, eid, { width: 96, height: 32 });
    world.step(0);
    expect(camera.chain).not.toBe(chain);
    expect(camera.chain!.texture.width).toBe(1536);
    expect(camera.chain!.texture.height).toBe(512);
    expect(camera.chain!.texture.mipLevelCount).toBe(8);
});
