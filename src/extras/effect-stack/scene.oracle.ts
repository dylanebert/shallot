import { expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { Scene, ScenePlugin, setEffect } from "../../../examples/post-processing/src/scene";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { attachTexture, EffectPasses, TonemappingSystem, Views } from "../../core/rendering";
import { offscreenTexture } from "../../core/rendering/view";
import type { Plugin } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "../../standard";
import { EffectStackPlugin } from ".";
import { bevyPipeline, renderBevy } from "./bevy.fixture";
import { cases, compare, texture } from "./parity.fixture";

const width = 1280,
    height = 720;
let output: GPUTexture | undefined;
const capture: Plugin = {
    name: "SceneCapture",
    systems: [
        {
            group: "draw",
            after: [EffectStackPlugin.systems![0]!],
            before: [TonemappingSystem],
            update(world) {
                if (!output) return;
                const camera = world.resource(Scene).camera;
                world.resource(EffectPasses).get(camera)!.before[0]!(
                    world,
                    camera,
                    world.resource(Views).get(camera)!,
                    offscreenTexture(world, camera)!.createView(),
                    output.createView(),
                );
            },
        },
    ],
};
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, ScenePlugin, capture] },
]);
function ppm(path: string, bits: Uint16Array): void {
    const pixels = new Uint8Array(width * height * 3);
    for (let i = 0; i < pixels.length; i++) {
        const b = bits[Math.floor(i / 3) * 4 + (i % 3)]!;
        const exponent = (b >> 10) & 31;
        const value =
            exponent === 0 ? (b & 1023) * 2 ** -24 : (1 + (b & 1023) / 1024) * 2 ** (exponent - 15);
        const mapped = value / (1 + value);
        pixels[i] = Math.round(
            255 * (mapped <= 0.0031308 ? mapped * 12.92 : 1.055 * mapped ** (1 / 2.4) - 0.055),
        );
    }
    writeFileSync(path, Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), pixels]));
}
test("the example's pre-tonemapping scene through Bevy and Shallot at default lens settings", async () => {
    const world = apps()[0]!.world;
    const camera = world.resource(Scene).camera;
    for (const i of [1, 2]) setEffect(world, i, false);
    attachTexture(world, camera, { width, height });
    world.step(0);
    output = texture(world, "rgba16float", width, height);
    world.step(0);
    const actual = new Uint16Array((await probeTexture(world, output)).bytes as ArrayBuffer);
    const input = offscreenTexture(world, camera)!;
    const reference = await renderBevy(
        world,
        await bevyPipeline(world),
        cases[0]!,
        input,
        texture(world, "rgba16float", width, height),
    );
    compare(actual, reference, "example lens-default");
    const directory = "test-results/effect-stack";
    mkdirSync(directory, { recursive: true });
    ppm(`${directory}/shallot-lens-default.ppm`, actual);
    ppm(`${directory}/bevy-lens-default.ppm`, reference);
    const source = new Uint16Array((await probeTexture(world, input)).bytes as ArrayBuffer);
    ppm(`${directory}/undistorted.ppm`, source);
    // At x=1/4, the default lens maps the bottom edge outside v=1.
    // The resulting bottom rows sample the same clamped source row (the near-floor smear).
    const x = width / 4;
    let identical = 0;
    for (let y = height - 20; y < height - 1; y++) {
        if (
            [0, 1, 2].every(
                (c) =>
                    reference[(y * width + x) * 4 + c] === reference[((y + 1) * width + x) * 4 + c],
            )
        )
            identical++;
    }
    expect(identical).toBeGreaterThan(10);
    console.info(
        `Bevy near-floor: ${identical}/19 adjacent bottom-row samples identical at x=${x}; captures in ${directory} (identical Reinhard + sRGB visualization).`,
    );
});
