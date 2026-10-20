import { test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { EffectStackPlugin } from ".";
import { bevyPipeline, renderBevy } from "./bevy.fixture";
import { cases, compare, gold, inputTexture, parityCapture, texture } from "./parity.fixture";

const capture = parityCapture();
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [EffectStackPlugin, capture.plugin] },
]);
test("Bevy b3bdd75be shaders reproduce the gold and Shallot agrees within one half-float step", async () => {
    const world = apps()[0]!.world;
    const pipeline = await bevyPipeline(world);
    const input = inputTexture(world);
    const output = texture(world, "rgba16float");
    for (const c of cases) {
        const reference = await renderBevy(world, pipeline, c, input, output);
        compare(reference, gold(c), `${c.name} Bevy`);
        compare(await capture.render(world, c), gold(c), `${c.name} Shallot`);
    }
});
