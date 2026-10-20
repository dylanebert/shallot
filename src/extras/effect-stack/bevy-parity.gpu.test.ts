import { test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { EffectStackPlugin } from ".";
import { cases, compare, gold, parityCapture } from "./parity.fixture";

const capture = parityCapture();
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [EffectStackPlugin, capture.plugin] },
]);
for (const c of cases)
    test(`${c.name} matches Bevy's HDR gold within one half-float step`, async () => {
        compare(await capture.render(apps()[0]!.world, c), gold(c), c.name);
    });
