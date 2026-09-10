// joints-bridge — stage-4 gym twin: the upstream `Bridge` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer.

import goldJson from "../../../../src/standard/physics/samples/joints-bridge.json";
import { register } from "../gym";
import { buildBridge } from "../physics-bridge";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildBridge,
    }),
);
